import torch
import torch.nn as nn

import torch.nn.functional as F
import math

# ====== Config ======
SR = 48000
N_FFT = 1536          # 32ms
WIN_MS = 32
HOP_MS = 20
HOP_LEN = 960         # 20ms at 48k
GROUP_FRAMES = 5      # 5 frames = 100ms
SEG_LEN = HOP_LEN * GROUP_FRAMES   # 4800 samples = 100ms
SEG_HOP = SEG_LEN                  # non-overlap 100ms stride

CTX_SEC = 1.0
CTX_FRAMES = int(CTX_SEC / (HOP_MS / 1000.0))  # 1s / 20ms = 50 frames
STEP_FRAMES = GROUP_FRAMES                     # 5 frames step = 100ms
K = CTX_FRAMES // STEP_FRAMES                  # 50/5=10 segments in context
LABEL_OFFSET = K - 1                           # 9

MIN_TOTAL_DBFS = -60.0
SMOOTHING = 1e-4

EPS = 1e-12

def _mel_frame_energy_from_wave(
    x: torch.Tensor,              # [B,T]
    n_fft: int,
    hop_len: int,
    win_len: int,
    window: torch.Tensor,         # [win_len]
    mel_fb: torch.Tensor,         # [n_mels, n_freq]  (n_freq = n_fft//2+1)
):
    """
    return:
      E_mel_frame: [B, F]  (mel-space energy per STFT frame)
    """
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        center=False,
        return_complex=True,
    )  # [B, n_freq, F]

    P = (X.real * X.real + X.imag * X.imag)  # [B, n_freq, F] power
    # mel power: [B, n_mels, F]
    mel_power = torch.matmul(mel_fb, P)      # mel_fb[n_mels,n_freq] @ P[B,n_freq,F] -> [B,n_mels,F]

    # mel-space energy per frame (sum over mel bins)
    E = mel_power.sum(dim=1)                 # [B, F]
    return E


def build_mel_energy_labels(
    padded_source: torch.Tensor,     # [B,3,T]  (voice/music/other)
    fe,                               # LogMelExtractor instance (must have .mel_fb and .window)
    n_fft: int = 1536,
    hop_len: int = 960,
    win_len: int = 1536,
    group_frames: int = 5,
    smoothing: float = 1e-4,
    min_total_dbfs: float = -60.0,
):
    """
    라벨을 "모델과 동일한 mel filterbank" 기반 energy ratio로 생성.

    returns:
      P_true: [B, N, 3]   soft ratio labels
      W     : [B, N]      0/1 mask
      E_seg : [B, 3, N]   (debug) mel-energy segment
    """
    B, C, T = padded_source.shape
    assert C == 3, "padded_source must be [B,3,T] (voice/music/other)"

    device = padded_source.device

    # feature extractor가 사용하는 mel filterbank / window를 그대로 재사용
    mel_fb = fe.mel_fb.to(device)     # [n_mels, n_freq]
    window = fe.window.to(device)     # [win_len]  (win_len must match)

    # --- vectorized over stems ---
    x = padded_source.reshape(B * 3, T)                 # [B*3,T]
    E_frame = _mel_frame_energy_from_wave(
        x, n_fft=n_fft, hop_len=hop_len, win_len=win_len,
        window=window, mel_fb=mel_fb
    )                                                   # [B*3,F]
    Fm = E_frame.shape[-1]
    E_frame = E_frame.reshape(B, 3, Fm)                 # [B,3,F]

    # --- group 5 frames -> 100ms segment energy ---
    n_seg = Fm // group_frames
    if n_seg <= 0:
        P_true = padded_source.new_zeros((B, 0, 3))
        W = padded_source.new_zeros((B, 0))
        E_seg = padded_source.new_zeros((B, 3, 0))
        return P_true, W, E_seg

    E_frame = E_frame[:, :, :n_seg * group_frames]
    # sum (or mean) over 5 frames; ratio는 둘 다 동일
    E_seg = E_frame.view(B, 3, n_seg, group_frames).sum(dim=-1)  # [B,3,N]

    # --- soft ratio labels ---
    S = E_seg.sum(dim=1, keepdim=True) + EPS
    P = (E_seg / S).permute(0, 2, 1).contiguous()                # [B,N,3]

    if smoothing and smoothing > 0:
        P = (1.0 - smoothing) * P + smoothing / 3.0
        P = P / (P.sum(dim=-1, keepdim=True) + EPS)

    # --- mask (silence/pad) ---
    Et = E_seg.sum(dim=1) + EPS
    Et_db = 10.0 * torch.log10(Et)
    W = (Et_db >= min_total_dbfs).float()                         # [B,N]

    return P, W, E_seg

import math
from torch.cuda.amp import autocast, GradScaler

def apply_warmup_mask(W: torch.Tensor, warmup_segments: int) -> torch.Tensor:
    if warmup_segments <= 0:
        return W
    W = W.clone()
    W[:, :min(warmup_segments, W.shape[1])] = 0.0
    return W

def pool_logits_by_frames(logits: torch.Tensor, group_frames: int = 5) -> torch.Tensor:
    # logits: [B,3,F] -> [B,3,N]
    B, C, Fm = logits.shape
    n_seg = Fm // group_frames
    if n_seg <= 0:
        return logits[:, :, :0]
    logits = logits[:, :, :n_seg * group_frames]
    return logits.view(B, C, n_seg, group_frames).mean(dim=-1)

def soft_ce_with_logits(logits: torch.Tensor, p_true: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    loss_per = -(p_true * logp).sum(dim=1)
    if weight is None:
        return loss_per.mean()
    denom = torch.clamp(weight.sum(), min=1.0)
    return (loss_per * weight).sum() / denom


def train_step_fullseq_mellabel(
    padded_mixture, padded_source, model, fe, optimizer, scaler, device,
    n_fft=1536, hop_len=960, win_len=1536, group_frames=5,
    smoothing=1e-4, min_total_dbfs=-60.0, grad_clip=5.0
):
    padded_mixture = padded_mixture.to(device, non_blocking=True)
    padded_source  = padded_source.to(device, non_blocking=True)

    # ---- MEL-space labels (same mel_fb as model feature extractor) ----
    P_true, W, _ = build_mel_energy_labels(
        padded_source,
        fe=fe,
        n_fft=n_fft,
        hop_len=hop_len,
        win_len=win_len,
        group_frames=group_frames,
        smoothing=smoothing,
        min_total_dbfs=min_total_dbfs,
    )  # P_true: [B,N,3], W:[B,N]
    if P_true.shape[1] == 0:
        return None

    # warmup (receptive field) masking
    rf_frames = model.receptive_field_frames()
    warmup_segments = math.ceil(max(0, rf_frames - 1) / group_frames)
    W = apply_warmup_mask(W, warmup_segments)

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=(device.type == "cuda")):
        # feature from mixture (no grad)
        with torch.no_grad():
            mel = fe(padded_mixture)      # [B,n_mels,F]
        logits_frame = model(mel)         # [B,3,F]

        logits_100ms = pool_logits_by_frames(logits_frame, group_frames=group_frames)  # [B,3,Nm]
        if logits_100ms.shape[-1] == 0:
            return None

        Nm = logits_100ms.shape[-1]
        Ns = P_true.shape[1]
        N = min(Nm, Ns)
        if N <= 0:
            return None

        logits_100ms = logits_100ms[:, :, :N]  # [B,3,N]
        y = P_true[:, :N, :]                   # [B,N,3]
        w = W[:, :N]                           # [B,N]

        # flatten
        B = logits_100ms.shape[0]
        logits_flat = logits_100ms.permute(0, 2, 1).reshape(B * N, 3)
        y_flat = y.reshape(B * N, 3)
        w_flat = w.reshape(B * N)

        loss = soft_ce_with_logits(logits_flat, y_flat, weight=w_flat)

    # backward
    if scaler is not None and device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return float(loss.detach().cpu().item())


def train_one_step(padded_mixture, padded_source, model, fe, optimizer, device):
    """
    padded_mixture: [B,T]
    padded_source:  [B,3,T]  (voice/music/other)
    """
    padded_mixture = padded_mixture.to(device)
    padded_source = padded_source.to(device)

    # 1) Feature (log-mel) from mixture
    with torch.no_grad():  # feature는 보통 grad 불필요
        mel = fe(padded_mixture)  # [B, n_mels, F]

    B, n_mels, Fm = mel.shape

    # 2) Make context windows (1s context, 100ms stride)
    if Fm < CTX_FRAMES:
        return None  # 너무 짧은 샘플 배치면 skip (또는 padding 처리)

    win = make_context_windows(mel, ctx_frames=CTX_FRAMES, step_frames=STEP_FRAMES)  # [B, Nw, n_mels, 50]
    B, Nw, _, _ = win.shape

    # 3) Label energies from stems in 100ms segments
    E = segment_mean_square_energy(padded_source, seg_len=SEG_LEN, seg_hop=SEG_HOP)  # [B,3,Ns]
    B2, C, Ns = E.shape
    assert B2 == B and C == 3

    P_true = make_soft_ratio_labels_from_energy(E, smoothing=SMOOTHING)  # [B,Ns,3]
    W = make_weights_from_total_energy(E, min_total_dbfs=MIN_TOTAL_DBFS) # [B,Ns]

    # 4) Align labels with windows (offset by 9 segments)
    # window index j corresponds to label segment index (LABEL_OFFSET + j)
    max_Nw = min(Nw, Ns - LABEL_OFFSET)
    if max_Nw <= 0:
        return None

    win = win[:, :max_Nw]  # [B, max_Nw, n_mels, 50]
    y = P_true[:, LABEL_OFFSET:LABEL_OFFSET + max_Nw, :]  # [B, max_Nw, 3]
    w = W[:, LABEL_OFFSET:LABEL_OFFSET + max_Nw]          # [B, max_Nw]

    # 5) Forward
    x_in = win.reshape(B * max_Nw, n_mels, CTX_FRAMES)  # [B*max_Nw, n_mels, 50]
    p_pred = model(x_in)                                # [B*max_Nw, 3] (softmax probs)

    y = y.reshape(B * max_Nw, 3)
    w = w.reshape(B * max_Nw)

    # 6) Loss + backward
    loss = soft_cross_entropy(p_pred, y, weight=w)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()



from torch.utils.data import TensorDataset, DataLoader

def fit(padded_mixture, padded_source, epochs=5, batch_size=8, lr=1e-3, weight_decay=1e-3, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature extractor (config 반영)
    fe = LogMelExtractor(
        sr=SR,
        n_fft=N_FFT,
        win_ms=WIN_MS,
        hop_ms=HOP_MS,
        n_mels=40,
        fmin=50.0,
        fmax=24000.0
    ).to(device)

    # Per-frame model
    model = TinyMelTCNFrame(
        n_mels=40,
        ch=64,
        dilations=(1,2,4,8),  # ~1.2s RF @20ms hop
        k=5,
        dropout=0.05
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    ds = TensorDataset(padded_mixture, padded_source)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for mix_b, src_b in dl:
            loss_val = train_step_fullseq(mix_b, src_b, model, fe, optimizer, scaler, device)
            if loss_val is None:
                continue
            running += loss_val
            n += 1

        avg = running / max(1, n)
        print(f"[Epoch {ep}] loss={avg:.5f}")

    return model, fe
