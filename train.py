import torch
import torch.nn as nn

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
