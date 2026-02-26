import os
import math
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

EPS = 1e-12

# =========================
# 1) Audio utils
# =========================
def load_wav_mono(path: str, target_sr: int):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"SR mismatch: {path} has {sr}Hz, expected {target_sr}Hz. Resample first.")
    return x

# =========================
# 2) Mel filterbank + extractor (training-consistent)
# =========================
def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float, device):
    """
    Returns: [n_mels, n_fft//2 + 1] float32
    Note: row-normalized (sum=1) like earlier code.
    """
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mels = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=torch.float32)
    hz = torch.tensor([mel_to_hz(m.item()) for m in mels], device=device, dtype=torch.float32)
    freqs = torch.linspace(0, sr / 2, n_freqs, device=device, dtype=torch.float32)

    fb = torch.zeros(n_mels, n_freqs, device=device, dtype=torch.float32)
    for i in range(n_mels):
        f_left, f_center, f_right = hz[i], hz[i + 1], hz[i + 2]
        left_slope = (freqs - f_left) / (f_center - f_left + 1e-9)
        right_slope = (f_right - freqs) / (f_right - f_center + 1e-9)
        fb[i] = torch.clamp(torch.min(left_slope, right_slope), min=0.0)

    fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-9)
    return fb

class LogMelExtractor(nn.Module):
    def __init__(self, sr, n_fft, win_len, hop_len, n_mels, fmin, fmax):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len

        window = torch.hann_window(win_len, dtype=torch.float32)
        self.register_buffer("window", window)

        fb = create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax, device=torch.device("cpu"))
        self.register_buffer("mel_fb", fb)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T] float32
        returns logmel: [B,n_mels,F]
        """
        window = self.window.to(device=x.device, dtype=x.dtype)
        mel_fb = self.mel_fb.to(device=x.device, dtype=x.dtype)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=window,
            center=False,
            return_complex=True,
        )  # [B,n_freq,F]
        P = (X.real * X.real + X.imag * X.imag)  # [B,n_freq,F]
        mel = torch.matmul(mel_fb, P)            # [B,n_mels,F]
        return torch.log(mel + 1e-6)

# =========================
# 3) Model (per-frame logits): TinyMelTCNFrame
#    NOTE: dilation list must match training!
# =========================
class DSConv1dBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=True)
        self.ln = nn.GroupNorm(1, ch)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x_in = x
        x = F.pad(x, (self.pad, 0))
        x = self.dw(x)
        x = self.pw(x)
        x = self.ln(x)
        x = F.relu(x)
        x = self.do(x)
        return x + x_in

class TinyMelTCNFrame(nn.Module):
    def __init__(self, n_mels=40, ch=64, dilations=(1, 2, 4, 8), k=5, dropout=0.05):
        super().__init__()
        self.n_mels = n_mels
        self.ch = ch
        self.dilations = tuple(dilations)
        self.k = k

        self.in_proj = nn.Conv1d(n_mels, ch, kernel_size=1)
        self.blocks = nn.ModuleList([DSConv1dBlock(ch, k=k, dilation=d, dropout=dropout) for d in self.dilations])
        self.out_proj = nn.Conv1d(ch, 3, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B,n_mels,F]
        returns logits: [B,3,F]
        """
        x = self.in_proj(mel)
        for b in self.blocks:
            x = b(x)
        return self.out_proj(x)

# =========================
# 4) Energy computation in MEL space (aligned to model grid)
# =========================
@torch.no_grad()
def mel_segment_energy_from_wave(x: torch.Tensor, fe: LogMelExtractor, group_frames: int):
    """
    x: [1,T] float32
    returns E_seg: [N] energy per 100ms segment in mel-space
    (STFT power -> mel power -> sum mel bins -> sum 5 frames)
    """
    window = fe.window.to(device=x.device, dtype=x.dtype)
    mel_fb = fe.mel_fb.to(device=x.device, dtype=x.dtype)

    X = torch.stft(
        x,
        n_fft=fe.n_fft,
        hop_length=fe.hop_len,
        win_length=fe.win_len,
        window=window,
        center=False,
        return_complex=True,
    )  # [1,n_freq,F]
    P = (X.real * X.real + X.imag * X.imag)             # [1,n_freq,F]
    mel_power = torch.matmul(mel_fb, P)                 # [1,n_mels,F]
    E_frame = mel_power.sum(dim=1).squeeze(0)           # [F]

    Fm = int(E_frame.shape[0])
    n_seg = Fm // group_frames
    if n_seg <= 0:
        return torch.empty((0,), device=x.device, dtype=x.dtype)

    E_frame = E_frame[:n_seg * group_frames]
    E_seg = E_frame.view(n_seg, group_frames).sum(dim=1)  # [N]
    return E_seg

@torch.no_grad()
def infer_pred_energy_db(wav_np: np.ndarray, model: nn.Module, fe: LogMelExtractor,
                         sr: int, hop_len: int, group_frames: int, device):
    """
    returns:
      t: [N] seconds (segment center)
      pred_db: dict voice/music/other -> [N] dB  (ratio * mix energy)
      pred_ratio: dict voice/music/other -> [N] ratio
    """
    x = torch.from_numpy(wav_np).to(device).unsqueeze(0)  # [1,T]

    mel = fe(x)                     # [1,n_mels,F]
    logits = model(mel)             # [1,3,F]

    Fm = logits.shape[-1]
    n_seg = Fm // group_frames
    if n_seg <= 0:
        raise ValueError("Audio too short: not enough frames to form one 100ms segment.")

    logits_seg = logits[:, :, :n_seg * group_frames].view(1, 3, n_seg, group_frames).mean(dim=-1)  # [1,3,N]
    p = torch.softmax(logits_seg, dim=1).squeeze(0)  # [3,N]

    # mix energy in mel space (same grid)
    E_mix = mel_segment_energy_from_wave(x, fe, group_frames=group_frames)  # [N]
    N = min(E_mix.numel(), p.shape[1])
    E_mix = E_mix[:N]
    p = p[:, :N]

    E_pred = p * E_mix.unsqueeze(0)  # [3,N]
    E_pred_db = 10.0 * torch.log10(E_pred + 1e-12)

    # time axis: segment duration = group_frames * hop_len / sr
    seg_dur = (group_frames * hop_len) / sr
    t = (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * seg_dur  # center time

    pred_db = {
        "voice": E_pred_db[0].detach().cpu().numpy(),
        "music": E_pred_db[1].detach().cpu().numpy(),
        "other": E_pred_db[2].detach().cpu().numpy(),
    }
    pred_ratio = {
        "voice": p[0].detach().cpu().numpy(),
        "music": p[1].detach().cpu().numpy(),
        "other": p[2].detach().cpu().numpy(),
    }
    return t.detach().cpu().numpy(), pred_db, pred_ratio

@torch.no_grad()
def infer_gt_energy_db_from_folder(gt_dir: str, fe: LogMelExtractor,
                                  sr: int, group_frames: int, device,
                                  mix_name="mix.wav",
                                  voice_name="mix_speech.wav",
                                  music_name="mix_music.wav",
                                  other_name="mix_others.wav"):
    """
    GT도 'ratio * mix energy'로 계산:
      ratio = E_stem / (E_v + E_m + E_o)  (mel-space, 100ms segments)
      E_gt = ratio * E_mix
    returns:
      t: [N] seconds
      gt_db: dict voice/music/other -> [N] dB
      gt_ratio: dict voice/music/other -> [N] ratio
    """
    mix = load_wav_mono(os.path.join(gt_dir, mix_name), target_sr=sr)
    v = load_wav_mono(os.path.join(gt_dir, voice_name), target_sr=sr)
    m = load_wav_mono(os.path.join(gt_dir, music_name), target_sr=sr)
    o = load_wav_mono(os.path.join(gt_dir, other_name), target_sr=sr)

    L = min(len(mix), len(v), len(m), len(o))
    mix, v, m, o = mix[:L], v[:L], m[:L], o[:L]

    mix_t = torch.from_numpy(mix).to(device).unsqueeze(0)
    v_t = torch.from_numpy(v).to(device).unsqueeze(0)
    m_t = torch.from_numpy(m).to(device).unsqueeze(0)
    o_t = torch.from_numpy(o).to(device).unsqueeze(0)

    E_mix = mel_segment_energy_from_wave(mix_t, fe, group_frames)
    Ev = mel_segment_energy_from_wave(v_t, fe, group_frames)
    Em = mel_segment_energy_from_wave(m_t, fe, group_frames)
    Eo = mel_segment_energy_from_wave(o_t, fe, group_frames)

    N = min(E_mix.numel(), Ev.numel(), Em.numel(), Eo.numel())
    E_mix, Ev, Em, Eo = E_mix[:N], Ev[:N], Em[:N], Eo[:N]

    S = Ev + Em + Eo + 1e-12
    pv, pm, po = Ev / S, Em / S, Eo / S

    Ev_gt = pv * E_mix
    Em_gt = pm * E_mix
    Eo_gt = po * E_mix

    gt_db = {
        "voice": (10.0 * torch.log10(Ev_gt + 1e-12)).detach().cpu().numpy(),
        "music": (10.0 * torch.log10(Em_gt + 1e-12)).detach().cpu().numpy(),
        "other": (10.0 * torch.log10(Eo_gt + 1e-12)).detach().cpu().numpy(),
    }
    gt_ratio = {
        "voice": pv.detach().cpu().numpy(),
        "music": pm.detach().cpu().numpy(),
        "other": po.detach().cpu().numpy(),
    }

    seg_dur = (group_frames * fe.hop_len) / sr
    t = (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * seg_dur
    return t.detach().cpu().numpy(), gt_db, gt_ratio

# =========================
# 5) Plotting
# =========================
def plot_series(t, series_db: dict, title: str):
    plt.figure(figsize=(12, 4))
    for k in ["voice", "music", "other"]:
        if k in series_db:
            plt.plot(t, series_db[k], label=k)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (dB)  [ratio × mix mel-energy]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_overlay(t_pred, pred_db, t_gt, gt_db, title="Pred vs GT (dB)"):
    plt.figure(figsize=(12, 4))
    for k in ["voice", "music", "other"]:
        if k in pred_db:
            plt.plot(t_pred, pred_db[k], label=f"pred-{k}")
        if k in gt_db:
            plt.plot(t_gt, gt_db[k], linestyle="--", label=f"gt-{k}")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (dB)  [ratio × mix mel-energy]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

# =========================
# 6) Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to model state_dict (.pt/.pth)")
    p.add_argument("--input_wav", type=str, required=True, help="path to input mixture wav (48kHz)")
    p.add_argument("--gt_dir", type=str, default=None, help="folder containing GT wavs (mix, mix_speech, mix_music, mix_others)")
    p.add_argument("--mix_name", type=str, default="mix.wav")
    p.add_argument("--voice_name", type=str, default="mix_speech.wav")
    p.add_argument("--music_name", type=str, default="mix_music.wav")
    p.add_argument("--other_name", type=str, default="mix_others.wav")

    # 반드시 학습 설정과 일치해야 함
    p.add_argument("--n_mels", type=int, default=40)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--dilations", type=str, default="1,2,4,8")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.05)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dilations = tuple(int(x.strip()) for x in args.dilations.split(",") if x.strip())

    # feature extractor
    fe = LogMelExtractor(
        sr=48000, n_fft=1536, win_len=1536, hop_len=960,
        n_mels=args.n_mels, fmin=50.0, fmax=24000.0
    ).to(device)

    # model
    model = TinyMelTCNFrame(
        n_mels=args.n_mels, ch=args.ch, dilations=dilations, k=args.k, dropout=args.dropout
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    wav = load_wav_mono(args.input_wav, target_sr=48000)

    # Pred
    t_pred, pred_db, _ = infer_pred_energy_db(
        wav, model, fe,
        sr=48000, hop_len=960, group_frames=5,
        device=device
    )
    plot_series(t_pred, pred_db, "Predicted energy (voice/music/other)")

    # GT
    if args.gt_dir is not None:
        t_gt, gt_db, _ = infer_gt_energy_db_from_folder(
            args.gt_dir, fe,
            sr=48000, group_frames=5, device=device,
            mix_name=args.mix_name,
            voice_name=args.voice_name,
            music_name=args.music_name,
            other_name=args.other_name,
        )
        plot_series(t_gt, gt_db, "Ground-truth energy (voice/music/other)")
        plot_overlay(t_pred, pred_db, t_gt, gt_db, "Pred vs GT energy (dB)")

if __name__ == "__main__":
    main()
