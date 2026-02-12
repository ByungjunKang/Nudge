import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10

def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float, device=None):
    """
    Returns: [n_mels, n_fft//2 + 1]
    """
    device = device or torch.device("cpu")
    n_freqs = n_fft // 2 + 1

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mels = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz = torch.tensor([mel_to_hz(m.item()) for m in mels], device=device)

    # FFT bin frequencies
    freqs = torch.linspace(0, sr / 2, n_freqs, device=device)

    fb = torch.zeros(n_mels, n_freqs, device=device)
    for i in range(n_mels):
        f_left, f_center, f_right = hz[i], hz[i+1], hz[i+2]

        left_slope = (freqs - f_left) / (f_center - f_left + EPS)
        right_slope = (f_right - freqs) / (f_right - f_center + EPS)

        fb[i] = torch.clamp(torch.min(left_slope, right_slope), min=0.0)

    # Normalize (optional; many pipelines omit or use slaney norm)
    fb = fb / (fb.sum(dim=1, keepdim=True) + EPS)
    return fb

class LogMelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=256, win_ms=25, hop_ms=10, n_mels=40, fmin=50.0, fmax=7600.0):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_len = int(sr * win_ms / 1000)
        self.hop_len = int(sr * hop_ms / 1000)
        self.n_mels = n_mels

        window = torch.hann_window(self.win_len)
        self.register_buffer("window", window)

        fb = create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_fb", fb)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] float32 in [-1, 1]
        returns log-mel: [B, n_mels, n_frames]
        """
        # STFT -> magnitude^2 (power)
        X = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len,
            window=self.window, center=False, return_complex=True
        )  # [B, n_freqs, n_frames]
        P = (X.real**2 + X.imag**2)  # power

        # mel
        mel = torch.matmul(self.mel_fb, P)  # [B, n_mels, n_frames]
        logmel = torch.log(mel + 1e-6)
        return logmel


class DSConv1dBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        pad = (k - 1) * dilation  # causal padding: pad left only
        self.pad = pad
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=True)
        self.ln = nn.GroupNorm(1, ch)  # lightweight "layernorm-ish"
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        # causal pad left
        x_in = x
        x = F.pad(x, (self.pad, 0))
        x = self.dw(x)
        x = self.pw(x)
        x = self.ln(x)
        x = F.relu(x)
        x = self.do(x)
        return x + x_in  # residual


class TinyMelTCN(nn.Module):
    def __init__(self, n_mels=40, ch=64, layers=(1, 2, 4), k=5, dropout=0.05):
        super().__init__()
        self.in_proj = nn.Conv1d(n_mels, ch, kernel_size=1)

        self.blocks = nn.ModuleList([
            DSConv1dBlock(ch, k=k, dilation=d, dropout=dropout) for d in layers
        ])

        self.out_proj = nn.Conv1d(ch, 3, kernel_size=1)

    def forward(self, mel):
        """
        mel: [B, n_mels, T] (e.g., T=100 for 1s context @10ms hop)
        returns probs: [B, 3]
        """
        x = self.in_proj(mel)
        for b in self.blocks:
            x = b(x)

        logits = self.out_proj(x)  # [B, 3, T]

        # 마지막 100ms(=10프레임 if hop=10ms) 평균으로 100ms decision
        last_k = min(10, logits.shape[-1])
        logits_100ms = logits[:, :, -last_k:].mean(dim=-1)  # [B, 3]
        p = F.softmax(logits_100ms, dim=1)
        return p


class StreamEnergyEstimator:
    def __init__(self, model: nn.Module, fe: LogMelExtractor,
                 sr=16000, context_sec=1.0, step_ms=100):
        self.model = model.eval()
        self.fe = fe

        self.sr = sr
        self.context_len = int(sr * context_sec)
        self.step_len = int(sr * step_ms / 1000)

        self.buf = torch.zeros(self.context_len, dtype=torch.float32)
        self.buf_filled = 0
        self.step_acc = 0

    @torch.no_grad()
    def push_audio(self, x_chunk: torch.Tensor):
        """
        x_chunk: [N] float32 mono [-1,1]
        yields list of dict outputs every 100ms
        """
        outs = []
        n = x_chunk.numel()
        idx = 0

        while idx < n:
            take = min(n - idx, self.context_len)  # safe
            chunk = x_chunk[idx: idx + take]
            idx += take

            # ring buffer shift-left + append
            L = chunk.numel()
            if L >= self.context_len:
                self.buf[:] = chunk[-self.context_len:]
                self.buf_filled = self.context_len
            else:
                self.buf = torch.roll(self.buf, shifts=-L, dims=0)
                self.buf[-L:] = chunk
                self.buf_filled = min(self.context_len, self.buf_filled + L)

            self.step_acc += L

            while self.step_acc >= self.step_len:
                self.step_acc -= self.step_len
                if self.buf_filled < self.context_len:
                    continue  # 아직 1초 컨텍스트 부족

                x = self.buf.unsqueeze(0)  # [1, T]
                mel = self.fe(x)          # [1, 40, frames] frames≈(1s/10ms)=100
                p = self.model(mel)       # [1,3]

                # 최근 100ms 에너지(절대크기)는 raw time-domain으로 계산
                last_100ms = self.buf[-self.step_len:]
                E_total = float((last_100ms * last_100ms).mean().item())

                pv, pm, po = p[0].tolist()
                outs.append({
                    "p_voice": pv, "p_music": pm, "p_other": po,
                    "E_total": E_total,
                    "E_voice": pv * E_total,
                    "E_music": pm * E_total,
                    "E_other": po * E_total,
                })

        return outs


class PCEN(nn.Module):
    def __init__(self, n_mels=40, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
        super().__init__()
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps
        self.register_buffer("M", torch.zeros(1, n_mels, 1))  # running smoother

    @torch.no_grad()
    def forward(self, mel_power: torch.Tensor):
        """
        mel_power: [B, n_mels, T] (power, not log)
        """
        B, C, T = mel_power.shape
        if self.M.shape[0] != B:
            self.M = self.M[:1].repeat(B, 1, 1)

        out = []
        M = self.M
        for t in range(T):
            x = mel_power[:, :, t:t+1]
            M = (1 - self.s) * M + self.s * x
            y = (x / (self.eps + M)**self.alpha + self.delta)**self.r - self.delta**self.r
            out.append(y)
        self.M = M.detach()
        return torch.cat(out, dim=-1)
