import math
import numpy as np

EPS = 1e-12

def _to_mono(x: np.ndarray) -> np.ndarray:
    # x: (T,) or (T,C)
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def frame_energy(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    x: (T,) float32 [-1,1]
    return: (N,) mean-square energy per frame
    """
    T = x.shape[0]
    if T < frame_len:
        return np.zeros((0,), dtype=np.float32)

    n = 1 + (T - frame_len) // hop_len
    E = np.empty((n,), dtype=np.float32)
    for i in range(n):
        s = i * hop_len
        f = x[s:s+frame_len]
        E[i] = np.mean(f * f, dtype=np.float64)
    return E

def soft_energy_ratio_labels(Ev: np.ndarray, Em: np.ndarray, Eo: np.ndarray,
                             smoothing: float = 1e-4) -> np.ndarray:
    """
    Ev, Em, Eo: (N,) energies
    return: P (N,3) soft labels where rows sum to 1
    """
    Ev = Ev.astype(np.float64)
    Em = Em.astype(np.float64)
    Eo = Eo.astype(np.float64)

    S = Ev + Em + Eo + EPS
    P = np.stack([Ev / S, Em / S, Eo / S], axis=1)  # (N,3)

    # label smoothing to avoid exact zeros
    if smoothing > 0:
        P = (1.0 - smoothing) * P + smoothing / 3.0
        P = P / (P.sum(axis=1, keepdims=True) + EPS)
    return P.astype(np.float32)

def build_100ms_labels_from_stems(voice: np.ndarray, music: np.ndarray, other: np.ndarray,
                                 sr: int,
                                 frame_ms: int = 100,
                                 hop_ms: int = 100,
                                 min_total_dbfs: float = -60.0,
                                 smoothing: float = 1e-4):
    """
    voice/music/other: float32 mono arrays (same sr), assumed time-aligned
    returns:
      P: (N,3) soft ratio labels (voice/music/other)
      W: (N,) weights (0 for ignore, >0 for train)
      meta: dict with energies etc.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms   / 1000)

    Ev = frame_energy(voice, frame_len, hop_len)
    Em = frame_energy(music, frame_len, hop_len)
    Eo = frame_energy(other, frame_len, hop_len)

    N = min(len(Ev), len(Em), len(Eo))
    Ev, Em, Eo = Ev[:N], Em[:N], Eo[:N]

    P = soft_energy_ratio_labels(Ev, Em, Eo, smoothing=smoothing)

    # 무음/너무 작은 구간 마스크: total energy가 너무 작으면 학습에서 제외
    Et = (Ev + Em + Eo).astype(np.float64)
    Et_db = 10.0 * np.log10(np.maximum(Et, EPS))  # dBFS-like (정규화에 따라 절대값은 달라질 수 있음)

    W = np.ones((N,), dtype=np.float32)
    W[Et_db < min_total_dbfs] = 0.0  # ignore

    meta = {
        "Ev": Ev, "Em": Em, "Eo": Eo,
        "Et": Et.astype(np.float32),
        "Et_db": Et_db.astype(np.float32),
    }
    return P, W, meta


import soundfile as sf

def load_wav_mono(path: str, target_sr: int = None) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = _to_mono(np.asarray(x, dtype=np.float32))
    if target_sr is not None and sr != target_sr:
        raise ValueError(f"Sample rate mismatch: {sr} != {target_sr}. Resample first.")
    return x, sr

def align_min_length(*signals: np.ndarray) -> list[np.ndarray]:
    L = min(s.shape[0] for s in signals)
    return [s[:L] for s in signals]

# Example:
# mix, sr = load_wav_mono("mix.wav")
# v, _ = load_wav_mono("voice.wav", target_sr=sr)
# m, _ = load_wav_mono("music.wav", target_sr=sr)
# o, _ = load_wav_mono("other.wav", target_sr=sr)
# mix, v, m, o = align_min_length(mix, v, m, o)


import torch
import torch.nn.functional as F

def soft_cross_entropy(p_pred: torch.Tensor, p_true: torch.Tensor, weight: torch.Tensor | None = None):
    """
    p_pred: (B,3) probabilities (after softmax)
    p_true: (B,3) soft labels (sum=1)
    weight: (B,) optional, 0이면 ignore
    """
    logp = torch.log(torch.clamp(p_pred, min=1e-8))
    loss_per = -(p_true * logp).sum(dim=1)  # (B,)

    if weight is None:
        return loss_per.mean()

    weight = weight.to(loss_per.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return (loss_per * weight).sum() / denom

def kl_div_loss(p_pred: torch.Tensor, p_true: torch.Tensor, weight: torch.Tensor | None = None):
    """
    KL(p_true || p_pred). Use p_pred.log() as input to kl_div.
    """
    logp = torch.log(torch.clamp(p_pred, min=1e-8))
    # kl_div returns per-element, we reduce manually for weighting
    kl_elem = F.kl_div(logp, p_true, reduction="none")  # (B,3)
    loss_per = kl_elem.sum(dim=1)  # (B,)

    if weight is None:
        return loss_per.mean()

    weight = weight.to(loss_per.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return (loss_per * weight).sum() / denom


def make_weights_from_total_energy(Et: np.ndarray, W_mask: np.ndarray,
                                   gamma: float = 0.5):
    """
    Et: (N,) total energy
    W_mask: (N,) 0/1 (ignore mask)
    gamma: 0~1; 0이면 모두 동일 가중, 1이면 에너지 비례
    """
    Et = Et.astype(np.float64)
    Et_norm = Et / (np.mean(Et[W_mask > 0]) + EPS)
    w = (Et_norm ** gamma).astype(np.float32)
    w *= W_mask.astype(np.float32)
    return w



def build_labels_from_wav_paths(mix_path, voice_path, music_path, other_path,
                                frame_ms=100, hop_ms=100,
                                min_total_dbfs=-60.0, smoothing=1e-4, gamma=0.5):
    mix, sr = load_wav_mono(mix_path)
    v, _ = load_wav_mono(voice_path, target_sr=sr)
    m, _ = load_wav_mono(music_path, target_sr=sr)
    o, _ = load_wav_mono(other_path, target_sr=sr)

    mix, v, m, o = align_min_length(mix, v, m, o)

    P, W_mask, meta = build_100ms_labels_from_stems(
        voice=v, music=m, other=o,
        sr=sr,
        frame_ms=frame_ms, hop_ms=hop_ms,
        min_total_dbfs=min_total_dbfs,
        smoothing=smoothing
    )
    W = make_weights_from_total_energy(meta["Et"], W_mask, gamma=gamma)
    return mix, P, W, meta, sr

# ===== training step 예시 =====
# model이 100ms마다 (B,3) p_pred 내는 구조라고 가정

def training_step_example(model, batch_mel, batch_P, batch_W):
    """
    batch_mel: (B, n_mels, Tctx)
    batch_P:   (B, 3) float32 soft labels
    batch_W:   (B,) float32 weights (0이면 ignore)
    """
    p_pred = model(batch_mel)  # already softmax probs (B,3)
    loss = soft_cross_entropy(p_pred, batch_P, weight=batch_W)
    return loss

