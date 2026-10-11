import numpy as np
from scipy.fft import dct, rfft, rfftfreq

EPS = 1e-12


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop
    shape = (n_frames, frame_len)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def frame_params(sr: int, frame_s: float = 0.032, hop_s: float = 0.010):
    frame_len = int(round(frame_s * sr))
    n_fft = 1 << int(np.ceil(np.log2(frame_len)))
    hop = max(1, int(round(hop_s * sr)))
    return frame_len, hop, n_fft


def stft_power(x: np.ndarray, sr: int, frame_s: float = 0.032, hop_s: float = 0.010):
    frame_len, hop, n_fft = frame_params(sr, frame_s, hop_s)
    frames = frame_signal(x, frame_len, hop)
    window = np.hanning(frame_len)
    spec = np.abs(rfft(frames * window, n=n_fft, axis=1)) ** 2
    freqs = rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(frames.shape[0]) * hop + frame_len / 2) / sr
    return spec, freqs, times, frames


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin=20.0, fmax=None):
    fmax = fmax or sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (c - lo + EPS)
        down = (hi - bins) / (hi - c + EPS)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def mfcc(power_spec: np.ndarray, sr: int, n_fft: int, n_mfcc: int = 13, n_mels: int = 40):
    fb = mel_filterbank(sr, n_fft, n_mels, fmax=min(8000.0, sr / 2))
    mel_energy = np.log(power_spec @ fb.T + EPS)
    return dct(mel_energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def spectral_features(power_spec: np.ndarray, freqs: np.ndarray) -> dict:
    mag = np.sqrt(power_spec)
    total = power_spec.sum(axis=1) + EPS
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + EPS)
    cumsum = np.cumsum(power_spec, axis=1)
    rolloff = freqs[np.argmax(cumsum >= 0.95 * total[:, None], axis=1)]
    flatness = np.exp(np.mean(np.log(power_spec + EPS), axis=1)) / (np.mean(power_spec, axis=1) + EPS)
    return {"centroid": centroid, "rolloff": rolloff, "flatness": flatness}


def rms_db(frames: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + EPS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)


def estimate_f0(frames: np.ndarray, sr: int, fmin=60.0, fmax=400.0, voicing_threshold=0.45):
    n = frames.shape[1]
    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), n - 1)
    frames = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :n]
    ac0 = ac[:, :1] + EPS
    nac = ac / ac0
    region = nac[:, lag_min:lag_max]
    best = np.argmax(region, axis=1)
    strength = region[np.arange(len(best)), best]
    lag = best + lag_min
    l = np.clip(lag, 1, n - 2)
    a, b, c = nac[np.arange(len(l)), l - 1], nac[np.arange(len(l)), l], nac[np.arange(len(l)), l + 1]
    denom = a - 2 * b + c
    safe = np.abs(denom) > EPS
    shift = np.divide(0.5 * (a - c), denom, out=np.zeros_like(denom), where=safe)
    f0 = sr / (l + np.clip(shift, -1, 1))
    energy = 10 * np.log10(np.mean(frames ** 2, axis=1) + EPS)
    voiced = (strength > voicing_threshold) & (energy > np.max(energy) - 45)
    return np.where(voiced, f0, 0.0), strength


def effective_bandwidth(power_spec: np.ndarray, freqs: np.ndarray, drop_db: float = 50.0) -> float:
    ltas = 10 * np.log10(np.mean(power_spec, axis=0) + EPS)
    k = max(3, len(ltas) // 200)
    smooth = np.convolve(ltas, np.ones(k) / k, mode="same")
    ref = np.max(smooth[freqs > 80]) if np.any(freqs > 80) else np.max(smooth)
    above = np.where(smooth > ref - drop_db)[0]
    return float(freqs[above[-1]]) if above.size else float(freqs[-1])


def band_limit(power_spec: np.ndarray, freqs: np.ndarray, min_drop_db: float = 25.0) -> float:
    ltas = 10 * np.log10(np.mean(power_spec, axis=0) + EPS)
    df = freqs[1] - freqs[0]
    w = max(3, int(1000 / df))
    n = len(ltas)
    if n < 4 * w:
        return float(freqs[-1])
    c = np.concatenate([[0.0], np.cumsum(ltas)])
    idx = np.arange(w, n - w)
    idx = idx[freqs[idx] > 1000]
    if idx.size == 0:
        return float(freqs[-1])
    below = (c[idx] - c[idx - w]) / w
    above = (c[idx + w] - c[idx]) / w
    drop = below - above
    best = int(np.argmax(drop))
    i = int(idx[best])
    rest = np.median(ltas[i:])
    if drop[best] >= min_drop_db and below[best] - rest >= min_drop_db:
        lo = max(0, i - 2 * w)
        ref = np.median(ltas[lo:max(lo + 1, i - w)])
        j = lo + int(np.argmax(ltas[lo:i + w] < ref - 6))
        return float(freqs[j])
    return float(freqs[-1])
