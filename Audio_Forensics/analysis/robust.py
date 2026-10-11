from math import gcd

import numpy as np
from scipy import signal

ML_SR = 16000
TARGET_RMS_DB = -23.0


def _resample(x: np.ndarray, sr: int, target: int = ML_SR) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(target, sr)
    return signal.resample_poly(x, target // g, sr // g)


def trim_silence(x: np.ndarray, sr: int, top_db: float = 40.0, margin_s: float = 0.1,
                 return_start: bool = False):
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) < 2 * frame:
        return (x, 0) if return_start else x
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    level = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    loud = np.where(level > level.max() - top_db)[0]
    if loud.size == 0:
        return (x, 0) if return_start else x
    margin = int(margin_s * sr)
    start = max(0, loud[0] * hop - margin)
    end = min(len(x), loud[-1] * hop + frame + margin)
    return (x[start:end], start) if return_start else x[start:end]


def normalize_for_ml(x: np.ndarray, sr: int, return_offset: bool = False):
    y, start = trim_silence(_resample(np.asarray(x, dtype=np.float64), sr), ML_SR, return_start=True)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    y = y * (10 ** (TARGET_RMS_DB / 20) / rms)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.99:
        y = y / peak * 0.99
    return (y, start / ML_SR) if return_offset else y


def augment(x: np.ndarray, rng: np.random.Generator, sr: int = ML_SR) -> np.ndarray:
    y = x.copy()
    if rng.random() < 0.6:
        cutoff = rng.choice([3400, 4000, 5500, 7000])
        sos = signal.butter(8, cutoff, btype="low", fs=sr, output="sos")
        y = signal.sosfilt(sos, y)
    if rng.random() < 0.3:
        sos = signal.butter(2, rng.uniform(60, 200), btype="high", fs=sr, output="sos")
        y = signal.sosfilt(sos, y)
    if rng.random() < 0.5:
        c = rng.uniform(0.3, 0.7)
        y = signal.lfilter([1, -c], [1], y) if rng.random() < 0.5 else signal.lfilter([1], [1, -c], y)
    if rng.random() < 0.7:
        snr = rng.uniform(10, 40)
        noise = rng.normal(0, 1, len(y))
        if rng.random() < 0.5:
            noise = signal.lfilter([1], [1, -0.95], noise)
        p_sig = np.mean(y ** 2) + 1e-12
        noise *= np.sqrt(p_sig / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-12))
        y = y + noise
    y = y * 10 ** (rng.uniform(-12, 6) / 20)
    peak = np.max(np.abs(y)) + 1e-12
    return y / peak * 0.99 if peak > 0.99 else y