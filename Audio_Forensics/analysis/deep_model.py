import os

import numpy as np
from scipy import signal

from .features import EPS, frame_signal, mel_filterbank

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch, nn = None, None
    TORCH_AVAILABLE = False

SR = 16000
FRAME = 400
HOP = 160
N_FFT = 512
SEG_S = 3.0
SEG_FRAMES = int(SEG_S * SR / HOP) - 2
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "crnn.pt")
_FB = mel_filterbank(SR, N_FFT, 64, fmin=20, fmax=SR / 2)

def _deltas(x: np.ndarray, n: int = 2) -> np.ndarray:
    pad = np.pad(x, ((0, 0), (n, n)), mode="edge")
    num = sum(k * (pad[:, n + k:pad.shape[1] - n + k] - pad[:, n - k:pad.shape[1] - n - k]) for k in range(1, n + 1))
    return num / (2 * sum(k * k for k in range(1, n + 1)))


def compute_features(x: np.ndarray, sr: int, kind: str = "mfcc") -> np.ndarray:
    from scipy.fft import dct, rfft
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g)
    x = signal.lfilter([1, -0.97], [1], x)
    frames = frame_signal(x, FRAME, HOP) * np.hamming(FRAME)
    power = np.abs(rfft(frames, n=N_FFT, axis=1)) ** 2
    logmel = np.log(power @ _FB.T + 1e-8).T
    if kind == "logmel":
        feat = logmel
    else:
        mf = dct(logmel, type=2, axis=0, norm="ortho")[:20]
        feat = np.vstack([mf, _deltas(mf), _deltas(_deltas(mf))])
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + EPS)
    return feat.astype(np.float32)


def segment(feat: np.ndarray, hop_frames: int = SEG_FRAMES // 2):
    T = feat.shape[1]
    if T < SEG_FRAMES:
        feat = np.pad(feat, ((0, 0), (0, SEG_FRAMES - T)), mode="wrap")
        T = SEG_FRAMES
    starts = list(range(0, T - SEG_FRAMES + 1, hop_frames)) or [0]
    segs = np.stack([feat[:, s:s + SEG_FRAMES] for s in starts])
    return segs, [s * HOP / SR for s in starts]

if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, pool):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(pool), nn.Dropout(0.1))

        def forward(self, x):
            return self.net(x)

    class CRNN(nn.Module):

        def __init__(self, n_feat: int = 60, hidden: int = 64):
            super().__init__()
            self.cnn = nn.Sequential(ConvBlock(1, 16, (2, 2)), ConvBlock(16, 32, (2, 2)), ConvBlock(32, 48, (2, 1)))
            f_out = n_feat // 8
            self.proj = nn.Linear(48 * f_out, 96)
            self.lstm = nn.LSTM(96, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            self.att = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            z = self.cnn(x.unsqueeze(1))
            b, c, f, t = z.shape
            z = z.permute(0, 3, 1, 2).reshape(b, t, c * f)
            z = torch.relu(self.proj(z))
            h, _ = self.lstm(z)
            w = torch.softmax(self.att(h).squeeze(-1), dim=1)
            pooled = (h * w.unsqueeze(-1)).sum(dim=1)
            return self.head(pooled).squeeze(-1)


class DeepDetector:

    def __init__(self, path: str = MODEL_PATH):
        self.model, self.meta, self.error = None, {}, None
        if not os.path.exists(path):
            return
        if not TORCH_AVAILABLE:
            self.error = "PyTorch не е инсталиран"
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.meta = ckpt["meta"]
            self.model = CRNN(n_feat=self.meta["n_feat"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        except Exception as exc:
            self.model, self.error = None, str(exc)

    @property
    def available(self):
        return self.model is not None

    def predict(self, x: np.ndarray, sr: int) -> dict:
        offset = 0.0
        if self.meta.get("input") == "ml16k":
            from .robust import ML_SR, normalize_for_ml
            (x, offset), sr = normalize_for_ml(x, sr, return_offset=True), ML_SR
        feat = compute_features(x, sr, self.meta.get("features", "mfcc"))
        segs, starts = segment(feat)
        starts = [s + offset for s in starts]
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.from_numpy(segs))).numpy()
        return {"probability": float(np.mean(probs)), "segment_probs": probs.tolist(),
                "segment_starts": starts, "segment_len": SEG_S}
