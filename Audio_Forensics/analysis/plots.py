import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
import numpy as np

from .features import EPS, stft_power

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
MFCC_CMAP = LinearSegmentedColormap.from_list(
    "mfcc", ["#104281", "#5598e7", "#f0efec", "#ea7978", "#a32726"])
MFCC_CMAP.set_bad("white")
MFCC_SMOOTH_S = 0.08

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "figure.facecolor": "white",
})


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mark(ax, regions, events):
    for r0, r1 in regions:
        ax.axvspan(r0, r1, color=RED, alpha=0.18, lw=0)
    for e in events:
        ax.axvline(e, color=RED, lw=1.2, ls="--")


def manipulation_plots(audio, result) -> dict:
    x, sr = audio.samples, audio.sr
    t = np.arange(len(x)) / sr
    regions = [r for f in result["findings"] for r in f["regions"]]
    events = [e for f in result["findings"] for e in f["events"]]
    plots = {}

    fig, ax = plt.subplots(figsize=(10, 2.4))
    step = max(1, len(x) // 20000)
    ax.plot(t[::step], x[::step], color=BLUE, lw=0.6)
    _mark(ax, regions, events)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Време (s)")
    ax.set_ylabel("Амплитуда")
    ax.set_title("Вълнова форма (червено – подозрителни участъци и моменти)", loc="left")
    plots["waveform"] = _to_b64(fig)

    spec, freqs, times, _ = stft_power(x, sr, frame_s=0.04, hop_s=0.02)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    db = 10 * np.log10(spec.T + EPS)
    vmax = np.percentile(db, 99.5)
    ax.pcolormesh(times, freqs / 1000, db, shading="auto", cmap="magma", vmin=vmax - 90, vmax=vmax,
                  rasterized=True)
    for e in events:
        ax.axvline(e, color="white", lw=1, ls="--")
    ax.grid(False)
    ax.set_xlabel("Време (s)")
    ax.set_ylabel("Честота (kHz)")
    ax.set_title("Спектрограма", loc="left")
    plots["spectrogram"] = _to_b64(fig)

    noise = result["series"]["noise"]
    if noise and len(noise["t"]) > 1:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(noise["t"], noise["noise_db"], color=BLUE, lw=2)
        nf = next(f for f in result["findings"] if f["id"] == "noise_floor")
        _mark(ax, [], nf["events"])
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("Време (s)")
        ax.set_ylabel("dBFS")
        ax.set_title("Ниво на фоновия шум във времето", loc="left")
        plots["noise"] = _to_b64(fig)

    enf = result["series"]["enf"]
    if enf:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.plot(enf["t"], enf["f"], color=AQUA, lw=2)
        ef = next(f for f in result["findings"] if f["id"] == "enf")
        _mark(ax, ef["regions"], ef["events"])
        ax.axhline(enf["nominal"], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("Време (s)")
        ax.set_ylabel("Hz")
        ax.set_title(f"Мрежова честота (ENF), номинал {enf['nominal']} Hz", loc="left")
        plots["enf"] = _to_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ltas = 10 * np.log10(np.mean(spec, axis=0) + EPS)
    ax.plot(freqs / 1000, ltas, color=BLUE, lw=1.4)
    bw = result["series"]["bandwidth"]["global_bw"]
    if bw < sr / 2 * 0.98:
        ax.axvline(bw / 1000, color=ORANGE, lw=1.5, ls="--")
        ax.text(bw / 1000, ax.get_ylim()[1], f" срез {bw / 1000:.1f} kHz", color=INK, va="top", fontsize=8)
    ax.set_xlabel("Честота (kHz)")
    ax.set_ylabel("dB")
    ax.set_title("Среден спектър (LTAS)", loc="left")
    plots["ltas"] = _to_b64(fig)
    return plots


def ai_plots(audio, ai) -> dict:
    s = ai["series"]
    times = np.array(s["times"])
    plots = {}

    f0 = np.array(s["f0"], dtype=float)
    f0[f0 <= 0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.plot(times, f0, color=ORANGE, lw=1.6)
    ax.set_xlim(0, times[-1] if len(times) else 1)
    ax.set_xlabel("Време (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Контур на основния тон (интонация)", loc="left")
    plots["f0"] = _to_b64(fig)

    M = np.array(s["mfcc"])[1:13]
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    k = max(1, int(round(MFCC_SMOOTH_S / dt)))
    speech = np.array(s["energy"]) > s["pause_threshold"]
    speech = binary_opening(binary_closing(speech, np.ones(k)), np.ones(k))
    if speech.sum() < 10:
        speech = np.ones(M.shape[1], dtype=bool)
    w = uniform_filter1d(speech.astype(float), k)
    M = uniform_filter1d(M * speech, k, axis=1) / (w + EPS)
    mu = M[:, speech].mean(axis=1, keepdims=True)
    sd = M[:, speech].std(axis=1, keepdims=True) + EPS
    Z = np.ma.masked_array((M - mu) / sd, mask=np.broadcast_to(~speech, M.shape))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=MFCC_CMAP, vmin=-2.5, vmax=2.5,
                   extent=(times[0] - dt / 2, times[-1] + dt / 2, 0.5, M.shape[0] + 0.5),
                   interpolation="nearest")
    ax.grid(False)
    ax.set_yticks([1, 4, 8, 12])
    ax.set_xlabel("Време (s)")
    ax.set_ylabel("MFCC №")
    ax.set_title("MFCC 1–12 (тембър на гласа) спрямо средното за записа · белите ивици са паузи", loc="left")
    cb = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2], fraction=0.025, pad=0.015)
    cb.ax.set_yticklabels(["−2σ", "−1σ", "средно", "+1σ", "+2σ"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    plots["mfcc"] = _to_b64(fig)

    votes = ai["votes"][::-1]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(votes) + 0.9))
    vals = [v["vote"] * v["weight"] for v in votes]
    colors = [ORANGE if v > 0 else BLUE for v in vals]
    ax.barh([v["tool"] for v in votes], vals, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    lim = max(0.6, max(abs(v) for v in vals) * 1.15) if vals else 1
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("← естествен запис                    принос × тегло                    синтетичен запис →")
    ax.set_title("Как всяка проверка „гласува“", loc="left")
    plots["votes"] = _to_b64(fig)

    deep = ai.get("deep")
    if deep:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        L = deep["segment_len"]
        for st, p in zip(deep["segment_starts"], deep["segment_probs"]):
            ax.plot([st + 0.1, st + L - 0.1], [p * 100, p * 100], color=ORANGE if p >= 0.5 else BLUE,
                    lw=4, solid_capstyle="round", alpha=0.85)
        ax.axhline(50, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(-3, 103)
        ax.set_xlim(0, max(times[-1] if len(times) else L, deep["segment_starts"][-1] + L))
        ax.set_xlabel("Време (s)")
        ax.set_ylabel("% синтетичен")
        ax.set_title("CNN + BiLSTM: вероятност за синтетична реч по 3-секундни прозорци", loc="left")
        plots["deep"] = _to_b64(fig)
    return plots
