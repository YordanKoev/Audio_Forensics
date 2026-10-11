from dataclasses import dataclass, field, asdict
from math import gcd
 
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
 
from .audio_io import AudioData
from .features import EPS, band_limit, effective_bandwidth, frame_params, rms_db, stft_power
 
STATUS_LABELS = {
    "ok": "Няма признаци",
    "info": "Информация",
    "warning": "Внимание",
    "suspicious": "Подозрително",
    "n/a": "Неприложимо",
}
 
 
@dataclass
class Finding:
    id: str
    title: str
    status: str
    score: float
    description: str
    details: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)
    events: list = field(default_factory=list)
 
    def to_dict(self):
        d = asdict(self)
        d["status_label"] = STATUS_LABELS.get(self.status, self.status)
        d["score"] = round(float(self.score), 3)
        return d
 
 
def _merge_times(times, min_gap):
    merged = []
    for t in sorted(times):
        if not merged or t - merged[-1] > min_gap:
            merged.append(float(t))
    return merged
 
 
def _runs(mask: np.ndarray):
    if mask.size == 0:
        return []
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
 
 
def detect_discontinuities(audio: AudioData, clip_fraction: float) -> Finding:
    x, sr = audio.samples, audio.sr
    d = np.zeros_like(x)
    d[4:] = np.diff(x, n=4)
    win = max(16, int(0.02 * sr))
    local_rms = np.sqrt(uniform_filter1d(d ** 2, win) + EPS)
    ratio = np.abs(d) / local_rms
    global_med = np.median(np.abs(d)) + EPS
    edge = int(0.01 * sr)
    cand = np.where((ratio > 14) & (np.abs(d) > 25 * global_med) & (np.abs(d) > 1e-3))[0]
    cand = cand[(cand > edge) & (cand < len(x) - edge)]
    events = _merge_times(cand / sr, 0.05)
 
    details = {"брой_прекъсвания": len(events)}
    if clip_fraction > 0.001:
        return Finding("discontinuity", "Резки прекъсвания на вълновата форма", "n/a", 0.0,
                       "Записът е силно изрязан (clipping), което създава естествени скокове. "
                       "Детекторът е изключен, за да не дава фалшиви тревоги.", details)
    if len(events) == 0:
        return Finding("discontinuity", "Резки прекъсвания на вълновата форма", "ok", 0.0,
                       "Не са открити резки скокове на сигнала, характерни за грубо залепване на сегменти.",
                       details)
    if len(events) > 25:
        return Finding("discontinuity", "Резки прекъсвания на вълновата форма", "info", 0.05,
                       f"Открити са {len(events)} импулсни смущения – по-вероятно е шум/пукане в записа, "
                       "отколкото монтаж.", details, events=events[:50])
    score = 0.35 if len(events) <= 2 else 0.5
    return Finding("discontinuity", "Резки прекъсвания на вълновата форма", "suspicious", score,
                   f"Открити са {len(events)} рязки скока на вълновата форма. Такива скокове възникват, "
                   "когато два сегмента са съединени без плавен преход (crossfade).", details, events=events)
 
 
def detect_digital_silence(audio: AudioData) -> Finding:
    x, sr = audio.samples, audio.sr
    zero = np.abs(x) < 1e-9
    min_len = int(0.005 * sr)
    regions = [(s / sr, e / sr) for s, e in _runs(zero) if e - s >= min_len and s > 0 and e < len(x)]
    total = sum(e - s for s, e in regions)
    nonzero_floor = np.percentile(np.abs(x[~zero]), 10) if np.any(~zero) else 0.0
    details = {"сегменти": len(regions), "обща_продължителност_s": round(total, 3)}
    if not regions:
        return Finding("digital_silence", "Цифрова тишина вътре в записа", "ok", 0.0,
                       "Няма участъци с абсолютна нула. В реален микрофонен запис винаги има фонов шум.",
                       details)
    if nonzero_floor > 1e-4:
        return Finding("digital_silence", "Цифрова тишина вътре в записа", "suspicious",
                       min(0.45, 0.25 + 0.05 * len(regions)),
                       f"Открити са {len(regions)} участъка с абсолютна цифрова тишина, докато останалата "
                       "част има фонов шум. Това е типично за изтрити/заглушени части или вмъкнати паузи.",
                       details, regions=regions[:50])
    return Finding("digital_silence", "Цифрова тишина вътре в записа", "info", 0.05,
                   "Има участъци с цифрова тишина, но целият запис е с много нисък шум "
                   "(вероятно студиен/синтетичен или с шумоподтискане).", details, regions=regions[:50])
 
 
def _noise_track(frame_db, times, mask):
    hop_s = times[1] - times[0] if len(times) > 1 else 0.01
    block = max(1, int(1.5 / hop_s))
    step = max(1, int(0.25 / hop_s))
    min_frames = max(10, block // 10)
    t_blocks, levels = [], []
    for start in range(0, max(1, len(frame_db) - block + 1), step):
        seg = frame_db[start:start + block][mask[start:start + block]]
        if seg.size < min_frames:
            continue
        t_blocks.append(times[start] + block * hop_s / 2)
        levels.append(np.percentile(seg, 5))
    return np.array(t_blocks), np.array(levels)
 
 
def _noise_steps(levels, L=4):
    steps = np.zeros_like(levels)
    for i in range(L, len(levels) - L + 1):
        steps[i] = np.median(levels[i:i + L]) - np.median(levels[i - L:i])
    d = np.diff(levels)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) + EPS
    thr = max(3.0, 6 * sigma)
    peaks, _ = signal.find_peaks(np.abs(steps), height=thr, distance=2 * L)
    return steps, thr, peaks
 
 
def _pause_mask(frame_db, valid, below_db=15.0, max_dev_db=1.5, min_frames=15):
    ref = np.percentile(frame_db[valid], 90)
    med = median_filter(frame_db, size=9, mode="nearest")
    cand = valid & (frame_db < ref - below_db) & (np.abs(frame_db - med) < max_dev_db)
    mask = np.zeros_like(cand)
    for s, e in _runs(cand):
        if e - s >= min_frames:
            mask[s:e] = True
    return mask
 
 
def analyze_noise_floor(audio: AudioData, frames, times) -> tuple:
    frame_db = rms_db(frames)
    valid = frame_db > -100
    title = "Промени във фоновия шум"
    if valid.sum() < 10:
        return Finding("noise_floor", title, "n/a", 0.0, "Недостатъчно данни за анализ.", {}), {}
 
    pauses = _pause_mask(frame_db, valid)
    t_blocks, levels = _noise_track(frame_db, times, pauses)
    series = {"t": t_blocks.tolist(), "noise_db": levels.tolist()}
 
    if len(levels) < 10:
        return Finding("noise_floor", title, "n/a", 0.0,
                       "В записа няма достатъчно паузи, за да се оцени фоновият шум надеждно.",
                       {"паузи_s": round(float(pauses.sum() * (times[1] - times[0])), 2)
                        if len(times) > 1 else 0.0}), series
    global_floor = float(np.percentile(frame_db, 5))
    if global_floor < -80:
        return Finding("noise_floor", title, "n/a", 0.0,
                       f"Записът практически няма фонов шум ({global_floor:.0f} dBFS), затова "
                       "проверката е неприложима.", {"шум_dBFS": round(global_floor, 1)}), series
 
    steps, thr, peaks = _noise_steps(levels)
    events = [float(t_blocks[p] - 0.125) for p in peaks]
    details = {
        "шум_dBFS": round(float(np.median(levels)), 1),
        "разсейване_dB": round(float(np.std(levels)), 2),
        "праг_dB": round(float(thr), 1),
        "скокове_dB": [round(float(steps[p]), 1) for p in peaks],
    }
    if not events:
        return Finding("noise_floor", title, "ok", 0.0,
                       "Нивото на фоновия шум е постоянно през целия запис – признак, че записът "
                       "е направен в една и съща среда с едно устройство.", details), series
 
    big = max(abs(steps[p]) for p in peaks)
    if _looks_like_noise_gate(frame_db, times, valid, levels, thr):
        details["спад_в_паузите_dB"] = round(float(big), 1)
        return Finding("noise_floor", title, "info", 0.05,
                       f"Фоновият шум изчезва в по-дългите паузи (спад до {big:.1f} dB) и се връща, "
                       "щом започне речта. Това е типично за микрофон с потискане на шума "
                       "(noise gate / noise suppression), а не за монтаж: непосредствено след "
                       "речта нивото на шума е еднакво в целия запис.", details), series
 
    score = min(0.6, 0.3 + 0.1 * len(events) + 0.02 * (big - thr))
    return Finding("noise_floor", title, "suspicious", score,
                   f"Открити са {len(events)} резки промени във фоновия шум (до {big:.1f} dB). "
                   "Това е силен индикатор за съединяване на части от различни записи/среди.",
                   details, events=events), series
 
 
def _looks_like_noise_gate(frame_db, times, valid, levels, thr) -> bool:
    loud = valid & (frame_db > np.percentile(frame_db[valid], 75) - 15)
    hop_s = times[1] - times[0] if len(times) > 1 else 0.01
    k = max(1, int(0.15 / hop_s))
    after_speech = np.convolve(loud.astype(float), np.ones(k + 1))[:len(loud)] > 0
    post = valid & ~loud & after_speech
    _, lev_post = _noise_track(frame_db, times, post)
    if len(lev_post) < 10:
        return False
    _, _, peaks_post = _noise_steps(lev_post)
    higher = levels > np.median(lev_post) + thr
    only_drops = not any(e - s >= 3 for s, e in _runs(higher))
    return len(peaks_post) == 0 and only_drops
 
COMMON_CUTOFFS = {4000: "телефонен запис / 8 kHz", 8000: "16 kHz (типично за TTS / VoIP)",
                  11025: "22.05 kHz", 12000: "24 kHz (типично за TTS)",
                  16000: "MP3 128 kbps", 19000: "MP3/AAC с висок битрейт"}
 
 
def analyze_bandwidth(audio: AudioData, spec, freqs, times, frames) -> tuple:
    sr = audio.sr
    nyq = sr / 2
    global_bw = band_limit(spec, freqs)
    frame_db = 10 * np.log10(spec.mean(axis=1) + EPS)
    fr_dbfs = rms_db(frames)
    has_noise = np.percentile(fr_dbfs, 5) > -80
    active_thr = np.median(frame_db)
    hop_s = times[1] - times[0] if len(times) > 1 else 0.01
    block = max(1, int(1.0 / hop_s))
    t_b, bw_b = [], []
    for start in range(0, max(1, len(times) - block + 1), block // 2 or 1):
        sl = slice(start, start + block)
        act = frame_db[sl] > active_thr
        if act.sum() < block // 4:
            continue
        t_b.append(times[start] + block * hop_s / 2)
        bw_b.append(effective_bandwidth(spec[sl][act], freqs))
    t_b, bw_b = np.array(t_b), np.array(bw_b)
    series = {"t": t_b.tolist(), "bandwidth_hz": bw_b.tolist(), "global_bw": global_bw}
 
    findings = []
    title = "Еднородност на честотната лента"
    if not has_noise:
        findings.append(Finding("bandwidth", title, "n/a", 0.0,
                                "Без фонов шум ширината на лентата зависи от самата реч, затова "
                                "сравнението между участъците е ненадеждно.", {}))
    elif len(bw_b) >= 4:
        med = np.median(bw_b)
        low = bw_b < 0.7 * med
        regions = []
        for s, e in _runs(low):
            if e - s >= 2:
                regions.append((float(t_b[s] - 0.5), float(t_b[e - 1] + 0.5)))
        details = {"медиана_Hz": round(float(med)), "минимум_Hz": round(float(bw_b.min()))}
        if regions:
            findings.append(Finding("bandwidth", title, "suspicious", 0.45,
                                    f"В {len(regions)} участъка честотната лента е значително по-тясна "
                                    f"(под {0.7 * med / 1000:.1f} kHz) от останалата част. Това показва "
                                    "вмъкнат фрагмент, записан с друго устройство или в друг формат.",
                                    details, regions=regions))
        else:
            findings.append(Finding("bandwidth", title, "ok", 0.0,
                                    "Честотната лента е еднаква в целия запис.", details))
    else:
        findings.append(Finding("bandwidth", title, "n/a", 0.0, "Недостатъчно активни участъци.", {}))
 
    title = "Следи от компресия / презаписване"
    ltas = 10 * np.log10(np.mean(spec, axis=0) + EPS)
    idx = np.searchsorted(freqs, global_bw)
    above = ltas[idx + 3: idx + 3 + max(3, int(500 / (freqs[1] - freqs[0])))]
    steep_drop = float(ltas[max(0, idx - 3)] - np.median(above)) if above.size else 0.0
    band = (freqs > max(3000, 0.5 * global_bw)) & (freqs < global_bw)
    hole_ratio = 0.0
    if band.sum() > 5:
        db = 10 * np.log10(spec[:, band] + EPS)
        active = frame_db > active_thr
        if active.sum() > 10:
            ref = np.median(db[active], axis=1, keepdims=True)
            hole_ratio = float(np.mean(db[active] < ref - 35))
    nearest = min(COMMON_CUTOFFS, key=lambda c: abs(c - global_bw))
    details = {"ефективна_лента_Hz": round(global_bw), "найквист_Hz": round(nyq),
               "рязкост_на_среза_dB": round(steep_drop, 1), "дял_спектрални_дупки": round(hole_ratio, 4)}
    limited = global_bw < 0.85 * nyq
    reasons = []
    if limited:
        txt = f"Съдържанието е ограничено до {global_bw / 1000:.1f} kHz при възможни {nyq / 1000:.1f} kHz"
        if abs(nearest - global_bw) < 0.06 * nearest:
            txt += f" (съвпада с {COMMON_CUTOFFS[nearest]})"
        reasons.append(txt + ".")
    if limited and steep_drop > 30:
        reasons.append(f"Срезът е много рязък ({steep_drop:.0f} dB) – характерно за кодеци (MP3/AAC) "
                       "или за повишаване на честотата на дискретизация.")
    if hole_ratio > 0.08:
        reasons.append(f"{hole_ratio * 100:.1f}% от спектралните клетки във високите честоти са „дупки“ – "
                       "типичен белег на перцептивна компресия.")
    if not reasons:
        findings.append(Finding("compression", title, "ok", 0.0,
                                "Пълна честотна лента без следи от компресия – съвместимо с оригинален WAV запис.",
                                details))
    else:
        score = 0.1 + 0.05 * len(reasons)
        findings.append(Finding("compression", title, "warning", score,
                                " ".join(reasons) + " Файлът вероятно не е оригиналният запис, а преобразуван/"
                                "презаписан (това само по себе си не доказва монтаж).", details))
    return findings, series
 
def analyze_enf(audio: AudioData) -> tuple:
    x, sr = audio.samples, audio.sr
    title = "ENF анализ (мрежова честота 50/60 Hz)"
    fs = 1000
    g = gcd(fs, sr)
    y = signal.resample_poly(x, fs // g, sr // g)
    if len(y) < 4 * fs:
        return Finding("enf", title, "n/a", 0.0, "Записът е твърде кратък за ENF анализ.", {}), None
 
    f, pxx = signal.welch(y, fs=fs, nperseg=min(len(y), 8 * fs))
    best = None
    for nominal in (50, 60):
        for k in (1, 2, 3):
            fc = nominal * k
            peak = pxx[(f > fc - 0.5) & (f < fc + 0.5)].max()
            neigh = pxx[((f > fc - 6) & (f < fc - 1.5)) | ((f > fc + 1.5) & (f < fc + 6))]
            snr = 10 * np.log10(peak / (np.median(neigh) + EPS) + EPS)
            if best is None or snr > best[2]:
                best = (nominal, k, snr)
    nominal, k, snr = best
    details = {"номинална_честота_Hz": nominal, "хармоник": k, "SNR_dB": round(float(snr), 1)}
    if snr < 20:
        return Finding("enf", title, "n/a", 0.0,
                       "В записа няма достатъчно силен сигнал от електрическата мрежа (записът е направен "
                       "на батерия или е филтриран), затова ENF проверката не е приложима.", details), None
 
    fc = nominal * k
    sos = signal.butter(4, [fc - 1.0, fc + 1.0], btype="bandpass", fs=fs, output="sos")
    nb = signal.sosfiltfilt(sos, y)
    analytic = signal.hilbert(nb)
    phase = np.unwrap(np.angle(analytic))
    t = np.arange(len(nb)) / fs
    resid = phase - 2 * np.pi * fc * t
    resid = median_filter(resid, size=51)
    inst_f = fc + np.gradient(resid) * fs / (2 * np.pi)
    win = fs
    t_tr, f_tr = [], []
    for s in range(0, len(nb) - win, win // 4):
        t_tr.append((s + win / 2) / fs)
        f_tr.append(np.median(inst_f[s:s + win]) / k)
    t_tr, f_tr = np.array(t_tr), np.array(f_tr)
 
    D = int(0.6 * fs)
    jumps = np.zeros_like(resid)
    left = resid[:-2 * D]
    right = resid[2 * D:]
    jumps[D:-D] = right - left
    slope = np.median(np.diff(resid[::fs // 10])) * 10 * 2 * 0.6
    dev = np.abs(jumps - slope)
    edge = D + fs // 2
    dev[:edge] = 0
    dev[-edge:] = 0
    env = np.abs(analytic)
    core = env[fs:-fs]
    present = median_filter((env > 0.25 * np.median(core)).astype(np.uint8), size=fs // 2) > 0
    continuity = float(np.mean(present[fs:-fs]))
    details["непрекъснатост"] = round(continuity, 2)
    if continuity < 0.5:
        return Finding("enf", title, "n/a", 0.0,
                       "Сигналът около 50/60 Hz не е постоянен (вероятно идва от самата реч, а не от "
                       "мрежата), затова ENF проверката не е приложима.", details), None
    gaps = [(s0 / fs, e0 / fs) for s0, e0 in _runs(~present)
            if e0 - s0 >= fs and s0 > fs and e0 < len(env) - fs]
    for g0, g1 in gaps:
        dev[max(0, int(g0 * fs) - D):int(g1 * fs) + D] = 0
    noise = 1.4826 * np.median(np.abs(dev[D:-D] - np.median(dev[D:-D]))) + EPS
    thr = max(0.9 * k, 8 * noise)
    peaks, _ = signal.find_peaks(dev, height=thr, distance=fs)
    events = [float(p / fs) for p in peaks]
    details.update({"фазов_праг_rad": round(float(thr), 2), "брой_фазови_скокове": len(events),
                    "честота_мин_Hz": round(float(f_tr.min()), 3), "честота_макс_Hz": round(float(f_tr.max()), 3)})
    trace = {"t": t_tr.tolist(), "f": f_tr.tolist(), "nominal": nominal,
             "env_t": t[::fs // 20].tolist(), "env": env[::fs // 20].tolist()}
    details["участъци_без_ENF"] = len(gaps)
    if events or gaps:
        parts = []
        if events:
            parts.append(f"открит(и) са {len(events)} фазов(и) скок(а) – при непрекъснат запис фазата на "
                         "мрежовата честота се променя плавно, а скок означава изрязване или вмъкване")
        if gaps:
            parts.append(f"в {len(gaps)} участък(а) мрежовият сигнал изчезва, макар че присъства в останалата "
                         "част – вероятно вмъкнат фрагмент от друг запис")
        score = min(0.7, 0.4 + 0.1 * len(events) + 0.1 * len(gaps))
        return Finding("enf", title, "suspicious", score,
                       f"ENF сигналът ({nominal} Hz) присъства, но " + "; ".join(parts) + ".",
                       details, events=events, regions=gaps), trace
    return Finding("enf", title, "ok", 0.0,
                   f"ENF сигналът ({nominal} Hz) присъства и фазата му е непрекъсната – няма следи от "
                   "изрязване или вмъкване.", details), trace
 
def detect_copy_move(audio: AudioData, spec, freqs, times) -> Finding:
    from sklearn.neighbors import NearestNeighbors
    from .features import mel_filterbank
 
    title = "Повторени (копирани) сегменти"
    x, sr = audio.samples, audio.sr
    frame_len, hop, n_fft = frame_params(sr)
    fb = mel_filterbank(sr, n_fft, 32, fmax=min(8000, sr / 2))
    feats = np.log(spec @ fb.T + EPS)
    energy = 10 * np.log10(spec.mean(axis=1) + EPS)
    active = np.where(energy > np.percentile(energy, 40))[0]
    if len(active) < 100:
        return Finding("copy_move", title, "n/a", 0.0, "Недостатъчно активен сигнал за анализ.", {})
    F = feats[active] - feats[active].mean(axis=1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + EPS
    nn = NearestNeighbors(n_neighbors=min(8, len(F))).fit(F)
    dist, ind = nn.kneighbors(F)
    min_sep = int(0.5 / (hop / sr))
    match = {}
    for row, (ds, js) in enumerate(zip(dist, ind)):
        i = active[row]
        for dd, jj in zip(ds, js):
            j = active[jj]
            if abs(j - i) > min_sep and dd < 0.15 and j > i:
                match[i] = j - i
                break
    min_run = int(0.3 / (hop / sr))
    max_gap = int(0.1 / (hop / sr))
    pairs, keys = [], sorted(match)
    run_start, prev = None, None
    for i in keys + [None]:
        if i is not None and prev is not None and i - prev <= max_gap and abs(match[i] - match[prev]) <= 1:
            prev = i
            continue
        if run_start is not None and prev - run_start >= min_run:
            pairs.append((run_start, prev, match[run_start]))
        run_start, prev = i, i
 
    confirmed = []
    for s, e, off in pairs:
        a0, a1 = s * hop + frame_len, e * hop
        if a1 - a0 < int(0.2 * sr):
            continue
        a = x[a0:a1]
        b0 = a0 + (off - 1) * hop
        b = x[max(0, b0): b0 + len(a) + 2 * hop]
        if len(b) < len(a):
            continue
        corr = signal.correlate(b, a, mode="valid")
        norm = np.sqrt(np.sum(a ** 2) * np.convolve(b ** 2, np.ones(len(a)), mode="valid")) + EPS
        c = float(np.max(corr / norm))
        if c > 0.97:
            confirmed.append({"източник_s": round(s * hop / sr, 2), "копие_s": round((s + off) * hop / sr, 2),
                              "продължителност_s": round((e - s) * hop / sr, 2), "корелация": round(c, 4)})
    details = {"кандидати": len(pairs), "потвърдени": len(confirmed), "двойки": confirmed[:10]}
    if not confirmed:
        return Finding("copy_move", title, "ok", 0.0,
                       "Не са открити идентични повторения на сегменти в записа.", details)
    regions = []
    for c in confirmed:
        regions.append((c["източник_s"], c["източник_s"] + c["продължителност_s"]))
        regions.append((c["копие_s"], c["копие_s"] + c["продължителност_s"]))
    return Finding("copy_move", title, "suspicious", min(0.8, 0.6 + 0.1 * len(confirmed)),
                   f"Открити са {len(confirmed)} сегмента, които са практически идентични копия на друга част "
                   "от записа (вкл. фоновия шум). В естествен запис това не се случва – признак за "
                   "copy-move монтаж.", details, regions=regions)
 
def analyze_format(audio: AudioData, clip_fraction: float) -> Finding:
    info = audio.info
    notes, score = [], 0.0
    if info["effective_bits"] < info["bit_depth"] and info["bit_depth"] > 16:
        notes.append(f"Файлът е {info['bit_depth']}-битов, но реално използва само {info['effective_bits']} бита "
                     "– преобразуван е от по-нискокачествен източник.")
        score += 0.05
    if info["identical_channels"]:
        notes.append("Двата канала са идентични (моно сигнал, записан като стерео) – резултат от обработка.")
        score += 0.05
    if clip_fraction > 0.001:
        notes.append(f"{clip_fraction * 100:.2f}% от отчетите са изрязани (clipping).")
    details = {"clipping_%": round(clip_fraction * 100, 3)}
    if not notes:
        return Finding("format", "Формат и метаданни", "ok", 0.0,
                       "Форматът на файла не показва следи от преобразуване.", details)
    return Finding("format", "Формат и метаданни", "info", score, " ".join(notes), details)
 
 
def clipping_fraction(x: np.ndarray) -> float:
    peak = np.max(np.abs(x))
    if peak < 0.98:
        return 0.0
    at_peak = np.abs(x) >= 0.999 * peak
    clipped = sum(e - s for s, e in _runs(at_peak) if e - s >= 3)
    return clipped / len(x)
 
 
def analyze_manipulation(audio: AudioData) -> dict:
    spec, freqs, times, frames = stft_power(audio.samples, audio.sr)
    clip = clipping_fraction(audio.samples)
 
    findings = [detect_discontinuities(audio, clip), detect_digital_silence(audio)]
    nf, noise_series = analyze_noise_floor(audio, frames, times)
    findings.append(nf)
    bw_findings, bw_series = analyze_bandwidth(audio, spec, freqs, times, frames)
    findings.extend(bw_findings)
    enf, enf_trace = analyze_enf(audio)
    findings.append(enf)
    findings.append(detect_copy_move(audio, spec, freqs, times))
    findings.append(analyze_format(audio, clip))
 
    score = 1.0 - np.prod([1.0 - f.score for f in findings])
    strong = [f for f in findings if f.status == "suspicious"]
    if score >= 0.6 or len(strong) >= 2:
        verdict, level = "Вероятно МАНИПУЛИРАН запис", "high"
    elif score >= 0.3:
        verdict, level = "СЪМНИТЕЛЕН запис – препоръчва се експертна проверка", "medium"
    else:
        verdict, level = "Няма открити признаци на манипулация", "low"
 
    return {
        "score": round(float(score) * 100, 1),
        "verdict": verdict,
        "level": level,
        "findings": [f.to_dict() for f in findings],
        "series": {"noise": noise_series, "bandwidth": bw_series, "enf": enf_trace},
    }