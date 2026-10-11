import os

import numpy as np

from .audio_io import AudioData
from .deep_model import MODEL_PATH as DEEP_MODEL_PATH, DeepDetector
from .features import (EPS, band_limit, estimate_f0, frame_params, mfcc, rms_db,
                       spectral_features, stft_power)
from .robust import ML_SR, normalize_for_ml

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "ai_detector.joblib")

TTS_CUTOFFS = {8000: "16 kHz", 11025: "22.05 kHz", 12000: "24 kHz"}

def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def extract_features(audio: AudioData) -> tuple:
    x, sr = audio.samples, audio.sr
    spec, freqs, times, frames = stft_power(x, sr)
    frame_len, hop, n_fft = frame_params(sr)
    hop_s = hop / sr
    energy = rms_db(frames)
    f0, strength = estimate_f0(frames, sr)
    voiced = f0 > 0
    feats = {}

    nyq = sr / 2
    bw = band_limit(spec, freqs)
    feats["bandwidth_hz"] = bw
    feats["bandwidth_ratio"] = bw / nyq

    feats["digital_silence_ratio"] = float(np.mean(energy < -100))
    non_digital = energy[energy > -100]
    floor = float(np.percentile(non_digital, 10)) if non_digital.size else -120.0
    feats["pause_noise_db"] = float(np.percentile(energy, 10))
    thr = max(floor + 10, energy.max() - 40)
    pauses = [(s, e) for s, e in _runs(energy < thr) if (e - s) * hop_s >= 0.15 and s > 0 and e < len(energy)]
    durations = np.array([(e - s) * hop_s for s, e in pauses])
    feats["n_pauses"] = len(pauses)
    feats["pause_cv"] = float(durations.std() / (durations.mean() + EPS)) if len(durations) >= 3 else 0.5

    feats["voiced_ratio"] = float(voiced.mean())
    both = voiced[1:] & voiced[:-1]
    if both.sum() > 20:
        f = f0[voiced]
        semis = 12 * np.log2(f / np.median(f))
        feats["f0_mean"] = float(np.median(f))
        feats["f0_std_semitones"] = float(1.4826 * np.median(np.abs(semis - np.median(semis))))
        rel = np.abs(np.diff(f0))[both] / f0[1:][both]
        rel = rel[rel < 0.2] 
        from scipy.ndimage import median_filter
        smooth = median_filter(f0, size=7)
        micro = np.abs(f0 - smooth)[voiced] / (smooth[voiced] + EPS)
        feats["f0_jitter"] = float(np.median(micro))
        feats["f0_delta"] = float(np.median(rel)) if rel.size else 0.0
        e_v = energy[1:][both] - energy[:-1][both]
        feats["shimmer_db"] = float(np.median(np.abs(e_v)))
    else:
        feats.update({"f0_mean": 0.0, "f0_std_semitones": 0.0, "f0_jitter": 0.0,
                      "f0_delta": 0.0, "shimmer_db": 0.0})

    active = energy > thr
    if active.sum() < 10:
        active = np.ones_like(energy, dtype=bool)
    sf = spectral_features(spec[active], freqs)
    feats["spectral_centroid"] = float(np.median(sf["centroid"]))
    feats["spectral_rolloff"] = float(np.median(sf["rolloff"]))
    hf = (freqs >= 4000) & (freqs < min(8000, nyq))
    if hf.sum() > 4:
        p = spec[active][:, hf]
        flat = np.exp(np.mean(np.log(p + EPS), axis=1)) / (np.mean(p, axis=1) + EPS)
        feats["hf_flatness"] = float(np.median(flat))
    else:
        feats["hf_flatness"] = 0.0
    feats["hf_energy_ratio"] = float(spec[active][:, freqs >= 4000].sum() / (spec[active].sum() + EPS))

    ltas = 10 * np.log10(np.mean(spec[active], axis=0) + EPS)
    band = (freqs > 1000) & (freqs < min(bw, nyq * 0.95))
    if band.sum() > 64:
        seg = ltas[band]
        k = 31
        trend = np.convolve(seg, np.ones(k) / k, mode="same")
        r = (seg - trend)[k:-k]
        r = r - r.mean()
        ac = np.correlate(r, r, mode="full")[len(r) - 1:]
        ac /= ac[0] + EPS
        lim = len(ac) // 4
        start = int(np.argmax(ac < 0)) if np.any(ac[:lim] < 0) else lim
        peaks = [i for i in range(start + 1, lim - 1) if ac[i - 1] < ac[i] >= ac[i + 1]]
        feats["ltas_periodicity"] = float(max(0.0, max(ac[i] for i in peaks))) if peaks else 0.0
    else:
        feats["ltas_periodicity"] = 0.0

    C = mfcc(spec, sr, n_fft)
    Ca = C[active]
    delta = np.diff(Ca[:, 1:], axis=0)
    feats["mfcc_delta_mean"] = float(np.mean(np.abs(delta))) if len(delta) else 0.0
    for i in range(13):
        feats[f"mfcc{i}_mean"] = float(Ca[:, i].mean())
        feats[f"mfcc{i}_std"] = float(Ca[:, i].std())

    series = {"times": times.tolist(), "f0": f0.tolist(), "energy": energy.tolist(),
              "mfcc": C.T.tolist(), "pause_threshold": thr}
    return feats, series

ML_EXCLUDE = {"bandwidth_hz", "bandwidth_ratio", "pause_noise_db", "digital_silence_ratio", "n_pauses",
              "spectral_centroid", "spectral_rolloff", "hf_energy_ratio", "f0_mean"}


def extract_ml_features(audio: AudioData, samples: np.ndarray = None) -> dict:
    x = normalize_for_ml(audio.samples, audio.sr) if samples is None else samples
    a = AudioData(sr=ML_SR, samples=x, channels=x[:, None])
    feats, _ = extract_features(a)
    return {k: v for k, v in feats.items() if k not in ML_EXCLUDE}

class AIDetectionAgent:
    def __init__(self, model_path: str = MODEL_PATH, deep_path: str = DEEP_MODEL_PATH):
        self.deep = DeepDetector(deep_path)
        self.model = None
        self.model_info = None
        if os.path.exists(model_path):
            try:
                import joblib
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.model_info = bundle
            except Exception as exc:
                self.model_info = {"error": str(exc)}

    def tool_bandwidth(self, f):
        bw, ratio = f["bandwidth_hz"], f["bandwidth_ratio"]
        for cut, sr_txt in TTS_CUTOFFS.items():
            if abs(bw - cut) < 0.06 * cut and ratio < 0.85:
                if cut == 8000:
                    return 0.2, 0.4, (f"Спектърът свършва при {bw / 1000:.1f} kHz – записът е направен или "
                                      "генериран при 16 kHz. Така работят някои TTS системи, но и много "
                                      "микрофони (Bluetooth слушалки, уеб камери, разговори), затова "
                                      "признакът е слаб.")
                return 0.6, 0.8, (f"Спектърът свършва рязко при {bw / 1000:.1f} kHz – точно колкото е "
                                  f"Найквист честотата на {sr_txt}. Повечето TTS/voice-cloning модели генерират "
                                  "аудио именно с такава честота, след което то се преобразува нагоре.")
        if ratio < 0.85:
            return 0.15, 0.4, (f"Честотната лента е ограничена до {bw / 1000:.1f} kHz, но не съвпада "
                               "с типичните за TTS стойности (може да е компресия или телефонен запис).")
        return -0.3, 0.6, (f"Записът използва пълната честотна лента ({bw / 1000:.1f} kHz) – "
                           "по-характерно за реален запис с микрофон.")

    def tool_silence(self, f):
        ds, nf = f["digital_silence_ratio"], f["pause_noise_db"]
        if ds > 0.05:
            return 0.6, 0.8, (f"{ds * 100:.0f}% от кадрите са абсолютна цифрова тишина. Реалните микрофони "
                              "винаги улавят фонов шум; генерираните записи често имат „перфектна“ тишина.")
        if nf < -85:
            return 0.4, 0.6, (f"Фоновият шум в паузите е изключително нисък ({nf:.0f} dBFS) – нетипично "
                              "за запис в реална среда.")
        if nf < -68:
            return 0.1, 0.4, (f"Фоновият шум е нисък ({nf:.0f} dBFS) – студиен запис, шумоподтискане "
                              "или синтез.")
        return -0.4, 0.6, (f"Има естествен фонов шум ({nf:.0f} dBFS), характерен за реална акустична среда.")

    def tool_jitter(self, f):
        j = f["f0_jitter"]
        if j < 0.001:
            return 0.6, 0.7, (f"Основният тон е неестествено стабилен (микро-вариации {j * 100:.2f}%). "
                              "Човешкият глас има естествен „jitter“, който синтезаторите често изглаждат.")
        if j < 0.002:
            return 0.25, 0.5, f"Микро-вариациите на основния тон са ниски ({j * 100:.2f}%)."
        if j > 0.004:
            return -0.35, 0.6, (f"Основният тон има естествени микро-вариации ({j * 100:.2f}%), "
                                "характерни за човешки глас.")
        return -0.1, 0.4, f"Микро-вариациите на основния тон са в нормални граници ({j * 100:.2f}%)."

    def tool_intonation(self, f):
        s = f["f0_std_semitones"]
        if s < 1.0:
            return 0.3, 0.4, f"Интонацията е монотонна (разсейване {s:.2f} полутона)."
        if s > 2.5:
            return -0.2, 0.4, f"Интонацията е жива и разнообразна (разсейване {s:.2f} полутона)."
        return 0.0, 0.3, f"Интонацията е в обичайни граници ({s:.2f} полутона)."

    def tool_shimmer(self, f):
        sh = f["shimmer_db"]
        if sh < 0.25:
            return 0.35, 0.5, (f"Силата на гласа се променя твърде плавно между кадрите ({sh:.2f} dB) – "
                               "признак на изгладен (вокодиран) сигнал.")
        if sh > 0.6:
            return -0.25, 0.5, f"Естествени колебания в силата на гласа ({sh:.2f} dB)."
        return 0.0, 0.3, f"Колебанията в силата на гласа са умерени ({sh:.2f} dB)."

    def tool_pauses(self, f):
        cv = f["pause_cv"]
        if cv < 0.2:
            return 0.4, 0.5, (f"Паузите между фразите са почти еднакво дълги (вариация {cv:.2f}) – "
                              "типично за машинно генерирана реч.")
        if cv > 0.5:
            return -0.25, 0.5, f"Паузите са с различна дължина (вариация {cv:.2f}), както при естествена реч."
        return 0.0, 0.3, f"Ритъмът на паузите е умерено регулярен (вариация {cv:.2f})."

    def tool_vocoder(self, f):
        p = f["ltas_periodicity"]
        if p > 0.5:
            return 0.45, 0.6, (f"В средния спектър има периодична структура (корелация {p:.2f}) – "
                               "възможен артефакт от невронен вокодер (upsampling слоеве).")
        return -0.05, 0.3, "Няма изразени периодични артефакти в спектъра."

    def tool_ml(self, f, audio: AudioData = None):
        names = self.model_info["features"]
        if self.model_info.get("input") == "ml16k" and audio is not None:
            f = extract_ml_features(audio)
        X = np.array([[f.get(n, 0.0) for n in names]])
        p = float(self.model.predict_proba(X)[0, 1])
        return p

    def run(self, audio: AudioData) -> dict:
        trace = []
        feats, series = extract_features(audio)

        trace.append({"type": "plan", "text":
                      f"Получих запис с продължителност {audio.duration:.1f} s при {audio.sr} Hz. "
                      "Първо проверявам дали съдържа реч, за да реша кои проверки да използвам."})
        speech = feats["voiced_ratio"] > 0.12 and feats["f0_mean"] > 0
        if speech:
            tools = [("Честотна лента", self.tool_bandwidth), ("Тишина и фонов шум", self.tool_silence),
                     ("Стабилност на основния тон (jitter)", self.tool_jitter),
                     ("Интонация", self.tool_intonation), ("Динамика на силата (shimmer)", self.tool_shimmer),
                     ("Ритъм на паузите", self.tool_pauses), ("Артефакти от вокодер", self.tool_vocoder)]
            trace.append({"type": "decision", "text":
                          f"Открита е реч ({feats['voiced_ratio'] * 100:.0f}% звучни кадри, среден тон "
                          f"{feats['f0_mean']:.0f} Hz). Ще изпълня всички {len(tools)} акустични проверки."})
        else:
            tools = [("Честотна лента", self.tool_bandwidth), ("Тишина и фонов шум", self.tool_silence),
                     ("Артефакти от вокодер", self.tool_vocoder)]
            trace.append({"type": "decision", "text":
                          "Почти няма звучна реч. Проверките за интонация и тон са неприложими – "
                          "изпълнявам само общите спектрални проверки и намалявам увереността си."})
        if feats["n_pauses"] < 3:
            tools = [t for t in tools if t[1] != self.tool_pauses]

        votes = []
        for name, tool in tools:
            s, w, text = tool(feats)
            votes.append({"tool": name, "vote": round(s, 2), "weight": w, "text": text})
            trace.append({"type": "observation", "tool": name, "vote": round(s, 2), "text": text})

        num = sum(v["vote"] * v["weight"] for v in votes)
        den = sum(v["weight"] for v in votes) + EPS
        rule_score = num / den
        rule_prob = 1 / (1 + np.exp(-4.0 * rule_score))
        if not speech:
            rule_prob = 0.5 + (rule_prob - 0.5) * 0.6

        components = [("Експертни правила", rule_prob, 0.3)]
        ml_prob = None
        if self.model is not None:
            ml_prob = self.tool_ml(feats, audio)
            components.append(("Random Forest (MFCC + признаци)", ml_prob, 0.25))
            trace.append({"type": "observation", "tool": "Класификатор върху MFCC признаци",
                          "vote": round(2 * ml_prob - 1, 2),
                          "text": f"Обученият класификатор ({self.model_info.get('name', 'модел')}) дава "
                                  f"{ml_prob * 100:.0f}% вероятност за синтетичен запис."})
        deep = None
        if self.deep.available:
            deep = self.deep.predict(audio.samples, audio.sr)
            dp = deep["probability"]
            sp = np.array(deep["segment_probs"])
            components.append(("CNN + BiLSTM невронна мрежа", dp, 0.45))
            spread = (f" Вероятността по 3-секундни прозорци е между {sp.min() * 100:.0f}% и {sp.max() * 100:.0f}%"
                      + (" – някои части изглеждат синтетични, а други не (възможна смесица)."
                         if sp.max() - sp.min() > 0.5 else "."))
            trace.append({"type": "observation", "tool": "Невронна мрежа CNN + BiLSTM",
                          "vote": round(2 * dp - 1, 2),
                          "text": f"Дълбоката мрежа анализира {len(sp)} прозореца ({self.deep.meta.get('features', 'mfcc').upper()}"
                                  f" → CNN → BiLSTM → attention) и дава {dp * 100:.0f}% вероятност за синтез." + spread})
        elif self.deep.error:
            trace.append({"type": "note", "text": f"Невронната мрежа не може да се зареди: {self.deep.error}."})

        if len(components) == 1:
            prob = rule_prob
            trace.append({"type": "note", "text":
                          "Няма обучени модели, затова решението се базира само на експертните правила. "
                          "Обучете модел с train_deep.py (CNN + BiLSTM) и/или train_model.py (Random Forest)."})
        else:
            wsum = sum(c[2] for c in components)
            prob = sum(c[1] * c[2] for c in components) / wsum
            trace.append({"type": "decision", "text": "Обединявам оценките с тегла: " + ", ".join(
                f"{n} {p * 100:.0f}% (тегло {w / wsum:.2f})" for n, p, w in components) + "."})

        pos = sorted([v for v in votes if v["vote"] > 0], key=lambda v: -v["vote"] * v["weight"])
        neg = sorted([v for v in votes if v["vote"] < 0], key=lambda v: v["vote"] * v["weight"])
        if prob >= 0.65:
            verdict, level = "Вероятно ГЕНЕРИРАН с изкуствен интелект", "high"
        elif prob >= 0.4:
            verdict, level = "НЕСИГУРНО - има смесени признаци", "medium"
        else:
            verdict, level = "Вероятно ЕСТЕСТВЕН (човешки) запис", "low"

        summary = f"Заключение: {verdict.lower()} ({prob * 100:.0f}%)."
        if pos:
            summary += " Основни аргументи „за“ синтез: " + "; ".join(v["tool"].lower() for v in pos[:3]) + "."
        if neg:
            summary += " Аргументи за естествен запис: " + "; ".join(v["tool"].lower() for v in neg[:3]) + "."
        trace.append({"type": "conclusion", "text": summary})

        return {
            "probability": round(prob * 100, 1),
            "rule_probability": round(rule_prob * 100, 1),
            "ml_probability": None if ml_prob is None else round(ml_prob * 100, 1),
            "verdict": verdict,
            "level": level,
            "votes": votes,
            "trace": trace,
            "features": {k: round(v, 5) if isinstance(v, float) else v for k, v in feats.items()
                         if not k.startswith("mfcc") or k == "mfcc_delta_mean"},
            "series": series,
            "model_loaded": self.model is not None,
            "deep_probability": None if deep is None else round(deep["probability"] * 100, 1),
            "deep": deep,
            "deep_meta": self.deep.meta if self.deep.available else None,
            "components": [{"name": n, "probability": round(p * 100, 1)} for n, p, _ in components],
        }