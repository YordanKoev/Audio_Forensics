LABELS = {
    "sample_rate": "Честота на дискретизация (Hz)",
    "channels": "Брой канали",
    "bit_depth": "Битова дълбочина",
    "effective_bits": "Реално използвани битове",
    "dtype": "Формат на отчетите",
    "duration_s": "Продължителност (s)",
    "identical_channels": "Еднакви канали",
    "truncated": "Анализирана само част от файла",
    "peak_dbfs": "Пиково ниво (dBFS)",
    "rms_dbfs": "Средно ниво (dBFS)",
    "bandwidth_hz": "Честотна лента (Hz)",
    "bandwidth_ratio": "Дял от възможната честотна лента",
    "digital_silence_ratio": "Дял на цифровата тишина",
    "pause_noise_db": "Фонов шум в паузите (dBFS)",
    "n_pauses": "Брой паузи",
    "pause_cv": "Неравномерност на паузите",
    "voiced_ratio": "Дял на озвучената реч",
    "f0_mean": "Среден основен тон (Hz)",
    "f0_std_semitones": "Разлика във височината на тона (полутонове)",
    "f0_jitter": "Нестабилност на тона (jitter)",
    "f0_delta": "Промяна на тона между кадрите",
    "shimmer_db": "Нестабилност на силата (shimmer, dB)",
    "spectral_centroid": "Спектрален център (Hz)",
    "spectral_rolloff": "Граница на 95% от енергията (Hz)",
    "hf_flatness": "Шумност на високите честоти",
    "hf_energy_ratio": "Дял на енергията над 4 kHz",
    "ltas_periodicity": "Периодичност в средния спектър",
    "mfcc_delta_mean": "Скорост на промяна на тембъра",
}


def label(key: str) -> str:
    return LABELS.get(key, key)


def display_value(value):
    if isinstance(value, bool):
        return "да" if value else "не"
    return value