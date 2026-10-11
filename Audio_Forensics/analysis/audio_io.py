import os
import struct
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile


@dataclass
class AudioData:
    sr: int
    samples: np.ndarray
    channels: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def _to_float(data: np.ndarray):
    dtype = data.dtype
    if dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0, 8
    if dtype == np.int16:
        return data.astype(np.float64) / 32768.0, 16
    if dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0, 32
    if np.issubdtype(dtype, np.floating):
        return data.astype(np.float64), 32 if dtype == np.float32 else 64
    raise ValueError(f"Неподдържан формат на данните: {dtype}")


def _effective_bits(raw: np.ndarray, container_bits: int, bits: int) -> int:
    if raw.dtype == np.uint8:
        return bits
    if np.issubdtype(raw.dtype, np.floating):
        x = raw[raw != 0].astype(np.float64)[:200000] * 32768.0
        x = x[np.abs(x) >= 4]
        if x.size and np.mean(np.abs(x - np.round(x)) <= 1 / 16) > 0.5:
            return 16
        return bits
    v = raw[raw != 0].astype(np.int64)[:200000] >> max(0, container_bits - bits)
    if v.size == 0:
        return bits
    combined = np.bitwise_or.reduce(np.abs(v))
    trailing = 0
    while trailing < bits and not (combined >> trailing) & 1:
        trailing += 1
    exact = bits - trailing
    for b in (16, 20):
        if b >= exact:
            break
        step = 1 << (bits - b)
        w = v[np.abs(v) >= 4 * step]
        r = (w + step // 2) % step - step // 2
        if w.size >= 1000 and np.mean(np.abs(r) <= step // 16) > 0.5:
            return b
    return exact


def _header_bits(path_or_file):
    own = isinstance(path_or_file, (str, os.PathLike))
    f = open(path_or_file, "rb") if own else path_or_file
    pos = None if own else f.tell()
    try:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            return None
        while True:
            head = f.read(8)
            if len(head) < 8:
                return None
            chunk_id, size = head[:4], struct.unpack("<I", head[4:])[0]
            if chunk_id == b"fmt ":
                fmt = f.read(size)
                tag, bits = struct.unpack("<H", fmt[:2])[0], struct.unpack("<H", fmt[14:16])[0]
                if tag == 0xFFFE and size >= 20:         
                    valid = struct.unpack("<H", fmt[18:20])[0]
                    bits = valid if 0 < valid <= bits else bits
                return bits
            f.seek(size + (size & 1), 1)                 
    except (OSError, struct.error):
        return None
    finally:
        if own:
            f.close()
        else:
            f.seek(pos)


def load_wav(path_or_file, max_seconds: float = 600.0) -> AudioData:
    header_bits = _header_bits(path_or_file)
    sr, raw = wavfile.read(path_or_file)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape[0] == 0:
        raise ValueError("Файлът не съдържа аудио данни.")

    truncated = False
    max_len = int(max_seconds * sr)
    if raw.shape[0] > max_len:
        raw = raw[:max_len]
        truncated = True

    data, container_bits = _to_float(raw)
    data = np.nan_to_num(data)
    mono = data.mean(axis=1)
    integer = np.issubdtype(raw.dtype, np.integer)
    bit_depth = header_bits if integer and header_bits else container_bits

    identical_channels = False
    if data.shape[1] == 2:
        diff_rms = np.sqrt(np.mean((data[:, 0] - data[:, 1]) ** 2))
        sig_rms = np.sqrt(np.mean(mono ** 2))
        identical_channels = bool(diff_rms < 5e-4 or diff_rms < 1e-3 * sig_rms)

    info = {
        "sample_rate": int(sr),
        "channels": int(data.shape[1]),
        "bit_depth": int(bit_depth),
        "effective_bits": int(min(bit_depth, _effective_bits(raw, container_bits, bit_depth))),
        "dtype": str(raw.dtype),
        "duration_s": round(raw.shape[0] / sr, 3),
        "identical_channels": identical_channels,
        "truncated": truncated,
        "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(mono)) + 1e-12)), 2),
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-12)), 2),
    }
    return AudioData(sr=int(sr), samples=mono, channels=data, info=info)