"""Laptop audio: record the wake word, build "wake word + command" trials, play them."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler (enough for speech playback / 16 kHz input)."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.arange(n_out) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(max(rms, 1e-9))


def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop).astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)        # ignore DC offset
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def _threshold(e: np.ndarray, rel_db: float, floor_db: float, snr_db: float) -> float:
    """Speech threshold: above the noise floor (10th-percentile frame) by snr_db,
    and no more than rel_db below the loudest frame."""
    noise = np.percentile(e, 10)
    thr = max(noise + snr_db, floor_db)
    return min(thr, e.max() - 6) if e.max() - 6 > noise else max(thr, e.max() - rel_db)


def trim(x: np.ndarray, sr: int = SR, pad_s: float = 0.08, rel_db: float = 35.0,
         floor_db: float = -50.0, snr_db: float = 12.0) -> np.ndarray:
    """Cut leading/trailing silence (20 ms frames).

    The threshold adapts to the recording's noise floor, so a laptop mic with
    fan/room noise above -50 dBFS still gets trimmed.
    """
    hop = int(0.02 * sr)
    if len(x) < hop * 2:
        return x
    e = _frame_db(x, hop)
    voiced = np.flatnonzero(e > _threshold(e, rel_db, floor_db, snr_db))
    if len(voiced) == 0:
        return x
    pad = int(pad_s * sr)
    a = max(voiced[0] * hop - pad, 0)
    b = min((voiced[-1] + 1) * hop + pad, len(x))
    return x[a:b]


def main_burst(x: np.ndarray, sr: int = SR, max_gap_s: float = 0.25, pad_s: float = 0.15,
               snr_db: float = 12.0) -> np.ndarray:
    """Keep only the loudest stretch of speech (for a one-word wake word take).

    Voiced frames closer than max_gap_s are merged into one segment; the
    segment with the most energy wins. Drops key clicks, breaths and noise
    bursts before/after the word.
    """
    hop = int(0.02 * sr)
    if len(x) < hop * 2:
        return x
    e = _frame_db(x, hop)
    voiced = np.flatnonzero(e > _threshold(e, 35.0, -50.0, snr_db))
    if len(voiced) == 0:
        return x
    gap = int(max_gap_s / 0.02)
    segs, start, prev = [], voiced[0], voiced[0]
    for v in voiced[1:]:
        if v - prev > gap:
            segs.append((start, prev))
            start = v
        prev = v
    segs.append((start, prev))
    power = 10 ** (e / 10)
    a, b = max(segs, key=lambda s: power[s[0]: s[1] + 1].sum())
    pad = int(pad_s * sr)
    return x[max(a * hop - pad, 0): min((b + 1) * hop + pad, len(x))]


def normalize(x: np.ndarray, target_dbfs: float = -20.0, peak_limit: float = 0.95) -> np.ndarray:
    """Scale speech to a target RMS level (measured on voiced frames), never clipping."""
    hop = int(0.02 * SR)
    n = max(len(x) // hop, 1)
    frames = x[: n * hop].reshape(n, -1) if len(x) >= hop else x.reshape(1, -1)
    e = np.mean(frames.astype(np.float64) ** 2, axis=1)
    voiced = e[e > e.max() * 10 ** (-30 / 10)] if e.max() > 0 else e
    rms = np.sqrt(voiced.mean()) if len(voiced) else 0.0
    if rms <= 0:
        return x
    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = np.abs(y).max()
    if peak > peak_limit:
        y *= peak_limit / peak
    return y.astype(np.float32)


def fade(x: np.ndarray, ms: float = 10.0) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(x) // 2)
    if n:
        x = x.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def patch(wake: np.ndarray, command: np.ndarray, gap_s: float, lead_s: float = 0.3,
          tail_s: float = 0.5) -> tuple[np.ndarray, dict]:
    """[lead silence][wake word][gap][command][tail silence], all 16 kHz.

    Returns the audio and the offsets (seconds) where each part starts/ends,
    used later to time the Pi's response from the end of the command.
    """
    z = lambda s: np.zeros(int(s * SR), dtype=np.float32)
    parts = [z(lead_s), fade(wake), z(gap_s), fade(command), z(tail_s)]
    out = np.concatenate(parts)
    t_wake0 = lead_s
    t_wake1 = t_wake0 + len(wake) / SR
    t_cmd0 = t_wake1 + gap_s
    t_cmd1 = t_cmd0 + len(command) / SR
    return out, {"wake_start": t_wake0, "wake_end": t_wake1, "cmd_start": t_cmd0,
                 "cmd_end": t_cmd1, "total": len(out) / SR}


def save(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype="PCM_16")


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample(x.astype(np.float32), sr, SR)


# ---------------------------------------------------------------- devices

def _sd():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:  # OSError: PortAudio library missing
        raise SystemExit(
            "sounddevice/PortAudio is not available. Install it with `pip install sounddevice` "
            "(Linux: also `sudo apt install libportaudio2`).\n" + str(e))
    return sd


def list_devices(kind: str) -> list[tuple[int, str, bool]]:
    """[(index, name, is_default)] for kind 'input' or 'output'."""
    sd = _sd()
    default = sd.default.device[0 if kind == "input" else 1]
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    return [(i, d["name"], i == default) for i, d in enumerate(sd.query_devices()) if d[key] > 0]


def record(seconds: float, device: int | None = None) -> np.ndarray:
    sd = _sd()
    info = sd.query_devices(device, "input")
    sr = int(info["default_samplerate"])
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return resample(x[:, 0], sr, SR)


class Player:
    """Plays 16 kHz audio on the chosen output and reports when playback really started."""

    def __init__(self, device: int | None = None, volume: float = 1.0):
        self.sd = _sd()
        self.device = device
        self.volume = volume
        info = self.sd.query_devices(device, "output")
        self.sr = int(info["default_samplerate"])

    def play(self, x: np.ndarray) -> float:
        """Play and block; return the laptop time (time.time()) of the first sample at the speaker."""
        y = np.clip(resample(x, SR, self.sr) * self.volume, -1, 1)
        stream = self.sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      device=self.device)
        stream.start()
        t0 = time.time() + float(stream.latency)
        stream.write(y.reshape(-1, 1))
        stream.stop()      # waits until the buffered audio has played
        stream.close()
        return t0
