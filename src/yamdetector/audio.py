"""Audio loading, channel handling, resampling, slicing and normalization.

This module is shared by stage 1 (ESC-50) and stage 2 (own recordings) so both
produce identical 16 kHz mono 1 s windows. It never depends on where the audio
came from or how many microphones recorded it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioError(RuntimeError):
    """Raised for unreadable audio or invalid slicing parameters."""


CHANNEL_MODES = ("index", "mix")


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any WAV/FLAC/OGG file as float32 ``(frames, channels)`` plus its rate."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")
    try:
        samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:  # soundfile raises its own error types
        raise AudioError(f"could not read {path}: {exc}") from exc
    if samples.size == 0:
        raise AudioError(f"{path} contains no audio samples")
    if sample_rate <= 0:
        raise AudioError(f"{path} reports an invalid sample rate: {sample_rate}")
    return samples, int(sample_rate)


def select_channels(
    samples: np.ndarray, mode: str = "index", channel: int = 0
) -> tuple[np.ndarray, str | None]:
    """Reduce ``(frames, channels)`` to mono.

    ``mode="index"`` takes one channel (``channel``), ``mode="mix"`` averages all
    of them. Returns ``(mono, warning)`` where ``warning`` is a human-readable
    note when the file had more than one channel (or when the requested channel
    does not exist and it fell back).
    """
    if samples.ndim != 2:
        raise AudioError(f"expected a (frames, channels) array, got shape {samples.shape}")
    if mode not in CHANNEL_MODES:
        raise AudioError(f"unknown channel mode {mode!r}; expected one of {CHANNEL_MODES}")

    channel_count = samples.shape[1]
    if mode == "mix":
        mono = samples.mean(axis=1, dtype=np.float32)
        if channel_count > 1:
            return mono, f"mixed {channel_count} channels to mono"
        return mono, None

    if channel < 0:
        raise AudioError(f"--channel must be >= 0, got {channel}")
    warning = None
    if channel >= channel_count:
        warning = (
            f"file has {channel_count} channel(s); requested channel {channel} does not "
            "exist, using channel 0 (pass --mix to average all channels)"
        )
        channel = 0
    elif channel_count > 1:
        warning = (
            f"file has {channel_count} channels; using channel {channel} "
            "(pass --mix to average all channels)"
        )
    return samples[:, channel].astype(np.float32, copy=True), warning


def resample(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample a mono signal to ``target_rate`` (float32)."""
    if mono.ndim != 1:
        raise AudioError(f"resample expects mono audio, got shape {mono.shape}")
    if source_rate <= 0 or target_rate <= 0:
        raise AudioError(f"invalid sample rates: {source_rate} -> {target_rate}")
    if source_rate == target_rate:
        return mono.astype(np.float32, copy=False)
    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    try:
        resampled = resample_poly(mono, up, down)
    except Exception as exc:  # pragma: no cover - scipy edge cases
        raise AudioError(f"could not resample {source_rate} Hz -> {target_rate} Hz: {exc}") from exc
    return np.ascontiguousarray(resampled, dtype=np.float32)


def rms(mono: np.ndarray) -> float:
    """Root-mean-square level of a window (0.0 for an empty/all-zero window)."""
    if mono.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))


def peak_normalize(
    mono: np.ndarray, target_peak: float = 0.95, max_gain_db: float = 20.0
) -> np.ndarray:
    """Scale so the peak sits at ``target_peak``, never boosting past ``max_gain_db``.

    Silent input is returned as zeros (no division by zero). Output never clips
    because ``target_peak`` is always <= 1.
    """
    if mono.size == 0:
        return mono.astype(np.float32, copy=True)
    peak = float(np.max(np.abs(mono)))
    if peak <= 0.0:
        return np.zeros_like(mono, dtype=np.float32)
    gain = min(target_peak / peak, 10.0 ** (max_gain_db / 20.0))
    return np.ascontiguousarray(mono * gain, dtype=np.float32)


def slice_windows(
    mono: np.ndarray,
    window_samples: int,
    hop_samples: int | None = None,
    pad_last: bool = True,
    min_tail_fraction: float = 0.25,
) -> np.ndarray:
    """Cut a mono signal into fixed-length windows, shape ``(n_windows, window_samples)``.

    Windows start every ``hop_samples`` samples (default: no overlap). A short
    trailing window is zero-padded when it keeps at least ``min_tail_fraction``
    of real audio; otherwise it is dropped. A recording shorter than one window
    yields a single zero-padded window (when ``pad_last`` is True).
    """
    window_samples = int(window_samples)
    hop_samples = int(hop_samples) if hop_samples is not None else window_samples
    if window_samples <= 0:
        raise AudioError(f"window_samples must be > 0, got {window_samples}")
    if hop_samples <= 0:
        raise AudioError(f"hop_samples must be > 0, got {hop_samples}")
    if not 0.0 <= min_tail_fraction <= 1.0:
        raise AudioError(f"min_tail_fraction must be in [0, 1], got {min_tail_fraction}")

    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    total = mono.size
    if total == 0:
        return np.zeros((0, window_samples), dtype=np.float32)

    starts = list(range(0, max(total - window_samples + 1, 0), hop_samples))
    frames = [mono[start : start + window_samples] for start in starts]

    if not starts:
        if not pad_last:
            raise AudioError(
                f"recording has {total} samples, shorter than one {window_samples}-sample "
                "window"
            )
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total] = mono
        frames.append(padded)
    else:
        # Audio left after the last full window: pad it into one extra window when
        # it is long enough to be meaningful, otherwise drop it.
        leftover = total - (starts[-1] + window_samples)
        if pad_last and leftover > 0 and leftover >= min_tail_fraction * window_samples:
            tail_start = starts[-1] + hop_samples
            tail_length = total - tail_start
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[:tail_length] = mono[tail_start : tail_start + tail_length]
            frames.append(padded)

    return np.stack(frames).astype(np.float32, copy=False)


def load_clip(
    path: str | Path,
    target_rate: int,
    channel_mode: str = "index",
    channel: int = 0,
) -> tuple[np.ndarray, str | None]:
    """Read a file, pick/average channels, resample to ``target_rate``."""
    samples, sample_rate = read_audio(path)
    mono, warning = select_channels(samples, channel_mode, channel)
    mono = resample(mono, sample_rate, target_rate)
    return mono, warning


def write_wav(path: str | Path, mono: np.ndarray, sample_rate: int) -> None:
    """Write float32 mono PCM_16 audio, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(mono, -1.0, 1.0).astype(np.float32)
    try:
        sf.write(str(path), clipped, int(sample_rate), subtype="PCM_16")
    except Exception as exc:  # pragma: no cover - disk/permission issues
        raise AudioError(f"could not write {path}: {exc}") from exc
