"""Microfrontend features (TensorFlow audio_microfrontend) with a disk cache.

The same frontend runs on the ESP32-S3 (TFLite Micro `MicroFrontend`), so the
settings here are the contract with the firmware:

* 16 kHz int16 input
* 30 ms window / 20 ms step / 40 channels, 125 Hz - 7500 Hz
* log + PCAN enabled (frontend defaults, kept identical to the device)

One 1 s window -> 49 frames x 40 channels. Each window is converted separately
because the op smooths each frame against the preceding ones; concatenating
windows would leak across their boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import PROJECT_ROOT

MICROFRONTEND_MODULE = "tensorflow.lite.experimental.microfrontend.python.ops.audio_microfrontend_op"


class FeatureError(RuntimeError):
    """Raised when features cannot be computed (missing op, bad audio, ...)."""


def _load_op():
    try:
        from tensorflow.lite.experimental.microfrontend.python.ops import (
            audio_microfrontend_op as frontend,
        )
    except Exception as exc:  # pragma: no cover - depends on the TF build
        raise FeatureError(
            "TensorFlow's audio_microfrontend op is unavailable in this environment.\n"
            f"Expected module: {MICROFRONTEND_MODULE}\n"
            "Install a full TensorFlow build (pip install tensorflow) - the op is not "
            "part of tensorflow-cpu-nightly/micro builds, and GPU runs need WSL2/Linux."
        ) from exc
    return frontend


def signature(cfg: dict) -> str:
    """Short hash of every setting that changes the feature values."""
    features = cfg["features"]
    audio = cfg["audio"]
    parts = [
        features["kind"],
        features["channels"],
        features["window_ms"],
        features["step_ms"],
        features["lower_band_limit_hz"],
        features["upper_band_limit_hz"],
        features["output_scale_shift"],
        int(features["left_context"]),
        int(features["right_context"]),
        features["value_scale"],
        audio["sample_rate"],
        audio["window_samples"],
    ]
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:10]


def frames_for_window(cfg: dict) -> int:
    return int(cfg["features"]["n_frames"])


def microfrontend(waves: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert ``(n, window_samples)`` float32 audio to ``(n, frames, channels)``.

    Values are divided by ``features.value_scale`` so they land around [0, 1]
    (the firmware divides by the same constant before quantizing).
    """
    waves = np.asarray(waves, dtype=np.float32)
    if waves.ndim == 1:
        waves = waves[None, :]
    expected = int(cfg["audio"]["window_samples"])
    if waves.shape[1] != expected:
        raise FeatureError(
            f"expected {expected}-sample windows, got shape {waves.shape}; "
            "audio.window_seconds and audio.sample_rate in config.yaml must match the "
            "windows written by the preparation scripts"
        )

    import tensorflow as tf

    frontend = _load_op()
    features = cfg["features"]
    out_frames = frames_for_window(cfg)
    out_channels = int(features["channels"])
    value_scale = float(features["value_scale"])

    result = np.empty((waves.shape[0], out_frames, out_channels), dtype=np.float32)
    for index, wave in enumerate(waves):
        pcm = np.clip(np.round(wave * 32767.0), -32768, 32767).astype(np.int16)
        frames = frontend.audio_microfrontend(
            tf.constant(pcm),
            sample_rate=int(cfg["audio"]["sample_rate"]),
            window_size=int(features["window_ms"]),
            window_step=int(features["step_ms"]),
            num_channels=out_channels,
            upper_band_limit=float(features["upper_band_limit_hz"]),
            lower_band_limit=float(features["lower_band_limit_hz"]),
            scale_shift=6,
            out_scale=1,
            out_type=tf.float32,
        ).numpy()
        if frames.shape != (out_frames, out_channels):
            raise FeatureError(
                "the microfrontend produced "
                f"{frames.shape} for a {expected}-sample window, expected "
                f"{(out_frames, out_channels)}; check features.window_ms/step_ms/channels"
            )
        result[index] = frames
    return result / value_scale


def _base_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["feature_cache_dir"]) / signature(cfg)


def _stat_key(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError as exc:
        raise FeatureError(f"cannot cache features, missing audio file: {path}") from exc
    return hashlib.sha1(f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:8]


def cache_path(cfg: dict, wav_path: str | Path) -> Path:
    """Where the features of one processed wav live (invalidated when the wav changes)."""
    wav_path = Path(wav_path)
    return _base_dir(cfg) / wav_path.parent.name / f"{wav_path.stem}.{_stat_key(wav_path)}.npy"


def augmented_cache_path(cfg: dict, wav_path: str | Path, copy_index: int) -> Path:
    """Cache slot for the ``copy_index``-th augmented version of a training window."""
    wav_path = Path(wav_path)
    return (
        _base_dir(cfg)
        / wav_path.parent.name
        / f"{wav_path.stem}.{_stat_key(wav_path)}.aug{copy_index}.npy"
    )


def load_feature(path: str | Path, cfg: dict) -> np.ndarray:
    """Load one cached feature file as ``(frames, channels, 1)`` float32."""
    array = np.load(Path(path), allow_pickle=False)
    expected = tuple(cfg["features"]["input_shape"])
    if array.shape != expected:
        raise FeatureError(
            f"cached feature {path} has shape {array.shape}, expected {expected}; "
            "delete the feature cache and recompute"
        )
    return array.astype(np.float32, copy=False)


def read_windows(wav_path: str | Path) -> np.ndarray:
    """Read a processed wav (already exactly one window) as float32 mono."""
    import soundfile as sf

    path = Path(wav_path)
    try:
        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise FeatureError(f"could not read {path}: {exc}") from exc
    if data.shape[1] > 1:
        raise FeatureError(
            f"{path} has {data.shape[1]} channels; the standardized dataset folder must "
            "contain mono 16 kHz wavs (re-run the preparation scripts)"
        )
    del sample_rate
    return data[:, 0]


def compute_missing(
    cfg: dict,
    jobs: Sequence[tuple[Path, Path]],
    force: bool = False,
    log: Callable[[str], None] = print,
    chunk: int = 64,
) -> int:
    """Compute cached features for ``(wav_path, cache_path)`` jobs that are missing.

    Returns the number of files computed. Windows are read/computed in small
    chunks so memory stays flat regardless of dataset size.
    """
    pending = [(wav, out) for wav, out in jobs if force or not Path(out).exists()]
    if not pending:
        return 0
    total = len(pending)
    for start in range(0, total, chunk):
        batch = pending[start : start + chunk]
        waves = np.stack([read_windows(wav) for wav, _ in batch])
        computed = microfrontend(waves, cfg)
        for (_, out), feature in zip(batch, computed):
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, feature[..., None].astype(np.float32))
        del waves, computed
        log(f"  features {min(start + chunk, total)}/{total}")
    return total


def feature_paths(cfg: dict, rows: Sequence[dict]) -> list[Path]:
    """Cache paths for manifest rows (relative manifest paths resolve to the project root)."""
    return [cache_path(cfg, resolve_row_path(row)) for row in rows]


def resolve_row_path(row: dict) -> Path:
    path = Path(row["path"])
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_split_features(cfg: dict, rows: Sequence[dict]) -> np.ndarray:
    """Stack cached features for a split into ``(n, frames, channels, 1)``."""
    paths = feature_paths(cfg, rows)
    if not paths:
        return np.zeros((0, *cfg["features"]["input_shape"]), dtype=np.float32)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FeatureError(
            f"{len(missing)} cached feature file(s) are missing, e.g. {missing[0]}\n"
            "run train.py first, or delete data/features and rerun it"
        )
    return np.stack([load_feature(path, cfg) for path in paths])


def human_bytes(count: int) -> str:
    if count > 1024 * 1024:
        return f"{count / (1024 * 1024):.1f} MiB"
    if count > 1024:
        return f"{count / 1024:.1f} KiB"
    return f"{count} B"
