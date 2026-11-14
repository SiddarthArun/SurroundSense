"""Waveform augmentation used to expand the training split only.

Augmentations are applied to the 1 s waveform *before* the microfrontend, so the
model sees features of realistic mixtures (background noise at 0-20 dB SNR),
level changes and small time shifts - the same variations a wearable sees.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from . import features


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``clean`` scaled to the requested signal-to-noise ratio."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean {clean.shape} and noise {noise.shape} must have equal length")
    clean_power = float(np.mean(np.square(clean, dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    if clean_power <= 1e-12:
        # Silent target: keep the noise at its own level, nothing to scale against.
        return np.clip(noise, -1.0, 1.0).astype(np.float32)
    if noise_power <= 1e-12:
        return clean.astype(np.float32, copy=True)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(clean + scaled_noise, -1.0, 1.0).astype(np.float32)


def augment_waveform(
    clean: np.ndarray,
    noise: np.ndarray | None,
    rng: np.random.Generator,
    snr_db_range: Sequence[float],
    gain_db_range: Sequence[float],
    shift_samples: int,
    noise_probability: float,
) -> np.ndarray:
    """Random time shift + gain + optional background-noise mixing."""
    wave = np.asarray(clean, dtype=np.float32).copy()

    if shift_samples > 0:
        shift = int(rng.integers(-shift_samples, shift_samples + 1))
        if shift:
            wave = np.roll(wave, shift)

    gain_db = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    wave = np.clip(wave * (10.0 ** (gain_db / 20.0)), -1.0, 1.0)

    if noise is not None and rng.random() < noise_probability:
        snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
        wave = mix_at_snr(wave, np.asarray(noise, dtype=np.float32), snr_db)

    return np.ascontiguousarray(wave, dtype=np.float32)


def build_noise_pool(
    cfg: dict, rows: Sequence[dict], seed: int, log=print
) -> np.ndarray | None:
    """Load a bounded set of background windows into RAM for noise mixing.

    Only ``augment.noise_pool_size`` windows are kept, so RAM use is fixed no
    matter how large the background class is. Returns ``(n, window_samples)``
    or ``None`` when there is no background material.
    """
    background_label = cfg["classes"]["background"]
    candidates = [row for row in rows if row["class"] == background_label]
    if not candidates:
        log(
            f"  warning: no '{background_label}' windows in the training split, "
            "background-noise mixing is disabled"
        )
        return None

    limit = int(cfg["augment"]["noise_pool_size"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[: min(limit, len(candidates))]
    pool = []
    for index in np.sort(order):
        row = candidates[int(index)]
        try:
            pool.append(features.read_windows(features.resolve_row_path(row)))
        except features.FeatureError as exc:  # pragma: no cover - corrupt file
            log(f"  warning: skipped noise window ({exc})")
    if not pool:
        return None
    log(f"  noise pool: {len(pool)} background windows ({pool[0].size} samples each)")
    return np.stack(pool)


def write_augmented_feature(
    cfg: dict,
    wav_path: Path,
    out_path: Path,
    copy_index: int,
    noise_pool: np.ndarray | None,
    seed: int,
) -> None:
    """Generate one augmented window and cache its features."""
    augment = cfg["augment"]
    shift_samples = int(round(float(augment["shift_ms"]) * cfg["audio"]["sample_rate"] / 1000.0))

    # Deterministic per (row, copy) across runs and machines: hash, not hash().
    digest = hashlib.sha1(f"{seed}|{wav_path}|{copy_index}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))

    clean = features.read_windows(wav_path)
    noise = None
    if noise_pool is not None:
        noise = noise_pool[int(rng.integers(0, noise_pool.shape[0]))]
    augmented = augment_waveform(
        clean,
        noise,
        rng,
        snr_db_range=augment["noise_snr_db"],
        gain_db_range=augment["gain_db"],
        shift_samples=shift_samples,
        noise_probability=float(augment["noise_probability"]),
    )
    feature = features.microfrontend(augmented, cfg)[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feature[..., None].astype(np.float32))
