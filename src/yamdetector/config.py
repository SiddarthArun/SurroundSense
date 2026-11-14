"""config.yaml loading, validation and derived values.

Every stage loads the config through :func:`load_config`, which resolves the
paths under ``paths:`` to absolute ``Path`` objects and validates the values
that the audio/feature code relies on. Failures raise :class:`ConfigError`
with a message that says which key is wrong.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SPLIT_NAMES = ("train", "val", "test")

# Dataset layouts tried in order when the configured ESC-50 paths do not exist.
_ESC50_CANDIDATES = (
    ("audio", "meta/esc50.csv"),
    ("data/ESC-50/audio", "data/ESC-50/meta/esc50.csv"),
    ("ESC-50/audio", "ESC-50/meta/esc50.csv"),
)

_PATH_KEYS = (
    "esc50_audio_dir",
    "esc50_meta_csv",
    "raw_recordings_dir",
    "processed_dir",
    "manifest_csv",
    "feature_cache_dir",
    "output_dir",
)


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing something the pipeline needs."""


def _get(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise ConfigError(f"config.yaml: '{where}' must be a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise ConfigError(f"config.yaml: missing required key '{where}{key}'")
    return mapping[key]


def _get_section(cfg: dict, name: str) -> dict:
    section = cfg.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"config.yaml: missing or invalid top-level section '{name}:'")
    return section


def _positive_number(section: dict, key: str, where: str) -> float:
    value = _get(section, key, where)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"config.yaml: '{where}{key}' must be a positive number, got {value!r}")
    return float(value)


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate config.yaml, returning a plain dict of settings.

    Keys added by this function: ``_path`` (config file used) and, per section,
    resolved paths plus derived sample counts (``audio.window_samples``,
    ``audio.hop_samples``, ``features.window_size``, ``features.stride``,
    ``features.n_frames``, ``features.input_shape``).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise ConfigError(
            f"config file not found: {config_path}\n"
            "Pass --config <path> or run the scripts from the project root."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        try:
            cfg = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - depends on user file
            raise ConfigError(f"could not parse {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"config.yaml must contain a YAML mapping, got {type(cfg).__name__}")

    cfg["_path"] = config_path

    # --- paths -------------------------------------------------------------
    paths = _get_section(cfg, "paths")
    root = config_path.parent
    resolved: dict[str, Path] = {}
    for key in _PATH_KEYS:
        value = _get(paths, key, "paths.")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"config.yaml: 'paths.{key}' must be a non-empty string")
        candidate = Path(value).expanduser()
        resolved[key] = candidate if candidate.is_absolute() else (root / candidate).resolve()
    paths.update(resolved)

    # --- audio -------------------------------------------------------------
    audio = _get_section(cfg, "audio")
    sample_rate = int(_positive_number(audio, "sample_rate", "audio."))
    window_seconds = _positive_number(audio, "window_seconds", "audio.")
    hop_seconds = _positive_number(audio, "hop_seconds", "audio.")
    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    if window_samples < sample_rate // 10:
        raise ConfigError(
            "config.yaml: 'audio.window_seconds' gives a window shorter than 100 ms; "
            "the microfrontend needs at least a few frames"
        )
    if hop_samples < 1:
        raise ConfigError("config.yaml: 'audio.hop_seconds' resolves to 0 samples")
    audio["sample_rate"] = sample_rate
    audio["window_samples"] = window_samples
    audio["hop_samples"] = hop_samples
    audio["peak_target"] = float(audio.get("peak_target", 0.95))
    audio["max_gain_db"] = float(audio.get("max_gain_db", 20.0))
    audio["silence_rms"] = float(audio.get("silence_rms", 0.0))
    if not 0.0 < audio["peak_target"] <= 1.0:
        raise ConfigError("config.yaml: 'audio.peak_target' must be in (0, 1]")

    # --- classes -----------------------------------------------------------
    classes = _get_section(cfg, "classes")
    background = _get(classes, "background", "classes.")
    if not isinstance(background, str) or not background:
        raise ConfigError("config.yaml: 'classes.background' must be a class name")
    esc50_map = _get(classes, "esc50_map", "classes.")
    if not isinstance(esc50_map, dict):
        raise ConfigError("config.yaml: 'classes.esc50_map' must map ESC-50 category -> label")
    for source_category, label in esc50_map.items():
        if not isinstance(label, str) or not label:
            raise ConfigError(
                f"config.yaml: 'classes.esc50_map.{source_category}' must map to a class name"
            )
    order = classes.get("order") or []
    if not isinstance(order, list) or not all(isinstance(item, str) and item for item in order):
        raise ConfigError("config.yaml: 'classes.order' must be a list of class names")
    background = background.strip()
    classes["background"] = background
    classes["esc50_map"] = {str(k): str(v) for k, v in esc50_map.items()}
    classes["order"] = [item.strip() for item in order]

    # --- features ----------------------------------------------------------
    features = _get_section(cfg, "features")
    channels = int(_positive_number(features, "channels", "features."))
    window_ms = _positive_number(features, "window_ms", "features.")
    step_ms = _positive_number(features, "step_ms", "features.")
    window_size = int(round(window_ms * sample_rate / 1000.0))
    stride = int(round(step_ms * sample_rate / 1000.0))
    if window_size < 1 or stride < 1:
        raise ConfigError("config.yaml: 'features.window_ms'/'step_ms' resolve to 0 samples")
    if stride > window_size:
        raise ConfigError("config.yaml: 'features.step_ms' cannot be larger than 'window_ms'")
    if window_size > window_samples:
        raise ConfigError(
            f"config.yaml: 'features.window_ms' ({window_ms} ms) is longer than the "
            f"audio window ({window_seconds * 1000:.0f} ms)"
        )
    n_frames = 1 + (window_samples - window_size) // stride
    if n_frames < 2:
        raise ConfigError(
            "config.yaml: the audio window is too short for the microfrontend settings "
            f"({n_frames} frame(s)); raise 'audio.window_seconds'"
        )
    features["kind"] = features.get("kind", "microfrontend")
    if features["kind"] != "microfrontend":
        raise ConfigError(
            f"config.yaml: unsupported 'features.kind' {features['kind']!r}; only "
            "'microfrontend' (tf.raw_ops.AudioMicrofrontend) is implemented"
        )
    features["channels"] = channels
    features["window_ms"] = window_ms
    features["step_ms"] = step_ms
    features["window_size"] = window_size
    features["stride"] = stride
    features["n_frames"] = n_frames
    features["input_shape"] = (n_frames, channels, 1)
    features["lower_band_limit_hz"] = float(features.get("lower_band_limit_hz", 125.0))
    features["upper_band_limit_hz"] = float(features.get("upper_band_limit_hz", 7500.0))
    features["output_scale_shift"] = float(features.get("output_scale_shift", 0.0))
    features["left_context"] = bool(features.get("left_context", True))
    features["right_context"] = bool(features.get("right_context", True))
    features["value_scale"] = float(features.get("value_scale", 255.0))
    if features["value_scale"] <= 0:
        raise ConfigError("config.yaml: 'features.value_scale' must be > 0")

    # --- splits ------------------------------------------------------------
    splits = _get_section(cfg, "splits")
    fold_map = _get(splits, "esc50_folds", "splits.")
    if not isinstance(fold_map, dict):
        raise ConfigError("config.yaml: 'splits.esc50_folds' must be a mapping of split -> folds")
    normalized_folds: dict[str, list[int]] = {}
    seen_folds: dict[int, str] = {}
    for split in SPLIT_NAMES:
        if split not in fold_map:
            raise ConfigError(f"config.yaml: 'splits.esc50_folds.{split}' is missing")
        folds = fold_map[split]
        if not isinstance(folds, list) or not folds:
            raise ConfigError(
                f"config.yaml: 'splits.esc50_folds.{split}' must be a non-empty list of folds"
            )
        normalized_folds[split] = []
        for fold in folds:
            if not isinstance(fold, int) or isinstance(fold, bool):
                raise ConfigError(
                    f"config.yaml: fold {fold!r} in 'splits.esc50_folds.{split}' must be an integer"
                )
            if fold in seen_folds:
                raise ConfigError(
                    f"config.yaml: fold {fold} is assigned to both '{seen_folds[fold]}' "
                    f"and '{split}'"
                )
            seen_folds[fold] = split
            normalized_folds[split].append(fold)
    splits["esc50_folds"] = normalized_folds

    ratios = _get(splits, "own_ratios", "splits.")
    if not isinstance(ratios, dict):
        raise ConfigError("config.yaml: 'splits.own_ratios' must be a mapping of split -> ratio")
    own_ratios: dict[str, float] = {}
    for split in SPLIT_NAMES:
        value = _get(ratios, split, "splits.own_ratios.")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(
                f"config.yaml: 'splits.own_ratios.{split}' must be a non-negative number"
            )
        own_ratios[split] = float(value)
    if abs(sum(own_ratios.values()) - 1.0) > 1e-6:
        raise ConfigError(
            "config.yaml: 'splits.own_ratios' must sum to 1.0, got "
            f"{sum(own_ratios.values()):.3f}"
        )
    splits["own_ratios"] = own_ratios
    splits["seed"] = int(splits.get("seed", 42))

    # --- prepare -----------------------------------------------------------
    prepare = _get_section(cfg, "prepare")
    ratio = _positive_number(prepare, "background_ratio", "prepare.")
    prepare["background_ratio"] = ratio
    prepare["esc50_source"] = str(prepare.get("esc50_source", "esc50"))
    prepare["own_source"] = str(prepare.get("own_source", "own"))

    # --- training ----------------------------------------------------------
    training = cfg.get("training") or {}
    if not isinstance(training, dict):
        raise ConfigError("config.yaml: 'training' must be a mapping")
    training.setdefault("batch_size", 16)
    training.setdefault("epochs", 50)
    training.setdefault("learning_rate", 0.001)
    training.setdefault("patience", 10)
    training.setdefault("seed", 42)
    training.setdefault("shuffle_buffer", 1024)
    training.setdefault("num_parallel_loads", 2)
    training.setdefault("max_class_weight", 10.0)
    if int(training["batch_size"]) < 1:
        raise ConfigError("config.yaml: 'training.batch_size' must be >= 1")
    if int(training["epochs"]) < 1:
        raise ConfigError("config.yaml: 'training.epochs' must be >= 1")
    cfg["training"] = training

    augment = cfg.get("augment") or {}
    if not isinstance(augment, dict):
        raise ConfigError("config.yaml: 'augment' must be a mapping")
    augment.setdefault("enabled", True)
    augment.setdefault("copies_per_sample", 2)
    augment.setdefault("noise_snr_db", [0.0, 20.0])
    augment.setdefault("noise_probability", 0.8)
    augment.setdefault("gain_db", [-6.0, 6.0])
    augment.setdefault("shift_ms", 100.0)
    augment.setdefault("noise_pool_size", 192)
    cfg["augment"] = augment

    cfg["audio"] = audio
    cfg["classes"] = classes
    cfg["features"] = features
    cfg["splits"] = splits
    cfg["prepare"] = prepare
    cfg["paths"] = paths
    return cfg


def esc50_paths(cfg: dict) -> tuple[Path, Path]:
    """Return the ESC-50 (audio_dir, meta_csv), falling back to known layouts.

    Raises ConfigError listing every location that was tried, so a moved
    dataset is a one-line fix rather than a traceback.
    """
    audio_dir = cfg["paths"]["esc50_audio_dir"]
    meta_csv = cfg["paths"]["esc50_meta_csv"]
    if audio_dir.is_dir() and meta_csv.is_file():
        return audio_dir, meta_csv

    root = cfg["_path"].parent
    tried = [f"{audio_dir} (audio) / {meta_csv} (meta)"]
    for audio_rel, meta_rel in _ESC50_CANDIDATES:
        candidate_audio = (root / audio_rel).resolve()
        candidate_meta = (root / meta_rel).resolve()
        tried.append(f"{candidate_audio} (audio) / {candidate_meta} (meta)")
        if candidate_audio.is_dir() and candidate_meta.is_file():
            return candidate_audio, candidate_meta

    raise ConfigError(
        "could not find the ESC-50 dataset. Looked for:\n  "
        + "\n  ".join(tried)
        + "\nDownload it from https://github.com/karolpiczak/ESC-50 and set "
        "'paths.esc50_audio_dir' + 'paths.esc50_meta_csv' in config.yaml, "
        "or pass --esc50-root <folder containing audio/ and meta/>."
    )


def label_order(cfg: dict, present: list[str]) -> list[str]:
    """Order the labels found in the manifest: config order first, then the rest."""
    unique = {label for label in present if label}
    ordered = [label for label in cfg["classes"]["order"] if label in unique]
    unknown = sorted(unique - set(ordered))
    return ordered + unknown


def set_seed(seed: int, tensorflow: bool = False) -> None:
    """Seed python, numpy and (optionally) TensorFlow."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if tensorflow:  # local import keeps audio-only stages light
        try:
            import tensorflow as tf
        except ImportError:  # pragma: no cover - TF is a hard requirement for training
            return
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
