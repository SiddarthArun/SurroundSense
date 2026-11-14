#!/usr/bin/env python
"""Stage 3: train the DS-CNN using only manifest.csv + config.yaml.

* classes come from the manifest/config (never hardcoded)
* microfrontend features are cached to data/features and reused across runs
* training windows are expanded once with background-noise mixing, gain and
  time-shift augmentation, and streamed from disk in small batches
* reports per-class precision/recall, a confusion matrix PNG, accuracy and the
  false-positive rate on background test windows
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yamdetector import augment, features, manifest, metrics  # noqa: E402
from yamdetector.config import (  # noqa: E402
    ConfigError,
    label_order,
    load_config,
    set_seed,
)
from yamdetector.model import build_model, describe  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the yamDetector DS-CNN (stage 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="train on the original training windows only",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="recompute every cached feature (and every augmented window)",
    )
    return parser.parse_args(argv)


def setup_tensorflow():
    """Import TF, use the GPU when there is one with small memory, else CPU."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "error: TensorFlow is not installed. Activate the project environment and run:\n"
            "  pip install -r requirements.txt"
        ) from exc

    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # already initialized; growth is best-effort
        device = "GPU: " + ", ".join(gpu.name for gpu in gpus)
    else:
        device = "CPU (no CUDA GPU visible - see README for WSL2/Linux GPU setup)"
    print(f"TensorFlow {tf.__version__} on {device}")
    return tf


def load_manifest_rows(cfg: dict) -> list[dict]:
    manifest_path = cfg["paths"]["manifest_csv"]
    rows = manifest.read_manifest(manifest_path)
    if not rows:
        raise SystemExit(
            f"error: {manifest_path} is missing or empty.\n"
            "Run stage 1 first:  python scripts/prepare_esc50.py"
        )
    missing = [row["path"] for row in rows if not features.resolve_row_path(row).exists()]
    if missing:
        preview = "\n  ".join(missing[:5])
        raise SystemExit(
            f"error: {len(missing)} wav file(s) referenced by {manifest_path} do not exist:\n  "
            f"{preview}\nRe-run the preparation scripts (or fix 'paths.processed_dir')."
        )
    return rows


def split_rows(cfg: dict, rows: list[dict]) -> dict[str, list[dict]]:
    splits = {name: [row for row in rows if row["split"] == name] for name in manifest.SPLIT_NAMES}
    for name, split_rows_ in splits.items():
        if not split_rows_:
            raise SystemExit(
                f"error: the '{name}' split has no windows.\n"
                "Check 'splits.esc50_folds' / 'splits.own_ratios' and re-run the preparation scripts."
            )
    return splits


def compute_class_weights(
    label_counts: Counter, num_classes: int, max_weight: float
) -> np.ndarray:
    """Square-root inverse-frequency weights, capped and normalized to mean 1.

    Full inverse frequency is too aggressive here: the background class is ~5x
    larger than the smallest target class, and flattening that completely makes
    the model fire on ambient noise (high false-positive rate). The square root
    halves the correction while still lifting the rare classes.
    """
    total = sum(label_counts.values())
    if total == 0:
        return np.ones(num_classes, dtype=np.float32)
    weights = np.ones(num_classes, dtype=np.float32)
    for index in range(num_classes):
        count = label_counts.get(index, 0)
        weights[index] = (total / (num_classes * count)) ** 0.5 if count else 1.0
    weights = np.clip(weights, 1.0 / max(max_weight, 1.0), max_weight)
    return (weights / weights.mean()).astype(np.float32)


def make_dataset(tf, items, input_shape, weights_by_label, cfg, training: bool):
    """Stream ``(feature, label, sample_weight)`` from cached .npy files."""
    paths = tf.constant([str(path) for path, _ in items])
    labels = tf.constant(np.asarray([label for _, label in items], dtype=np.int32))
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        buffer = min(len(items), int(cfg["training"]["shuffle_buffer"]))
        dataset = dataset.shuffle(buffer, seed=int(cfg["training"]["seed"]), reshuffle_each_iteration=True)

    weights_tensor = tf.constant(weights_by_label)

    def _load(path, label):
        feature = tf.numpy_function(
            lambda item: np.load(item.decode("utf-8"), allow_pickle=False), [path], tf.float32
        )
        feature.set_shape(tuple(input_shape))
        return feature, label, tf.gather(weights_tensor, label)

    dataset = dataset.map(_load, num_parallel_calls=int(cfg["training"]["num_parallel_loads"]))
    batch_size = int(cfg["training"]["batch_size"])
    return dataset.batch(batch_size, drop_remainder=False).prefetch(1)


def build_training_items(cfg, rows, label_index, force: bool) -> list[tuple[Path, int]]:
    """Original training windows plus augmented copies (cached on disk)."""
    items: list[tuple[Path, int]] = []
    for row in rows:
        wav_path = features.resolve_row_path(row)
        items.append((features.cache_path(cfg, wav_path), label_index[row["class"]]))

    copies = int(cfg["augment"]["copies_per_sample"]) if cfg["augment"]["enabled"] else 0
    if copies <= 0:
        print("augmentation: disabled")
        return items

    seed = int(cfg["training"]["seed"])
    print(
        f"augmentation: {copies} extra cop{'y' if copies == 1 else 'ies'} per training window "
        f"(noise {cfg['augment']['noise_snr_db'][0]:g}-{cfg['augment']['noise_snr_db'][1]:g} dB SNR, "
        f"gain {cfg['augment']['gain_db'][0]:g}-{cfg['augment']['gain_db'][1]:g} dB, "
        f"shift +/-{cfg['augment']['shift_ms']:g} ms)"
    )
    noise_pool = augment.build_noise_pool(cfg, rows, seed)

    total = len(rows) * copies
    done = 0
    for copy_index in range(copies):
        for row in rows:
            wav_path = features.resolve_row_path(row)
            out_path = features.augmented_cache_path(cfg, wav_path, copy_index)
            if force or not out_path.exists():
                augment.write_augmented_feature(
                    cfg, wav_path, out_path, copy_index, noise_pool, seed
                )
            items.append((out_path, label_index[row["class"]]))
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  augmented windows {done}/{total}")
    return items


def evaluate(model, x_test, y_test, labels: list[str], background_index: int):
    probabilities = model.predict(x_test, batch_size=64, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    matrix = metrics.confusion_matrix(y_test, predictions, len(labels))
    overall = metrics.accuracy(matrix)
    per_class = metrics.per_class_metrics(matrix, labels)
    fpr = metrics.false_positive_rate_background(matrix, background_index)
    return matrix, overall, per_class, fpr, predictions


def print_report(matrix, overall, per_class, fpr, labels, background_index) -> None:
    print("\nper-class metrics (test split):")
    print(f"  {'class':<18}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for entry in per_class:
        print(
            f"  {entry['label']:<18}{entry['precision']:>10.3f}{entry['recall']:>9.3f}"
            f"{entry['f1']:>8.3f}{entry['support']:>9}"
        )

    print("\nconfusion matrix (rows = true, columns = predicted):")
    header = "".join(f"{label[:9]:>10}" for label in labels)
    print(f"  {'':<16}{header}")
    for index, label in enumerate(labels):
        row = "".join(f"{int(value):>10}" for value in matrix[index])
        print(f"  {label[:15]:<16}{row}")

    print(f"\noverall accuracy: {overall:.4f}")
    if background_index is None:
        print("background false-positive rate: n/a (no background class in the manifest)")
    else:
        background_total = int(matrix[background_index, :].sum())
        false_positives = background_total - int(matrix[background_index, background_index])
        if background_total:
            print(
                f"background false-positive rate: {fpr:.4f} "
                f"({false_positives}/{background_total} background windows predicted as a target class)"
            )
        else:
            print(
                "background false-positive rate: n/a (no background windows in the test split)"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.epochs is not None:
        if args.epochs < 1:
            print("error: --epochs must be >= 1", file=sys.stderr)
            return 2
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            print("error: --batch-size must be >= 1", file=sys.stderr)
            return 2
        cfg["training"]["batch_size"] = args.batch_size
    if args.no_augment:
        cfg["augment"]["enabled"] = False

    tf = setup_tensorflow()
    set_seed(int(cfg["training"]["seed"]), tensorflow=True)

    rows = load_manifest_rows(cfg)
    labels = label_order(cfg, [row["class"] for row in rows])
    if len(labels) < 2:
        print(
            f"error: only {len(labels)} class(es) in the manifest ({labels}); "
            "training needs at least 2 - run scripts/prepare_esc50.py first",
            file=sys.stderr,
        )
        return 2
    label_index = {label: index for index, label in enumerate(labels)}
    background = cfg["classes"]["background"]
    background_index = label_index.get(background)

    splits = split_rows(cfg, rows)
    train_classes = {row["class"] for row in splits["train"]}
    absent = [label for label in labels if label not in train_classes]
    if absent:
        print(
            f"error: class(es) {absent} have no windows in the train split; "
            "check splits.esc50_folds / splits.own_ratios",
            file=sys.stderr,
        )
        return 2

    print(f"classes ({len(labels)}): {', '.join(labels)}")
    for name in manifest.SPLIT_NAMES:
        counts = Counter(row["class"] for row in splits[name])
        detail = ", ".join(f"{label}={counts.get(label, 0)}" for label in labels)
        print(f"  {name:<5} {len(splits[name]):>5} windows  ({detail})")

    # --- features ----------------------------------------------------------
    print("\ncomputing/caching microfrontend features ...")
    jobs = [
        (features.resolve_row_path(row), features.cache_path(cfg, features.resolve_row_path(row)))
        for row in rows
    ]
    computed = features.compute_missing(cfg, jobs, force=args.force_features)
    print(
        f"  {computed} window(s) computed, {len(jobs) - computed} loaded from "
        f"{cfg['paths']['feature_cache_dir']}"
    )

    x_val = features.load_split_features(cfg, splits["val"])
    x_test = features.load_split_features(cfg, splits["test"])
    y_val = np.asarray([label_index[row["class"]] for row in splits["val"]], dtype=np.int32)
    y_test = np.asarray([label_index[row["class"]] for row in splits["test"]], dtype=np.int32)
    print(f"  val {x_val.shape}, test {x_test.shape}")

    # --- training data -----------------------------------------------------
    print("\nbuilding training windows ...")
    train_items = build_training_items(cfg, splits["train"], label_index, args.force_features)
    class_weights = compute_class_weights(
        Counter(label_index[row["class"]] for row in splits["train"]),
        len(labels),
        float(cfg["training"]["max_class_weight"]),
    )
    weights_by_label = class_weights
    print(
        "class weights: "
        + ", ".join(f"{labels[i]}={class_weights[i]:.2f}" for i in range(len(labels)))
    )

    train_ds = make_dataset(tf, train_items, cfg["features"]["input_shape"], weights_by_label, cfg, True)
    val_weights = np.ones(len(y_val), dtype=np.float32)

    # --- model -------------------------------------------------------------
    model = build_model(cfg, len(labels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["training"]["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nmodel: {describe(model)} (budget 80,000), input {cfg['features']['input_shape']}")
    if model.count_params() > 80_000:
        print("warning: parameter count exceeds the 80k budget for the MCU target", file=sys.stderr)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["training"]["patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path), monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    print(f"\ntraining up to {cfg['training']['epochs']} epochs, batch {cfg['training']['batch_size']}")
    history = model.fit(
        train_ds,
        validation_data=(x_val, y_val, val_weights),
        epochs=int(cfg["training"]["epochs"]),
        callbacks=callbacks,
        verbose=2,
    )
    epochs_run = len(history.history.get("loss", []))
    print(f"trained for {epochs_run} epoch(s)")

    # best weights are restored by EarlyStopping; make sure they are on disk
    if not best_model_path.exists():
        model.save(best_model_path)
    print(f"saved Keras model: {best_model_path}")

    # --- evaluation --------------------------------------------------------
    matrix, overall, per_class, fpr, _ = evaluate(
        model, x_test, y_test, labels, background_index if background_index is not None else 0
    )
    print_report(matrix, overall, per_class, fpr, labels, background_index)

    confusion_path = metrics.plot_confusion_matrix(
        matrix, labels, output_dir / "confusion_matrix.png"
    )
    print(f"confusion matrix PNG: {confusion_path}")
    print("\nnext: python export.py   (float + int8 TFLite, model.h, labels.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
