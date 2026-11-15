#!/usr/bin/env python
"""Stage 4: export the trained model for firmware.

* saves the float TFLite model and an int8-quantized one (full integer
  quantization with a representative dataset taken from training features)
* evaluates both on the test split and warns when int8 drops more than 3 points
* writes artifacts/model.h (int8 model as a C array) and artifacts/labels.txt
* prints the int8 input scale / zero point the firmware needs
* optionally copies model.h into a firmware sketch folder (--firmware-dir)
* warns when a model class is missing from libraries/protocol/protocol.h
  CLASS_NAMES, or CLASS_ALERT differs from classes.alert
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yamdetector import features, manifest, metrics  # noqa: E402
from yamdetector.config import ConfigError, label_order, load_config, set_seed  # noqa: E402

ACCURACY_WARNING_POINTS = 3.0
PROTOCOL_HEADER = PROJECT_ROOT / "libraries" / "protocol" / "protocol.h"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export float/int8 TFLite + model.h + labels.txt (stage 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument(
        "--model",
        default=None,
        help="Keras model to export (default: <paths.output_dir>/model.keras)",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=200,
        help="training windows used to calibrate int8 quantization (default: 200)",
    )
    parser.add_argument(
        "--firmware-dir",
        default=None,
        help="also copy model.h into this sketch folder, e.g. room_capture "
        "(artifacts/ is gitignored, the sketch folder is not)",
    )
    return parser.parse_args(argv)


def representative_dataset(x_train: np.ndarray, count: int, seed: int):
    """Yield single training windows for the int8 calibrator."""
    limit = min(int(count), len(x_train))
    if limit <= 0:
        raise SystemExit("error: no training features available for the representative dataset")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_train), size=limit, replace=False)

    def generator():
        for index in indices:
            yield [x_train[int(index) : int(index) + 1].astype(np.float32)]

    return generator


def convert_float(tf, keras_model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return converter.convert()


def convert_int8(tf, keras_model, x_train: np.ndarray, count: int, seed: int) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train, count, seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_tflite(tf, model_content: bytes, x: np.ndarray, y: np.ndarray, batch: int = 64):
    """Run the test windows through a TFLite interpreter; returns (accuracy, predictions)."""
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=2)
    interpreter.allocate_tensors()
    predictions = []
    for start in range(0, len(x), batch):
        chunk = x[start : start + batch]
        details = interpreter.get_input_details()[0]
        if details["dtype"] == np.int8:
            scale, zero_point = details["quantization"]
            if scale == 0:
                raise SystemExit("error: int8 model has an uncalibrated input (scale == 0)")
            quantized = np.clip(np.round(chunk / scale + zero_point), -128, 127).astype(np.int8)
        else:
            quantized = chunk.astype(details["dtype"])

        if tuple(quantized.shape) != tuple(details["shape"]):
            interpreter.resize_tensor_input(details["index"], quantized.shape, strict=False)
            interpreter.allocate_tensors()
            details = interpreter.get_input_details()[0]

        interpreter.set_tensor(details["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_output_details()[0]
        predictions.append(interpreter.get_tensor(output["index"]).copy())

    scores = np.concatenate(predictions, axis=0)
    predicted = np.argmax(scores, axis=1)
    return float(np.mean(predicted == y)), predicted


def write_labels(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def write_model_header(
    path: Path,
    model_bytes: bytes,
    labels: list[str],
    input_scale: float,
    input_zero_point: int,
    input_shape: tuple[int, ...],
) -> None:
    """Write the int8 model as a C array plus the constants the firmware needs."""
    frames, channels, _ = input_shape
    lines = [
        "// Generated by export.py - do not edit.",
        "// yamDetector int8 TFLite model for TFLite Micro on the ESP32-S3.",
        "#ifndef YAMDETECTOR_MODEL_H",
        "#define YAMDETECTOR_MODEL_H",
        "",
        "#include <cstdint>",
        "",
        "// Feature/input contract (must match the device's microfrontend output).",
        f"constexpr int kYamInputFrames = {frames};",
        f"constexpr int kYamInputChannels = {channels};",
        f"constexpr float kYamInputScale = {input_scale:.10g}f;",
        f"constexpr int32_t kYamInputZeroPoint = {input_zero_point};",
        f"constexpr int kYamOutputClasses = {len(labels)};",
        "// Output order. yam_infer.h matches these names to CLASS_NAMES in protocol.h.",
        "static const char *const kYamLabels[kYamOutputClasses] = {"
        + ", ".join(f'"{label}"' for label in labels)
        + "};",
        "",
        "alignas(8) const unsigned char g_yam_model_tflite[] = {",
    ]
    per_line = 12
    for start in range(0, len(model_bytes), per_line):
        chunk = model_bytes[start : start + per_line]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int g_yam_model_tflite_len = {len(model_bytes)};",
            "",
            "#endif  // YAMDETECTOR_MODEL_H",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def protocol_class_names(header: Path) -> list[str] | None:
    """Read CLASS_NAMES from the firmware's protocol.h; None if it can't be found."""
    if not header.exists():
        return None
    text = header.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"CLASS_NAMES\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}", text)
    if not match:
        return None
    return re.findall(r'"([^"]*)"', match.group(1))


def protocol_alert_flags(header: Path) -> list[bool] | None:
    """Read CLASS_ALERT from the firmware's protocol.h; None if it can't be found."""
    if not header.exists():
        return None
    text = header.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"CLASS_ALERT\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}", text)
    if not match:
        return None
    return [token == "true" for token in re.findall(r"\b(?:true|false)\b", match.group(1))]


def check_protocol_tiers(
    labels: list[str], alert_labels: set[str], header: Path = PROTOCOL_HEADER
) -> bool:
    """Warn when the firmware's alert tiers differ from ``classes.alert``."""
    flags = protocol_alert_flags(header)
    names = protocol_class_names(header)
    if flags is None or names is None:
        print(f"note: no CLASS_ALERT found in {header}; skipped the alert tier check")
        return True
    if len(flags) != len(names):
        print(
            f"WARNING: {header.relative_to(PROJECT_ROOT)} CLASS_ALERT has {len(flags)} entries "
            f"but CLASS_NAMES has {len(names)}",
            file=sys.stderr,
        )
        return False
    tier = dict(zip(names, flags))
    mismatched = [
        label for label in labels if label in tier and tier[label] != (label in alert_labels)
    ]
    if not mismatched:
        print(f"firmware alert tiers match classes.alert ({header.relative_to(PROJECT_ROOT)})")
        return True
    print(
        "WARNING: protocol.h CLASS_ALERT disagrees with classes.alert for: "
        + ", ".join(mismatched),
        file=sys.stderr,
    )
    return False


def check_protocol_labels(labels: list[str], header: Path = PROTOCOL_HEADER) -> bool:
    """Warn when the model has a class the firmware's protocol.h does not list.

    yam_infer.h matches the model's output names (kYamLabels in model.h) to
    CLASS_NAMES, so the order may differ and a model may have fewer classes,
    but every label has to exist there.
    """
    names = protocol_class_names(header)
    if names is None:
        print(f"note: no CLASS_NAMES found in {header}; skipped the firmware class check")
        return True
    missing = [label for label in labels if label not in names]
    if not missing:
        print(f"every model class is in the firmware class list ({header.relative_to(PROJECT_ROOT)})")
        return True
    print(
        "WARNING: the firmware's class list has no entry for: " + ", ".join(missing) + "\n"
        f"  protocol.h      : {', '.join(names)}\n"
        f"  Append them to CLASS_NAMES, the CLS_ enum and CLASS_ALERT in "
        f"{header.relative_to(PROJECT_ROOT)} (appending keeps the ids the wristband already "
        "knows), then reflash both boards. The harness refuses to start until then.",
        file=sys.stderr,
    )
    return False


def copy_header_to_firmware(header_path: Path, firmware_dir: str) -> Path | None:
    """Copy model.h into a sketch folder so it can be committed and #included."""
    target_dir = Path(firmware_dir).expanduser()
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    if not target_dir.is_dir():
        print(
            f"WARNING: --firmware-dir {target_dir} is not a folder; model.h was not copied",
            file=sys.stderr,
        )
        return None
    target = target_dir / header_path.name
    shutil.copy2(header_path, target)
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        import tensorflow as tf
    except ImportError:
        print(
            "error: TensorFlow is not installed. Activate the project environment and run:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    tf.get_logger().setLevel("ERROR")
    set_seed(int(cfg["training"]["seed"]), tensorflow=True)

    output_dir = Path(cfg["paths"]["output_dir"])
    keras_path = Path(args.model) if args.model else output_dir / "model.keras"
    if not keras_path.exists():
        print(
            f"error: trained model not found: {keras_path}\nRun stage 3 first:  python train.py",
            file=sys.stderr,
        )
        return 2

    rows = manifest.read_manifest(cfg["paths"]["manifest_csv"])
    if not rows:
        print(
            f"error: {cfg['paths']['manifest_csv']} is missing or empty; "
            "run scripts/prepare_esc50.py",
            file=sys.stderr,
        )
        return 2
    try:
        manifest.assert_unique_paths(rows)
    except manifest.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Same selection train.py used: the per-class caps decide which windows are
    # part of the dataset, so the exported model is measured on exactly the
    # windows training held back.
    budgets = manifest.class_budgets(
        rows,
        background_label=cfg["classes"]["background"],
        background_ratio=float(cfg["prepare"]["background_ratio"]),
        max_per_class=int(cfg["prepare"]["max_windows_per_class"]),
        max_background=int(cfg["prepare"]["max_background_windows"]),
    )
    rows, selection = manifest.cap_class_windows(rows, budgets, int(cfg["splits"]["seed"]))
    print(f"dataset: {len(selection)} classes, {len(rows)} window(s) after the per-class caps")
    labels = label_order(cfg, [row["class"] for row in rows])

    test_rows = [row for row in rows if row["split"] == "test"]
    train_rows = [row for row in rows if row["split"] == "train"]
    if not test_rows:
        print("error: the test split is empty; cannot evaluate the exported models", file=sys.stderr)
        return 2

    label_index = {label: index for index, label in enumerate(labels)}
    x_test = features.load_split_features(cfg, test_rows)
    y_test = np.asarray([label_index[row["class"]] for row in test_rows], dtype=np.int32)
    x_train = features.load_split_features(cfg, train_rows)
    print(
        f"model: {keras_path}\n"
        f"labels ({len(labels)}): {', '.join(labels)}\n"
        f"test windows: {len(x_test)}, representative pool: {len(x_train)} train windows"
    )

    keras_model = tf.keras.models.load_model(keras_path)
    expected = tuple(cfg["features"]["input_shape"])
    if tuple(keras_model.input_shape[1:]) != expected:
        print(
            f"error: {keras_path} expects input {keras_model.input_shape[1:]}, but config.yaml "
            f"produces {expected}.\nRetrain the model or fix the feature settings.",
            file=sys.stderr,
        )
        return 2

    print("\nconverting ...")
    float_bytes = convert_float(tf, keras_model)
    int8_bytes = convert_int8(
        tf,
        keras_model,
        x_train,
        args.representative_samples,
        int(cfg["training"]["seed"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = output_dir / "model_float.tflite"
    int8_path = output_dir / "model_int8.tflite"
    float_path.write_bytes(float_bytes)
    int8_path.write_bytes(int8_bytes)
    print(f"  {float_path} ({features.human_bytes(len(float_bytes))})")
    print(f"  {int8_path} ({features.human_bytes(len(int8_bytes))})")

    float_accuracy, _ = evaluate_tflite(tf, float_bytes, x_test, y_test)
    int8_accuracy, int8_predictions = evaluate_tflite(tf, int8_bytes, x_test, y_test)
    print("\ntest accuracy:")
    print(f"  float TFLite : {float_accuracy:.4f}")
    print(f"  int8  TFLite : {int8_accuracy:.4f}")
    drop_points = (float_accuracy - int8_accuracy) * 100.0
    print(f"  delta        : {-drop_points:+.2f} points")
    if drop_points > ACCURACY_WARNING_POINTS:
        print(
            f"WARNING: int8 quantization loses {drop_points:.2f} points of accuracy "
            f"(more than {ACCURACY_WARNING_POINTS:g}). Try more representative samples, "
            "more training data, or lower 'features.value_scale'.",
            file=sys.stderr,
        )

    background_index = label_index.get(cfg["classes"]["background"])
    if background_index is not None:
        matrix = metrics.confusion_matrix(y_test, int8_predictions, len(labels))
        fpr = metrics.false_positive_rate_background(matrix, background_index)
        print(f"  int8 background FPR: {fpr:.4f}")

    interpreter = tf.lite.Interpreter(model_content=int8_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    input_scale = float(scale)
    input_zero_point = int(zero_point)

    header_path = output_dir / "model.h"
    labels_path = output_dir / "labels.txt"
    write_model_header(
        header_path,
        int8_bytes,
        labels,
        input_scale,
        input_zero_point,
        tuple(cfg["features"]["input_shape"]),
    )
    write_labels(labels_path, labels)
    print(f"\n{header_path} ({features.human_bytes(header_path.stat().st_size)})")
    print(f"{labels_path} ({len(labels)} labels, model output order)")
    if args.firmware_dir:
        copied = copy_header_to_firmware(header_path, args.firmware_dir)
        if copied is not None:
            print(f"copied model.h to {copied}")
    check_protocol_labels(labels)
    check_protocol_tiers(labels, set(cfg["classes"]["alert"]))

    print("\nint8 input quantization (needed by the firmware):")
    print(f"  input shape      : {tuple(int(dim) for dim in input_details['shape'])} (int8)")
    print(f"  scale            : {input_scale:.10g}")
    print(f"  zero point       : {input_zero_point}")
    print(
        "  quantize a feature value f as: int8 = clip(round(f / scale) + zero_point, -128, 127)\n"
        "  features are microfrontend frames divided by "
        f"{cfg['features']['value_scale']:g} (config.yaml features.value_scale)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
