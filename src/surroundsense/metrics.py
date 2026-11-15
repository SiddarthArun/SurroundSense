"""Confusion matrix, per-class precision/recall and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size != y_pred.size:
        raise ValueError(f"{y_true.size} labels vs {y_pred.size} predictions")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y_true.size:
        np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_metrics(
    matrix: np.ndarray, labels: Sequence[str]
) -> list[dict]:
    """Precision/recall/F1/support per class, with 0.0 for undefined ratios."""
    results = []
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual,
            }
        )
    return results


def accuracy(matrix: np.ndarray) -> float:
    total = int(matrix.sum())
    return float(np.trace(matrix) / total) if total else 0.0


def false_positive_rate_background(matrix: np.ndarray, background_index: int) -> float:
    """Fraction of true background windows predicted as any other class.

    This is the number that matters for a wearable: how often the device cries
    wolf on ordinary ambient sound.
    """
    background_total = int(matrix[background_index, :].sum())
    if background_total == 0:
        return float("nan")
    true_positive = int(matrix[background_index, background_index])
    return float((background_total - true_positive) / background_total)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: str | Path,
    title: str = "Test confusion matrix",
    normalize: bool = True,
) -> Path:
    """Save the confusion matrix as a PNG (row-normalized by default)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(matrix)
    if normalize:
        row_totals = matrix.sum(axis=1, keepdims=True)
        display = np.divide(
            matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals > 0
        )
    else:
        display = matrix.astype(float)

    size = max(4.0, 0.9 * len(labels) + 2.0)
    figure, axis = plt.subplots(figsize=(size, size * 0.85), dpi=140)
    image = axis.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)

    threshold = 0.5 if normalize else (display.max() / 2.0 if display.size else 0.5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = display[row, column]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
