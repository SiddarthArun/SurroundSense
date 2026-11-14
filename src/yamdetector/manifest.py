"""manifest.csv handling: the single contract between data prep and training.

Columns: path, class, source, source_file_id, split.

* ``path``           - processed wav, relative to the project root
* ``class``          - label the model learns
* ``source``         - "esc50" or "own" (anything the preprocessing stages write)
* ``source_file_id`` - the recording the window came from; splits never split it
* ``split``          - train / val / test
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MANIFEST_COLUMNS = ("path", "class", "source", "source_file_id", "split")
SPLIT_NAMES = ("train", "val", "test")


class ManifestError(RuntimeError):
    """Raised for malformed manifests or impossible split requests."""


def read_manifest(path: str | Path) -> list[dict]:
    """Read manifest.csv into a list of row dicts (empty list if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in MANIFEST_COLUMNS if column not in fieldnames]
        if missing:
            raise ManifestError(
                f"{path} is missing column(s) {', '.join(missing)}; "
                f"expected {', '.join(MANIFEST_COLUMNS)}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            cleaned = {column: (row.get(column) or "").strip() for column in MANIFEST_COLUMNS}
            if not cleaned["path"]:
                continue
            if cleaned["split"] not in SPLIT_NAMES:
                raise ManifestError(
                    f"{path}:{line_number} has split {cleaned['split']!r}; "
                    f"expected one of {SPLIT_NAMES}"
                )
            if not cleaned["class"]:
                raise ManifestError(f"{path}:{line_number} has an empty class column")
            rows.append(cleaned)
    return rows


def to_manifest_path(path: str | Path, project_root: str | Path) -> str:
    """Path as stored in manifest.csv: relative to the project root, posix slashes."""
    path = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(os.path.relpath(path, root))
    return relative.as_posix()


def write_manifest(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows sorted deterministically; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("source", ""),
            row.get("class", ""),
            row.get("source_file_id", ""),
            row.get("path", ""),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    return len(ordered)


def split_for_esc50_fold(fold: int, fold_map: dict[str, Sequence[int]]) -> str:
    """Map an ESC-50 fold (1-5) to its split using ``splits.esc50_folds``."""
    for split in SPLIT_NAMES:
        if int(fold) in {int(item) for item in fold_map.get(split, ())}:
            return split
    raise ManifestError(
        f"ESC-50 fold {fold} is not assigned to any split in 'splits.esc50_folds'"
    )


def _split_counts(count: int, ratios: dict[str, float]) -> dict[str, int]:
    """Turn ratios into integer counts, giving val/test at least one file when possible."""
    if count <= 0:
        return {split: 0 for split in SPLIT_NAMES}
    if count < 3:
        # Not enough recordings to hold out; all of them must stay in train.
        return {"train": count, "val": 0, "test": 0}

    raw = {split: count * ratios[split] for split in SPLIT_NAMES}
    counts = {split: int(np.floor(raw[split])) for split in SPLIT_NAMES}
    remainder = count - sum(counts.values())
    for split in sorted(SPLIT_NAMES, key=lambda name: raw[name] - counts[name], reverse=True):
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1
    for split in ("val", "test"):
        if counts[split] == 0 and ratios[split] > 0:
            donor = max(("train", "val", "test"), key=lambda name: counts[name])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split] += 1
    return counts


def assign_splits_by_file(
    file_ids: Sequence[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign whole source recordings to train/val/test (~70/15/15 by default).

    Deterministic for a given ``(file_ids, ratios, seed)`` set: the ids are
    sorted, shuffled with the seed, then cut into the three splits. Every window
    of a recording therefore lands in the same split, so no recording leaks
    across splits through different windows.
    """
    unique_ids = sorted({str(file_id) for file_id in file_ids})
    if not unique_ids:
        return {}
    counts = _split_counts(len(unique_ids), ratios)
    rng = np.random.default_rng(seed)
    shuffled = [unique_ids[index] for index in rng.permutation(len(unique_ids))]

    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_NAMES:
        for file_id in shuffled[cursor : cursor + counts[split]]:
            assignment[file_id] = split
        cursor += counts[split]
    return assignment


def subsample_background(
    rows: Sequence[dict],
    ratio: float,
    seed: int,
    background_label: str,
    target_labels: Sequence[str],
) -> tuple[list[dict], dict]:
    """Cap background windows at ``ratio`` x the average target-class size.

    Selection happens per split, in proportion to how many background windows
    each split already holds, so the fold-based split ratio is preserved.
    Returns ``(kept_rows, stats)``.
    """
    background_rows = [row for row in rows if row["class"] == background_label]
    target_rows = [row for row in rows if row["class"] in set(target_labels)]
    if not background_rows:
        return list(rows), {"background_kept": 0, "background_available": 0, "budget": 0}

    per_class = Counter(row["class"] for row in target_rows)
    if not per_class:
        return list(rows), {
            "background_kept": len(background_rows),
            "background_available": len(background_rows),
            "budget": len(background_rows),
        }
    average_target = sum(per_class.values()) / len(per_class)
    budget = int(round(ratio * average_target))
    if budget >= len(background_rows):
        return list(rows), {
            "background_kept": len(background_rows),
            "background_available": len(background_rows),
            "budget": budget,
        }

    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in background_rows:
        by_split[row["split"]].append(row)
    total_available = len(background_rows)

    rng = np.random.default_rng(seed)
    kept: list[dict] = []
    for split in SPLIT_NAMES:
        available = sorted(by_split.get(split, []), key=lambda row: row["path"])
        if not available:
            continue
        share = int(round(budget * len(available) / total_available))
        share = min(max(share, 1), len(available))
        indices = rng.permutation(len(available))[:share]
        kept.extend(available[int(index)] for index in sorted(indices))

    kept_paths = {row["path"] for row in kept}
    remaining = [
        row
        for row in rows
        if row["class"] != background_label or row["path"] in kept_paths
    ]
    return remaining, {
        "background_kept": len(kept),
        "background_available": total_available,
        "budget": budget,
    }
