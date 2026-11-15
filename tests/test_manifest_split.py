"""Unit tests for manifest IO and the split/subsampling logic."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surroundsense import manifest  # noqa: E402

FOLD_MAP = {"train": [1, 2, 3], "val": [4], "test": [5]}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_row(path: str, label: str, split: str, file_id: str = "f") -> dict:
    return {
        "path": path,
        "class": label,
        "source": "esc50",
        "source_file_id": file_id,
        "split": split,
    }


class ManifestIOTests(unittest.TestCase):
    def test_roundtrip(self):
        rows = [
            make_row("data/processed/siren/a.wav", "siren", "train", "a"),
            make_row("data/processed/background/b.wav", "background", "test", "b"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            written = manifest.write_manifest(path, rows)
            self.assertEqual(written, 2)
            loaded = manifest.read_manifest(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded[0]), set(manifest.MANIFEST_COLUMNS))
            self.assertEqual([row["split"] for row in loaded], ["test", "train"])  # sorted

    def test_missing_manifest_returns_empty(self):
        self.assertEqual(manifest.read_manifest("does/not/exist.csv"), [])

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text("path,class,split\nx.wav,siren,train\n", encoding="utf-8")
            with self.assertRaises(manifest.ManifestError):
                manifest.read_manifest(path)

    def test_invalid_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                "path,class,source,source_file_id,split\nx.wav,siren,esc50,a,holdout\n",
                encoding="utf-8",
            )
            with self.assertRaises(manifest.ManifestError):
                manifest.read_manifest(path)

    def test_manifest_paths_are_relative_posix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "data" / "processed" / "siren" / "a.wav"
            self.assertEqual(
                manifest.to_manifest_path(nested, root), "data/processed/siren/a.wav"
            )


class Esc50FoldTests(unittest.TestCase):
    def test_folds_map_to_splits(self):
        self.assertEqual(manifest.split_for_esc50_fold(1, FOLD_MAP), "train")
        self.assertEqual(manifest.split_for_esc50_fold(3, FOLD_MAP), "train")
        self.assertEqual(manifest.split_for_esc50_fold(4, FOLD_MAP), "val")
        self.assertEqual(manifest.split_for_esc50_fold(5, FOLD_MAP), "test")

    def test_unmapped_fold_raises(self):
        with self.assertRaises(manifest.ManifestError):
            manifest.split_for_esc50_fold(9, FOLD_MAP)


class AssignSplitsTests(unittest.TestCase):
    def test_every_file_gets_exactly_one_split(self):
        files = [f"rec{i}" for i in range(20)]
        assignment = manifest.assign_splits_by_file(files, RATIOS, seed=42)
        self.assertEqual(set(assignment), set(files))
        counts = Counter(assignment.values())
        self.assertEqual(counts["train"], 14)
        self.assertEqual(counts["val"], 3)
        self.assertEqual(counts["test"], 3)

    def test_deterministic_for_same_seed(self):
        files = [f"rec{i}" for i in range(30)]
        first = manifest.assign_splits_by_file(files, RATIOS, seed=7)
        second = manifest.assign_splits_by_file(files, RATIOS, seed=7)
        self.assertEqual(first, second)

    def test_different_seed_changes_assignment(self):
        files = [f"rec{i}" for i in range(30)]
        first = manifest.assign_splits_by_file(files, RATIOS, seed=1)
        second = manifest.assign_splits_by_file(files, RATIOS, seed=2)
        self.assertNotEqual(first, second)

    def test_all_windows_of_a_file_share_a_split(self):
        # the function is per file id, so asserting the id map is enough
        files = ["takeA", "takeA", "takeB", "takeC", "takeD"]
        assignment = manifest.assign_splits_by_file(files, RATIOS, seed=3)
        self.assertEqual(len(assignment), 4)
        self.assertIn(assignment["takeA"], manifest.SPLIT_NAMES)

    def test_tiny_class_keeps_everything_in_train(self):
        assignment = manifest.assign_splits_by_file(["only_one"], RATIOS, seed=0)
        self.assertEqual(assignment, {"only_one": "train"})

    def test_three_files_get_one_per_split(self):
        assignment = manifest.assign_splits_by_file(["a", "b", "c"], RATIOS, seed=0)
        self.assertEqual(sorted(assignment.values()), ["test", "train", "val"])

    def test_empty_input(self):
        self.assertEqual(manifest.assign_splits_by_file([], RATIOS, seed=0), {})


class LabelMappingTests(unittest.TestCase):
    def test_unmapped_labels_become_background(self):
        label = manifest.class_for_labels(
            ["Wind", "Rain"], {"Siren": "siren"}, ["siren", "background"], "background"
        )
        self.assertEqual(label, "background")

    def test_order_breaks_multi_label_ties(self):
        mapping = {"Screaming": "scream", "Speech": "speech", "Music": "music"}
        order = ["scream", "speech", "music", "background"]
        self.assertEqual(
            manifest.class_for_labels(["Music", "Speech", "Screaming"], mapping, order, "background"),
            "scream",
        )
        self.assertEqual(
            manifest.class_for_labels(["Music", "Speech"], mapping, order, "background"), "speech"
        )

    def test_label_absent_from_order_still_maps(self):
        self.assertEqual(
            manifest.class_for_labels(["Alarm"], {"Alarm": "alarm"}, [], "background"), "alarm"
        )


class UniquePathTests(unittest.TestCase):
    def test_clean_manifest_passes(self):
        rows = [
            make_row("data/processed/siren/a.wav", "siren", "train", "a"),
            make_row("data/processed/siren/a2.wav", "siren", "train", "a"),
        ]
        manifest.assert_unique_paths(rows)  # no raise

    def test_duplicate_path_in_two_splits_raises(self):
        rows = [
            make_row("data/processed/background/a.wav", "background", "val", "a"),
            make_row("data/processed/background/a.wav", "background", "test", "a"),
        ]
        with self.assertRaises(manifest.ManifestError):
            manifest.assert_unique_paths(rows)

    def test_recording_straddling_splits_raises(self):
        rows = [
            make_row("data/processed/siren/a_w0.wav", "siren", "train", "clipA"),
            make_row("data/processed/siren/a_w1.wav", "siren", "test", "clipA"),
        ]
        with self.assertRaises(manifest.ManifestError):
            manifest.assert_unique_paths(rows)


class ClassBudgetTests(unittest.TestCase):
    def test_background_budget_is_ratio_times_average_class(self):
        rows = []
        for label in ("siren", "knock"):
            for index in range(30):
                rows.append(make_row(f"data/processed/{label}/{index}.wav", label, "train", f"{label}{index}"))
        for index in range(500):
            rows.append(make_row(f"data/processed/background/{index}.wav", "background", "train", f"bg{index}"))
        budgets = manifest.class_budgets(rows, "background", 3.0, 2000)
        self.assertEqual(budgets["siren"], 2000)  # the flat cap never grows a class
        self.assertEqual(budgets["background"], 90)  # 3 x 30

    def test_background_has_its_own_larger_cap(self):
        rows = []
        for label in ("siren", "knock"):
            for index in range(30):
                rows.append(make_row(f"data/processed/{label}/{index}.wav", label, "train", f"{label}{index}"))
        rows.extend(
            make_row(f"data/processed/background/{index}.wav", "background", "train", f"bg{index}")
            for index in range(5000)
        )
        # 80 x 30 = 2400 exceeds the flat 2000 cap but stays under the background cap
        budgets = manifest.class_budgets(rows, "background", 80.0, 2000, max_background=6000)
        self.assertEqual(budgets["background"], 2400)
        self.assertEqual(budgets["siren"], 2000)

    def test_background_never_exceeds_its_ceiling(self):
        rows = []
        for index in range(30):
            rows.append(make_row(f"data/processed/siren/{index}.wav", "siren", "train", f"s{index}"))
        rows.extend(
            make_row(f"data/processed/background/{index}.wav", "background", "train", f"bg{index}")
            for index in range(5000)
        )
        budgets = manifest.class_budgets(rows, "background", 80.0, 2000, max_background=600)
        self.assertEqual(budgets["background"], 600)

    def test_flat_cap_limits_a_huge_class(self):
        rows = [make_row(f"data/processed/music/{i}.wav", "music", "train", f"m{i}") for i in range(50)]
        rows.append(make_row("data/processed/siren/1.wav", "siren", "train", "s1"))
        budgets = manifest.class_budgets(rows, "background", 3.0, 20)
        self.assertEqual(budgets["music"], 20)


class ClassCapTests(unittest.TestCase):
    def build_rows(self, per_class: dict[str, int]):
        rows = []
        for label, count in per_class.items():
            for split, share in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
                for index in range(int(count * share)):
                    rows.append(
                        make_row(
                            f"data/processed/{label}/{split}{index}.wav",
                            label,
                            split,
                            f"{label}{split}{index}",
                        )
                    )
        return rows

    def test_caps_each_class_and_keeps_it_under_budget(self):
        rows = self.build_rows({"music": 300, "siren": 20})
        kept, stats = manifest.cap_class_windows(rows, {"music": 80, "siren": 80}, 42)
        music = [row for row in kept if row["class"] == "music"]
        self.assertLessEqual(len(music), 80)
        self.assertEqual(stats["music"]["kept"], len(music))
        self.assertEqual(len([row for row in kept if row["class"] == "siren"]), 20)

    def test_keeps_split_proportions(self):
        rows = self.build_rows({"music": 300, "siren": 20})
        kept, _ = manifest.cap_class_windows(rows, {"music": 60, "siren": 60}, 42)
        splits = Counter(row["split"] for row in kept if row["class"] == "music")
        self.assertGreater(splits["train"], splits["val"])
        self.assertGreater(splits["val"], 0)
        self.assertGreater(splits["test"], 0)

    def test_deterministic(self):
        rows = self.build_rows({"music": 200, "speech": 200})
        first, _ = manifest.cap_class_windows(rows, {"music": 50, "speech": 50}, 7)
        second, _ = manifest.cap_class_windows(rows, {"music": 50, "speech": 50}, 7)
        self.assertEqual([row["path"] for row in first], [row["path"] for row in second])

    def test_no_cap_when_class_is_small(self):
        rows = self.build_rows({"siren": 20})
        kept, stats = manifest.cap_class_windows(rows, {"siren": 80}, 42)
        self.assertEqual(len(kept), len(rows))
        self.assertEqual(stats["siren"]["kept"], 20)


if __name__ == "__main__":
    unittest.main()
