"""Unit tests for audio slicing/normalization (shared by stage 1 and 2)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yamdetector import audio  # noqa: E402


class SliceWindowsTests(unittest.TestCase):
    def test_exact_multiple_gives_no_extra_window(self):
        windows = audio.slice_windows(np.zeros(48000, dtype=np.float32), 16000)
        self.assertEqual(windows.shape, (3, 16000))
        self.assertEqual(windows.dtype, np.float32)

    def test_hop_controls_overlap(self):
        samples = np.arange(32000, dtype=np.float32)
        windows = audio.slice_windows(samples, 16000, hop_samples=8000)
        self.assertEqual(windows.shape, (3, 16000))
        np.testing.assert_array_equal(windows[1, :100], samples[8000:8100])
        np.testing.assert_array_equal(windows[2, :100], samples[16000:16100])

    def test_partial_tail_is_zero_padded(self):
        windows = audio.slice_windows(np.ones(40000, dtype=np.float32), 16000)
        self.assertEqual(windows.shape, (3, 16000))
        self.assertAlmostEqual(float(windows[2, :8000].sum()), 8000.0)
        self.assertAlmostEqual(float(windows[2, 8000:].sum()), 0.0)

    def test_tiny_tail_is_dropped(self):
        windows = audio.slice_windows(np.ones(48010, dtype=np.float32), 16000)
        self.assertEqual(windows.shape, (3, 16000))

    def test_recording_shorter_than_window_is_padded(self):
        windows = audio.slice_windows(np.ones(4000, dtype=np.float32), 16000)
        self.assertEqual(windows.shape, (1, 16000))
        self.assertAlmostEqual(float(windows[0, :4000].sum()), 4000.0)
        self.assertAlmostEqual(float(windows[0, 4000:].sum()), 0.0)

    def test_short_recording_without_padding_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.slice_windows(np.ones(100, dtype=np.float32), 16000, pad_last=False)

    def test_empty_audio_gives_no_windows(self):
        self.assertEqual(audio.slice_windows(np.zeros(0, dtype=np.float32), 16000).shape, (0, 16000))

    def test_invalid_parameters_raise(self):
        with self.assertRaises(audio.AudioError):
            audio.slice_windows(np.zeros(1000, dtype=np.float32), 0)
        with self.assertRaises(audio.AudioError):
            audio.slice_windows(np.zeros(1000, dtype=np.float32), 512, hop_samples=0)

    def test_two_dimensional_input_is_flattened(self):
        windows = audio.slice_windows(np.ones((16000, 1), dtype=np.float32), 16000)
        self.assertEqual(windows.shape, (1, 16000))


class LevelTests(unittest.TestCase):
    def test_rms_matches_known_value(self):
        self.assertAlmostEqual(audio.rms(np.full(100, 0.5, dtype=np.float32)), 0.5, places=6)
        self.assertEqual(audio.rms(np.zeros(10, dtype=np.float32)), 0.0)

    def test_peak_normalize_hits_target(self):
        normalized = audio.peak_normalize(np.array([0.0, 0.25, -0.25], dtype=np.float32), 0.95)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.95, places=6)

    def test_peak_normalize_respects_max_gain(self):
        # a very quiet window must not be amplified by more than 20 dB
        quiet = np.full(100, 0.001, dtype=np.float32)
        normalized = audio.peak_normalize(quiet, target_peak=0.95, max_gain_db=20.0)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.01, places=6)

    def test_peak_normalize_handles_silence(self):
        normalized = audio.peak_normalize(np.zeros(64, dtype=np.float32))
        self.assertTrue(np.all(normalized == 0.0))

    def test_silence_gate_skips_quiet_target_windows(self):
        quiet = np.full(16000, 0.001, dtype=np.float32)
        loud = np.full(16000, 0.1, dtype=np.float32)
        self.assertLess(audio.rms(quiet), 0.005)
        self.assertGreater(audio.rms(loud), 0.005)


class ChannelTests(unittest.TestCase):
    def test_mix_averages_all_channels(self):
        samples = np.array([[0.1, 0.3, 0.5, 0.7], [-0.1, -0.3, -0.5, -0.7]], dtype=np.float32)
        mono, warning = audio.select_channels(samples, "mix")
        np.testing.assert_allclose(mono, [0.4, -0.4], atol=1e-6)
        self.assertIn("4 channels", warning)

    def test_default_channel_zero_and_warning(self):
        samples = np.array([[1.0, 9.0], [2.0, 9.0]], dtype=np.float32)
        mono, warning = audio.select_channels(samples, "index", 0)
        np.testing.assert_allclose(mono, [1.0, 2.0])
        self.assertIsNotNone(warning)

    def test_specific_channel(self):
        samples = np.array([[1.0, 9.0], [2.0, 9.0]], dtype=np.float32)
        mono, _ = audio.select_channels(samples, "index", 1)
        np.testing.assert_allclose(mono, [9.0, 9.0])

    def test_missing_channel_falls_back_with_warning(self):
        samples = np.array([[1.0, 9.0], [2.0, 9.0]], dtype=np.float32)
        mono, warning = audio.select_channels(samples, "index", 3)
        np.testing.assert_allclose(mono, [1.0, 2.0])
        self.assertIn("does not exist", warning)

    def test_mono_has_no_warning(self):
        samples = np.array([[0.1], [0.2]], dtype=np.float32)
        _, warning = audio.select_channels(samples, "index", 0)
        self.assertIsNone(warning)

    def test_unknown_mode_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.select_channels(np.zeros((4, 2), dtype=np.float32), "stereo")


class ResampleTests(unittest.TestCase):
    def test_resample_44100_to_16000_one_second(self):
        samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
        resampled = audio.resample(samples, 44100, 16000)
        self.assertEqual(resampled.shape, (16000,))
        self.assertEqual(resampled.dtype, np.float32)

    def test_same_rate_is_passthrough(self):
        samples = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(audio.resample(samples, 16000, 16000), samples)

    def test_invalid_rate_raises(self):
        with self.assertRaises(audio.AudioError):
            audio.resample(np.zeros(10, dtype=np.float32), 0, 16000)


if __name__ == "__main__":
    unittest.main()
