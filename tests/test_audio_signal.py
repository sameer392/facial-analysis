"""Pure audio signal functions (phase H9.2): configuration, timing, energy, pitch and baseline VAD.

Synthetic signals only, computed in memory. No FFmpeg, no files, no real recordings. These results
show that the functions behave as specified on known inputs. They are not accuracy claims for speech.
"""

import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "extraction"))

import audio_signal as sig  # noqa: E402
from audio_signal import AudioConfig, SignalError  # noqa: E402

RATE = 16000


def frames_of(x, cfg):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64),
                                                    cfg.window_samples)[::cfg.hop_samples]


def tone(freq, seconds=1.0, amp=0.5, rate=RATE):
    t = np.arange(int(rate * seconds)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_and_derived_values_are_correct(self):
        cfg = AudioConfig().validate()
        self.assertEqual(cfg.window_samples, 640)   # 40 ms at 16 kHz
        self.assertEqual(cfg.hop_samples, 160)      # 10 ms at 16 kHz
        self.assertEqual(cfg.lag_min, 40)           # 16000 / 400
        self.assertEqual(cfg.lag_max, 266)          # 16000 / 60
        self.assertEqual(cfg.as_dict()["hop_samples"], 160)

    def test_rates_outside_the_supported_range_are_refused(self):
        for rate in (4000, 96000):
            with self.assertRaises(SignalError):
                AudioConfig(analysis_rate=rate).validate()

    def test_a_hop_shorter_than_one_sample_is_refused(self):
        with self.assertRaises(SignalError):
            AudioConfig(analysis_rate=8000, hop_ms=0.01).validate()

    def test_inverted_pitch_range_is_refused(self):
        with self.assertRaises(SignalError):
            AudioConfig(pitch_min_hz=400, pitch_max_hz=60).validate()

    def test_a_window_too_short_for_the_lowest_pitch_is_refused(self):
        with self.assertRaises(SignalError):
            AudioConfig(window_ms=10, pitch_min_hz=60).validate()

    def test_thresholds_out_of_range_are_refused(self):
        with self.assertRaises(SignalError):
            AudioConfig(pitch_threshold=0).validate()
        with self.assertRaises(SignalError):
            AudioConfig(vad_db=3).validate()
        with self.assertRaises(SignalError):
            AudioConfig(vad_min_speech_ms=-1).validate()


class TimingTests(unittest.TestCase):
    def test_sample_times_match_exact_rational_arithmetic(self):
        for rate in (8000, 16000, 22050, 44100, 48000):
            for base in (0, 21333, 10**9 + 7):
                for j in (0, 1, 2, 999, 123457, rate * 3600 + 17):
                    exact = base + math.floor(Fraction(j) * 10**6 / rate + Fraction(1, 2))
                    got = int(sig.sample_time_us(np.array([j]), rate, base)[0])
                    self.assertEqual(got, exact, (rate, base, j))

    def test_one_hour_has_no_cumulative_drift(self):
        for rate in (44100, 16000, 48000):
            n = rate * 3600
            self.assertEqual(int(sig.sample_time_us(np.array([n]), rate, 0)[0]), 3600 * 10**6)

    def test_half_microsecond_rounds_up(self):
        # Sample 1 at 16 kHz is 62.5 us: it rounds half up to 63.
        self.assertEqual(int(sig.sample_time_us(np.array([1]), 16000, 0)[0]), 63)

    def test_non_zero_base_is_added_exactly(self):
        base = 250_000 + 21_333
        self.assertEqual(int(sig.sample_time_us(np.array([320]), 16000, base)[0]), base + 20000)

    def test_frame_grid_counts_complete_windows_only(self):
        cfg = AudioConfig().validate()
        self.assertEqual(sig.frame_grid(640, cfg), (1, 0))
        self.assertEqual(sig.frame_grid(799, cfg), (1, 159))
        self.assertEqual(sig.frame_grid(800, cfg), (2, 0))
        self.assertEqual(sig.frame_grid(639, cfg), (0, 639))

    def test_a_different_hop_changes_the_frame_count(self):
        cfg = AudioConfig(hop_ms=20).validate()
        frames, _ = sig.frame_grid(RATE, cfg)
        self.assertEqual(frames, (RATE - 640) // 320 + 1)


class EnergyTests(unittest.TestCase):
    def test_rms_of_a_sine_is_amplitude_over_root_two(self):
        cfg = AudioConfig().validate()
        rms, db, silent = sig.energy(frames_of(tone(200, amp=0.5), cfg), cfg)
        self.assertAlmostEqual(float(np.median(rms)), 0.5 / np.sqrt(2), places=3)
        self.assertAlmostEqual(float(np.median(db)), 20 * np.log10(0.5) - 10 * np.log10(2), places=2)
        self.assertFalse(silent.any())

    def test_silence_gets_the_floor_and_is_flagged_not_measured(self):
        cfg = AudioConfig().validate()
        rms, db, silent = sig.energy(frames_of(np.zeros(RATE), cfg), cfg)
        self.assertTrue(np.all(db == sig.SILENCE_FLOOR_DB))
        self.assertTrue(silent.all())
        self.assertTrue(np.all(rms == 0))

    def test_non_finite_samples_are_refused(self):
        cfg = AudioConfig().validate()
        x = tone(200)
        x[100] = np.nan
        with self.assertRaises(SignalError):
            sig.energy(frames_of(x, cfg), cfg)

    def test_other_window_and_hop_sizes_work(self):
        for window_ms, hop_ms in ((40, 5), (50, 10), (64, 16)):  # windows must hold two periods of 60 Hz
            cfg = AudioConfig(window_ms=window_ms, hop_ms=hop_ms).validate()
            rms, db, _ = sig.energy(frames_of(tone(200, amp=0.5), cfg), cfg)
            self.assertAlmostEqual(float(np.median(rms)), 0.5 / np.sqrt(2), places=3, msg=(window_ms, hop_ms))


class PitchTests(unittest.TestCase):
    def estimate(self, x, cfg=None):
        cfg = (cfg or AudioConfig()).validate()
        frames = frames_of(x, cfg)
        _, _, silent = sig.energy(frames, cfg)
        return sig.pitch(frames, cfg, silent)

    def test_known_tones_are_estimated_accurately(self):
        for freq in (90, 120, 220, 300, 380):
            f0, conf = self.estimate(tone(freq, seconds=1.0))
            voiced = f0[~np.isnan(f0)]
            self.assertTrue(voiced.size > 0.9 * f0.size, freq)
            self.assertLess(float(np.max(np.abs(voiced - freq))), 0.5, f"{freq} Hz estimated as {np.median(voiced)}")
            self.assertGreater(float(conf.min()), 0.95)

    def test_a_tone_is_not_reported_as_a_multiple_of_its_period(self):
        f0, _ = self.estimate(tone(220, seconds=1.0))
        self.assertLess(abs(float(np.nanmedian(f0)) - 220), 1.0)

    def test_silence_has_no_pitch(self):
        f0, conf = self.estimate(np.zeros(RATE))
        self.assertTrue(np.all(np.isnan(f0)))

    def test_white_noise_is_not_treated_as_voiced(self):
        rng = np.random.default_rng(3)
        f0, conf = self.estimate(rng.standard_normal(RATE) * 0.2)
        self.assertEqual(int(np.sum(~np.isnan(f0))), 0)
        self.assertLess(float(conf.max()), 0.5)

    def test_the_threshold_is_configurable(self):
        rng = np.random.default_rng(4)
        noisy = tone(220, amp=0.5) + rng.standard_normal(RATE) * 0.5  # low signal-to-noise ratio
        strict, _ = self.estimate(noisy, AudioConfig(pitch_threshold=0.95))
        lax, _ = self.estimate(noisy, AudioConfig(pitch_threshold=0.2))
        self.assertLess(np.sum(~np.isnan(strict)), np.sum(~np.isnan(lax)))

    def test_voiced_values_respect_the_configured_range(self):
        f0, _ = self.estimate(tone(120, seconds=1.0), AudioConfig(pitch_min_hz=100, pitch_max_hz=150))
        voiced = f0[~np.isnan(f0)]
        self.assertTrue(np.all(voiced >= 100 * 0.999))
        self.assertTrue(np.all(voiced <= 150 * 1.001))

    def test_a_tone_below_the_range_is_not_reported_inside_it(self):
        f0, _ = self.estimate(tone(45, seconds=1.0), AudioConfig(pitch_min_hz=60))
        voiced = f0[~np.isnan(f0)]
        self.assertTrue(np.all(voiced >= 60 * 0.999))


class VadTests(unittest.TestCase):
    def cfg(self, **kw):
        return AudioConfig(**kw).validate()

    def segments(self, flags, **kw):
        return sig.vad_segments(np.asarray(flags, dtype=bool), self.cfg(**kw))

    def test_silence_only_gives_one_silence_segment(self):
        segs = self.segments([False] * 300)
        self.assertEqual(segs, [{"kind": "silence", "start_frame": 0, "end_frame": 300}])

    def test_speech_then_silence(self):
        flags = [False] * 30 + [True] * 50 + [False] * 100  # 300 ms leading silence, above the 200 ms minimum
        segs = self.segments(flags)
        self.assertEqual([(s["kind"], s["start_frame"], s["end_frame"]) for s in segs],
                         [("silence", 0, 30), ("speech", 30, 80), ("silence", 80, 180)])

    def test_multiple_speech_segments_are_kept_apart_by_long_silence(self):
        flags = [True] * 30 + [False] * 60 + [True] * 30
        speech = [s for s in self.segments(flags) if s["kind"] == "speech"]
        self.assertEqual(len(speech), 2)

    def test_a_short_gap_is_merged(self):
        flags = [True] * 30 + [False] * 10 + [True] * 30  # gap is 100 ms, below the 200 ms merge gap
        speech = [s for s in self.segments(flags) if s["kind"] == "speech"]
        self.assertEqual(speech, [{"kind": "speech", "start_frame": 0, "end_frame": 70}])

    def test_a_gap_longer_than_the_merge_threshold_is_not_merged(self):
        flags = [True] * 30 + [False] * 30 + [True] * 30  # gap is 300 ms
        speech = [s for s in self.segments(flags) if s["kind"] == "speech"]
        self.assertEqual(len(speech), 2)

    def test_speech_shorter_than_the_minimum_is_dropped(self):
        flags = [False] * 50 + [True] * 5 + [False] * 50  # 50 ms burst, minimum 100 ms
        segs = self.segments(flags)
        self.assertFalse(any(s["kind"] == "speech" for s in segs))

    def test_silence_shorter_than_the_minimum_is_not_reported(self):
        flags = [True] * 30 + [False] * 15 + [True] * 30  # 150 ms gap: merged if merge gap is 150 ms
        segs = self.segments(flags, vad_merge_gap_ms=0, vad_min_silence_ms=200)
        self.assertFalse(any(s["kind"] == "silence" for s in segs))

    def test_the_energy_threshold_is_configurable(self):
        energy_db = np.array([-30.0, -45.0, -60.0])
        self.assertEqual(sig.vad_frames(energy_db, -40).tolist(), [True, False, False])
        self.assertEqual(sig.vad_frames(energy_db, -50).tolist(), [True, True, False])

    def test_speech_like_bursts_are_detected_and_silence_is_not(self):
        cfg = self.cfg()
        x = np.concatenate([np.zeros(RATE // 2), tone(200, seconds=1.0, amp=0.3), np.zeros(RATE // 2)])
        frames = frames_of(x, cfg)
        _, db, _ = sig.energy(frames, cfg)
        speech = sig.vad_frames(db, cfg.vad_db)
        segs = sig.vad_segments(speech, cfg)
        speech_segs = [s for s in segs if s["kind"] == "speech"]
        self.assertEqual(len(speech_segs), 1)
        start_ms = speech_segs[0]["start_frame"] * cfg.hop_ms
        self.assertLess(abs(start_ms - 500), 2 * cfg.hop_ms + cfg.window_ms)


if __name__ == "__main__":
    unittest.main()
