"""Hand tracking and motion metrics (phase H2). Deterministic synthetic landmark sequences only."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import hand_metrics as hm  # noqa: E402
import hand_tracking as ht  # noqa: E402

W, H = 640, 480
TEMPLATE = np.array([  # (21, 2) hand shape in units of hand size; wrist at origin, fingers pointing up (-y)
    (0.00, 0.00), (-0.20, -0.10), (-0.30, -0.25), (-0.38, -0.40), (-0.45, -0.52),
    (-0.10, -0.40), (-0.10, -0.62), (-0.10, -0.80), (-0.10, -0.95),
    (0.00, -0.45), (0.00, -0.70), (0.00, -0.90), (0.00, -1.05),
    (0.10, -0.40), (0.12, -0.62), (0.12, -0.80), (0.12, -0.92),
    (0.20, -0.35), (0.24, -0.52), (0.26, -0.66), (0.27, -0.76),
], dtype=float)
SIZE = 80.0  # pixels between wrist and middle MCP, fixed for all synthetic hands


def hand_at(centre_xy, scale=1.0):
    """21 landmarks in image pixels, centred on the hand's centroid position."""
    pts = TEMPLATE * SIZE * scale
    pts = pts - pts.mean(axis=0) + np.asarray(centre_xy, dtype=float)
    return pts


def det(frame, t, label, centre_xy, status="valid", score=0.95, scale=1.0, outside=0):
    pts = hand_at(centre_xy, scale)
    if outside:
        pts[:outside] = [W + 30, 100]
    return ht.Detection(frame=frame, t_ms=t, label=label, status=status, score=score,
                        points=pts, outside=outside, reason="" if status in ht.USABLE_STATUSES else status)


def run_tracks(sequences, fps=10):
    """sequences: list of per-frame dicts {side: Detection or None}. Frame i has time i * 1000/fps."""
    frames = list(range(len(sequences)))
    timestamps = {f: int(round(f * 1000 / fps)) for f in frames}
    dets, statuses = {}, {}
    for f, per in enumerate(sequences):
        dets[f] = []
        for label, d in per.items():
            if d is None:
                statuses[(label, f)] = "not_detected"
            else:
                dets[f].append(d)
                statuses[(label, f)] = d.status
    events, obs = ht.track_hands(dets, frames, timestamps, statuses)
    motion = ht.motion_rows(obs)
    return events, obs, motion, frames, timestamps, statuses


def track_path(obs, track_id):
    return [o for o in obs if o["track_id"] == track_id]


class MetricTests(unittest.TestCase):
    def test_to_pixels_scales_x_by_width_and_y_by_height(self):
        p = hm.to_pixels(np.array([[0.5, 0.25, 0.9]] * 21), 640, 480)
        self.assertAlmostEqual(float(p[0, 0]), 320.0)
        self.assertAlmostEqual(float(p[0, 1]), 120.0)
        self.assertEqual(p.shape, (21, 2))

    def test_displacement_and_velocity_units(self):
        d = hm.displacement(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
        self.assertAlmostEqual(d, 5.0)
        self.assertAlmostEqual(hm.velocity(5.0, 100.0), 50.0)  # 5 px in 0.1 s = 50 px/s
        self.assertAlmostEqual(hm.acceleration(50.0, 30.0, 100.0), 200.0)  # 20 px/s over 0.1 s

    def test_missing_inputs_give_nan_not_zero(self):
        self.assertTrue(math.isnan(hm.displacement(np.array([np.nan, 0.0]), np.array([0.0, 0.0]))))
        self.assertTrue(math.isnan(hm.velocity(3.0, 0.0)))
        self.assertTrue(math.isnan(hm.velocity(3.0, float("nan"))))
        self.assertTrue(math.isnan(hm.acceleration(1.0, float("nan"), 100.0)))

    def test_direction_convention(self):
        self.assertAlmostEqual(hm.direction_deg(1.0, 0.0), 0.0)     # right
        self.assertAlmostEqual(hm.direction_deg(0.0, 1.0), 90.0)    # down (image y)
        self.assertAlmostEqual(hm.direction_deg(-1.0, 0.0), 180.0)  # left
        self.assertTrue(math.isnan(hm.direction_deg(0.0, 0.0)))     # no movement: undefined

    def test_joint_angles(self):
        self.assertAlmostEqual(hm.joint_angle(np.array([0, 0.]), np.array([1, 0.]), np.array([2, 0.])), 180.0)
        self.assertAlmostEqual(hm.joint_angle(np.array([1, 0.]), np.array([0, 0.]), np.array([0, 1.])), 90.0)
        self.assertTrue(math.isnan(hm.joint_angle(np.array([0, 0.]), np.array([0, 0.]), np.array([1, 0.]))))

    def test_finger_angles_cover_all_joints(self):
        angles = hm.finger_angles(hand_at((320, 240)))
        self.assertEqual(len(angles), 10)
        for value in angles.values():
            self.assertTrue(0.0 <= value <= 180.0)

    def test_openness_is_larger_for_an_open_hand(self):
        open_hand = hand_at((0, 0))
        closed = open_hand.copy()
        for tip, mcp in ((4, 2), (8, 6), (12, 10), (16, 14), (20, 18)):
            closed[tip] = closed[mcp] * 0.5 + closed[0] * 0.5  # folded towards the wrist
        self.assertGreater(hm.openness(open_hand), hm.openness(closed))
        self.assertGreater(hm.openness(open_hand), 0.0)

    def test_orientation_follows_the_palm_direction(self):
        upright = hand_at((0, 0))
        self.assertAlmostEqual(hm.orientation_deg(upright), -90.0, places=6)  # palm points up (-y)
        self.assertAlmostEqual(hm.wrap_angle_deg(170.0 - (-170.0)), -20.0)

    def test_wrist_and_fingertips_are_taken_from_the_model_order(self):
        pts = hand_at((100, 100))
        self.assertTrue(np.allclose(hm.wrist(pts), pts[0]))
        self.assertTrue(np.allclose(hm.fingertips(pts)[2], pts[12]))


class StationaryAndMovingTests(unittest.TestCase):
    def test_stationary_hand_keeps_one_track_with_zero_velocity(self):
        seq = [{"Right": det(f, 0, "Right", (300, 240))} for f in range(8)]
        events, obs, motion, *_ = run_tracks(seq)
        self.assertEqual({o["track_id"] for o in obs}, {1})
        speeds = [r["velocity_px_per_s"] for r in motion if r["motion_reason"] == "ok"]
        self.assertTrue(all(abs(v) < 1e-6 for v in speeds))
        self.assertEqual(motion[0]["motion_reason"], "first_observation")

    def test_horizontal_motion_has_the_right_speed_and_direction(self):
        seq = [{"Right": det(f, 0, "Right", (100 + 5 * f, 240))} for f in range(6)]  # 5 px per frame at 10 fps
        _events, _obs, motion, *_ = run_tracks(seq)
        ok = [r for r in motion if r["motion_reason"] == "ok"]
        for r in ok:
            self.assertAlmostEqual(r["velocity_px_per_s"], 50.0, places=3)
            self.assertAlmostEqual(r["direction_deg"], 0.0, places=6)
            if math.isfinite(r["acceleration_px_per_s2"]):
                self.assertAlmostEqual(r["acceleration_px_per_s2"], 0.0, places=3)
        self.assertTrue(math.isnan(ok[0]["acceleration_px_per_s2"]))  # needs two velocities first

    def test_vertical_motion_points_down(self):
        seq = [{"Right": det(f, 0, "Right", (300, 100 + 4 * f))} for f in range(5)]
        _e, _o, motion, *_ = run_tracks(seq)
        for r in motion:
            if r["motion_reason"] == "ok":
                self.assertAlmostEqual(r["direction_deg"], 90.0, places=6)

    def test_diagonal_motion_direction_is_45_degrees(self):
        seq = [{"Right": det(f, 0, "Right", (200 + 3 * f, 150 + 3 * f))} for f in range(5)]
        _e, _o, motion, *_ = run_tracks(seq)
        for r in motion:
            if r["motion_reason"] == "ok":
                self.assertAlmostEqual(r["direction_deg"], 45.0, places=6)

    def test_accelerating_hand_gives_constant_acceleration(self):
        xs, x = [], 100.0
        for f in range(6):
            xs.append(x)
            x += 2.0 * (f + 1)  # step grows by 2 px each frame: acceleration 2 px/frame^2 = 200 px/s^2 at 10 fps
        seq = [{"Right": det(f, 0, "Right", (xs[f], 240))} for f in range(6)]
        _e, _o, motion, *_ = run_tracks(seq)
        acc = [r["acceleration_px_per_s2"] for r in motion if math.isfinite(r["acceleration_px_per_s2"])]
        self.assertGreater(len(acc), 0)
        for a in acc:
            self.assertAlmostEqual(a, 200.0, places=2)

    def test_first_two_observations_have_no_acceleration(self):
        seq = [{"Right": det(f, 0, "Right", (100 + 5 * f, 240))} for f in range(3)]
        _e, _o, motion, *_ = run_tracks(seq)
        self.assertTrue(math.isnan(motion[0]["velocity_px_per_s"]))
        self.assertTrue(math.isnan(motion[1]["acceleration_px_per_s2"]))


class TwoHandTests(unittest.TestCase):
    def test_two_hands_moving_independently_keep_their_labels(self):
        seq = [{"Left": det(f, 0, "Left", (150, 200 + 3 * f)), "Right": det(f, 0, "Right", (450, 300 - 2 * f))}
               for f in range(8)]
        events, obs, *_ = run_tracks(seq)
        ids = {o["track_id"]: o["label"] for o in obs}
        self.assertEqual(len(ids), 2)
        self.assertEqual(sorted(ids.values()), ["Left", "Right"])
        self.assertEqual(len([e for e in events if e["event"] == "created"]), 2)
        self.assertFalse(any(e["uncertain_reason"] for e in events if e["event"] == "matched"))

    def test_hands_that_cross_keep_their_labels_and_are_flagged(self):
        # The left-labelled hand moves from x=150 to x=450 and the right-labelled one the other way.
        seq = []
        for f in range(9):
            lx = 150 + 37.5 * f
            rx = 450 - 37.5 * f
            seq.append({"Left": det(f, 0, "Left", (lx, 240)), "Right": det(f, 0, "Right", (rx, 240))})
        events, obs, *_ = run_tracks(seq)
        left_ids = {o["track_id"] for o in obs if o["label"] == "Left"}
        right_ids = {o["track_id"] for o in obs if o["label"] == "Right"}
        self.assertEqual(len(left_ids), 1)
        self.assertEqual(len(right_ids), 1)
        self.assertNotEqual(left_ids, right_ids)
        flagged = [o for o in obs if "hands_close" in o["uncertain_reason"]]
        self.assertGreater(len(flagged), 0)


class MissingAndReturnTests(unittest.TestCase):
    def test_temporary_missing_detection_keeps_the_track_and_flags_the_gap(self):
        seq = []
        for f in range(8):
            if f in (3, 4):
                seq.append({"Right": None})
            else:
                seq.append({"Right": det(f, 0, "Right", (300 + 4 * f, 240))})
        events, obs, motion, *_ = run_tracks(seq)
        self.assertEqual({o["track_id"] for o in obs}, {1})
        interrupted = [e for e in events if e["event"] == "interrupted"]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["frame"], 3)
        self.assertEqual(interrupted[0]["reason"], "not_detected")
        reacq = [e for e in events if e["event"] == "reacquired"]
        self.assertEqual(len(reacq), 1)
        self.assertEqual(reacq[0]["gap_frames"], 2)
        after = [r for r in motion if r["frame"] == 5][0]
        self.assertEqual(after["motion_reason"], "after_gap")
        self.assertTrue(math.isnan(after["velocity_px_per_s"]))  # no velocity across the gap, and no zero

    def test_hand_that_returns_after_a_long_absence_gets_a_new_track(self):
        gap = ht.MAX_GAP_FRAMES + 2
        seq = [{"Right": det(0, 0, "Right", (300, 240))}]
        seq += [{"Right": None} for _ in range(gap)]
        seq += [{"Right": det(1 + gap, 0, "Right", (300, 240))}]
        events, obs, *_ = run_tracks(seq)
        self.assertEqual(len({o["track_id"] for o in obs}), 2)
        term = [e for e in events if e["event"] == "terminated" and e["reason"] == "max_gap_exceeded"]
        self.assertEqual(len(term), 1)

    def test_hand_reappearing_after_leaving_the_frame_is_not_guessed_when_far(self):
        seq = [{"Right": det(0, 0, "Right", (300, 240))}, {"Right": det(1, 0, "Right", (300, 240))},
               {"Right": None}, {"Right": det(3, 0, "Right", (560, 60))}]
        events, obs, *_ = run_tracks(seq)
        ids = [o["track_id"] for o in obs]
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[1], ids[2])

    def test_low_confidence_detection_is_not_used_for_tracking(self):
        seq = [{"Right": det(0, 0, "Right", (300, 240))},
               {"Right": det(1, 0, "Right", (300, 240), status="low_confidence", score=0.3)},
               {"Right": det(2, 0, "Right", (300, 240))}]
        events, obs, motion, *_ = run_tracks(seq)
        self.assertEqual([o["frame"] for o in obs], [0, 2])
        interrupted = [e for e in events if e["event"] == "interrupted"]
        self.assertEqual(interrupted[0]["reason"], "low_confidence")

    def test_insufficient_landmarks_are_not_used(self):
        d = det(0, 0, "Right", (300, 240), outside=ht.PARTIAL_LANDMARK_LIMIT + 1)
        self.assertEqual(d.reason, "")
        bad = ht.make_detection(0, 0, "Right", "outside_frame", 0.9,
                                np.full((21, 3), np.nan), W, H)
        self.assertEqual(bad.reason, "insufficient_landmarks")
        many_out = np.zeros((21, 3))
        many_out[:, 0] = 1.5
        self.assertEqual(ht.make_detection(0, 0, "Right", "outside_frame", 0.9, many_out, W, H).reason,
                         "insufficient_landmarks")

    def test_partial_view_is_flagged_but_still_tracked(self):
        seq = [{"Right": det(f, 0, "Right", (300 + 4 * f, 240), outside=3)} for f in range(4)]
        _e, obs, motion, *_ = run_tracks(seq)
        self.assertEqual(len(obs), 4)
        self.assertTrue(all(r["partial_view"] == 1 for r in motion))


class IdentityUncertaintyTests(unittest.TestCase):
    def test_label_conflict_is_flagged_rather_than_hidden(self):
        seq = [{"Left": det(0, 0, "Left", (300, 240))},
               {"Right": det(1, 0, "Right", (302, 240))}]
        _e, obs, *_ = run_tracks(seq)
        self.assertEqual(obs[0]["track_id"], obs[1]["track_id"])
        self.assertIn("label_conflict", obs[1]["uncertain_reason"])


class TimestampTests(unittest.TestCase):
    def test_timestamps_that_do_not_increase_are_refused(self):
        dets = {0: [det(0, 0, "Right", (300, 240))], 1: [det(1, 0, "Right", (300, 240))]}
        with self.assertRaises(ht.TrackingError):
            ht.track_hands(dets, [0, 1], {0: 100, 1: 100}, {("Right", 0): "valid", ("Right", 1): "valid"})

    def test_irregular_intervals_are_counted(self):
        seq = [{"Right": det(f, 0, "Right", (300 + 2 * f, 240))} for f in range(5)]
        events, obs, motion, frames, _ts, statuses = run_tracks(seq)
        irregular_ts = {0: 0, 1: 100, 2: 150, 3: 400, 4: 500}
        q = ht.quality_summary(events, obs, motion, frames, irregular_ts, statuses)
        self.assertGreaterEqual(q["timestamps"]["irregular_intervals"], 1)

    def test_missing_frame_indices_are_interruptions(self):
        dets = {0: [det(0, 0, "Right", (300, 240))], 3: [det(3, 0, "Right", (306, 240))]}
        frames = [0, 3]
        ts = {0: 0, 3: 300}
        statuses = {("Right", 0): "valid", ("Right", 3): "valid"}
        events, obs = ht.track_hands(dets, frames, ts, statuses)
        self.assertEqual({o["track_id"] for o in obs}, {1})
        self.assertTrue(any(e["event"] == "interrupted" and e["frame"] == 1 for e in events))
        motion = ht.motion_rows(obs)
        self.assertEqual(motion[1]["motion_reason"], "after_gap")


class QualityAndSerialisationTests(unittest.TestCase):
    def test_quality_separates_the_kinds_of_loss(self):
        seq = [{"Right": det(0, 0, "Right", (300, 240))},
               {"Right": None},
               {"Right": det(2, 0, "Right", (300, 240), status="low_confidence", score=0.2)},
               {"Right": det(3, 0, "Right", (302, 240))}]
        events, obs, motion, frames, ts, statuses = run_tracks(seq)
        q = ht.quality_summary(events, obs, motion, frames, ts, statuses)
        self.assertEqual(q["missing_frame_reasons"].get("not_detected"), 1)
        self.assertEqual(q["missing_frame_reasons"].get("low_confidence"), 1)
        self.assertEqual(q["tracks_created"], 1)
        self.assertEqual(q["smoothing"], "none")
        self.assertIn("not physical", q["coordinates"])

    def test_motion_rows_match_the_declared_columns(self):
        seq = [{"Right": det(f, 0, "Right", (300 + 3 * f, 240))} for f in range(3)]
        _e, _o, motion, *_ = run_tracks(seq)
        for row in motion:
            self.assertEqual(set(row), set(ht.MOTION_COLUMNS))

    def test_track_events_use_only_declared_event_names(self):
        seq = [{"Right": det(0, 0, "Right", (300, 240))}, {"Right": None}, {"Right": det(2, 0, "Right", (300, 240))}]
        events, *_ = run_tracks(seq)
        self.assertTrue({e["event"] for e in events} <= set(ht.EVENTS))
        self.assertEqual(set(events[0]), set(ht.TRACK_COLUMNS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
