"""Hand gesture recognition (phase H3). Synthetic hands built from finger configurations. No real recordings."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import gesture_rules as gr  # noqa: E402

PALM_PX = 80.0
CENTRE = (320.0, 240.0)
MCP = {"index": (-0.35, -0.95), "middle": (0.0, -1.0), "ring": (0.33, -0.93), "pinky": (0.6, -0.8)}
THUMB_CMC = (-0.3, -0.15)
THUMB_DIR_EXTENDED = (-0.8, -0.6)
THUMB_DIR_UP = (-0.15, -1.0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.hypot(*v)


def _perp(d):
    return np.array([-d[1], d[0]])


def finger_chain(mcp, state, direction):
    """(pip, dip, tip) for a non-thumb finger in palm units."""
    d = _unit(direction)
    mcp = np.asarray(mcp, dtype=float)
    if state == "extended":
        pip = mcp + d * 0.45
        dip = pip + d * 0.30
        tip = dip + d * 0.25
    else:  # curled: the fingertip folds back towards the palm
        pip = mcp + d * 0.45
        dip = pip + _perp(d) * 0.30
        tip = mcp - d * 0.05 + _perp(d) * 0.35
    return pip, dip, tip


def thumb_chain(state, direction):
    """(mcp, ip, tip) for the thumb in palm units."""
    d = _unit(direction)
    cmc = np.asarray(THUMB_CMC, dtype=float)
    if state == "extended":
        mcp = cmc + d * 0.40
        ip = mcp + d * 0.35
        tip = ip + d * 0.30
    else:
        mcp = cmc + d * 0.35
        ip = mcp + _perp(d) * 0.30
        tip = mcp - d * 0.05 + _perp(d) * 0.35
    return mcp, ip, tip


def make_hand(states, thumb_up=False, rotate_deg=0.0, mirror=False, centre=CENTRE, scale=PALM_PX):
    """21 landmarks in image pixels. states: dict finger -> 'extended' or 'curled'.
    Rotation is in the image plane about the wrist. Mirror flips x before rotation."""
    pts = np.zeros((21, 2))
    pts[0] = (0.0, 0.0)
    # thumb
    t_dir = THUMB_DIR_UP if thumb_up else THUMB_DIR_EXTENDED
    t_state = states["thumb"]
    mcp, ip, tip = thumb_chain(t_state, t_dir if t_state == "extended" else (-0.8, -0.6))
    pts[1], pts[2], pts[3], pts[4] = THUMB_CMC, mcp, ip, tip
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for finger, first in base.items():
        m = np.asarray(MCP[finger], dtype=float)
        pip, dip, tip_f = finger_chain(m, states[finger], m)
        pts[first] = m
        pts[first + 1], pts[first + 2], pts[first + 3] = pip, dip, tip_f
    if mirror:
        pts[:, 0] *= -1.0
    theta = math.radians(rotate_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    pts = pts @ rot.T
    return pts * scale + np.asarray(centre, dtype=float)


ALL_EXTENDED = {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"}
ALL_CURLED = {"thumb": "curled", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"}
POINTING = {"thumb": "curled", "index": "extended", "middle": "curled", "ring": "curled", "pinky": "curled"}
PEACE = {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled"}
THUMBS_UP_STATES = {"thumb": "extended", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"}


def classify(points, status="valid", partial=False):
    return gr.classify_points(points, status, partial)


class GestureClassTests(unittest.TestCase):
    def test_open_palm(self):
        self.assertEqual(classify(make_hand(ALL_EXTENDED)).gesture, "open_palm")

    def test_closed_fist(self):
        self.assertEqual(classify(make_hand(ALL_CURLED)).gesture, "closed_fist")

    def test_pointing(self):
        self.assertEqual(classify(make_hand(POINTING)).gesture, "pointing")

    def test_thumbs_up(self):
        self.assertEqual(classify(make_hand(THUMBS_UP_STATES, thumb_up=True)).gesture, "thumbs_up")

    def test_peace(self):
        self.assertEqual(classify(make_hand(PEACE)).gesture, "peace")

    def test_decided_frames_have_a_reason_and_margin(self):
        c = classify(make_hand(POINTING))
        self.assertEqual(c.reason, "")
        self.assertGreater(c.margin, 0.0)
        self.assertEqual(c.states["index"], "extended")

    def test_each_class_is_separated_from_the_others(self):
        hands = {"open_palm": ALL_EXTENDED, "closed_fist": ALL_CURLED, "pointing": POINTING, "peace": PEACE}
        for name, states in hands.items():
            with self.subTest(gesture=name):
                self.assertEqual(classify(make_hand(states)).gesture, name)


class InvarianceTests(unittest.TestCase):
    def test_rotated_hands_keep_their_class(self):
        cases = [(ALL_EXTENDED, False, "open_palm"), (ALL_CURLED, False, "closed_fist"),
                 (POINTING, False, "pointing"), (THUMBS_UP_STATES, True, "thumbs_up"), (PEACE, False, "peace")]
        for angle in (0, 30, 90, 180, -45, 135):
            for states, up, name in cases:
                with self.subTest(gesture=name, angle=angle):
                    self.assertEqual(classify(make_hand(states, thumb_up=up, rotate_deg=angle)).gesture, name)

    def test_translation_and_scale_do_not_change_the_class(self):
        for scale in (40.0, 120.0):
            self.assertEqual(classify(make_hand(PEACE, centre=(100, 90), scale=scale)).gesture, "peace")

    def test_mirrored_hands_keep_their_class(self):
        cases = [(ALL_EXTENDED, False, "open_palm"), (POINTING, False, "pointing"), (THUMBS_UP_STATES, True, "thumbs_up")]
        for states, up, name in cases:
            with self.subTest(gesture=name):
                pts = make_hand(states, thumb_up=up)
                mirrored = make_hand(states, thumb_up=up, mirror=True)
                self.assertEqual(classify(mirrored).gesture, name)
                self.assertEqual(classify(pts).gesture, classify(mirrored).gesture)

    def test_left_and_right_labels_give_the_same_class(self):
        pts = make_hand(PEACE)
        left = gr.HandObservation(1, 0, 0, "Left", "valid", pts)
        right = gr.HandObservation(2, 0, 0, "Right", "valid", pts)
        rows = gr.classify_observations([left, right])
        self.assertEqual(rows[0]["frame_class"], rows[1]["frame_class"])
        self.assertEqual(rows[0]["frame_class"], "peace")


class UnknownAndQualityTests(unittest.TestCase):
    def test_halfway_finger_is_indeterminate_so_unknown(self):
        states = dict(PEACE)
        pts = make_hand(states)
        # move the index tip to a middle distance: neither extended nor curled
        pts[8] = pts[5] + (pts[8] - pts[5]) * 0.5 + np.array([0.0, 0.0])
        pts[7] = pts[6] + (pts[7] - pts[6]) * 0.5
        c = classify(pts)
        self.assertEqual(c.gesture, "unknown")
        self.assertEqual(c.reason, "indeterminate_finger")

    def test_pattern_without_a_rule_is_unknown(self):
        states = {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "curled"}
        c = classify(make_hand(states))
        self.assertEqual(c.gesture, "unknown")
        self.assertEqual(c.reason, "no_rule_matched")

    def test_low_confidence_is_unknown_with_reason(self):
        c = classify(make_hand(POINTING), status="low_confidence")
        self.assertEqual((c.gesture, c.quality), ("unknown", "low_confidence"))

    def test_missing_landmarks_are_unknown_not_guessed(self):
        pts = make_hand(POINTING)
        pts[8] = np.nan
        c = classify(pts)
        self.assertEqual((c.gesture, c.reason), ("unknown", "insufficient_landmarks"))

    def test_tiny_hand_is_insufficient(self):
        c = classify(make_hand(POINTING, scale=2.0))
        self.assertEqual(c.reason, "insufficient_landmarks")

    def test_partially_visible_hand_is_flagged_and_classified_when_fingers_are_decided(self):
        pts = make_hand(POINTING)
        pts[0:, 0] = np.minimum(pts[0:, 0], 700.0)   # nothing is outside the frame here
        c = classify(pts, status="outside_frame", partial=True)
        self.assertEqual(c.gesture, "pointing")
        self.assertEqual(c.quality, "partial_view")

    def test_partially_visible_hand_with_a_cut_off_finger_is_unknown(self):
        pts = make_hand(PEACE)
        pts[8] = (-40.0, -40.0)  # index tip cut off the frame: its state cannot be decided from the image
        pts[7] = (-30.0, -20.0)
        c = classify(pts, status="outside_frame", partial=True)
        self.assertEqual(c.gesture, "unknown")


class TemporalTests(unittest.TestCase):
    def _rows(self, labels, track=1, step_ms=100):
        obs = []
        for i, g in enumerate(labels):
            if g is None:
                continue
            states = {"open_palm": ALL_EXTENDED, "pointing": POINTING, "closed_fist": ALL_CURLED,
                      "peace": PEACE}.get(g)
            pts = make_hand(states) if states else make_hand(
                {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "curled"})
            obs.append(gr.HandObservation(track, i, i * step_ms, "Right", "valid", pts))
        return gr.classify_observations(obs)

    def test_sustained_gesture_becomes_one_event_with_frames_and_times(self):
        rows = self._rows(["pointing"] * 8)
        events, rows = gr.build_events(rows)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual((ev["gesture"], ev["start_frame"], ev["end_frame"]), ("pointing", 0, 7))
        self.assertEqual(ev["start_timestamp_ms"], 0)
        self.assertEqual(ev["duration_ms"], 700)
        self.assertEqual(ev["supporting_observations"], 8)
        self.assertEqual(gr.validate_events(events), [])
        self.assertTrue(all(r["event_id"] == ev["event_id"] for r in rows))

    def test_single_noisy_frame_does_not_split_a_gesture(self):
        rows = self._rows(["pointing"] * 4 + ["peace"] + ["pointing"] * 4)
        events, _ = gr.build_events(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gesture"], "pointing")
        self.assertEqual(events[0]["bridged_observations"], 1)

    def test_short_unknown_interruption_is_bridged(self):
        rows = self._rows(["open_palm"] * 5 + [None] + ["open_palm"] * 5)
        # the None above is a missing frame: missing frames are never bridged, so this makes two events
        events, _ = gr.build_events(rows)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["end_reason"], "gap")

    def test_unknown_frames_within_the_limit_are_bridged(self):
        rows = self._rows(["open_palm"] * 4)
        extra = [dict(rows[-1])]
        for k in range(2):
            u = dict(rows[-1]); u["frame"] += 1 + k; u["timestamp_ms"] += 100 * (1 + k)
            u["frame_class"] = "unknown"; u["reason"] = "indeterminate_finger"; u["event_id"] = ""
            extra.append(u)
        cont = dict(rows[-1]); cont["frame"] += 3; cont["timestamp_ms"] += 300; cont["event_id"] = ""
        events, _ = gr.build_events(rows + extra[1:] + [cont])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["bridged_observations"], 2)
        self.assertEqual(events[0]["supporting_observations"], 4 + 1)

    def test_long_unknown_stretch_ends_the_gesture(self):
        base = self._rows(["open_palm"] * 5)
        last = base[-1]
        unknowns = []
        for k in range(gr.MAX_BRIDGE + 2):
            u = dict(last); u["frame"] = last["frame"] + 1 + k; u["timestamp_ms"] = last["timestamp_ms"] + 100 * (k + 1)
            u["frame_class"] = "unknown"; u["event_id"] = ""
            unknowns.append(u)
        after = self._rows(["open_palm"] * 5)
        for r in after:
            r["frame"] += 100; r["timestamp_ms"] += 10000
        events, _ = gr.build_events(base + unknowns + after)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["end_reason"], "unknown_run")

    def test_close_gestures_of_different_class_are_not_merged(self):
        rows = self._rows(["pointing"] * 6 + ["peace"] * 6)
        events, _ = gr.build_events(rows)
        self.assertEqual([e["gesture"] for e in events], ["pointing", "peace"])
        self.assertEqual(events[0]["end_reason"], "class_changed")

    def test_short_run_is_not_an_event(self):
        rows = self._rows(["pointing"] * 3 + ["open_palm"] * 3)
        events, _ = gr.build_events(rows)
        self.assertEqual(events, [])

    def test_tracks_are_independent(self):
        a = self._rows(["pointing"] * 6, track=1)
        b = self._rows(["peace"] * 6, track=2)
        events, _ = gr.build_events(a + b)
        self.assertEqual(sorted(e["gesture"] for e in events), ["peace", "pointing"])

    def test_invalid_timestamps_are_refused(self):
        rows = self._rows(["pointing"] * 6)
        rows[3]["timestamp_ms"] = rows[2]["timestamp_ms"]
        with self.assertRaises(gr.GestureError):
            gr.build_events(rows)


class EventValidationTests(unittest.TestCase):
    def test_validator_catches_inconsistent_events(self):
        bad = {"event_id": "E1", "track_id": 1, "label": "Right", "gesture": "unknown", "start_frame": 5,
               "end_frame": 2, "start_timestamp_ms": 0, "end_timestamp_ms": 10, "duration_ms": 3,
               "supporting_observations": 1, "bridged_observations": 0, "support_quality": 1.5,
               "margin_median": 0.1, "end_reason": "x"}
        issues = gr.validate_events([bad])
        self.assertGreaterEqual(len(issues), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
