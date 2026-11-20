"""Visibility gate for pose landmarks that fall outside the frame.

Rule: a joint whose normalised x or y lies outside [0, 1] is treated as not visible, and its
coordinates are set to NaN, unless its visibility is at least the threshold. Joints inside the
frame are kept whatever their visibility. Visibility and presence are kept as recorded, so the
reason for a gated joint can still be seen.

This is an analysis step. It returns a derived copy and never changes the stored pose data arrays, which
hold the raw model output. The threshold is an empirical choice, not a validated constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_VISIBILITY_THRESHOLD = 0.5
N_POSE_LANDMARKS = 33
COORDINATE_CHANNELS = (0, 1, 2)  # x, y, z
VISIBILITY_CHANNEL = 3


@dataclass(frozen=True)
class GateReport:
    threshold: float
    gated_joint_samples: int
    frames_with_gated_joints: int
    total_joint_samples: int

    @property
    def gated_fraction(self) -> float:
        return self.gated_joint_samples / self.total_joint_samples if self.total_joint_samples else 0.0


def gate_pose(pose: np.ndarray, threshold: float = DEFAULT_VISIBILITY_THRESHOLD):
    """Return (gated copy, GateReport) for an array of shape (T, 33, 5) with channels x, y, z, visibility, presence."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")
    arr = np.asarray(pose)
    if arr.ndim != 3 or arr.shape[1:] != (N_POSE_LANDMARKS, 5):
        raise ValueError(f"pose must have shape (T, {N_POSE_LANDMARKS}, 5), got {arr.shape}")

    out = arr.astype(np.float32, copy=True)
    x, y, vis = arr[:, :, 0], arr[:, :, 1], arr[:, :, VISIBILITY_CHANNEL]
    with np.errstate(invalid="ignore"):
        outside = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        low_visibility = vis < threshold
    gated = outside & low_visibility  # NaN comparisons are False, so missing joints are not counted
    out[gated, 0:3] = np.nan

    report = GateReport(
        threshold=float(threshold),
        gated_joint_samples=int(gated.sum()),
        frames_with_gated_joints=int(gated.any(axis=1).sum()),
        total_joint_samples=int(np.isfinite(x).sum()),
    )
    return out, report
