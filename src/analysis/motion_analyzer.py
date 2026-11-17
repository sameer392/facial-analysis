"""Head-pose compensation, neutral-face calibration and landmark motion metrics.

All functions work on NumPy arrays, so they can be tested without the dataset files.
Coordinates are MediaPipe normalised landmarks (x, y in image units, z relative depth).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Blendshape name prefixes that describe deliberate facial movement (mouth, jaw, eyes, brows).
ACTIVITY_PREFIXES = ("jaw", "mouth", "eyeBlink", "browDown")


def compensate_head_pose(landmarks: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Express each frame's landmarks in head-local coordinates.

    The rotation from MediaPipe's facial transformation matrix is removed after centring
    each face on its own centroid. Translation and scale are not compensated, so this is
    an approximation that isolates shape change from head rotation.
    Frames without a face or without a valid matrix are returned as NaN.
    """
    out = np.full_like(landmarks, np.nan, dtype=float)
    for t in range(landmarks.shape[0]):
        pts = landmarks[t]
        if not np.isfinite(pts).all() or not np.isfinite(matrices[t]).all():
            continue
        rotation = matrices[t, :3, :3]
        centred = pts - pts.mean(axis=0)
        out[t] = centred @ rotation  # row-vector form of R^T @ p
    return out


def neutral_frame_mask(blendshapes: pd.DataFrame, usable: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """Pick the calmest tracked frames: lowest summed mouth/jaw/eye/brow activity."""
    cols = [c for c in blendshapes.columns if c.startswith(ACTIVITY_PREFIXES)]
    if not cols:
        return np.zeros(len(usable), dtype=bool)
    activity = blendshapes[cols].sum(axis=1).to_numpy()
    valid = usable & np.isfinite(activity)
    if not valid.any():
        return np.zeros(len(usable), dtype=bool)
    threshold = np.quantile(activity[valid], quantile)
    return valid & (activity <= threshold)


def neutral_shape(compensated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average head-local landmark positions over the neutral frames."""
    if not mask.any():
        return np.full(compensated.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(compensated[mask], axis=0)


def landmark_kinematics(points: np.ndarray, timestamps_s: np.ndarray, usable: np.ndarray):
    """Per-frame mean landmark displacement, velocity and acceleration.

    displacement[t] is the mean distance each landmark moved since frame t-1.
    velocity[t] = displacement[t] / dt, acceleration[t] = change in velocity / dt.
    Values are NaN where the current or previous frame has no usable face.
    """
    T = points.shape[0]
    displacement = np.full(T, np.nan)
    velocity = np.full(T, np.nan)
    acceleration = np.full(T, np.nan)
    if T < 2:
        return displacement, velocity, acceleration

    dt = np.diff(timestamps_s)
    dt = np.where(dt > 0, dt, np.nan)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=2)  # (T-1, N)
    ok = usable[1:] & usable[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_step = np.nanmean(steps, axis=1)
    mean_step = np.where(ok, mean_step, np.nan)

    displacement[1:] = mean_step
    velocity[1:] = mean_step / dt
    if T > 2:
        acceleration[2:] = np.diff(velocity[1:]) / dt[1:]
    return displacement, velocity, acceleration
