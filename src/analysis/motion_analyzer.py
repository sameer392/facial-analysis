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


# Sign flip of the image x-axis relative to the transformation matrix's rotation frame.
# Found by a read-only study on local sessions (docs/validation.md): it gave the lowest
# residual scatter in all three sessions tested. It is an empirical convention, not a
# documented MediaPipe guarantee.
AXIS_FLIP = np.diag([-1.0, 1.0, 1.0])


def nearest_rotation(matrix3: np.ndarray) -> np.ndarray:
    """Closest proper rotation to a 3x3 matrix (removes any scale or shear, keeps det = +1)."""
    U, _, Vt = np.linalg.svd(matrix3)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def compensate_head_pose(landmarks: np.ndarray, matrices: np.ndarray, frame_size,
                         axis_flip: np.ndarray = AXIS_FLIP) -> np.ndarray:
    """Head-pose compensated landmark shape, dimensionless and scale-normalised.

    Steps for each frame:
    1. Convert normalised landmarks to pixel-proportional units: x*W, y*H, z*W. MediaPipe
       normalises x and y by different sizes, so without this step a rotation is distorted.
    2. Centre on the frame's own landmark centroid, which removes translation.
    3. Apply the head rotation (nearest proper rotation of the matrix) with the axis flip.
    4. Divide by the RMS distance from the centroid, which removes scale.

    Frames without a face or without a usable matrix are NaN.
    """
    W, H = frame_size
    out = np.full(landmarks.shape, np.nan, dtype=float)
    for t in range(landmarks.shape[0]):
        pts = landmarks[t]
        mat = matrices[t]
        if not np.isfinite(pts).all() or not np.isfinite(mat).all():
            continue
        iso = np.stack([pts[:, 0] * W, pts[:, 1] * H, pts[:, 2] * W], axis=1)
        centred = iso - iso.mean(axis=0)
        R = nearest_rotation(mat[:3, :3])
        shape = centred @ (axis_flip @ R)  # each row p gives (flip @ R)^T p = R^T flip p
        rms = np.sqrt(np.mean(np.sum(shape ** 2, axis=1)))
        if rms > 0:
            out[t] = shape / rms
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
    velocity[t] = displacement[t] / (t_t - t_{t-1}): the speed over the interval ending at frame t.
    It is assigned to the interval midpoint m_t = (t_{t-1} + t_t) / 2.
    acceleration[t] = (velocity[t] - velocity[t-1]) / (m_t - m_{t-1}), where m_t - m_{t-1}
    = (t_t - t_{t-2}) / 2. Using the midpoints keeps acceleration correct for irregular timestamps.

    Velocity and acceleration are speed magnitudes: they are never negative and do not give direction.
    Values are NaN where the current or previous frame has no usable face, or where timestamps do not increase.
    """
    T = points.shape[0]
    displacement = np.full(T, np.nan)
    velocity = np.full(T, np.nan)
    acceleration = np.full(T, np.nan)
    if T < 2:
        return displacement, velocity, acceleration

    timestamps_s = np.asarray(timestamps_s, dtype=float)
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
        midpoint_span = (timestamps_s[2:] - timestamps_s[:-2]) / 2.0  # m_t - m_{t-1} for t = 2..T-1
        midpoint_span = np.where(midpoint_span > 0, midpoint_span, np.nan)
        acceleration[2:] = np.diff(velocity[1:]) / midpoint_span
    return displacement, velocity, acceleration
