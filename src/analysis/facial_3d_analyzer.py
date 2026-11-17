"""3D facial landmark and head-pose analysis.

Head pose is read from MediaPipe's facial transformation matrix. Landmarks are
returned as stored (raw) or head-pose compensated (see motion_analyzer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def head_pose_angles(matrices: np.ndarray) -> pd.DataFrame:
    """Yaw, pitch and roll in degrees for each frame (NaN where the matrix is missing)."""
    R = matrices[:, :3, :3]
    pitch = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    yaw = np.degrees(np.arctan2(-R[:, 2, 0], np.hypot(R[:, 2, 1], R[:, 2, 2])))
    roll = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return pd.DataFrame({"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll})


def frame_points(points: np.ndarray, frame_index: int) -> pd.DataFrame:
    """One frame's landmarks as a table with id, x, y, z columns."""
    pts = points[frame_index]
    return pd.DataFrame({
        "id": np.arange(pts.shape[0]),
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
    })


def depth_range_per_frame(points: np.ndarray) -> np.ndarray:
    """Front-to-back depth span of the face (max z minus min z) for each frame."""
    z = points[:, :, 2]
    with np.errstate(all="ignore"):
        return np.nanmax(z, axis=1) - np.nanmin(z, axis=1)
