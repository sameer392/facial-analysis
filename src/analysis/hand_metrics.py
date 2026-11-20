"""Hand motion metrics (phase H2). Pure functions on landmark arrays. No file access.

Coordinate systems
    Input landmarks are MediaPipe normalised image coordinates: x / frame width, y / frame height, origin at
    the top-left, y pointing down. Normalised x and y have different physical scales on a non-square frame,
    so every distance here is computed in IMAGE PIXELS: x * width and y * height. Pixels are an image measure.
    They are not physical distances, and the velocities are not physical speeds.

    World landmarks (metres, from the model) are not used by these metrics.

Units
    distances and displacements: image pixels
    velocity: image pixels per second
    acceleration: image pixels per second squared
    angles: degrees
    openness: a ratio (dimensionless)

Missing landmarks
    A function returns NaN when an input it needs is not finite, or when a divisor is zero. Nothing is
    replaced by zero.

Smoothing
    None. These functions take the raw per-frame values. Smoothing, if ever added, must be recorded
    in the metadata as a separate step.
"""

from __future__ import annotations

import math

import numpy as np

WRIST = 0
MIDDLE_MCP = 9
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
# Joint chains per finger: (before, joint, after) indices. Angle is taken at the middle index.
FINGER_JOINTS = {
    "thumb": [(0, 1, 2), (1, 2, 3)],          # wrist-cmc-mcp, cmc-mcp-ip, mcp-ip-tip
    "index": [(5, 6, 7), (6, 7, 8)],
    "middle": [(9, 10, 11), (10, 11, 12)],
    "ring": [(13, 14, 15), (14, 15, 16)],
    "pinky": [(17, 18, 19), (18, 19, 20)],
}
JOINT_ANGLE_NAMES = {
    "thumb": ("thumb_mcp_angle_deg", "thumb_ip_angle_deg"),
    "index": ("index_pip_angle_deg", "index_dip_angle_deg"),
    "middle": ("middle_pip_angle_deg", "middle_dip_angle_deg"),
    "ring": ("ring_pip_angle_deg", "ring_dip_angle_deg"),
    "pinky": ("pinky_pip_angle_deg", "pinky_dip_angle_deg"),
}


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(21, 2) or (T, 21, 2) normalised x, y to image pixels. z is not returned."""
    arr = np.asarray(landmarks, dtype=float)[..., :2].copy()
    arr[..., 0] *= float(width)
    arr[..., 1] *= float(height)
    return arr


def centroid(points: np.ndarray) -> np.ndarray:
    """Mean of all landmarks. Returns (2,) or (T, 2). NaN if any landmark is not finite."""
    return np.mean(points, axis=-2)


def wrist(points: np.ndarray) -> np.ndarray:
    return points[..., WRIST, :]


def fingertips(points: np.ndarray) -> np.ndarray:
    """(…, 5, 2) in the order thumb, index, middle, ring, pinky."""
    return points[..., [FINGERTIPS[f] for f in ("thumb", "index", "middle", "ring", "pinky")], :]


def displacement(current: np.ndarray, previous: np.ndarray) -> float:
    """Euclidean distance between two (2,) points, in pixels. NaN when either is not finite."""
    diff = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    if not np.all(np.isfinite(diff)):
        return float("nan")
    return float(np.hypot(diff[0], diff[1]))


def velocity(disp_px: float, dt_ms: float) -> float:
    """Pixels per second. NaN unless dt_ms is a positive finite number."""
    if not (math.isfinite(disp_px) and math.isfinite(dt_ms) and dt_ms > 0):
        return float("nan")
    return disp_px / (dt_ms / 1000.0)


def acceleration(v_now: float, v_prev: float, dt_ms: float) -> float:
    """Pixels per second squared, from two consecutive velocities. NaN unless all inputs are usable."""
    if not (math.isfinite(v_now) and math.isfinite(v_prev) and math.isfinite(dt_ms) and dt_ms > 0):
        return float("nan")
    return (v_now - v_prev) / (dt_ms / 1000.0)


def direction_deg(dx: float, dy: float) -> float:
    """Direction of a displacement in degrees: 0 = towards +x (right), 90 = towards +y (down, image convention).
    NaN when there is no movement (zero length), because the direction is then undefined."""
    if not (math.isfinite(dx) and math.isfinite(dy)) or (dx == 0.0 and dy == 0.0):
        return float("nan")
    return float(math.degrees(math.atan2(dy, dx)))


def wrap_angle_deg(value: float) -> float:
    """Wrap an angle difference to (-180, 180]. NaN passes through."""
    if not math.isfinite(value):
        return float("nan")
    wrapped = (value + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b, between segments b-a and b-c, in degrees (0 to 180). Two-dimensional (image plane).
    NaN when a segment has zero length or an input is not finite."""
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    nu, nv = float(np.hypot(*u)), float(np.hypot(*v))
    if not (math.isfinite(nu) and math.isfinite(nv)) or nu == 0.0 or nv == 0.0:
        return float("nan")
    cos = float(np.dot(u, v)) / (nu * nv)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


def finger_angles(points: np.ndarray) -> dict[str, float]:
    """Two joint angles per finger (degrees), from the image-plane landmark positions of one hand (21, 2)."""
    out = {}
    for finger, chain in FINGER_JOINTS.items():
        names = JOINT_ANGLE_NAMES[finger]
        for name, (i, j, k) in zip(names, chain):
            out[name] = joint_angle(points[i], points[j], points[k])
    return out


def openness(points: np.ndarray) -> float:
    """Mean wrist-to-fingertip distance divided by the wrist-to-middle-MCP distance (palm length).
    Dimensionless. About 1.0 for a loosely open hand, lower for a closed hand. NaN if the palm length is zero."""
    palm = float(np.hypot(*(points[MIDDLE_MCP] - points[WRIST])))
    if not math.isfinite(palm) or palm == 0.0:
        return float("nan")
    tips = fingertips(points)
    dists = np.hypot(*(tips - points[WRIST]).T)
    if not np.all(np.isfinite(dists)):
        return float("nan")
    return float(np.mean(dists) / palm)


def orientation_deg(points: np.ndarray) -> float:
    """In-plane orientation: direction from the wrist to the middle-finger MCP, in degrees (image convention).
    Describes rotation in the image plane only. It is not a 3D orientation."""
    v = points[MIDDLE_MCP] - points[WRIST]
    return direction_deg(float(v[0]), float(v[1]))
