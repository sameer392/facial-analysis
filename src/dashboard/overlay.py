"""Facial landmark overlay rendering for the dashboard.

Modes:
    off        no overlay
    points     one dot per landmark (the original overlay style)
    wireframe  the full MediaPipe face tessellation (2,556 edges)
    contours   face oval, lips, eyes, eyebrows and nose, each in its own colour
    hybrid     faint full mesh with the contours drawn on top

Inputs are the existing landmark data from facial_data.json (normalised x and y). Nothing here
modifies the data or the frame passed in. The connection topology comes from MediaPipe's own
FaceLandmarksConnections list and is read once.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

N_LANDMARKS = 478
MODES = ("off", "points", "wireframe", "contours", "hybrid")

# BGR colours, chosen to be restrained against video frames.
MESH_COLOUR = (170, 170, 170)
REGION_COLOURS = {
    "oval": (235, 235, 235),
    "lips": (95, 95, 200),
    "left_eye": (170, 140, 0),
    "right_eye": (170, 140, 0),
    "left_eyebrow": (45, 85, 140),
    "right_eyebrow": (45, 85, 140),
    "nose": (90, 185, 215),
}
POINT_COLOUR = (0, 255, 0)  # unchanged from the original overlay

HYBRID_MESH_FACTOR = 0.35  # the mesh is drawn at this fraction of the chosen opacity in hybrid mode


@lru_cache(maxsize=1)
def load_topology() -> dict:
    """Edge index arrays (shape (E, 2)) from MediaPipe's official connection lists. Read once."""
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C

    def to_edges(connections) -> np.ndarray:
        return np.array([(c.start, c.end) for c in connections], dtype=np.int32)

    return {
        "mesh": to_edges(C.FACE_LANDMARKS_TESSELATION),
        "contours": to_edges(C.FACE_LANDMARKS_CONTOURS),
        "oval": to_edges(C.FACE_LANDMARKS_FACE_OVAL),
        "lips": to_edges(C.FACE_LANDMARKS_LIPS),
        "left_eye": to_edges(C.FACE_LANDMARKS_LEFT_EYE),
        "right_eye": to_edges(C.FACE_LANDMARKS_RIGHT_EYE),
        "left_eyebrow": to_edges(C.FACE_LANDMARKS_LEFT_EYEBROW),
        "right_eyebrow": to_edges(C.FACE_LANDMARKS_RIGHT_EYEBROW),
        "nose": to_edges(C.FACE_LANDMARKS_NOSE),
    }


def landmarks_to_pixels(frame_data: dict, width: int, height: int):
    """Landmarks as an array of pixel coordinates, shape (478, 2). Missing ids are NaN.

    Returns None when no face was detected or no landmarks are stored.
    """
    if not frame_data.get("face_detected", False):
        return None
    points = frame_data.get("landmarks", [])
    if not points:
        return None
    px = np.full((N_LANDMARKS, 2), np.nan, dtype=float)
    for p in points:
        idx = int(p["id"])
        if 0 <= idx < N_LANDMARKS:
            px[idx, 0] = float(p["x"]) * width
            px[idx, 1] = float(p["y"]) * height
    return px


def _draw_edges(canvas: np.ndarray, px: np.ndarray, edges: np.ndarray, colour, thickness: int) -> None:
    for a, b in edges:
        p, q = px[a], px[b]
        if np.isfinite(p).all() and np.isfinite(q).all():
            cv2.line(canvas, (int(round(p[0])), int(round(p[1]))),
                     (int(round(q[0])), int(round(q[1]))), colour, thickness, cv2.LINE_AA)


def _draw_points(canvas: np.ndarray, px: np.ndarray, radius: int) -> None:
    for x, y in px:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, POINT_COLOUR, -1, cv2.LINE_AA)


def _compose(base: np.ndarray, draw, alpha: float) -> np.ndarray:
    """Draw onto a copy of base, then blend that copy into base with the given opacity."""
    if alpha <= 0.0:
        return base
    layer = base.copy()
    draw(layer)
    return cv2.addWeighted(base, 1.0 - alpha, layer, alpha, 0.0)


def _validate(mode: str, opacity: float, thickness: int, point_radius: int) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity must be between 0 and 1")
    if thickness < 1:
        raise ValueError("thickness must be at least 1")
    if point_radius < 1:
        raise ValueError("point_radius must be at least 1")


def render_overlay(frame_bgr: np.ndarray, frame_data: dict, mode: str = "hybrid",
                   opacity: float = 0.8, thickness: int = 1, point_radius: int = 2) -> np.ndarray:
    """Return a new image with the chosen landmark overlay. The input frame is not modified."""
    _validate(mode, opacity, thickness, point_radius)
    out = frame_bgr.copy()
    if mode == "off":
        return out
    h, w = frame_bgr.shape[:2]
    px = landmarks_to_pixels(frame_data, w, h)
    if px is None:
        return out

    topo = load_topology()
    if mode == "points":
        return _compose(out, lambda c: _draw_points(c, px, point_radius), opacity)

    if mode == "wireframe":
        return _compose(out, lambda c: _draw_edges(c, px, topo["mesh"], MESH_COLOUR, thickness), opacity)

    def draw_contours(canvas):
        for region, colour in REGION_COLOURS.items():
            _draw_edges(canvas, px, topo[region], colour, thickness)

    if mode == "contours":
        return _compose(out, draw_contours, opacity)

    # hybrid: faint mesh first, then the coloured contours on top
    out = _compose(out, lambda c: _draw_edges(c, px, topo["mesh"], MESH_COLOUR, 1),
                   opacity * HYBRID_MESH_FACTOR)
    return _compose(out, draw_contours, opacity)
