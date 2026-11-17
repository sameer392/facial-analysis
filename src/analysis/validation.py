"""Checks on extracted facial data. Each function returns a list of problems (empty means OK)."""

from __future__ import annotations

import math

N_LANDMARKS = 478
COORD_LIMIT = 0.5  # MediaPipe x/y are normalised to [0, 1]; allow some overshoot at the frame edge


def validate_frame(frame: dict) -> list[str]:
    issues = []
    where = f"frame {frame.get('frame', '?')}"
    detected = bool(frame.get("face_detected", False))
    points = frame.get("landmarks", [])

    if not detected:
        if points:
            issues.append(f"{where}: landmarks present although no face was detected")
        return issues

    if len(points) != N_LANDMARKS:
        issues.append(f"{where}: expected {N_LANDMARKS} landmarks, found {len(points)}")

    ids = [p.get("id") for p in points]
    if sorted(ids) != list(range(N_LANDMARKS)):
        issues.append(f"{where}: landmark ids are not 0..{N_LANDMARKS - 1}")

    for p in points:
        x, y, z = p.get("x"), p.get("y"), p.get("z")
        if any(v is None or not math.isfinite(v) for v in (x, y, z)):
            issues.append(f"{where}: landmark {p.get('id')} has a missing or non-finite coordinate")
            break
        if not (-COORD_LIMIT <= x <= 1 + COORD_LIMIT and -COORD_LIMIT <= y <= 1 + COORD_LIMIT):
            issues.append(f"{where}: landmark {p.get('id')} is outside the frame (x={x:.3f}, y={y:.3f})")
            break

    matrix = frame.get("transformation_matrix")
    if matrix is not None and (len(matrix) != 4 or any(len(row) != 4 for row in matrix)):
        issues.append(f"{where}: transformation matrix is not 4x4")
    return issues


def validate_frames(frames: list[dict]) -> list[str]:
    issues = []
    for expected_index, frame in enumerate(frames):
        if frame.get("frame") != expected_index:
            issues.append(f"frame numbers are not sequential at position {expected_index}")
            break
        issues.extend(validate_frame(frame))
    return issues


def validate_timestamps(timestamps_ms: list[float], fps: float, tolerance_ms: float = 1.0) -> list[str]:
    """Timestamps must start at or after 0, strictly increase, and step by 1000/fps."""
    issues = []
    if not timestamps_ms:
        return ["no timestamps"]
    if timestamps_ms[0] < 0:
        issues.append("first timestamp is negative")
    if fps <= 0:
        return issues + ["fps must be positive"]
    expected_step = 1000.0 / fps
    for i in range(1, len(timestamps_ms)):
        step = timestamps_ms[i] - timestamps_ms[i - 1]
        if step <= 0:
            issues.append(f"timestamps do not increase at position {i}")
            break
        if abs(step - expected_step) > tolerance_ms:
            issues.append(f"step {step:.2f} ms at position {i} differs from {expected_step:.2f} ms")
            break
    return issues
