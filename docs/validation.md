# Validation of head-pose compensation and motion metrics

This document records what was checked, how, and what the checks do and do not establish.
It covers the mathematics of the analysis modules and the tracking-quality checks. It does not
claim that the measured facial motion is clinically or biometrically accurate.

## 1. Conventions

**Landmarks** (`facial_data.json`) are MediaPipe normalised image coordinates:
`x` is divided by the frame width `W`, `y` by the frame height `H`, and `z` is a relative depth
on the same scale as `x`.

**Transformation matrix** (`transformation_matrix`) is a 4×4 matrix. Its top-left 3×3 block is
the head rotation (plus any scale). The translation column is not used.

**Isotropic units.** Before any rotation, landmarks are converted to pixel-proportional units:
`(x·W, y·H, z·W)`. Without this step, a rotation is distorted for any frame whose width and
height differ.

**Axis flip.** The image x-axis is mirrored relative to the matrix's rotation frame:
`F = diag(-1, 1, 1)`. This was found empirically (section 3). It is not a documented MediaPipe
guarantee, and it is the most important convention in this document.

## 2. Formulas

For each frame with a face and a usable matrix:

1. `I = (x·W, y·H, z·W)` for all 478 landmarks.
2. `C = I − mean(I)` removes translation.
3. `R = nearest proper rotation of M[:3,:3]` (SVD: `U·Vᵀ`, with det = +1). This removes any scale in the matrix.
4. `Q = C · (F·R)` (row vectors). When the landmarks came from a head rotation `R`, `Q` recovers the canonical shape.
5. `S = Q / sqrt(mean_n |Q_n|²)` removes scale. The result is dimensionless (face-size units).

**Motion** (per frame `t`, uses `S`):

- `displacement_t` = mean over landmarks of `|S_t − S_{t−1}|`
- `velocity_t` = `displacement_t / (τ_t − τ_{t−1})`, where `τ` is time in seconds
- `acceleration_t` = `(velocity_t − velocity_{t−1}) / ((τ_t − τ_{t−2}) / 2)`

The acceleration denominator is the time between velocity midpoints, which is correct for
irregular timestamps. Velocity and acceleration are speed magnitudes and carry no direction.

**Neutral calibration:** the tracked frames with the lowest summed mouth, jaw, eye-blink and
brow-down activity (lowest 10%). The neutral shape is the mean of those frames.

**Blinks:** a blink is a run of frames where the mean of `eyeBlinkLeft` and `eyeBlinkRight` is at
least 0.5. The threshold has not been validated against manual annotation.

**Head angles** (`yaw`, `pitch`, `roll`) are read from the matrix with the standard ZYX
decomposition. Their sign convention is not verified against an external reference, and it
should be treated with care.

## 3. Empirical check of the convention

A read-only study on local sessions compared 128 sign and transpose combinations of the rotation,
using calm frames only. Calm frames were selected as in section 2. The best candidate had
`F = diag(-1, 1, 1)` and used `R` (not `Rᵀ`) in every session. The output-side sign did not
change the scatter, because flipping output axes does not change spread.

Scale-free scatter across calm frames (mean per-landmark standard deviation of the
RMS-normalised shape; lower is steadier). Sessions are anonymised.

| Session | Calm frames | No compensation | Previous formula | New formula |
|---|---|---|---|---|
| Person A / session 001 | 7 | 0.00640 | 0.00738 | 0.00651 |
| Person A / session 002 | 55 | 0.01039 | 0.01179 | 0.00607 |
| Person B / session 001 | 40 | 0.02737 | 0.02468 | 0.00712 |

What this shows:

- The **previous formula was worse than no compensation** in two of three sessions, because it
  had no axis flip and used distorted coordinates.
- The **new formula reduced scatter by 42% and 74%** where there were enough calm frames
  (55 and 40). This is evidence that the convention is right, but it comes from three sessions
  of two people.
- With **7 calm frames** (Person A / session 001) the new formula made no difference. This is
  too few frames to judge anything.

This is a convention check, not proof of MediaPipe's internal coordinate system. A stronger check
would use a recording with a known, measured head rotation.

## 4. Tests

`tests/test_scientific.py` (19 tests) uses a synthetic face shape and the convention above:

- rotation about X, Y and Z axes, and a combined rotation
- translation along each axis, scale changes, and a combined case
- matrix scale factor ignored
- missing matrix and missing landmarks give NaN without affecting other frames
- aspect ratio matters (transposed frame size gives a different result)
- the axis flip is required for this convention
- neutral calibration selects the calmest frames, and the neutral shape is their mean
- untracked frames are never used for calibration
- constant speed and constant acceleration on regular timestamps
- constant acceleration on irregular timestamps (this test failed on the previous formula)
- velocity on irregular timestamps
- gaps in tracking produce NaN around the gap
- non-increasing timestamps produce NaN

Tests were written before the implementation was changed. The tests that encode the convention
prove the mathematics is internally consistent. They do not prove that MediaPipe uses this
convention; section 3 is the evidence for that.

## 5. Data quality (tracking only)

`tracking_quality.json` reports how reliably the face was found and described:

- detection rate, longest run without a face, and the list of interruptions
- transformation-matrix availability (over all frames and over detected frames)
- landmark completeness: detected frames with all 478 finite landmarks
- timestamp checks: strictly increasing, and steps equal to 1000/fps within 1 ms
- median frame-to-frame landmark jitter

These measure tracking, not motion. A session can score 100 and still give a poor head-pose estimate.

## 6. Limitations

- The axis-flip convention rests on three sessions of two people.
- Only rotation is compensated for head pose at the matrix level. Translation is removed by
  centring, and scale by normalisation.
- Velocity and acceleration are speed magnitudes, not signed quantities.
- Neutral calibration needs enough calm frames. With fewer than about 20, the reference shape
  is unreliable.
- Blink detection and the 0.5 threshold have not been checked against manual annotation.
- Head-angle sign conventions have not been checked against an external reference.
- The synthetic demo follows the same convention, so it validates the software path, not the
  convention itself.
- Saved outputs in `datasets/` were produced by the previous formula. Regenerate them with
  `analyze_faces.py <person> <session> --force`.
