"""Eye geometry: patch transforms, the lid closure line, and temporal filtering.

Everything here is pure numpy and OpenCV geometry with no I/O, so it is fully
unit-testable without a camera.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .landmarks import EyeGeometry, FaceFrame

# Canonical patch size. Deliberately taller than the eye aperture so the patch
# carries brow and upper-cheek context, which the lid crease and the brow's
# cast shadow both need.
PATCH_W, PATCH_H = 128, 80

# Patch covers this multiple of the eye's corner-to-corner width.
PATCH_SPAN = 1.8


# ---------------------------------------------------------------------------
# One Euro filter
# ---------------------------------------------------------------------------


class _LowPass:
    __slots__ = ("_y", "_has")

    def __init__(self) -> None:
        self._y: np.ndarray | float = 0.0
        self._has = False

    def __call__(self, x, alpha: float):
        if not self._has:
            self._y, self._has = x, True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last(self):
        return self._y if self._has else None

    def reset(self) -> None:
        self._has = False


class OneEuroFilter:
    """Adaptive low-pass (Casiez et al.).

    Smooths hard when the signal is slow, which kills jitter, and opens up when
    the signal is fast, which kills lag. Works on scalars or numpy arrays.

    Tuning procedure, from the authors: set beta to 0 and mincutoff to about
    1Hz, hold the tracked thing still and lower mincutoff until jitter is gone,
    then move it fast and raise beta to remove lag.
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.0, dcutoff: float = 1.0) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None

    def __call__(self, x, timestamp: float | None = None):
        """Filter one sample. Pass the real timestamp in seconds when you have
        it; frame intervals are not actually uniform and assuming 1/30 makes
        the filter mistune itself under load."""
        x = np.asarray(x, dtype=np.float64) if isinstance(x, np.ndarray) else float(x)

        dt = 1.0 / self.freq
        if timestamp is not None:
            if self._last_t is not None:
                measured = timestamp - self._last_t
                # Guard against clock stalls and duplicate timestamps.
                if 1e-4 < measured < 1.0:
                    dt = measured
            self._last_t = timestamp

        prev = self._x.last
        if prev is None:
            dx = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        else:
            dx = (x - prev) / dt

        edx = self._dx(dx, self._alpha(self.dcutoff, dt))
        magnitude = np.abs(edx)
        cutoff = self.mincutoff + self.beta * magnitude

        if isinstance(cutoff, np.ndarray):
            tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
            alpha = 1.0 / (1.0 + tau / dt)
        else:
            alpha = self._alpha(max(cutoff, 1e-6), dt)

        return self._x(x, alpha)


# Starting parameters. Head pose is smoothed heavily because heads have
# inertia. Lids need medium. Iris needs light, because an over-smoothed iris
# reads as dead eyes, which is exactly the artifact that would sink this.
FILTER_PRESETS = {
    "pose": dict(mincutoff=0.8, beta=0.008),
    "lid": dict(mincutoff=1.5, beta=0.05),
    "iris": dict(mincutoff=2.0, beta=0.10),
    # For geometry held in PATCH space, which patch_transform has already
    # normalised for head pose, scale and roll. The aperture is nearly static
    # there, so beta is zero: there is no fast motion to keep up with, and the
    # adaptive term would only re-admit the noise we are trying to remove.
    "aperture": dict(mincutoff=0.7, beta=0.0),
}


# ---------------------------------------------------------------------------
# Patch transforms
# ---------------------------------------------------------------------------


def eye_axis(eye: EyeGeometry) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit vectors along and perpendicular to the eye, plus the roll angle.

    The along-axis points from the lateral (outer) corner to the medial
    (inner) corner. The perpendicular points "up" in image terms, meaning
    toward the brow, which is negative y.
    """
    v = eye.medial_corner - eye.lateral_corner
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0]), np.array([0.0, -1.0]), 0.0
    along = v / n
    # Rotate -90 degrees to get the brow-ward normal for either eye.
    up = np.array([along[1], -along[0]])
    if eye.side == "left":
        # For the subject's left eye the along-axis points the other way in
        # image space, so flip the normal to keep "up" brow-ward.
        up = -up
    return along, up, math.degrees(math.atan2(v[1], v[0]))


def patch_transform(eye: EyeGeometry,
                    size: tuple[int, int] = (PATCH_W, PATCH_H)) -> np.ndarray:
    """2x3 affine mapping the source frame into a canonical upright eye patch.

    Use with cv2.warpAffine to extract, and cv2.invertAffineTransform to put a
    rendered patch back. Roll is compensated here rather than being indexed in
    the patch bank, which saves a whole descriptor dimension.

    Both eyes are canonicalised to the SAME anatomical layout: the lateral
    (outer) corner is always on the left of the patch and the medial corner on
    the right. Without this the two eyes come out as mirror images of each
    other, which would break both the left-right patch mirroring that doubles
    pose coverage and any sharing of patches between eyes.
    """
    pw, ph = size
    _, _, angle = eye_axis(eye)

    span = max(eye.width * PATCH_SPAN, 1e-3)
    scale = pw / span

    # Rotate about the eye centre so the eye axis becomes horizontal, scale so
    # the span fills the patch width, then translate the centre to the middle.
    turn = angle + 180.0 if eye.side == "left" else angle
    m = cv2.getRotationMatrix2D(
        (float(eye.center[0]), float(eye.center[1])), turn, scale)
    m[0, 2] += pw / 2.0 - eye.center[0]
    m[1, 2] += ph / 2.0 - eye.center[1]

    if eye.side == "left":
        # Half a turn puts the eye upright but leaves lateral on the right.
        # Flip horizontally in patch space to match the right eye's layout.
        flip = np.array([[-1.0, 0.0, pw], [0.0, 1.0, 0.0]])
        m = _compose(flip, m)
    return m


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines: apply inner, then outer."""
    a = np.vstack([inner, [0.0, 0.0, 1.0]])
    b = np.vstack([outer, [0.0, 0.0, 1.0]])
    return (b @ a)[:2]


def extract_patch(image: np.ndarray, eye: EyeGeometry,
                  size: tuple[int, int] = (PATCH_W, PATCH_H)) -> np.ndarray:
    """Pull the canonical, upright, scale-normalised patch for one eye."""
    m = patch_transform(eye, size)
    return cv2.warpAffine(image, m, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def to_patch_coords(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Map source-image points into patch space with a 2x3 affine."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# The lid closure line
# ---------------------------------------------------------------------------


def closure_line(eye: EyeGeometry, rise: float = 0.15) -> np.ndarray:
    """Where the lids meet when the eye closes.

    Anatomy drives three choices here. The upper lid does the great majority of
    the travel while the lower lid rises only slightly, so the closure line
    sits near the lower lid rather than midway. It takes the LOWER lid's gentle
    downward-convex curvature, not a straight line and not the upper lid's
    curve. And it is pinned at both canthi, because the corners barely move
    during closure; only the middle rises.

    rise is the fraction of the open palpebral height that the mid-lid climbs.
    """
    lower = eye.lower_lid.astype(np.float64)
    _, up, _ = eye_axis(eye)
    height = max(eye.height, 1e-3)

    n = len(lower)
    # Weight is zero at both corners and peaks mid-lid, so the canthi stay put.
    s = np.linspace(0.0, 1.0, n)
    weight = np.sin(math.pi * s)

    return (lower + up[None, :] * (rise * height * weight)[:, None]).astype(np.float32)


def lid_mask_polygon(eye: EyeGeometry, rise: float = 0.15,
                     expand: float = 1.0) -> np.ndarray:
    """Polygon covering the region we repaint.

    It must span the ENTIRE open aperture, upper lid to lower lid, plus a
    little beyond each. Bounding the bottom at the closure line instead leaves
    an unmasked strip of real sclera below it, which shows through the
    composite as a dark slit and reads as a half-open eye. The closure line is
    where the lash line gets drawn, not where the repainted region ends.

    The medial corner is deliberately left near the boundary so the caruncle
    and tear duct keep their real pixels. Erasing them looks instantly wrong.
    """
    upper = eye.upper_lid.astype(np.float64)
    lower = eye.lower_lid.astype(np.float64)
    _, up, _ = eye_axis(eye)
    height = max(eye.height, 1e-3)

    # Push each boundary a little past the lid: above to cover the crease,
    # below to clear the lower lash line and leave no residual slit.
    upper_out = upper + up[None, :] * (0.25 * height * expand)
    lower_out = lower - up[None, :] * (0.12 * height * expand)

    # Both contours run lateral to medial, so reverse one to close the loop.
    poly = np.vstack([upper_out, lower_out[::-1]])
    return poly.astype(np.float32)


def feathered_mask(shape: tuple[int, int], polygon: np.ndarray,
                   feather: float = 5.0) -> np.ndarray:
    """Soft-edged float mask in [0, 1] for a polygon.

    The boundary follows the periorbital contour rather than a rectangle;
    rectangular masks are a documented source of visible seams.
    """
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    if feather > 0:
        k = int(max(3, round(feather * 2) | 1))
        blurred = cv2.GaussianBlur(mask, (k, k), feather / 2.0)
        return (blurred.astype(np.float32) / 255.0)
    return mask.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Face-local filtering
# ---------------------------------------------------------------------------


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation)."""
    m, _ = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2).astype(np.float32),
        dst.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.LMEDS)
    return m if m is not None else np.array([[1.0, 0, 0], [0, 1.0, 0]])


class StabilizedFace:
    """Filters landmarks in a face-local frame rather than in image space.

    This matters more than filter parameter tuning. A rigid transform is
    estimated from stable anchors and filtered heavily, then the residual
    landmark offsets are filtered lightly in the face-local frame and
    recomposed. The effect is that fast head turns are not lagged by lid
    smoothing, and lid jitter is not amplified by head motion.

    The blink signal is deliberately NOT filtered. A blink closes in about
    100ms, roughly three frames at 30fps, and any 1-2Hz low-pass smears that
    into a slow droop.
    """

    ANCHORS = [33, 263, 1, 168]  # eye corners, nose tip, nasion

    def __init__(self, freq: float = 30.0) -> None:
        self.freq = freq
        self._rigid = OneEuroFilter(freq, **FILTER_PRESETS["pose"])
        self._residual = OneEuroFilter(freq, **FILTER_PRESETS["lid"])
        self._iris = OneEuroFilter(freq, **FILTER_PRESETS["iris"])
        self._pose = OneEuroFilter(freq, **FILTER_PRESETS["pose"])
        self._reference: np.ndarray | None = None

    def reset(self) -> None:
        for f in (self._rigid, self._residual, self._iris, self._pose):
            f.reset()
        self._reference = None

    def __call__(self, face: FaceFrame, timestamp: float) -> FaceFrame:
        pts = face.landmarks
        if pts is None:
            return face

        if self._reference is None:
            self._reference = pts[self.ANCHORS].copy()

        # Rigid part: where the face is. Heavy smoothing.
        rigid = similarity_transform(self._reference, pts[self.ANCHORS])

        # The reference is whatever the first frame happened to be, and the
        # residuals are only ever expressed in that frame, so ordinary drift is
        # harmless: measured over a live session the similarity scale stayed
        # within 0.89 to 1.02. Re-anchor only if it becomes degenerate, which
        # would make the inverse transform below meaningless. A one-frame jump
        # is much better than warping the composite with a bad transform.
        scale = float(np.hypot(rigid[0, 0], rigid[1, 0]))
        if not np.isfinite(scale) or not 0.2 < scale < 5.0:
            self.reset()
            self._reference = pts[self.ANCHORS].copy()
            rigid = similarity_transform(self._reference, pts[self.ANCHORS])
        rigid_s = self._rigid(rigid.reshape(-1), timestamp).reshape(2, 3)

        # Residual part: how the face is deformed, in the face-local frame.
        inv = cv2.invertAffineTransform(rigid_s)
        local = to_patch_coords(pts, inv)
        local_s = self._residual(local.reshape(-1), timestamp).reshape(-1, 2)

        # Iris gets its own lighter filter so gaze stays snappy.
        iris_idx = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
        local_s[iris_idx] = self._iris(
            local[iris_idx].reshape(-1), timestamp).reshape(-1, 2)

        smoothed = to_patch_coords(local_s.astype(np.float32), rigid_s)

        pose = self._pose(np.array([face.yaw, face.pitch, face.roll]), timestamp)

        from .landmarks import Landmarker
        return FaceFrame(
            right=Landmarker._eye(smoothed, "right", face.blendshapes),
            left=Landmarker._eye(smoothed, "left", face.blendshapes),
            yaw=float(pose[0]), pitch=float(pose[1]), roll=float(pose[2]),
            blendshapes=face.blendshapes,
            landmarks=smoothed,
        )

class ApertureTracker:
    """Per-eye open-aperture geometry, held in patch space across blinks.

    One cause, two measured symptoms. The mask polygon was derived from the
    live lid contours every frame, so on this webcam at 20fps it moved:

        still                0.067 px/frame
        very fast head motion 0.800 px/frame   (12x worse)
        during a blink        1.883 px/frame   (11x worse)

    Blinks are the worst case because the real aperture collapses while the
    effect still needs to repaint the whole open region. Motion is bad because
    landmark noise is amplified by head movement.

    Patch space is already normalised for head pose, scale and roll, so
    geometry held here follows the head without inheriting its noise. Two gates
    do the work: updates are smoothed, so per-frame jitter cannot deform the
    repainted region, and they are suspended while the eye is closing, so a
    blink cannot either. A squint or a smile reshapes the aperture over several
    hundred milliseconds and passes the filter; a 100ms blink does not.

    This replaces freezing the whole FaceFrame during a blink, which also froze
    head pose and made the composite stick and then snap if the head moved.
    """

    # Below this the lids are open enough to trust as a sample of the resting
    # aperture. Deliberately well under BLINK_ENTER (0.55) so the approach to a
    # closure is excluded too, not merely the fully closed phase: most of the
    # jitter happens while the lid is on its way down.
    OPEN_BELOW = 0.25

    def __init__(self, freq: float = 30.0) -> None:
        self.freq = freq
        self._filters: dict[str, OneEuroFilter] = {}
        self._held: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._filters.clear()
        self._held.clear()

    def _track(self, key: str, live: np.ndarray, blink: float,
               timestamp: float) -> np.ndarray:
        held = self._held.get(key)
        if held is not None and held.shape != live.shape:
            held = None  # landmark count changed; start over

        filt = self._filters.get(key)
        if filt is None or held is None:
            filt = OneEuroFilter(self.freq, **FILTER_PRESETS["aperture"])
            self._filters[key] = filt

        # While the eye is closing, feed the filter its own held value rather
        # than skipping the update. That keeps its internal clock current, so
        # the blink does not look like a long pause followed by a jump.
        sample = held if (held is not None and blink >= self.OPEN_BELOW) else live
        out = filt(np.asarray(sample, np.float32).reshape(-1),
                   timestamp).reshape(live.shape).astype(np.float32)
        self._held[key] = out
        return out

    def polygon(self, eye: EyeGeometry, m: np.ndarray, blink: float,
                timestamp: float, rise: float, expand: float) -> np.ndarray:
        """Mask polygon in patch coordinates, stable through blinks."""
        live = to_patch_coords(lid_mask_polygon(eye, rise, expand), m)
        return self._track(f"poly:{eye.side}", live, blink, timestamp)

    def iris(self, eye: EyeGeometry, m: np.ndarray, blink: float,
             timestamp: float) -> tuple[np.ndarray, float]:
        """Iris centre in patch coordinates plus its radius.

        Held for the same reason: the iris landmarks are meaningless once the
        lids cover them, and the globe bulge is driven from the iris position.
        """
        live = to_patch_coords(eye.iris_center.reshape(1, 2), m)
        centre = self._track(f"iris:{eye.side}", live, blink, timestamp)[0]
        radius = eye.iris_radius * (PATCH_W / max(eye.width * 1.8, 1e-3))
        return centre, max(float(radius), 2.0)
