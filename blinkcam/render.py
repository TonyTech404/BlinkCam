"""Rendering the closed eye: retrieve, warp, relight, composite.

The central idea is two-band relighting. Illumination is almost entirely
low-frequency and identity or anatomy is almost entirely high-frequency, so we
take the low band from a fit to the LIVE frame's surrounding skin and the high
band from the reference patch. That decouples them at essentially zero cost
with no solver and no linear system, and it is what makes uneven lighting and
exposure drift a non-problem rather than a tuning nightmare.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .bank import Descriptor, PatchBank, annulus_mask
from .geometry import (PATCH_H, PATCH_W, ApertureTracker, closure_line,
                       extract_patch, feathered_mask, patch_transform,
                       to_patch_coords)
from .landmarks import EyeGeometry, FaceFrame

_ANNULUS = annulus_mask()


@dataclass
class RenderConfig:
    rise: float = 0.15          # closure line height, fraction of aperture
    feather: float = 4.0        # composite boundary softness, pixels
    mask_expand: float = 1.0    # how far past the upper lid to repaint
    top_k: int = 3              # candidates blended per eye
    bulge_strength: float = 0.10
    bell_bias: float = 0.30     # globe rides up on closure in ~75% of people
    # Confidence thresholds, measured rather than guessed. Against a 476-patch
    # bank the retrieval distance for a live frame the bank covers well is 1.1
    # to 1.5, while genuinely out-of-gamut poses score 3.5 and above. The first
    # values here were 1.0 and 2.5, which faded good matches to 67 percent and
    # produced a half-lidded eye with the iris showing through, the exact
    # symptom of a partly-applied composite.
    good_distance: float = 1.8  # at or below this, full strength
    fade_distance: float = 3.2  # at or above this, effect off
    # Confidence is smoothed with a fast attack and a slow release. A single
    # bad frame (a blink, motion blur, a momentary landmark wobble) must not be
    # able to drop the effect, because the eyes then flick open for a few
    # frames, which is far more noticeable than a slightly wrong lid.
    confidence_attack: float = 0.5   # toward a better match, per frame
    confidence_release: float = 0.06  # toward a worse match, per frame
    low_band_sigma: float = PATCH_W / 6.0


def _design_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quadratic basis in normalised patch coordinates."""
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _patch_bounds(inv: np.ndarray, width: int, height: int,
                  margin: int = 2) -> tuple[int, int, int, int] | None:
    """Where a canonical patch lands in the source frame, as an integer box.

    Returns None when the patch falls entirely outside the frame, which happens
    when the face is leaving shot.
    """
    corners = np.array([[0.0, 0.0], [PATCH_W, 0.0],
                        [PATCH_W, PATCH_H], [0.0, PATCH_H]], np.float64)
    mapped = corners @ inv[:, :2].T + inv[:, 2]

    x0 = int(np.floor(mapped[:, 0].min())) - margin
    y0 = int(np.floor(mapped[:, 1].min())) - margin
    x1 = int(np.ceil(mapped[:, 0].max())) + margin
    y1 = int(np.ceil(mapped[:, 1].max())) + margin

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def fit_illumination(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit a per-channel quadratic to the masked region and evaluate it
    everywhere.

    This synthesises "what the shading would be here, under today's light, at
    this head pose" from skin that is visible in the live frame. Because it
    never consults the reference patch's illumination, a reference harvested
    under different lighting still composites correctly.
    """
    h, w = patch.shape[:2]
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 32:  # too few samples to fit; fall back to a flat mean
        return np.broadcast_to(patch.reshape(-1, 3).mean(axis=0),
                               (h, w, 3)).astype(np.float32).copy()

    nx = (xs / w - 0.5).astype(np.float32)
    ny = (ys / h - 0.5).astype(np.float32)
    basis = _design_matrix(nx, ny)

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    full = _design_matrix((gx / w - 0.5).ravel(), (gy / h - 0.5).ravel())

    out = np.empty((h * w, 3), np.float32)
    samples = patch[ys, xs].astype(np.float32)
    for c in range(3):
        coeffs, *_ = np.linalg.lstsq(basis, samples[:, c], rcond=None)
        out[:, c] = full @ coeffs
    return out.reshape(h, w, 3)


def split_bands(patch: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Low band (illumination) and high band (texture, lashes, crease)."""
    f = patch.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    return low, f - low


def two_band_relight(reference: np.ndarray, live: np.ndarray,
                     sigma: float) -> np.ndarray:
    """Put the reference eyelid under the live frame's lighting.

    Low band comes from a quadratic fitted to the live periorbital ring. High
    band comes from the reference, rescaled so its local contrast matches the
    live neighbourhood, which prevents a reference shot under harder light from
    reading as over-sharpened against softer live light.
    """
    _, ref_high = split_bands(reference, sigma)
    _, live_high = split_bands(live, sigma)

    target_low = fit_illumination(live, _ANNULUS)

    ref_energy = float(np.sqrt((ref_high ** 2 * _ANNULUS[..., None]).mean()) + 1e-6)
    live_energy = float(np.sqrt((live_high ** 2 * _ANNULUS[..., None]).mean()) + 1e-6)
    contrast = float(np.clip(live_energy / ref_energy, 0.6, 1.6))

    return target_low + ref_high * contrast


def globe_bulge(eye: EyeGeometry, config: RenderConfig, m: np.ndarray,
                centre: np.ndarray | None = None,
                radius: float | None = None) -> np.ndarray:
    """Subtle convex shading for the eyeball under the closed lid.

    The lid is a thin membrane draped over a sphere, and that sphere still
    moves. We can see the user's real iris in the source frame, so we drive the
    bulge from it: physically correct, personally synchronised micro-motion for
    no extra sensing. It is heavily damped, because a closed lid shows only a
    fraction of globe motion, and biased upward for Bell's phenomenon, which
    rotates the globe up and out on closure in about 75% of people.
    """
    if centre is None:
        centre = to_patch_coords(eye.iris_center.reshape(1, 2), m)[0]
    if radius is None:
        radius = max(eye.iris_radius * (PATCH_W / max(eye.width * 1.8, 1e-3)), 2.0)

    cx = float(centre[0])
    cy = float(centre[1]) - config.bell_bias * radius

    gy, gx = np.mgrid[0:PATCH_H, 0:PATCH_W].astype(np.float32)
    r2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / (radius * radius)
    dome = np.clip(1.0 - r2, 0.0, 1.0) ** 0.5
    # Light from above: the top of the dome catches light, the bottom falls off.
    gradient = np.clip((cy - gy) / max(radius, 1e-3), -1.0, 1.0)
    return (dome * gradient).astype(np.float32)


class EyeRenderer:
    """Renders both eyes closed for a frame."""

    def __init__(self, bank: PatchBank, config: RenderConfig | None = None) -> None:
        self.bank = bank
        self.config = config or RenderConfig()
        self._last_choice: dict[str, tuple[int, ...]] = {}
        self._confidence: dict[str, float] = {"right": 0.0, "left": 0.0}
        self._distance: dict[str, float] = {"right": 0.0, "left": 0.0}
        # Holds the open-aperture geometry in patch space so a blink or a fast
        # head turn cannot deform the region we repaint. See ApertureTracker.
        self.aperture = ApertureTracker()
        self._sticky: dict[str, set[int]] = {}

    def confidence(self, side: str) -> float:
        return self._confidence.get(side, 0.0)

    def distance(self, side: str) -> float:
        """Retrieval distance of the best match. Above RenderConfig's
        fade_distance the bank has no patch for this pose."""
        return self._distance.get(side, 0.0)

    def render(self, frame: np.ndarray, face: FaceFrame,
               opacity: float = 1.0,
               timestamp: float | None = None) -> np.ndarray:
        """Composite closed eyes onto a copy of the frame.

        timestamp drives the aperture filtering; it defaults to the wall clock
        so callers that do not track time still get stable geometry.
        """
        if timestamp is None:
            timestamp = time.monotonic()
        if opacity <= 0.0 or len(self.bank) == 0:
            return frame
        out = frame.copy()
        for eye in face.eyes():
            self._render_eye(out, frame, face, eye, opacity, timestamp)
        return out

    def _pose_gate(self, face: FaceFrame, eye: EyeGeometry) -> float:
        """Fade an eye out as it self-occludes.

        Gated on head pose rather than landmark confidence, because the mesh
        will confidently hallucinate landmarks for a fully occluded eye, which
        is worse than reporting nothing.
        """
        yaw = face.yaw
        # The far eye is the one being turned away from the camera.
        far = (eye.side == "right" and yaw > 0) or (eye.side == "left" and yaw < 0)
        a = abs(yaw)
        if a >= 70.0:
            return 0.0
        if not far:
            return 1.0
        if a <= 30.0:
            return 1.0
        if a >= 55.0:
            return 0.0
        return float(1.0 - (a - 30.0) / 25.0)

    def _render_eye(self, out: np.ndarray, source: np.ndarray, face: FaceFrame,
                    eye: EyeGeometry, opacity: float,
                    timestamp: float) -> None:
        cfg = self.config
        gate = self._pose_gate(face, eye)
        if gate <= 0.0:
            self._confidence[eye.side] = 0.0
            return

        m = patch_transform(eye)
        live = extract_patch(source, eye).astype(np.float32)

        candidates = self.bank.query(
            Descriptor.build(face, eye), live.astype(np.uint8), k=cfg.top_k,
            sticky=self._sticky.get(eye.side))
        self._sticky[eye.side] = set(self.bank.last_indices)
        if not candidates:
            self._confidence[eye.side] = 0.0
            return

        # Blend the top candidates rather than hard-selecting one, so ranks can
        # swap without a visible pop.
        reference = np.zeros_like(live)
        for patch, weight, _ in candidates:
            reference += patch.image.astype(np.float32) * weight

        best_distance = candidates[0][2]
        self._distance[eye.side] = best_distance
        span = max(cfg.fade_distance - cfg.good_distance, 1e-3)
        target = float(np.clip((cfg.fade_distance - best_distance) / span, 0.0, 1.0))

        # Asymmetric smoothing: rise quickly, fall slowly.
        previous = self._confidence.get(eye.side, 0.0)
        rate = (cfg.confidence_attack if target >= previous
                else cfg.confidence_release)
        quality = previous + (target - previous) * rate
        self._confidence[eye.side] = quality
        alpha = opacity * gate * quality
        if alpha <= 0.0:
            return

        rendered = two_band_relight(reference, live, cfg.low_band_sigma)

        iris_centre, iris_radius = self.aperture.iris(
            eye, m, face.blink, timestamp)
        bulge = globe_bulge(eye, cfg, m, iris_centre, iris_radius)
        rendered += (bulge * (cfg.bulge_strength * 255.0))[..., None]
        rendered = np.clip(rendered, 0, 255)

        # Held in patch space, so it neither collapses during a blink nor
        # inherits landmark noise amplified by head motion.
        polygon = self.aperture.polygon(
            eye, m, face.blink, timestamp, cfg.rise, cfg.mask_expand)
        mask = feathered_mask((PATCH_H, PATCH_W), polygon, cfg.feather)
        mask *= alpha

        blended = live * (1.0 - mask[..., None]) + rendered * mask[..., None]

        # Warp the finished patch back and composite using the same mask, so
        # only pixels we actually changed are touched.
        #
        # Both the warp and the alpha blend are restricted to the eye's
        # bounding box. Compositing at full frame size means several
        # full-resolution float conversions per eye, which dominated the frame
        # budget: it was over 3ms per eye at 1024x820, against about 0.15ms for
        # the warp itself.
        inv = cv2.invertAffineTransform(m)
        h, w = out.shape[:2]
        box = _patch_bounds(inv, w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0

        local = inv.copy()
        local[0, 2] -= x0
        local[1, 2] -= y0

        back = cv2.warpAffine(blended.astype(np.uint8), local, (bw, bh),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        back_mask = cv2.warpAffine(mask, local, (bw, bh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)[..., None]

        roi = out[y0:y1, x0:x1]
        np.copyto(roi, (roi * (1.0 - back_mask)
                        + back * back_mask).astype(np.uint8))


def draw_debug(frame: np.ndarray, face: FaceFrame,
               renderer: EyeRenderer | None = None) -> np.ndarray:
    """Overlay tracked geometry. Used by the preview window."""
    vis = frame.copy()
    for eye in face.eyes():
        cv2.polylines(vis, [np.round(eye.ring).astype(np.int32)], True,
                      (0, 255, 0), 1)
        cv2.polylines(vis, [np.round(closure_line(eye)).astype(np.int32)], False,
                      (0, 0, 255), 1)
        cv2.circle(vis, tuple(np.round(eye.iris_center).astype(int)),
                   max(1, int(eye.iris_radius)), (255, 255, 0), 1)
    text = (f"yaw {face.yaw:+5.1f}  pitch {face.pitch:+5.1f}  "
            f"roll {face.roll:+5.1f}  blink {face.blink:.2f}  ear {face.ear:.2f}")
    if renderer is not None:
        text += (f"  conf R{renderer.confidence('right'):.2f}"
                 f" L{renderer.confidence('left'):.2f}"
                 f"  dist R{renderer.distance('right'):.2f}"
                 f" L{renderer.distance('left'):.2f}")
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis
