"""Procedurally synthesised closed eyelid: the fallback renderer.

This is Technique B. It is deliberately NOT the primary method, because no
classical inpainting algorithm will produce eyelashes: lashes are not a
continuation of the surrounding structure, they are a novel high-frequency
feature that has to be drawn rather than diffused. The result is plausible but
slightly dead compared with the user's real eyelid.

Its job is to cover poses and expressions the harvested patch bank does not
reach, so an out-of-gamut pose degrades to "slightly synthetic" instead of
"visibly broken". It also lets the full render path be exercised before any
calibration recording exists.

What makes a closed lid read as real, in descending order of perceptual
weight: the lash line, the crease, the globe bulge shaded correctly, lid skin
being slightly pinker and more translucent than surrounding skin, and a
surviving tear duct.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .bank import annulus_mask
from .geometry import (PATCH_H, PATCH_W, closure_line,
                       patch_transform, to_patch_coords)
from .landmarks import EyeGeometry

_ANNULUS = annulus_mask()


@dataclass
class ProceduralConfig:
    lash_count: int = 110
    lash_length: float = 4.5        # patch pixels, before jitter
    lash_darkness: float = 0.62     # 0 none, 1 black
    crease_offset: float = 9.0      # patch pixels above the closure line
    crease_darkness: float = 0.20
    crease_sigma: float = 2.2
    # Eyelid skin is the thinnest on the body: pinker, more translucent than
    # the periorbital skin around it. Filling with plain mean skin looks waxy.
    lid_warmth: float = 1.02        # red multiplier
    lid_coolness: float = 0.99      # blue multiplier
    texture_sigma: float = 3.0      # band split for borrowed lid texture
    texture_strength: float = 0.9
    fill_feather: float = 1.6       # aperture edge softness, patch pixels
    bulge_strength: float = 0.13
    bell_bias: float = 0.30
    seed: int = 0


def aperture_mask(eye: EyeGeometry, m: np.ndarray) -> np.ndarray:
    """The whole visible aperture of the open eye: everything to be erased.

    This spans upper lid to lower lid, not upper lid to closure line. Stopping
    at the closure line leaves a sliver of sclera showing beneath it, which
    reads as a partly-open eye.
    """
    upper = to_patch_coords(eye.upper_lid, m)
    lower = to_patch_coords(eye.lower_lid, m)
    poly = np.vstack([upper, lower[::-1]])
    mask = np.zeros((PATCH_H, PATCH_W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask.astype(np.float32) / 255.0


def _column_profile(contour: np.ndarray, width: int) -> np.ndarray:
    """Contour y per column, interpolated across the patch width."""
    order = np.argsort(contour[:, 0])
    return np.interp(np.arange(width, dtype=np.float32),
                     contour[order, 0], contour[order, 1]).astype(np.float32)


def _reflected_lid_texture(patch: np.ndarray, upper: np.ndarray,
                           lower: np.ndarray, closure: np.ndarray) -> np.ndarray:
    """Fold real lid skin into the aperture from both sides.

    This is anatomically the right texture source. The skin covering the
    aperture when the eye shuts IS the lid skin folded away while it was open,
    so above the closure line we mirror the upper-lid skin downward, and below
    it we mirror the lower-lid skin upward. Both bring in real skin of the
    correct colour, texture and pore structure.

    A purely low-frequency fill reads as a flat plastic blob against textured
    surrounding skin, which is the most obvious artifact of naive inpainting.
    Unlike PatchMatch this is deterministic, so it cannot boil frame to frame.
    """
    h, w = patch.shape[:2]
    upper_y = _column_profile(upper, w)
    lower_y = _column_profile(lower, w)
    closure_y = _column_profile(closure, w)

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    from_above = 2.0 * upper_y[None, :] - gy
    from_below = 2.0 * lower_y[None, :] - gy

    above_closure = gy <= closure_y[None, :]
    map_y = np.where(above_closure, from_above, from_below)
    # Outside the aperture entirely, sample in place.
    outside = (gy < upper_y[None, :]) | (gy > lower_y[None, :])
    map_y = np.where(outside, gy, map_y)

    return cv2.remap(patch, gx, np.clip(map_y, 0, h - 1).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _fill_skin(patch: np.ndarray, aperture: np.ndarray, upper: np.ndarray,
               lower: np.ndarray, closure: np.ndarray,
               config: ProceduralConfig) -> np.ndarray:
    """Replace the aperture with skin that matches the local lighting.

    Low frequency comes from a quadratic illumination fit over the surrounding
    periorbital ring, which is correctly lit and far cheaper than Telea or
    Navier-Stokes inpainting, and unlike them cannot drag iris colour inward.
    High frequency comes from the folded-down upper-lid skin, so the fill has
    real pore texture instead of reading as a smooth patch.
    """
    from .render import fit_illumination, split_bands

    low = fit_illumination(patch, _ANNULUS)
    tint = np.array([config.lid_coolness, 1.0, config.lid_warmth], np.float32)
    low = low * tint  # BGR order, so index 2 is red

    _, texture = split_bands(
        _reflected_lid_texture(patch, upper, lower, closure),
        config.texture_sigma)
    skin = low + texture * config.texture_strength

    # Feather the aperture edge. A hard boundary reads as a cutout, and the
    # arc along the upper lid is exactly where it shows most.
    soft = cv2.GaussianBlur(aperture, (0, 0), config.fill_feather)
    a = np.clip(soft, 0.0, 1.0)[..., None]
    return patch * (1.0 - a) + skin * a


def _draw_lashes(canvas: np.ndarray, line: np.ndarray, up: np.ndarray,
                 config: ProceduralConfig) -> np.ndarray:
    """Stochastic lash band along the closure line.

    A clean anti-aliased dark curve reads as eyeliner, not lashes. Real closed
    lashes are a dark, slightly irregular, soft-edged band that projects
    downward and forward, so each lash gets its own length, angle and opacity.
    """
    rng = np.random.default_rng(config.seed)
    layer = np.zeros(canvas.shape[:2], np.float32)

    # Resample the closure line to get evenly spaced lash roots.
    t = np.linspace(0.0, 1.0, len(line))
    ts = np.linspace(0.0, 1.0, config.lash_count)
    roots = np.stack([np.interp(ts, t, line[:, 0]),
                      np.interp(ts, t, line[:, 1])], axis=1)

    # Lashes are sparse and short at the medial end, longer laterally.
    taper = 0.45 + 0.55 * np.sin(np.pi * np.clip(ts * 1.05, 0, 1)) ** 0.6
    down = -up  # lashes project away from the brow

    for i, root in enumerate(roots):
        if rng.random() < 0.12:
            continue  # gaps: real lash lines are not uniform
        length = config.lash_length * taper[i] * rng.uniform(0.55, 1.45)
        # Splay: lashes fan outward rather than all pointing the same way.
        angle = rng.normal(0.0, 0.30)
        direction = np.array([
            down[0] * np.cos(angle) - down[1] * np.sin(angle),
            down[0] * np.sin(angle) + down[1] * np.cos(angle),
        ])
        tip = root + direction * length
        opacity = rng.uniform(0.45, 1.0)
        cv2.line(layer, tuple(np.round(root).astype(int)),
                 tuple(np.round(tip).astype(int)), float(opacity), 1,
                 cv2.LINE_AA)

    # Soften slightly: individual lashes are thinner than a pixel at this scale.
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    layer = np.clip(layer, 0.0, 1.0) * config.lash_darkness
    return canvas * (1.0 - layer[..., None])


def _draw_crease(canvas: np.ndarray, line: np.ndarray, up: np.ndarray,
                 config: ProceduralConfig) -> np.ndarray:
    """Soft shadow for the superior palpebral sulcus.

    Closure deepens and lowers the crease. Its depth and height above the lash
    line are strongly identity-specific, and monolid users have no visible
    crease at all, so this is kept subtle: drawing a pronounced crease on
    someone who has none destroys their identity. The patch bank does not have
    this problem, which is one more reason it is the primary path.
    """
    layer = np.zeros(canvas.shape[:2], np.float32)
    shifted = line + up[None, :] * config.crease_offset
    pts = np.round(shifted).astype(np.int32)
    cv2.polylines(layer, [pts], False, 1.0, 2, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), config.crease_sigma)
    peak = float(layer.max())
    if peak > 1e-6:
        layer /= peak
    layer *= config.crease_darkness
    return canvas * (1.0 - layer[..., None])


def _apply_bulge(canvas: np.ndarray, eye: EyeGeometry, m: np.ndarray,
                 config: ProceduralConfig) -> np.ndarray:
    """Shade the eyeball beneath the lid, driven by the real iris position."""
    centre = to_patch_coords(eye.iris_center.reshape(1, 2), m)[0]
    radius = max(eye.iris_radius * (PATCH_W / max(eye.width * 1.8, 1e-3)), 2.0)
    cx = float(centre[0])
    cy = float(centre[1]) - config.bell_bias * radius

    gy, gx = np.mgrid[0:PATCH_H, 0:PATCH_W].astype(np.float32)
    r2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / (radius * radius)
    dome = np.clip(1.0 - r2, 0.0, 1.0) ** 0.5
    gradient = np.clip((cy - gy) / max(radius, 1e-3), -1.0, 1.0)
    shade = (dome * gradient * config.bulge_strength * 255.0)[..., None]
    return canvas + shade


def synthesize_closed_patch(live_patch: np.ndarray, eye: EyeGeometry,
                            rise: float = 0.15,
                            config: ProceduralConfig | None = None
                            ) -> np.ndarray:
    """Build a closed-eyelid patch from an open-eye patch.

    Order matters: fill the aperture, shade the globe under the lid, then lay
    the crease and the lash line on top, because those two are the highest
    contrast features and must not be washed out by later steps.
    """
    cfg = config or ProceduralConfig()
    m = patch_transform(eye)
    patch = live_patch.astype(np.float32)

    line = to_patch_coords(closure_line(eye, rise), m)
    upper = to_patch_coords(eye.upper_lid, m)
    lower = to_patch_coords(eye.lower_lid, m)
    # The brow-ward direction in patch space is simply up, since the patch
    # transform has already removed roll.
    up = np.array([0.0, -1.0], np.float32)

    out = _fill_skin(patch, aperture_mask(eye, m), upper, lower, line, cfg)
    out = _apply_bulge(out, eye, m, cfg)
    out = _draw_crease(out, line, up, cfg)
    out = _draw_lashes(out, line, up, cfg)
    return np.clip(out, 0, 255)
