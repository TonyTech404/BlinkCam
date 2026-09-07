"""Geometry and filtering tests. Pure math, no camera needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blinkcam import geometry as G
from blinkcam.landmarks import EyeGeometry


def make_eye(side: str = "right", cx: float = 100.0, cy: float = 100.0,
             width: float = 40.0, height: float = 12.0,
             roll_deg: float = 0.0) -> EyeGeometry:
    """A synthetic almond-shaped eye for geometry tests."""
    t = np.linspace(0.0, 1.0, 9)
    half = width / 2.0
    x = -half + t * width
    lower = np.stack([x, (height / 2.0) * np.sin(math.pi * t)], axis=1)
    tu = np.linspace(0.0, 1.0, 7)
    xu = -half + tu * width
    upper = np.stack([xu, -(height / 2.0) * np.sin(math.pi * tu)], axis=1)

    if side == "left":  # subject's left eye axis points image-leftward
        lower[:, 0] *= -1.0
        upper[:, 0] *= -1.0

    a = math.radians(roll_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    lower = lower @ rot.T + [cx, cy]
    upper = upper @ rot.T + [cx, cy]

    # The brow is placed relative to the eye CENTRE and scaled with the eye,
    # because a real brow does not move when the lids do. Anchoring it to the
    # upper lid instead makes the fixture unable to express the very property
    # the descriptor depends on: that brow height is independent of lid state.
    brow_x = np.linspace(-half, half, 7)
    brow = np.stack([brow_x, np.full(7, -0.45 * width)], axis=1)
    if side == "left":
        brow[:, 0] *= -1.0
    brow = brow @ rot.T + [cx, cy]

    return EyeGeometry(
        side=side, ring=np.vstack([lower, upper[::-1]]).astype(np.float32),
        upper_lid=upper.astype(np.float32), lower_lid=lower.astype(np.float32),
        lateral_corner=lower[0].astype(np.float32),
        medial_corner=lower[-1].astype(np.float32),
        iris_center=np.array([cx, cy], np.float32), iris_radius=6.0,
        brow=brow.astype(np.float32),
        ear=height / width, blink=0.0,
    )


# ---- One Euro filter -------------------------------------------------------


def test_one_euro_passes_through_first_sample():
    f = G.OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0)
    assert f(5.0, 0.0) == pytest.approx(5.0)


def test_one_euro_reduces_noise_on_a_constant_signal():
    """At mincutoff=1.5Hz and 30Hz sampling the equivalent EMA smoothing factor
    is about 0.24, so white noise should shrink by roughly 2.7x. beta's
    adaptive term correctly relaxes that a little, since white noise looks like
    fast motion. Anything above 2x is the filter working."""
    rng = np.random.default_rng(0)
    f = G.OneEuroFilter(freq=30.0, **G.FILTER_PRESETS["lid"])
    noisy = 10.0 + rng.normal(0, 1.0, 300)
    out = [f(v, i / 30.0) for i, v in enumerate(noisy)]
    tail = np.array(out[100:])
    assert tail.std() < noisy[100:].std() / 2.0
    assert abs(tail.mean() - 10.0) < 0.3


def test_one_euro_smooths_harder_at_a_lower_cutoff():
    """The pose preset must be visibly calmer than the iris preset, because an
    over-smoothed iris reads as dead eyes."""
    rng = np.random.default_rng(1)
    noisy = 10.0 + rng.normal(0, 1.0, 300)
    stds = {}
    for name in ("pose", "iris"):
        f = G.OneEuroFilter(freq=30.0, **G.FILTER_PRESETS[name])
        stds[name] = np.array([f(v, i / 30.0)
                               for i, v in enumerate(noisy)][100:]).std()
    assert stds["pose"] < stds["iris"]


def test_one_euro_tracks_fast_motion_without_excessive_lag():
    """beta is what buys this. With beta=0 the ramp would lag much further."""
    f = G.OneEuroFilter(freq=30.0, **G.FILTER_PRESETS["lid"])
    ramp = np.linspace(0.0, 100.0, 60)
    out = [f(v, i / 30.0) for i, v in enumerate(ramp)]
    assert abs(out[-1] - ramp[-1]) < 12.0


def test_one_euro_handles_arrays_elementwise():
    f = G.OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.01)
    for i in range(20):
        out = f(np.array([1.0, 2.0, 3.0]), i / 30.0)
    assert out.shape == (3,)
    assert out == pytest.approx(np.array([1.0, 2.0, 3.0]), abs=1e-6)


def test_one_euro_ignores_absurd_timestamp_gaps():
    """A clock stall must not blow the filter up."""
    f = G.OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.01)
    f(0.0, 0.0)
    out = f(1.0, 10_000.0)
    assert np.isfinite(out)


# ---- patch transform -------------------------------------------------------


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("roll", [0.0, 15.0, -20.0])
def test_patch_transform_centres_and_uprights_the_eye(side, roll):
    eye = make_eye(side, roll_deg=roll)
    m = G.patch_transform(eye)
    corners = G.to_patch_coords(
        np.stack([eye.lateral_corner, eye.medial_corner]), m)

    # Both corners land on the patch's horizontal midline: roll compensated.
    assert corners[0][1] == pytest.approx(G.PATCH_H / 2, abs=0.5)
    assert corners[1][1] == pytest.approx(G.PATCH_H / 2, abs=0.5)
    # Symmetric about the centre, and spanning 1/PATCH_SPAN of the width.
    assert corners[0][0] == pytest.approx(
        G.PATCH_W / 2 - G.PATCH_W / (2 * G.PATCH_SPAN), abs=0.5)


@pytest.mark.parametrize("side", ["right", "left"])
def test_both_eyes_canonicalise_to_the_same_layout(side):
    """Lateral corner always patch-left. Without this the two eyes come out as
    mirror images and left-right patch mirroring breaks."""
    eye = make_eye(side)
    m = G.patch_transform(eye)
    corners = G.to_patch_coords(
        np.stack([eye.lateral_corner, eye.medial_corner]), m)
    assert corners[0][0] < corners[1][0]


def test_patch_transform_is_invertible():
    eye = make_eye("right", roll_deg=12.0)
    m = G.patch_transform(eye)
    import cv2
    inv = cv2.invertAffineTransform(m)
    pts = np.array([[100.0, 100.0], [120.0, 90.0]], np.float32)
    assert G.to_patch_coords(G.to_patch_coords(pts, m), inv) == pytest.approx(
        pts, abs=1e-3)


def test_patch_scale_is_normalised_across_distances():
    """A face near and far must produce the same patch geometry."""
    near = G.patch_transform(make_eye("right", width=80.0))
    far = G.patch_transform(make_eye("right", width=20.0))
    # Different transforms, but corners land identically. Checked via corners.
    n = G.to_patch_coords(np.stack([make_eye("right", width=80.0).lateral_corner,
                                    make_eye("right", width=80.0).medial_corner]), near)
    f = G.to_patch_coords(np.stack([make_eye("right", width=20.0).lateral_corner,
                                    make_eye("right", width=20.0).medial_corner]), far)
    assert n == pytest.approx(f, abs=0.5)


# ---- closure line ----------------------------------------------------------


@pytest.mark.parametrize("side", ["right", "left"])
def test_closure_line_is_pinned_at_both_canthi(side):
    """The corners barely move during closure; only the middle rises."""
    eye = make_eye(side)
    cl = G.closure_line(eye)
    assert np.linalg.norm(cl[0] - eye.lower_lid[0]) < 1e-4
    assert np.linalg.norm(cl[-1] - eye.lower_lid[-1]) < 1e-4


@pytest.mark.parametrize("side", ["right", "left"])
def test_closure_line_rises_toward_the_upper_lid(side):
    """It must move from the lower lid toward the aperture, not away."""
    eye = make_eye(side)
    cl = G.closure_line(eye, rise=0.15)
    mid = len(cl) // 2
    _, up, _ = G.eye_axis(eye)
    travelled = float(np.dot(cl[mid] - eye.lower_lid[mid], up))
    assert travelled > 0
    assert travelled == pytest.approx(0.15 * eye.height, rel=0.3)


def test_closure_line_stays_below_the_upper_lid():
    """Overshooting past the upper lid would invert the lid."""
    eye = make_eye("right")
    cl = G.closure_line(eye, rise=0.15)
    assert cl[:, 1].min() > eye.upper_lid[:, 1].min()


def test_closure_line_rise_scales_with_aperture():
    small = G.closure_line(make_eye("right", height=6.0), rise=0.15)
    large = G.closure_line(make_eye("right", height=18.0), rise=0.15)
    mid = len(small) // 2
    assert abs(large[mid][1] - 100.0) < abs(small[mid][1] - 100.0) + 10


# ---- masks -----------------------------------------------------------------


def test_feathered_mask_is_bounded_and_soft():
    eye = make_eye("right")
    m = G.patch_transform(eye)
    poly = G.to_patch_coords(G.lid_mask_polygon(eye), m)
    mask = G.feathered_mask((G.PATCH_H, G.PATCH_W), poly, feather=4.0)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert 0.0 < mask.mean() < 0.5
    # A soft edge means intermediate values exist.
    assert ((mask > 0.05) & (mask < 0.95)).sum() > 20


def test_mask_polygon_covers_the_aperture():
    import cv2
    eye = make_eye("right")
    poly = G.lid_mask_polygon(eye)
    assert cv2.pointPolygonTest(poly.astype(np.float32),
                                (float(eye.iris_center[0]),
                                 float(eye.iris_center[1])), False) >= 0


def test_mask_polygon_spares_the_medial_canthus():
    """The caruncle and tear duct must keep their real pixels."""
    import cv2
    eye = make_eye("right")
    poly = G.lid_mask_polygon(eye)
    beyond = eye.medial_corner + (eye.medial_corner - eye.lateral_corner) * 0.06
    assert cv2.pointPolygonTest(poly.astype(np.float32),
                                (float(beyond[0]), float(beyond[1])), False) < 0


def test_eye_axis_up_vector_points_browward_for_both_eyes():
    for side in ("right", "left"):
        eye = make_eye(side)
        _, up, _ = G.eye_axis(eye)
        # Brow is above the lid, and image y grows downward.
        to_brow = eye.brow.mean(axis=0) - eye.center
        assert float(np.dot(up, to_brow)) > 0


def test_mask_polygon_covers_the_entire_open_aperture():
    """Bounding the mask at the closure line leaves an unmasked strip of real
    sclera below it, which shows through the composite as a dark slit and reads
    as a half-open eye. The mask must reach past both lids."""
    import cv2

    for side in ("right", "left"):
        eye = make_eye(side)
        poly = G.lid_mask_polygon(eye).astype(np.float32)
        # Every point on both lid contours must fall inside the mask.
        for contour, name in ((eye.upper_lid, "upper"), (eye.lower_lid, "lower")):
            for point in contour:
                inside = cv2.pointPolygonTest(
                    poly, (float(point[0]), float(point[1])), False)
                assert inside >= 0, f"{side} {name} lid point outside the mask"


def test_mask_polygon_extends_past_the_lower_lid():
    eye = make_eye("right")
    poly = G.lid_mask_polygon(eye)
    # The mask's lowest extent must sit below the lower lid, not above it.
    assert poly[:, 1].max() > eye.lower_lid[:, 1].max()


def test_mask_polygon_grows_with_expand():
    small = G.lid_mask_polygon(make_eye("right"), expand=0.5)
    large = G.lid_mask_polygon(make_eye("right"), expand=2.0)
    height = lambda p: p[:, 1].max() - p[:, 1].min()
    assert height(large) > height(small)
