"""Patch bank and rendering tests, especially two-band relighting."""

from __future__ import annotations

import numpy as np
import pytest

from blinkcam import render as R
from blinkcam.bank import Descriptor, PatchBank, annulus_stats
from blinkcam.geometry import PATCH_H, PATCH_W


def synthetic_patch(seed: int = 0, base: tuple[int, int, int] = (110, 140, 175)
                    ) -> np.ndarray:
    """Skin-toned patch with fine texture, standing in for a periorbital crop."""
    rng = np.random.default_rng(seed)
    patch = np.zeros((PATCH_H, PATCH_W, 3), np.float32)
    patch[:] = base
    patch += rng.normal(0, 6.0, patch.shape)
    # A dark horizontal band standing in for the lash line.
    patch[PATCH_H // 2 - 2: PATCH_H // 2 + 2] *= 0.45
    return np.clip(patch, 0, 255).astype(np.uint8)


def lighting_ramp(patch: np.ndarray, strength: float = 60.0) -> np.ndarray:
    """Apply a left-to-right illumination gradient."""
    x = np.linspace(-1.0, 1.0, PATCH_W, dtype=np.float32)
    ramp = (x * strength)[None, :, None]
    return np.clip(patch.astype(np.float32) + ramp, 0, 255).astype(np.uint8)


# ---- two-band relighting ---------------------------------------------------


def test_relight_is_near_identity_when_reference_matches_live():
    patch = synthetic_patch()
    out = R.two_band_relight(patch.astype(np.float32), patch.astype(np.float32),
                             R.RenderConfig().low_band_sigma)
    assert np.abs(out - patch.astype(np.float32)).mean() < 6.0


def test_relight_adopts_the_live_illumination_gradient():
    """The whole point: a reference shot under flat light must pick up the
    live frame's lighting gradient, because illumination is low-frequency."""
    reference = synthetic_patch(seed=0)
    live = lighting_ramp(synthetic_patch(seed=1), strength=60.0)
    out = R.two_band_relight(reference.astype(np.float32),
                             live.astype(np.float32),
                             R.RenderConfig().low_band_sigma)

    def tilt(img):
        g = np.asarray(img, np.float32).mean(axis=2)
        return float(g[:, -PATCH_W // 4:].mean() - g[:, :PATCH_W // 4].mean())

    # Output tilt should track the live frame's, not the flat reference's.
    assert tilt(out) == pytest.approx(tilt(live), abs=15.0)
    assert abs(tilt(out)) > abs(tilt(reference)) + 20.0


def test_relight_keeps_reference_high_frequency_detail():
    """Lashes and the crease come from the reference and must survive."""
    reference = synthetic_patch(seed=3)
    live = np.full((PATCH_H, PATCH_W, 3), 150, np.uint8)  # texture-free
    out = R.two_band_relight(reference.astype(np.float32),
                             live.astype(np.float32),
                             R.RenderConfig().low_band_sigma)
    row = out[PATCH_H // 2].mean(axis=1)
    above = out[PATCH_H // 2 - 8].mean(axis=1)
    # The dark lash band is still darker than the skin above it.
    assert row.mean() < above.mean() - 15.0


def test_relight_survives_a_colour_cast():
    reference = synthetic_patch(seed=4)
    live = np.clip(synthetic_patch(seed=5).astype(np.float32)
                   * np.array([1.35, 1.0, 0.8]), 0, 255).astype(np.uint8)
    out = R.two_band_relight(reference.astype(np.float32),
                             live.astype(np.float32),
                             R.RenderConfig().low_band_sigma)
    # Per-channel means should follow the live frame, not the reference.
    for c in range(3):
        assert out[..., c].mean() == pytest.approx(
            float(live[..., c].mean()), abs=14.0)


def test_relight_output_stays_in_range_after_clipping():
    reference = synthetic_patch(seed=6)
    live = np.full((PATCH_H, PATCH_W, 3), 250, np.uint8)
    out = np.clip(R.two_band_relight(reference.astype(np.float32),
                                     live.astype(np.float32),
                                     R.RenderConfig().low_band_sigma), 0, 255)
    assert out.min() >= 0 and out.max() <= 255


def test_fit_illumination_recovers_a_known_gradient():
    from blinkcam.bank import annulus_mask
    target = lighting_ramp(np.full((PATCH_H, PATCH_W, 3), 128, np.uint8), 50.0)
    fitted = R.fit_illumination(target.astype(np.float32), annulus_mask())
    assert np.abs(fitted - target.astype(np.float32)).mean() < 8.0


def test_fit_illumination_falls_back_when_mask_is_empty():
    patch = synthetic_patch()
    empty = np.zeros((PATCH_H, PATCH_W), np.float32)
    out = R.fit_illumination(patch.astype(np.float32), empty)
    assert out.shape == patch.shape
    assert np.isfinite(out).all()


# ---- patch bank ------------------------------------------------------------


def descriptor(yaw=0.0, pitch=0.0, brow=0.3, squint=0.0, scale=40.0):
    return Descriptor(yaw, pitch, brow, squint, scale)


def test_bank_add_also_stores_the_mirrored_patch():
    bank = PatchBank()
    bank.add(synthetic_patch(), descriptor(yaw=20.0), "right")
    assert len(bank) == 2
    assert {p.side for p in bank.patches} == {"right", "left"}
    mirrored = next(p for p in bank.patches if p.mirrored)
    assert mirrored.descriptor.yaw == pytest.approx(-20.0)


def test_bank_retrieves_the_closest_pose():
    bank = PatchBank()
    for yaw in (-30.0, 0.0, 30.0):
        bank.add(synthetic_patch(seed=int(yaw) + 40), descriptor(yaw=yaw),
                 "right", include_mirror=False)
    results = bank.query(descriptor(yaw=28.0), synthetic_patch(), k=1)
    assert results[0][0].descriptor.yaw == pytest.approx(30.0)


def test_bank_query_weights_sum_to_one():
    bank = PatchBank()
    for i in range(5):
        bank.add(synthetic_patch(seed=i), descriptor(yaw=i * 10.0), "right",
                 include_mirror=False)
    results = bank.query(descriptor(yaw=12.0), synthetic_patch(), k=3)
    assert len(results) == 3
    assert sum(w for _, w, _ in results) == pytest.approx(1.0)
    # Distances come back sorted, nearest first.
    assert [d for _, _, d in results] == sorted(d for _, _, d in results)


def test_bank_boundary_term_rejects_a_mismatched_neighbourhood():
    """A pose-perfect patch whose surrounding skin looks nothing like the live
    frame should lose to a slightly worse pose that matches. This is what stops
    us picking a patch with a stray hair or a hand shadow in it."""
    bank = PatchBank()
    dark = np.clip(synthetic_patch(seed=7).astype(np.float32) * 0.35,
                   0, 255).astype(np.uint8)
    bank.add(dark, descriptor(yaw=0.0), "right", include_mirror=False)
    bank.add(synthetic_patch(seed=8), descriptor(yaw=9.0), "right",
             include_mirror=False)

    live = synthetic_patch(seed=9)
    best = bank.query(descriptor(yaw=0.0), live, k=1)[0][0]
    assert best.descriptor.yaw == pytest.approx(9.0)


def test_empty_bank_query_returns_nothing():
    assert PatchBank().query(descriptor(), synthetic_patch()) == []


def test_bank_round_trips_through_disk(tmp_path):
    bank = PatchBank()
    for i in range(4):
        bank.add(synthetic_patch(seed=i), descriptor(yaw=i * 5.0, pitch=-i),
                 "right")
    path = str(tmp_path / "bank.npz")
    bank.save(path)
    loaded = PatchBank.load(path)

    assert len(loaded) == len(bank)
    for a, b in zip(bank.patches, loaded.patches):
        assert np.array_equal(a.image, b.image)
        assert a.side == b.side and a.mirrored == b.mirrored
        assert a.descriptor.yaw == pytest.approx(b.descriptor.yaw)
        assert a.descriptor.scale == pytest.approx(b.descriptor.scale)


def test_saving_an_empty_bank_is_refused(tmp_path):
    with pytest.raises(ValueError):
        PatchBank().save(str(tmp_path / "empty.npz"))


def test_coverage_buckets_by_pose_cell():
    bank = PatchBank()
    for yaw in (0.0, 2.0, 30.0):
        bank.add(synthetic_patch(), descriptor(yaw=yaw), "right",
                 include_mirror=False)
    cells = bank.coverage(yaw_step=10.0, pitch_step=10.0)
    assert cells[(0, 0)] == 2  # 0 and 2 degrees share a cell
    assert cells[(3, 0)] == 1


def test_annulus_stats_tracks_brightness():
    dim = annulus_stats(np.full((PATCH_H, PATCH_W, 3), 60, np.uint8))
    bright = annulus_stats(np.full((PATCH_H, PATCH_W, 3), 200, np.uint8))
    assert bright[0] > dim[0]


# ---- pose gating -----------------------------------------------------------


@pytest.mark.parametrize("yaw,side,expected", [
    (0.0, "right", 1.0), (0.0, "left", 1.0),
    (20.0, "right", 1.0),      # turned, but not far enough to occlude
    (60.0, "right", 0.0),      # far eye is self-occluded
    (60.0, "left", 1.0),       # near eye still fine
    (80.0, "left", 0.0),       # extreme profile disables both
])
def test_pose_gate_fades_the_self_occluded_eye(yaw, side, expected):
    from blinkcam.landmarks import FaceFrame
    from tests.test_geometry import make_eye

    renderer = R.EyeRenderer(PatchBank())
    face = FaceFrame(right=make_eye("right"), left=make_eye("left"),
                     yaw=yaw, pitch=0.0, roll=0.0)
    eye = face.right if side == "right" else face.left
    assert renderer._pose_gate(face, eye) == pytest.approx(expected)


def test_pose_gate_ramps_smoothly_rather_than_cutting():
    from blinkcam.landmarks import FaceFrame
    from tests.test_geometry import make_eye

    renderer = R.EyeRenderer(PatchBank())
    values = []
    for yaw in range(30, 56, 5):
        face = FaceFrame(right=make_eye("right"), left=make_eye("left"),
                         yaw=float(yaw), pitch=0.0, roll=0.0)
        values.append(renderer._pose_gate(face, face.right))
    assert values == sorted(values, reverse=True)
    assert 0.0 < values[2] < 1.0


def test_renderer_passes_frame_through_when_bank_is_empty():
    from blinkcam.landmarks import FaceFrame
    from tests.test_geometry import make_eye

    frame = np.full((200, 200, 3), 100, np.uint8)
    renderer = R.EyeRenderer(PatchBank())
    face = FaceFrame(right=make_eye("right"), left=make_eye("left", cx=140.0),
                     yaw=0.0, pitch=0.0, roll=0.0)
    assert np.array_equal(renderer.render(frame, face), frame)


def test_renderer_leaves_frame_untouched_at_zero_opacity():
    from blinkcam.landmarks import FaceFrame
    from tests.test_geometry import make_eye

    frame = np.full((200, 200, 3), 100, np.uint8)
    bank = PatchBank()
    bank.add(synthetic_patch(), descriptor(), "right")
    renderer = R.EyeRenderer(bank)
    face = FaceFrame(right=make_eye("right"), left=make_eye("left", cx=140.0),
                     yaw=0.0, pitch=0.0, roll=0.0)
    assert np.array_equal(renderer.render(frame, face, opacity=0.0), frame)


def test_renderer_modifies_only_the_eye_regions():
    from blinkcam.landmarks import FaceFrame
    from tests.test_geometry import make_eye

    rng = np.random.default_rng(2)
    frame = rng.integers(80, 170, (200, 260, 3), dtype=np.uint8)
    bank = PatchBank()
    bank.add(synthetic_patch(), descriptor(), "right")
    bank.add(synthetic_patch(seed=1), descriptor(), "left")

    face = FaceFrame(right=make_eye("right", cx=90.0, cy=100.0),
                     left=make_eye("left", cx=170.0, cy=100.0),
                     yaw=0.0, pitch=0.0, roll=0.0)
    out = R.EyeRenderer(bank).render(frame, face, opacity=1.0)

    changed = np.any(out != frame, axis=2)
    assert changed.any(), "renderer did nothing"
    # Nothing outside a generous box around the eyes may change.
    outside = changed.copy()
    outside[60:140, 40:220] = False
    assert not outside.any()


# ---- confidence thresholds -------------------------------------------------


def test_confidence_thresholds_are_ordered_and_measured():
    """These were guessed at 1.0 and 2.5, which faded genuinely good matches to
    67 percent and produced a visibly half-lidded eye. Measured against a real
    bank, good live matches score 1.1 to 1.5 and out-of-gamut poses 3.5 plus."""
    cfg = R.RenderConfig()
    assert cfg.good_distance < cfg.fade_distance
    assert cfg.good_distance >= 1.6, "good live matches reach ~1.5"
    assert cfg.fade_distance <= 3.4, "out-of-gamut poses start around 3.5"


def _quality(cfg, distance):
    span = max(cfg.fade_distance - cfg.good_distance, 1e-3)
    return float(np.clip((cfg.fade_distance - distance) / span, 0.0, 1.0))


@pytest.mark.parametrize("distance,expected", [
    (0.0, 1.0), (1.2, 1.0), (1.5, 1.0),   # good matches: full strength
    (3.5, 0.0), (5.0, 0.0),               # out of gamut: fully faded
])
def test_quality_curve_matches_measured_distances(distance, expected):
    assert _quality(R.RenderConfig(), distance) == pytest.approx(expected)


def test_quality_fades_smoothly_in_between():
    cfg = R.RenderConfig()
    mid = _quality(cfg, (cfg.good_distance + cfg.fade_distance) / 2)
    assert 0.2 < mid < 0.8


# ---- confidence smoothing --------------------------------------------------


def _renderer_with_one_patch():
    bank = PatchBank()
    bank.add(synthetic_patch(), descriptor(), "right", include_mirror=False)
    return R.EyeRenderer(bank, R.RenderConfig())


def test_confidence_rises_fast_and_falls_slow():
    """A single bad frame, a blink or motion blur, must not drop the effect:
    the eyes flicking open for a few frames is far more noticeable than a
    slightly wrong lid."""
    cfg = R.RenderConfig()
    assert cfg.confidence_attack > cfg.confidence_release * 4

    # Falling from full confidence takes many frames.
    value, frames = 1.0, 0
    while value > 0.5 and frames < 500:
        value += (0.0 - value) * cfg.confidence_release
        frames += 1
    assert frames > 8, "confidence collapses too quickly to survive a blink"

    # Rising is quick.
    value, frames = 0.0, 0
    while value < 0.9 and frames < 500:
        value += (1.0 - value) * cfg.confidence_attack
        frames += 1
    assert frames <= 5


# ---- selection stickiness --------------------------------------------------


def test_sticky_indices_are_reported_for_the_caller():
    bank = PatchBank()
    for i in range(6):
        bank.add(synthetic_patch(seed=i), descriptor(yaw=i * 8.0), "right",
                 include_mirror=False)
    bank.query(descriptor(yaw=10.0), synthetic_patch(), k=3)
    assert len(bank.last_indices) == 3
    assert all(0 <= i < len(bank) for i in bank.last_indices)


def test_stickiness_breaks_ties_toward_the_previous_choice():
    """Measured churn was 11% of frames with weights moving only 0.005, so this
    is a refinement, not the fix for visible instability. It still costs
    nothing to stop the set flickering between near-identical candidates."""
    bank = PatchBank()
    # Two candidates at nearly the same distance from the query.
    bank.add(synthetic_patch(seed=1), descriptor(yaw=9.9), "right",
             include_mirror=False)
    bank.add(synthetic_patch(seed=2), descriptor(yaw=10.1), "right",
             include_mirror=False)
    live = synthetic_patch(seed=3)

    plain = bank.query(descriptor(yaw=10.0), live, k=1)[0][0]
    other_index = next(i for i, p in enumerate(bank.patches) if p is not plain)
    stuck = bank.query(descriptor(yaw=10.0), live, k=1,
                       sticky={other_index}, stickiness=1.0)[0][0]
    assert stuck is not plain, "a large stickiness discount should flip a tie"


def test_stickiness_cannot_override_a_clearly_better_match():
    bank = PatchBank()
    bank.add(synthetic_patch(seed=1), descriptor(yaw=0.0), "right",
             include_mirror=False)
    bank.add(synthetic_patch(seed=2), descriptor(yaw=80.0), "right",
             include_mirror=False)
    live = synthetic_patch(seed=3)
    best = bank.query(descriptor(yaw=0.0), live, k=1,
                      sticky={1}, stickiness=0.10)[0][0]
    assert best.descriptor.yaw == pytest.approx(0.0)


def test_out_of_range_sticky_indices_are_ignored():
    """A bank reloaded between frames could leave stale indices behind."""
    bank = PatchBank()
    bank.add(synthetic_patch(), descriptor(), "right", include_mirror=False)
    result = bank.query(descriptor(), synthetic_patch(), k=1, sticky={99, 1000})
    assert len(result) == 1
