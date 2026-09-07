"""ApertureTracker tests.

The tracker exists because the repainted region was derived from live lid
contours every frame, which made it jitter 11x more during a blink and 12x more
during fast head motion. These tests pin the two mechanisms that fixed it:
gating updates on the eye being open, and smoothing in patch space.
"""

from __future__ import annotations

import numpy as np
import pytest

from blinkcam.geometry import (ApertureTracker, FILTER_PRESETS,
                               lid_mask_polygon, patch_transform,
                               to_patch_coords)
from tests.test_geometry import make_eye


def _poly(tracker, eye, blink, t):
    return tracker.polygon(eye, patch_transform(eye), blink, t, 0.15, 1.0)


def test_first_call_returns_the_live_polygon():
    """No history yet, so there is nothing to hold; the tracker must not emit
    zeros or block the first frame."""
    tracker = ApertureTracker(30.0)
    eye = make_eye("right")
    out = _poly(tracker, eye, 0.0, 0.0)
    live = to_patch_coords(lid_mask_polygon(eye, 0.15, 1.0), patch_transform(eye))
    assert out.shape == live.shape
    assert out == pytest.approx(live, abs=1e-3)


def test_polygon_is_held_while_the_eye_closes():
    """The measured worst case: 1.883 px/frame of movement during a blink,
    because the real aperture collapses while the effect still needs to repaint
    the whole open region."""
    tracker = ApertureTracker(30.0)
    open_eye = make_eye("right", height=22.0)

    # Settle on the open aperture.
    for i in range(40):
        settled = _poly(tracker, open_eye, 0.02, i / 30.0)

    # Now the lids collapse and the blink signal rises.
    moved = []
    for i in range(10):
        shut = make_eye("right", height=22.0 - 2.1 * i)
        out = _poly(tracker, shut, 0.85, (40 + i) / 30.0)
        moved.append(float(np.abs(out - settled).max()))

    assert max(moved) < 1e-3, "a blink must not move the repainted region"


def test_polygon_tracks_again_once_the_eye_reopens():
    tracker = ApertureTracker(30.0)
    for i in range(40):
        _poly(tracker, make_eye("right", height=22.0), 0.02, i / 30.0)

    # A genuinely different resting aperture, eyes open.
    wider = make_eye("right", height=30.0)
    for i in range(90):
        out = _poly(tracker, wider, 0.02, (40 + i) / 30.0)

    target = to_patch_coords(lid_mask_polygon(wider, 0.15, 1.0),
                             patch_transform(wider))
    assert out == pytest.approx(target, abs=1.0), (
        "a sustained change in aperture, such as a squint, must be followed")


def test_smoothing_reduces_jitter_in_patch_space():
    """Patch space is pose-normalised, so the aperture is nearly static there
    and can be smoothed hard without any lag penalty."""
    rng = np.random.default_rng(0)
    tracker = ApertureTracker(30.0)

    raw_steps, out_steps = [], []
    prev_raw = prev_out = None
    for i in range(120):
        jitter = rng.normal(0.0, 0.4, 2)
        eye = make_eye("right", cx=100.0 + jitter[0], cy=100.0 + jitter[1])
        raw = to_patch_coords(lid_mask_polygon(eye, 0.15, 1.0),
                              patch_transform(eye))
        out = _poly(tracker, eye, 0.02, i / 30.0)
        if prev_raw is not None and i > 30:
            raw_steps.append(np.linalg.norm(raw - prev_raw, axis=1).mean())
            out_steps.append(np.linalg.norm(out - prev_out, axis=1).mean())
        prev_raw, prev_out = raw, out.copy()

    assert np.mean(out_steps) < np.mean(raw_steps) / 1.5


def test_iris_is_held_through_a_blink():
    """The globe bulge is driven from the iris, and iris landmarks are
    meaningless once the lids cover them."""
    tracker = ApertureTracker(30.0)
    eye = make_eye("right")
    m = patch_transform(eye)
    for i in range(40):
        settled, radius = tracker.iris(eye, m, 0.02, i / 30.0)

    shifted = make_eye("right")
    shifted.iris_center = shifted.iris_center + np.array([9.0, 5.0], np.float32)
    during, _ = tracker.iris(shifted, patch_transform(shifted), 0.9, 41 / 30.0)
    assert during == pytest.approx(settled, abs=1e-3)
    assert radius >= 2.0


def test_a_changed_point_count_restarts_cleanly():
    """Landmark counts should not change, but a shape mismatch must not raise
    or return a stale array of the wrong size."""
    tracker = ApertureTracker(30.0)
    eye = make_eye("right")
    _poly(tracker, eye, 0.0, 0.0)

    trimmed = make_eye("right")
    trimmed.upper_lid = trimmed.upper_lid[:4]
    out = tracker.polygon(trimmed, patch_transform(trimmed), 0.0, 0.1, 0.15, 1.0)
    expected = to_patch_coords(lid_mask_polygon(trimmed, 0.15, 1.0),
                               patch_transform(trimmed))
    assert out.shape == expected.shape


def test_reset_clears_all_held_state():
    tracker = ApertureTracker(30.0)
    eye = make_eye("right")
    _poly(tracker, eye, 0.0, 0.0)
    tracker.iris(eye, patch_transform(eye), 0.0, 0.0)
    assert tracker._held
    tracker.reset()
    assert not tracker._held and not tracker._filters


def test_open_threshold_excludes_the_approach_to_a_closure():
    """Most of the jitter happens while the lid is on its way down, so the gate
    has to sit well below the blink-detection threshold of 0.55."""
    from blinkcam.transition import BLINK_ENTER
    assert ApertureTracker.OPEN_BELOW < BLINK_ENTER / 2.0


def test_aperture_preset_has_no_adaptive_term():
    """beta re-admits noise at speed, and there is no fast motion to keep up
    with in a pose-normalised frame."""
    assert FILTER_PRESETS["aperture"]["beta"] == 0.0
    assert FILTER_PRESETS["aperture"]["mincutoff"] < FILTER_PRESETS["lid"]["mincutoff"]
