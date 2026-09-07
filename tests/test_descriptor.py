"""Descriptor tests.

The descriptor decides which harvested patch gets used, and one of its
dimensions was silently comparing incommensurable quantities. These tests pin
down the property that matters: a query taken with the eyes OPEN must be
comparable to a reference harvested with the eyes CLOSED.
"""

from __future__ import annotations

import numpy as np
import pytest

from blinkcam.bank import Descriptor, WEIGHTS
from blinkcam.landmarks import FaceFrame
from tests.test_geometry import make_eye


def _face(height: float) -> FaceFrame:
    """A face whose eyes are open by `height` pixels of aperture."""
    return FaceFrame(right=make_eye("right", height=height),
                     left=make_eye("left", cx=180.0, height=height),
                     yaw=0.0, pitch=0.0, roll=0.0)


def test_brow_height_ignores_lid_state():
    """The whole point. Measured against the upper lid it moved by 0.20 between
    open and closed eyes, spending 1.38 of the 3.2 distance budget on every
    frame and fading the effect out during a blink. The eye corners do not move
    when the lids do, so the measurement must use them."""
    wide_open = _face(height=22.0).right.brow_height
    half = _face(height=11.0).right.brow_height
    shut = _face(height=1.0).right.brow_height

    assert wide_open == pytest.approx(half, abs=0.02)
    assert wide_open == pytest.approx(shut, abs=0.02)


def test_brow_height_still_tracks_brow_movement():
    """Lid-independence must not cost us the expression signal it exists for."""
    eye = make_eye("right")
    baseline = eye.brow_height

    raised = make_eye("right")
    raised.brow = raised.brow - np.array([0.0, 10.0], np.float32)
    assert raised.brow_height > baseline + 0.15

    lowered = make_eye("right")
    lowered.brow = lowered.brow + np.array([0.0, 6.0], np.float32)
    assert lowered.brow_height < baseline - 0.10


def test_brow_height_is_scale_invariant():
    """A face near and far must produce the same expression reading."""
    near = make_eye("right", width=80.0, height=24.0).brow_height
    far = make_eye("right", width=20.0, height=6.0).brow_height
    assert near == pytest.approx(far, rel=0.05)


def test_descriptor_distance_between_open_and_closed_is_small():
    """A live open-eye query against a closed-eye reference of the same pose and
    expression should be nearly zero apart, since only the lids differ."""
    open_eyes = _face(height=22.0)
    closed_eyes = _face(height=1.0)

    a = Descriptor.build(open_eyes, open_eyes.right).vector()
    b = Descriptor.build(closed_eyes, closed_eyes.right).vector()
    assert float(np.linalg.norm((a - b) * WEIGHTS)) < 0.25


def test_mirrored_descriptor_flips_only_yaw():
    d = Descriptor(20.0, -5.0, 0.5, 0.1, 60.0)
    m = d.mirrored()
    assert m.yaw == pytest.approx(-20.0)
    assert m.pitch == pytest.approx(d.pitch)
    assert m.brow_height == pytest.approx(d.brow_height)
    assert m.cheek_squint == pytest.approx(d.cheek_squint)
    assert m.scale == pytest.approx(d.scale)
