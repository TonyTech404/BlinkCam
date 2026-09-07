"""Blink detection and transition timing tests."""

from __future__ import annotations

import pytest

from blinkcam.landmarks import FaceFrame
from blinkcam.transition import (BLINK_ENTER, BLINK_EXIT, BlinkDetector,
                                 EffectTransition, State, TransitionConfig)
from tests.test_geometry import make_eye


def face(blink: float = 0.0, ear: float = 0.30) -> FaceFrame:
    right, left = make_eye("right"), make_eye("left")
    right.blink = left.blink = blink
    right.ear = left.ear = ear
    return FaceFrame(right=right, left=left, yaw=0.0, pitch=0.0, roll=0.0,
                     blendshapes={"eyeBlinkRight": blink,
                                  "eyeBlinkLeft": blink})


# ---- blink detection -------------------------------------------------------


def test_detector_uses_hysteresis_not_a_single_threshold():
    """A signal hovering at the boundary must not chatter."""
    d = BlinkDetector()
    assert not d.update(face(blink=0.50), 0.0)   # below enter
    assert d.update(face(blink=0.60), 0.1)       # crosses enter
    assert d.update(face(blink=0.45), 0.2)       # between: stays closed
    assert not d.update(face(blink=0.30), 0.3)   # crosses exit


def test_detector_thresholds_are_ordered():
    assert BLINK_EXIT < BLINK_ENTER


def test_detector_falls_back_to_ear_without_blendshapes():
    d = BlinkDetector()
    f = face(blink=0.0, ear=0.10)
    f.blendshapes = {}
    f.right.blink = f.left.blink = 0.0
    assert d.update(f, 0.0)


def test_detector_reports_closed_duration():
    d = BlinkDetector()
    d.update(face(blink=0.9), 10.0)
    assert d.closed_for(10.25) == pytest.approx(0.25)


def test_detector_resets_when_the_face_is_lost():
    d = BlinkDetector()
    d.update(face(blink=0.9), 0.0)
    assert d.closed
    assert not d.update(None, 0.1)
    assert d.closed_for(0.2) == 0.0


# ---- transition ------------------------------------------------------------


def test_starts_off_and_transparent():
    t = EffectTransition()
    assert t.state is State.OFF
    assert t.update(face(), 0.0) == 0.0
    assert not t.enabled


def test_toggle_arms_and_waits_for_a_real_blink():
    t = EffectTransition()
    t.toggle(0.0)
    assert t.state is State.ARMING
    # Eyes open: still waiting, and nothing is rendered yet.
    assert t.update(face(blink=0.0), 0.1) == 0.0
    assert t.state is State.ARMING


def test_riding_a_real_blink_engages_instantly_at_full_opacity():
    """The transition is perfect by construction: the eyes are already shut, so
    coming up at full strength shows nothing."""
    t = EffectTransition()
    t.toggle(0.0)
    assert t.update(face(blink=0.9), 0.2) == pytest.approx(1.0)
    assert t.state is State.ON


def test_falls_back_to_a_synthetic_closure_after_the_arm_timeout():
    t = EffectTransition(TransitionConfig(arm_timeout=2.0))
    t.toggle(0.0)
    t.update(face(blink=0.0), 1.0)
    assert t.state is State.ARMING
    t.update(face(blink=0.0), 2.1)
    assert t.state is State.CLOSING


def test_synthetic_closure_ramps_up_and_completes():
    t = EffectTransition(TransitionConfig(ride_real_blink=False,
                                          close_duration=0.10))
    t.toggle(0.0)
    assert t.state is State.CLOSING
    mid = t.update(face(), 0.05)
    assert 0.0 < mid < 1.0
    assert t.update(face(), 0.11) == pytest.approx(1.0)
    assert t.state is State.ON


def _samples(t: EffectTransition, duration: float, steps: int = 8) -> list[float]:
    return [t.update(face(), duration * i / steps) for i in range(steps + 1)]


def test_closing_accelerates():
    """Ease-in: the lid starts from rest, so little happens in the first
    moments and most of the travel is late."""
    t = EffectTransition(TransitionConfig(ride_real_blink=False,
                                          close_duration=1.0))
    t.toggle(0.0)
    s = _samples(t, 1.0)
    first_quarter = s[2] - s[0]
    last_quarter = s[8] - s[6]
    assert last_quarter > first_quarter * 2.0
    assert s == sorted(s)  # monotonically closing


def test_opening_decelerates():
    """Ease-out: the lid leaves fast and settles, rather than stopping dead."""
    t = EffectTransition(TransitionConfig(open_duration=1.0))
    t.state = State.ON
    t.toggle(0.0)
    assert t.state is State.OPENING
    s = _samples(t, 1.0)
    first_quarter = s[0] - s[2]
    last_quarter = s[6] - s[8]
    assert first_quarter > last_quarter * 2.0
    assert s == sorted(s, reverse=True)  # monotonically opening


def test_opening_is_slower_than_closing_by_default():
    cfg = TransitionConfig()
    assert cfg.open_duration > cfg.close_duration * 1.5


def test_toggling_off_returns_to_off_after_the_open_phase():
    t = EffectTransition(TransitionConfig(open_duration=0.22))
    t.state = State.ON
    t.toggle(0.0)
    assert t.update(face(), 0.10) > 0.0
    assert t.update(face(), 0.30) == 0.0
    assert t.state is State.OFF


def test_toggling_during_opening_re_engages():
    t = EffectTransition()
    t.state = State.ON
    t.toggle(0.0)
    assert t.state is State.OPENING
    t.toggle(0.05)
    assert t.state is State.ARMING


def test_force_off_is_immediate():
    t = EffectTransition()
    t.state = State.ON
    t.force_off()
    assert t.state is State.OFF
    assert t.opacity == 0.0


def test_geometry_freezes_only_during_a_real_blink_while_on():
    t = EffectTransition()
    t.state = State.ON
    t.update(face(blink=0.0), 0.0)
    assert not t.should_freeze_geometry()
    t.update(face(blink=0.9), 0.1)
    assert t.should_freeze_geometry()


def test_geometry_does_not_freeze_while_off():
    t = EffectTransition()
    t.update(face(blink=0.9), 0.0)
    assert not t.should_freeze_geometry()


def test_opacity_stays_in_range_across_a_full_cycle():
    t = EffectTransition(TransitionConfig(ride_real_blink=False))
    t.toggle(0.0)
    now = 0.0
    for _ in range(40):
        now += 1 / 30
        assert 0.0 <= t.update(face(), now) <= 1.0
    t.toggle(now)
    for _ in range(40):
        now += 1 / 30
        assert 0.0 <= t.update(face(), now) <= 1.0
    assert t.state is State.OFF
