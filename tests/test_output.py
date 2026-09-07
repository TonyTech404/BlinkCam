"""Virtual camera output tests.

These cover the two things that actually broke against the real OBS sink, both
of which failed silently rather than raising: publishing at an unusual
resolution, and reconnecting too soon after a previous session.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from blinkcam import output as O


# ---- output size ----------------------------------------------------------


def test_supported_sizes_are_offered():
    assert (1920, 1080) in O.VirtualCamera.SUPPORTED
    assert (1280, 720) in O.VirtualCamera.SUPPORTED


def test_unusual_size_snaps_to_1080p(capsys):
    """Publishing 820x1024 made the extension log a pixel buffer size mismatch
    for every frame and consumers received solid green. It never raised, so the
    guard has to be here."""
    cam = O.VirtualCamera(820, 1024)
    assert (cam.width, cam.height) == (1920, 1080)
    assert "not a size" in capsys.readouterr().out


def test_supported_size_is_left_alone(capsys):
    cam = O.VirtualCamera(1280, 720)
    assert (cam.width, cam.height) == (1280, 720)
    assert capsys.readouterr().out == ""


def test_default_is_1080p():
    cam = O.VirtualCamera()
    assert (cam.width, cam.height) == (1920, 1080)


# ---- letterboxing --------------------------------------------------------


def test_fit_passes_a_matching_frame_through_untouched():
    cam = O.VirtualCamera(1280, 720)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    assert cam.fit(frame) is frame


def test_fit_always_returns_the_output_size():
    cam = O.VirtualCamera(1920, 1080)
    for shape in [(1024, 820), (480, 640), (2160, 3840), (100, 100)]:
        out = cam.fit(np.full((*shape, 3), 120, np.uint8))
        assert out.shape == (1080, 1920, 3)


def test_fit_preserves_aspect_ratio_rather_than_stretching():
    """A portrait source must not be distorted into 16:9."""
    cam = O.VirtualCamera(1920, 1080)
    # Bright content on a portrait canvas; measure the lit region.
    frame = np.full((1024, 820, 3), 200, np.uint8)
    out = cam.fit(frame)

    grey = out.max(axis=2)
    cols = np.where(grey.max(axis=0) > 12)[0]
    rows = np.where(grey.max(axis=1) > 12)[0]
    width = cols.max() - cols.min() + 1
    height = rows.max() - rows.min() + 1

    assert height == pytest.approx(1080, abs=2)          # fills the height
    assert width / height == pytest.approx(820 / 1024, rel=0.02)


def test_fit_centres_the_content():
    cam = O.VirtualCamera(1920, 1080)
    out = cam.fit(np.full((1024, 820, 3), 200, np.uint8))
    cols = np.where(out.max(axis=2).max(axis=0) > 12)[0]
    left = cols.min()
    right = 1920 - 1 - cols.max()
    assert left == pytest.approx(right, abs=2)


def test_fit_pads_with_black():
    cam = O.VirtualCamera(1920, 1080)
    out = cam.fit(np.full((1024, 820, 3), 200, np.uint8))
    assert out[540, 0].tolist() == [0, 0, 0]
    assert out[540, 1919].tolist() == [0, 0, 0]


def test_fit_handles_a_wider_than_output_source():
    cam = O.VirtualCamera(1920, 1080)
    out = cam.fit(np.full((1080, 3840, 3), 200, np.uint8))
    rows = np.where(out.max(axis=2).max(axis=1) > 12)[0]
    # 32:9 source letterboxes top and bottom, filling the width.
    assert rows.min() > 0 and rows.max() < 1079


# ---- reconnect cooldown ---------------------------------------------------


def test_cooldown_waits_after_a_recent_close(monkeypatch, tmp_path):
    """Reconnecting too soon leaves consumers on OBS's placeholder with no
    error raised anywhere, so the wait has to be enforced."""
    state = tmp_path / "closed"
    state.write_text(str(time.time()))
    monkeypatch.setattr(O, "_STATE", str(state))
    monkeypatch.setattr(O, "SINK_COOLDOWN", 0.4)

    start = time.monotonic()
    O.VirtualCamera._wait_for_cooldown(announce=False)
    assert time.monotonic() - start >= 0.3


def test_cooldown_does_not_wait_when_the_close_was_long_ago(monkeypatch, tmp_path):
    state = tmp_path / "closed"
    state.write_text("0")
    os.utime(state, (0, 0))
    monkeypatch.setattr(O, "_STATE", str(state))
    monkeypatch.setattr(O, "SINK_COOLDOWN", 5.0)

    start = time.monotonic()
    O.VirtualCamera._wait_for_cooldown(announce=False)
    assert time.monotonic() - start < 0.1


def test_cooldown_is_a_no_op_with_no_state_file(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "_STATE", str(tmp_path / "absent"))
    start = time.monotonic()
    O.VirtualCamera._wait_for_cooldown(announce=False)
    assert time.monotonic() - start < 0.1


def test_send_before_start_is_an_error():
    with pytest.raises(RuntimeError, match="not started"):
        O.VirtualCamera().send(np.zeros((1080, 1920, 3), np.uint8))


def test_setup_help_points_at_the_real_settings_pane():
    """macOS reports the extension under General > Login Items & Extensions,
    not the Privacy & Security pane this originally documented."""
    assert "Login Items & Extensions" in O.SETUP_HELP
    assert "QUIT OBS" in O.SETUP_HELP


# ---- landmarker timestamp discipline --------------------------------------


def test_landmarker_clamps_non_increasing_timestamps():
    """MediaPipe's video mode raises if a timestamp does not exceed the last
    one. A tool that mixed absolute monotonic time with time-since-start
    crashed part way through a run, so the clamp lives in the wrapper and
    callers no longer supply clocks at all."""
    from blinkcam.landmarks import Landmarker

    # Bypass the model load; only the timestamp bookkeeping is under test.
    landmarker = Landmarker.__new__(Landmarker)
    landmarker._last_timestamp = -1
    landmarker._origin = time.monotonic()

    seen = [landmarker._next_timestamp(v)
            for v in (92_837_000, 5, 5, 4, 92_837_001)]

    assert seen == sorted(seen), "timestamps must never go backwards"
    assert len(set(seen)) == len(seen), "and must be strictly increasing"


def test_landmarker_generates_its_own_clock_when_none_is_given():
    from blinkcam.landmarks import Landmarker

    landmarker = Landmarker.__new__(Landmarker)
    landmarker._last_timestamp = -1
    landmarker._origin = time.monotonic()

    seen = [landmarker._next_timestamp(None) for _ in range(5)]
    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen)
    assert all(v >= 0 for v in seen)
