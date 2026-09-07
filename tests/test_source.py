"""Frame source tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from blinkcam.capture import CameraConfig
from blinkcam.source import FileSource, open_source


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 120, np.uint8))
    return str(path)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available")
    for i in range(5):
        writer.write(np.full((48, 64, 3), 40 + i * 40, np.uint8))
    writer.release()
    return str(path)


def test_image_source_reports_its_geometry(image):
    with FileSource(image, fps=30) as src:
        assert src.actual()["width"] == 64
        assert src.actual()["height"] == 48
        assert src.measure_fps() == 30.0


def test_image_source_serves_the_same_frame_repeatedly(image):
    with FileSource(image, fps=1000) as src:
        first, second = src.read(), src.read()
    assert np.array_equal(first, second)


def test_read_new_paces_to_the_declared_rate(image):
    """Timings only mean something if playback is paced."""
    with FileSource(image, fps=50) as src:
        assert src.read_new() is not None    # first call is due immediately
        assert src.read_new() is None        # too soon for the next
        import time
        time.sleep(0.03)
        assert src.read_new() is not None


def test_video_source_loops_by_default(video):
    with FileSource(video, fps=1000, loop=True) as src:
        frames = [src.read() for _ in range(12)]
    assert all(f is not None for f in frames)
    assert not src.exhausted


def test_video_source_reports_exhaustion_when_not_looping(video):
    with FileSource(video, fps=1000, loop=False) as src:
        for _ in range(20):
            if src.read() is None:
                break
        assert src.exhausted


def test_missing_file_is_rejected():
    with pytest.raises(RuntimeError, match="no such file"):
        FileSource("/definitely/not/here.png")


def test_undecodable_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(RuntimeError):
        FileSource(str(bad))


def test_file_source_satisfies_the_camera_interface(image):
    """The main loop must not care which source it has."""
    with FileSource(image) as src:
        for name in ("read", "read_new", "actual", "measure_fps",
                     "lock_exposure_and_white_balance"):
            assert callable(getattr(src, name))
        assert src.dropped == 0
        assert src.lock_exposure_and_white_balance() == {}


def test_open_source_picks_the_file_when_given_a_path(image):
    with open_source(image, CameraConfig()) as src:
        assert isinstance(src, FileSource)
