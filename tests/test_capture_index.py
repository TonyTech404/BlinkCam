"""Capture device selection.

This exists because a live call went into a feedback loop. BlinkCam captured
its own virtual camera output and republished it. The cause was assuming that
AVFoundation's device indices are the ones cv2.VideoCapture opens. They are
not, and the difference is not theoretical: AVFoundation reported the virtual
camera at index 0 and the webcam at 1, while OpenCV opened the webcam at 0 and
the virtual camera at 1.
"""

from __future__ import annotations

import inspect

import numpy as np

from blinkcam import capture as C


def test_fingerprint_is_order_independent(monkeypatch):
    """AVFoundation's enumeration order changes within a single session, so an
    order-sensitive fingerprint misses the cache on nearly every call and
    re-runs a five second probe each time."""
    a = [C.Device(0, "OBS Virtual Camera", "uid-obs"),
         C.Device(1, "UGREEN Camera 4K", "uid-ugreen")]
    b = [C.Device(0, "UGREEN Camera 4K", "uid-ugreen"),
         C.Device(1, "OBS Virtual Camera", "uid-obs")]

    monkeypatch.setattr(C, "list_devices", lambda: a)
    first = C._device_fingerprint()
    monkeypatch.setattr(C, "list_devices", lambda: b)
    assert C._device_fingerprint() == first


def test_fingerprint_changes_when_a_camera_is_added(monkeypatch):
    one = [C.Device(0, "UGREEN Camera 4K", "uid-ugreen")]
    two = one + [C.Device(1, "Ipon Camera", "uid-ipon")]
    monkeypatch.setattr(C, "list_devices", lambda: one)
    first = C._device_fingerprint()
    monkeypatch.setattr(C, "list_devices", lambda: two)
    assert C._device_fingerprint() != first


def test_default_index_skips_the_measured_virtual_index(monkeypatch):
    """The whole point: skip the index OpenCV actually opens the virtual camera
    at, not the one AVFoundation names."""
    monkeypatch.setattr(C, "virtual_capture_index", lambda: 0)
    assert C.default_camera_index() == 1
    monkeypatch.setattr(C, "virtual_capture_index", lambda: 1)
    assert C.default_camera_index() == 0
    monkeypatch.setattr(C, "virtual_capture_index", lambda: None)
    assert C.default_camera_index() == 0


def test_marker_is_not_something_a_real_scene_produces():
    """Detection relies on the marker being unmistakable. Measured correlation
    was 0.992 against the virtual camera and 0.044 against the webcam."""
    marker = C._marker_frame(320, 180)
    assert marker.shape == (180, 320, 3)
    # Coarse blocks: high variance between regions, flat within them.
    assert marker.std() > 40
    assert C._looks_like(marker, marker) > 0.99


def test_looks_like_rejects_an_unrelated_frame():
    marker = C._marker_frame(320, 180)
    rng = np.random.default_rng(7)
    scene = rng.integers(40, 90, (180, 320, 3), dtype=np.uint8)
    assert C._looks_like(scene, marker) < 0.5


def test_looks_like_is_safe_on_a_flat_frame():
    """A blank frame correlates with nothing meaningfully, and must not be
    reported as a match or raise on a zero standard deviation."""
    marker = C._marker_frame(320, 180)
    flat = np.full((180, 320, 3), 17, np.uint8)
    assert C._looks_like(flat, marker) == 0.0


def test_device_is_virtual_is_documented_as_display_only():
    """It reads a NAME against AVFoundation's index, which is not the index
    OpenCV opens. Using it to choose a capture device is what caused the loop."""
    doc = inspect.getdoc(C.Device.is_virtual.fget) or ""
    assert "display only" in doc.lower()
    assert "virtual_capture_index" in doc


def test_list_devices_docstring_warns_the_indices_are_not_opencvs():
    doc = inspect.getdoc(C.list_devices) or ""
    assert "do NOT match" in doc or "do not match" in doc

