"""Webcam capture with latest-frame-only delivery.

Two macOS-specific concerns drive this module.

First, OpenCV's AVFoundation backend buffers two to four frames. If we read
frames in the main loop we inherit 65-130ms of latency that is not ours and
cannot be tuned away. So a background thread consumes frames as fast as the
camera produces them and keeps only the newest, and the main loop takes that.

Second, the backend silently ignores several property writes. CAP_PROP_FOURCC
in particular returns success while doing nothing, and a resolution request can
quietly land at a lower frame rate. So every property is read back after being
set and the caller is told what it actually got.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Device:
    index: int
    name: str
    uid: str

    @property
    def is_virtual(self) -> bool:
        """Whether this is a virtual camera rather than real hardware.

        Capturing from our own output would feed the effect into itself, and it
        cannot be detected from the picture: while the effect is off the output
        is byte-identical to the input, so any content comparison reports a
        perfect match no matter which device is open. Identity has to come from
        the device, not the frames.
        """
        lowered = self.name.lower()
        return any(word in lowered for word in
                   ("obs virtual", "virtual camera", "blinkcam"))


def list_devices() -> list[Device]:
    """Video capture devices with their real names, in OpenCV's index order.

    OpenCV cannot report device names on macOS, and indices are not stable:
    installing the virtual camera renumbered everything here. AVFoundation is
    asked directly so a device can be chosen by name.
    """
    try:
        from AVFoundation import AVCaptureDevice
    except Exception:
        # No pyobjc: fall back to probing, which yields no names.
        return [Device(i, f"camera {i}", "")
                for i, _, _ in describe_cameras()]

    devices = AVCaptureDevice.devicesWithMediaType_("vide")
    return [Device(i, str(d.localizedName()), str(d.uniqueID()))
            for i, d in enumerate(devices)]


def default_camera_index() -> int:
    """First real camera, skipping virtual ones."""
    for device in list_devices():
        if not device.is_virtual:
            return device.index
    return 0


def resolve_device(selector: int | str | None) -> Device | None:
    """Look up a device by index, by name substring, or pick a sensible default."""
    devices = list_devices()
    if not devices:
        return None
    if selector is None:
        return next((d for d in devices if not d.is_virtual), devices[0])
    if isinstance(selector, int):
        return next((d for d in devices if d.index == selector), None)
    needle = selector.lower()
    return next((d for d in devices if needle in d.name.lower()), None)


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    allow_virtual: bool = False


class Camera:
    """Threaded capture that always hands back the most recent frame."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_seq_read = -1
        self._dropped = 0

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> "Camera":
        device = resolve_device(self.config.index)
        if device is not None:
            self.name = device.name
            if device.is_virtual and not self.config.allow_virtual:
                names = ", ".join(
                    f"[{d.index}] {d.name}" for d in list_devices())
                raise RuntimeError(
                    f"Camera {device.index} is '{device.name}', a virtual "
                    f"camera. Capturing it would feed BlinkCam's own output "
                    f"back into itself.\nAvailable devices: {names}\n"
                    f"Pass --camera {default_camera_index()} for the real "
                    f"webcam.")

        cap = cv2.VideoCapture(self.config.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.index}. "
                "Check System Settings > Privacy & Security > Camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Ask for the shallowest buffer the backend will give us. Often ignored
        # on AVFoundation, which is why the drain thread below exists anyway.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="blinkcam-capture")
        self._thread.start()

        # Wait for the first frame so callers never race an empty camera.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self
            time.sleep(0.01)
        self.stop()
        raise RuntimeError("Camera opened but delivered no frames within 5s.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- frames --------------------------------------------------------

    def _drain(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                if self._seq != self._last_seq_read:
                    # Previous frame was never consumed. We are the bottleneck.
                    self._dropped += 1
                self._frame = frame
                self._seq += 1

    def read(self) -> np.ndarray | None:
        """Newest frame as BGR, or None if nothing has arrived yet."""
        with self._lock:
            if self._frame is None:
                return None
            self._last_seq_read = self._seq
            return self._frame

    def read_new(self) -> np.ndarray | None:
        """Newest frame, but only if it has not been returned before."""
        with self._lock:
            if self._frame is None or self._seq == self._last_seq_read:
                return None
            self._last_seq_read = self._seq
            return self._frame

    @property
    def dropped(self) -> int:
        """Frames the camera produced that the main loop never saw."""
        return self._dropped

    # ---- properties ----------------------------------------------------

    def actual(self) -> dict[str, float]:
        """What the driver actually gave us, which may not be what we asked."""
        if self._cap is None:
            return {}
        return {
            "width": self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

    def measure_fps(self, seconds: float = 2.0) -> float:
        """Time real delivery. Never trust CAP_PROP_FPS on this backend."""
        start, count, last = time.monotonic(), 0, -1
        while time.monotonic() - start < seconds:
            with self._lock:
                if self._seq != last:
                    last = self._seq
                    count += 1
            time.sleep(0.002)
        return count / seconds

    def lock_exposure_and_white_balance(self) -> dict[str, bool]:
        """Freeze auto-exposure and auto-white-balance.

        Exposure hunting is the most likely cause of the composited eye patch
        drifting in colour against the surrounding skin, because the patch was
        harvested under one exposure and is being blended into another.

        The AVFoundation backend does not reliably support these, so the result
        reports per-property whether the write actually took effect.
        """
        if self._cap is None:
            return {}
        result: dict[str, bool] = {}
        for name, prop, value in (
            ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
            ("auto_wb", cv2.CAP_PROP_AUTO_WB, 0.0),
        ):
            before = self._cap.get(prop)
            self._cap.set(prop, value)
            after = self._cap.get(prop)
            result[name] = after != before or after == value
        return result


def describe_cameras(max_index: int = 4) -> list[tuple[int, int, int]]:
    """Probe indices and report (index, width, height) for those that open.

    Stops at the first gap once something has been found, and silences the
    backend while probing: AVFoundation logs a wall of "out device of bound"
    errors for every index past the last real camera, which looks alarming in
    the doctor output and means nothing.
    """
    with _silenced_stderr():
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            opened = cap.isOpened()
            if opened:
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append((i, frame.shape[1], frame.shape[0]))
            cap.release()
            if not opened and found:
                break  # past the last real device
        return found


@contextlib.contextmanager
def _silenced_stderr():
    """Silence the capture backend while probing.

    AVFoundation writes "out device of bound" straight to the process's stderr
    for every index past the last real camera. That is not routed through
    OpenCV's logging, so setLogLevel cannot hide it, and it looks alarming in
    the doctor output while meaning nothing. Redirecting the file descriptor is
    the only thing that works.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
