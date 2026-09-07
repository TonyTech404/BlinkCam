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
import tempfile
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
        """Whether this device NAME belongs to a virtual camera.

        Useful for display only. It must NOT be used to decide what to capture:
        `index` here is AVFoundation's, and OpenCV opens a different device at
        the same number. Use virtual_capture_index() for that.
        """
        lowered = self.name.lower()
        return any(word in lowered for word in
                   ("obs virtual", "virtual camera", "blinkcam"))


def list_devices() -> list[Device]:
    """Video capture devices with their real names, in AVFoundation's order.

    The indices here are AVFoundation's own and do NOT match the ones
    cv2.VideoCapture opens; see probe_virtual_index. Use this to show the user
    what is attached, never to choose a capture device.
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
    """Lowest OpenCV index that is not our own virtual camera output.

    Deliberately NOT "the first device AVFoundation calls non-virtual": those
    indices do not match the ones OpenCV opens. See probe_virtual_index.
    """
    virtual = virtual_capture_index()
    for index in range(4):
        if index != virtual:
            return index
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
        # Refuse to capture our own output. The check is against the MEASURED
        # OpenCV index, not against AVFoundation's device names: those indices
        # disagree, and trusting the names is what put a live call into a
        # feedback loop. See probe_virtual_index.
        if not self.config.allow_virtual:
            virtual = virtual_capture_index()
            if virtual is not None and self.config.index == virtual:
                raise RuntimeError(
                    f"Capture index {self.config.index} is BlinkCam's own "
                    f"virtual camera output; capturing it would feed the "
                    f"effect back into itself.\n"
                    f"Use --camera {default_camera_index()} instead.")

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

# ---------------------------------------------------------------------------
# Which OpenCV index is our own output
# ---------------------------------------------------------------------------

_VIRTUAL_CACHE = os.path.join(tempfile.gettempdir(), "blinkcam-virtual-index")


def _device_fingerprint() -> str:
    """Identifies the SET of cameras attached, so a cached probe is invalidated
    when one is plugged in or removed.

    Sorted deliberately. AVFoundation's enumeration order is not stable within
    a session, so an order-dependent fingerprint misses the cache on almost
    every call and re-runs a five second probe each time.
    """
    return "|".join(sorted(f"{d.name}:{d.uid}" for d in list_devices()))


def _marker_frame(width: int, height: int) -> np.ndarray:
    """A pattern no real scene produces: coarse random colour blocks."""
    rng = np.random.default_rng(12345)
    blocks = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
    return cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)


def _looks_like(frame: np.ndarray, marker: np.ndarray) -> float:
    a = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    b = cv2.resize(marker, (64, 36), interpolation=cv2.INTER_AREA)
    x = a.astype(np.float32).ravel()
    y = b.astype(np.float32).ravel()
    if x.std() < 1.0 or y.std() < 1.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def probe_virtual_index(max_index: int = 4, settle: float = 3.5) -> int | None:
    """Which OpenCV capture index is the virtual camera, measured not guessed.

    OpenCV's AVFoundation indices do NOT correspond to AVFoundation's own
    enumeration. Measured on macOS 26.4: AVFoundation reported [0] OBS Virtual
    Camera and [1] UGREEN Camera 4K, while OpenCV's index 1 was the virtual
    camera and index 0 the webcam. Every AVFoundation enumeration tried
    (devicesWithMediaType_, devices(), AVCaptureDeviceDiscoverySession) agreed
    with each other and disagreed with OpenCV, so there is no ordering to copy.
    AVFoundation's own order also changed within a single session.

    Selecting a device by name therefore cannot work, and getting it wrong is
    not a small matter: BlinkCam captured its own output and republished it,
    live, mid-call.

    So measure the correspondence. Publish a marker no real scene produces and
    find the index that returns it. Correlation was 0.992 for the virtual
    camera against 0.044 for the webcam, so the test is decisive rather than
    marginal.

    Returns None if the virtual camera cannot be opened, in which case there is
    nothing to avoid capturing.
    """
    from .output import VirtualCamera, obs_is_running

    if obs_is_running():
        return None

    marker = _marker_frame(1920, 1080)
    found: int | None = None
    stop = threading.Event()

    def keep_publishing(camera) -> None:
        # The marker must still be going out WHILE we read, or the sink falls
        # back to OBS's placeholder the moment we stop and the probe finds
        # nothing. Publishing before reading is not enough.
        while not stop.is_set():
            camera.send(marker)
            camera.sleep_until_next_frame()

    try:
        with VirtualCamera(1920, 1080, 30) as vcam:
            pump = threading.Thread(target=keep_publishing, args=(vcam,),
                                    daemon=True)
            pump.start()
            time.sleep(settle)

            with _silenced_stderr():
                for index in range(max_index):
                    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    ok = False
                    frame = None
                    for _ in range(6):
                        ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None and _looks_like(frame, marker) > 0.9:
                        found = index
                        break
            stop.set()
            pump.join(timeout=1.0)
    except Exception:
        return None
    return found


def virtual_capture_index(refresh: bool = False) -> int | None:
    """probe_virtual_index, cached against the current set of cameras.

    The probe costs a few seconds and briefly publishes a marker pattern, so it
    runs only when the camera set has changed since last time.
    """
    fingerprint = _device_fingerprint()
    if not refresh:
        try:
            with open(_VIRTUAL_CACHE) as handle:
                cached_print, cached_index = handle.read().split("\n", 1)
            if cached_print == fingerprint:
                return None if cached_index.strip() == "none" else int(cached_index)
        except (OSError, ValueError):
            pass

    index = probe_virtual_index()
    try:
        with open(_VIRTUAL_CACHE, "w") as handle:
            handle.write(f"{fingerprint}\n{'none' if index is None else index}")
    except OSError:
        pass
    return index
