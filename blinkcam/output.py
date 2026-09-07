"""Virtual camera output via the OBS CoreMediaIO sink stream.

Building our own camera extension would need a paid Apple Developer account
for the system-extension entitlement (a free Apple ID is refused that
capability), full Xcode, System Integrity Protection relaxed, and developer
mode re-enabled after every reboot.

None of that is necessary. OBS Studio ships a signed and notarized camera
extension whose sink stream accepts frames from any process: its
authorizedToStartStream returns true unconditionally and its connect(to:) is
empty. pyvirtualcam has driven that sink since 0.14. So we push frames in with
no entitlement, no signature, no SIP change, and no OBS process running.

The one-time cost is that the user must launch OBS once to trigger the
extension installation and approve it. After that OBS is a lazily-launched
system daemon and its app never needs to run again.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np

# Measured on macOS 26.4 with OBS 32.2.2: after a client disconnects, the sink
# needs a few seconds of quiet before it will accept a new one. Reconnect
# sooner and pyvirtualcam opens without error while consumers keep seeing OBS's
# "no signal" placeholder. That silent failure would be baffling mid-meeting,
# so a restart waits the cooldown out instead.
SINK_COOLDOWN = 5.0

# Frames take roughly two to three seconds to start reaching consumers after
# the pusher starts, even once the sink has accepted us.
SINK_WARMUP = 3.0

_STATE = os.path.join(tempfile.gettempdir(), "blinkcam-sink-closed")

SETUP_HELP = """
One-time virtual camera setup, in order:

  1. brew install --cask obs
  2. Launch OBS.
  3. Click 'Start Virtual Camera' (bottom right), then 'Stop Virtual Camera'.
  4. If macOS asks for approval, allow it. To check or change it later:
     System Settings > General > Login Items & Extensions > Camera
     Extensions. Some macOS 26 builds instead list it under Privacy &
     Security > Extensions as a 'Media Extension', so look in both.
  5. Click 'Stop Virtual Camera', then QUIT OBS. It holds the single
     camera instance while running, so BlinkCam cannot use the sink
     until OBS exits.

You may need to restart the Mac once after step 4.
"""


def obs_is_running() -> bool:
    """OBS provides a single camera instance, so a running OBS competes with us
    for the sink. Better to fail loudly than to show a frozen image."""
    try:
        return subprocess.run(["pgrep", "-x", "OBS"], capture_output=True,
                              timeout=5).returncode == 0
    except Exception:
        return False


class VirtualCamera:
    """Pushes BGR frames to the virtual camera.

    The output resolution is a fixed contract, deliberately NOT inherited from
    whatever the capture source happens to be. Publishing at an unusual size
    breaks the sink: at 820x1024 the extension logs "Pixel buffer size
    mismatch" for every frame and consumers receive solid green. Standard sizes
    such as 1920x1080 and 1280x720 both work. Meeting apps also read the device
    format once, so a stable size matters beyond correctness.

    Frames arrive as BGR because that is what OpenCV produces; pyvirtualcam
    wants RGB, so the conversion happens here rather than being every caller's
    problem. Frames of a different shape are letterboxed rather than stretched,
    so a portrait or oddly-sized source keeps its proportions.
    """

    # Sizes confirmed working against OBS 32.2.2's sink on macOS 26.4.
    SUPPORTED = ((1920, 1080), (1280, 720), (640, 480))

    def __init__(self, width: int = 1920, height: int = 1080,
                 fps: int = 30) -> None:
        if (width, height) not in self.SUPPORTED:
            print(f"Virtual camera: {width}x{height} is not a size the OBS "
                  f"sink handles reliably; publishing 1920x1080 instead and "
                  f"letterboxing into it.")
            width, height = 1920, 1080
        self.width, self.height, self.fps = width, height, fps
        self._cam = None
        self.device = ""

    @staticmethod
    def _wait_for_cooldown(announce: bool = True) -> None:
        """Hold off if a previous session closed very recently."""
        try:
            closed_at = os.path.getmtime(_STATE)
        except OSError:
            return
        remaining = SINK_COOLDOWN - (time.time() - closed_at)
        if remaining <= 0:
            return
        if announce:
            print(f"Virtual camera was in use {SINK_COOLDOWN - remaining:.0f}s "
                  f"ago; waiting {remaining:.0f}s for the sink to settle.")
        time.sleep(remaining)

    def start(self) -> "VirtualCamera":
        if obs_is_running():
            raise RuntimeError(
                "OBS is running and will compete for the virtual camera sink. "
                "Quit OBS and try again.")
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError(
                "pyvirtualcam is not installed; run "
                "pip install -r requirements.txt") from exc

        self._wait_for_cooldown()

        try:
            self._cam = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.RGB)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not open the virtual camera: {exc}\n{SETUP_HELP}") from exc

        self.device = self._cam.device
        return self

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None
            # Record the close so a quick restart knows to wait.
            try:
                with open(_STATE, "w") as handle:
                    handle.write(str(time.time()))
            except OSError:
                pass

    def __enter__(self) -> "VirtualCamera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def fit(self, bgr: np.ndarray) -> np.ndarray:
        """Letterbox a frame to the output size, preserving aspect ratio."""
        import cv2

        h, w = bgr.shape[:2]
        if h == self.height and w == self.width:
            return bgr

        scale = min(self.width / w, self.height / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        x = (self.width - new_w) // 2
        y = (self.height - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def send(self, bgr: np.ndarray) -> None:
        """Send one BGR frame, letterboxed to the output size if needed."""
        if self._cam is None:
            raise RuntimeError("virtual camera not started")
        self._cam.send(np.ascontiguousarray(self.fit(bgr)[:, :, ::-1]))

    def sleep_until_next_frame(self) -> None:
        if self._cam is not None:
            self._cam.sleep_until_next_frame()
