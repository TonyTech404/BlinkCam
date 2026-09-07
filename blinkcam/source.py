"""Frame sources: the live camera, or a file.

A file source makes the pipeline verifiable without anyone sitting in front of
the webcam, and it is what the stress suite needs: replaying the same recorded
clip every build is the only way to compare quality across changes. Tuning the
effect against a fixed clip also beats tuning against a live face, because the
input stops moving between attempts.

All sources expose the same two methods the main loop uses, so it does not care
which it has.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .capture import Camera, CameraConfig


class FileSource:
    """Replays an image or a video file at a fixed rate.

    An image is served repeatedly, which is useful for measuring render cost
    and for eyeballing a single frame. A video loops.
    """

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, path: str, fps: int = 30, loop: bool = True) -> None:
        if not os.path.exists(path):
            raise RuntimeError(f"no such file: {path}")
        self.path = path
        self.fps = max(1, fps)
        self.loop = loop
        self._interval = 1.0 / self.fps
        self._next = 0.0
        self._capture: cv2.VideoCapture | None = None
        self._still: np.ndarray | None = None
        self._exhausted = False

        if os.path.splitext(path)[1].lower() in self.IMAGE_SUFFIXES:
            still = cv2.imread(path, cv2.IMREAD_COLOR)
            if still is None:
                raise RuntimeError(f"could not decode image: {path}")
            self._still = still
        else:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {path}")
            self._capture = capture

    def start(self) -> "FileSource":
        self._next = time.monotonic()
        return self

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FileSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- the frame-source interface ------------------------------------

    def read(self) -> np.ndarray | None:
        return self._still if self._still is not None else self._read_video()

    def read_new(self) -> np.ndarray | None:
        """Pace playback to the declared rate, so timings mean something."""
        if self._exhausted:
            return None
        now = time.monotonic()
        if now < self._next:
            return None
        self._next = max(now, self._next + self._interval)
        return self.read()

    def _read_video(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            if not self.loop:
                self._exhausted = True
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                self._exhausted = True
                return None
        return frame

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def dropped(self) -> int:
        return 0

    def actual(self) -> dict[str, float]:
        frame = self.read()
        if frame is None:
            return {}
        return {"width": float(frame.shape[1]), "height": float(frame.shape[0]),
                "fps": float(self.fps)}

    def measure_fps(self, seconds: float = 1.0) -> float:
        return float(self.fps)

    def lock_exposure_and_white_balance(self) -> dict[str, bool]:
        return {}


def open_source(path: str | None, config: CameraConfig):
    """Camera when path is None, otherwise the file."""
    if path:
        return FileSource(path, fps=config.fps)
    return Camera(config)
