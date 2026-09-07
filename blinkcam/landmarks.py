"""MediaPipe Face Landmarker wrapper producing per-eye geometry.

The delegate and pixel format here are not stylistic choices. Measured on
macOS 26.4 / arm64, the working combinations differ by MediaPipe version, and
only one of them avoids a severe memory leak:

    version   delegate  format   result
    1.0.1     CPU       any      crashes, "graph_service.h: Check failed:
                                 service_ Service is unavailable"
    1.0.1     GPU       SRGB     crashes, "unsupported ImageFrame format: 1"
    1.0.1     GPU       SRGBA    works, LEAKS 3.4 MB per frame
    0.10.35   GPU       SRGB     crashes, same format error
    0.10.35   GPU       SRGBA    works, LEAKS 3.4 MB per frame
    0.10.35   CPU       SRGB     works, no leak            <-- what we use
    0.10.35   CPU       SRGBA    works, no leak

The leak is in the GPU path, not in the version: the CoreVideo pixel buffer
backing each input frame is never released, so the graph retains one full-size
image per call. On 1.0.1 there is NO non-leaking configuration, because CPU
crashes outright there. That is why this project pins mediapipe==0.10.35 and
runs on CPU. It cost 48 GB of RAM and a wedged machine to establish; do not
"modernise" either setting without re-running tools/check_leak.py.

CPU inference costs about 3.8 ms per frame against 2.2 ms on GPU. Irrelevant
here: the webcam delivers 20 fps, a 50 ms budget.

Landmark index conventions were verified by measurement rather than trusted
from documentation, because published sources disagree about which iris index
belongs to which eye. On MediaPipe's own naming, which is ANATOMICAL (the
subject's own left and right, not the image's), the verified mapping is:

    subject RIGHT eye  ring corners 33 (lateral) / 133 (medial), iris center 468
    subject LEFT  eye  ring corners 263 (lateral) / 362 (medial), iris center 473

In an unmirrored frame the subject's right eye appears on the image LEFT. All
geometry in BlinkCam is computed in unmirrored source coordinates; mirroring is
a display-time concern only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Verified landmark indices. Subject-anatomical naming, as MediaPipe uses.
# ---------------------------------------------------------------------------

# Lower lid, lateral corner to medial corner.
LOWER_LID_RIGHT = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOWER_LID_LEFT = [263, 249, 390, 373, 374, 380, 381, 382, 362]

# Upper lid, lateral corner to medial corner.
UPPER_LID_RIGHT = [246, 161, 160, 159, 158, 157, 173]
UPPER_LID_LEFT = [466, 388, 387, 386, 385, 384, 398]

# Closed 16-point ring: lower lid lateral to medial, then upper lid back.
EYE_RING_RIGHT = LOWER_LID_RIGHT + UPPER_LID_RIGHT[::-1]
EYE_RING_LEFT = LOWER_LID_LEFT + UPPER_LID_LEFT[::-1]

CORNER_LATERAL_RIGHT, CORNER_MEDIAL_RIGHT = 33, 133
CORNER_LATERAL_LEFT, CORNER_MEDIAL_LEFT = 263, 362

# Lid mid-points, used for the eye aspect ratio.
UPPER_MID_RIGHT, LOWER_MID_RIGHT = 159, 145
UPPER_MID_LEFT, LOWER_MID_LEFT = 386, 374

IRIS_CENTER_RIGHT, IRIS_RING_RIGHT = 468, [469, 470, 471, 472]
IRIS_CENTER_LEFT, IRIS_RING_LEFT = 473, [474, 475, 476, 477]

BROW_RIGHT = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_LEFT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

NOSE_TIP = 1

# Blendshape indices confirmed by measurement.
BLENDSHAPE_EYE_BLINK_LEFT = 9
BLENDSHAPE_EYE_BLINK_RIGHT = 10


@dataclass
class EyeGeometry:
    """One eye, in unmirrored source-image pixel coordinates."""

    side: str  # "right" or "left", anatomical
    ring: np.ndarray  # (16, 2) closed contour
    upper_lid: np.ndarray  # (7, 2) lateral to medial
    lower_lid: np.ndarray  # (9, 2) lateral to medial
    lateral_corner: np.ndarray  # (2,) outer, barely moves during closure
    medial_corner: np.ndarray  # (2,) inner, at the tear duct
    iris_center: np.ndarray  # (2,)
    iris_radius: float
    brow: np.ndarray  # (10, 2)
    ear: float  # eye aspect ratio; ~0.3 open, <0.2 closed
    blink: float  # learned blendshape blink scalar, 0 open to 1 closed

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.lateral_corner - self.medial_corner))

    @property
    def height(self) -> float:
        """Palpebral height, the open aperture's vertical extent."""
        return float(self.lower_lid[:, 1].max() - self.upper_lid[:, 1].min())

    @property
    def center(self) -> np.ndarray:
        return (self.lateral_corner + self.medial_corner) / 2.0

    @property
    def brow_height(self) -> float:
        """Brow height above the canthal line, normalised by eye width.

        Measured against the eye CORNERS, not the upper lid. The corners barely
        move when the eye opens or closes, whereas the upper lid moves by the
        whole aperture. An earlier version measured brow to upper lid and was
        therefore incommensurable between a live query (eyes open) and a
        harvested reference (eyes closed): it differed by a systematic 0.20,
        spent 1.38 of the 3.2 distance budget before any real pose difference,
        and swung far enough during a blink to fade the effect out completely.

        This is an expression proxy, so it must track brow raise and lowering
        while ignoring lid state entirely.
        """
        w = self.width
        if not w:
            return 0.0
        return float(self.center[1] - self.brow[:, 1].mean()) / w


@dataclass
class FaceFrame:
    """Everything the renderer needs from one video frame."""

    right: EyeGeometry
    left: EyeGeometry
    yaw: float  # degrees, positive turning to the subject's left
    pitch: float  # degrees, positive looking up
    roll: float  # degrees, positive head tilting to the subject's right
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmarks: np.ndarray | None = None  # (478, 2) full mesh, source pixels

    @property
    def blink(self) -> float:
        return max(self.right.blink, self.left.blink)

    @property
    def ear(self) -> float:
        return (self.right.ear + self.left.ear) / 2.0

    def eyes(self) -> tuple[EyeGeometry, EyeGeometry]:
        return (self.right, self.left)


def _eye_aspect_ratio(pts: np.ndarray, upper: int, lower: int,
                      lateral: int, medial: int) -> float:
    """Soukupova and Cech eye aspect ratio: vertical over horizontal extent."""
    horizontal = np.linalg.norm(pts[lateral] - pts[medial])
    if horizontal < 1e-6:
        return 0.0
    return float(np.linalg.norm(pts[upper] - pts[lower]) / horizontal)


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch and roll in degrees from MediaPipe's 4x4 face transform.

    The matrix carries scale, so rotation columns are normalised first.

    UNVERIFIED: the intended convention is that yaw is positive when the head
    turns to the subject's left, but this has NOT been confirmed against real
    head movement. The sign is a convention of MediaPipe's matrix, not a
    guarantee, and it matters: EyeRenderer._pose_gate decides which eye is
    self-occluded from the sign of yaw, so an inverted sign fades the visible
    eye and keeps painting the hidden one.

    Run tools/check_pose.py to settle it, then replace this notice with the
    measured result.
    """
    r = np.array(m[:3, :3], dtype=np.float64)
    for c in range(3):
        n = np.linalg.norm(r[:, c])
        if n > 1e-9:
            r[:, c] /= n

    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal-locked, looking straight up or down
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _roll_from_landmarks(pts: np.ndarray) -> float:
    """Roll straight from the interocular line. More reliable than the matrix
    and used to cross-check it."""
    v = pts[CORNER_LATERAL_LEFT] - pts[CORNER_LATERAL_RIGHT]
    return math.degrees(math.atan2(v[1], v[0]))


class Landmarker:
    """Stateful per-video-stream face landmarker.

    Uses RunningMode.VIDEO with num_faces=1, which enables MediaPipe's internal
    tracking so the detector does not run on every frame. The tracking and
    smoothing path only applies when num_faces is 1, so do not raise it.
    """

    def __init__(self, model_path: str = "models/face_landmarker.task") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(
                model_asset_path=model_path,
                # CPU, deliberately, and only viable on mediapipe 0.10.x.
                # The GPU path leaks a full frame per call. See the module
                # docstring for the full compatibility matrix.
                delegate=mpp.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._closed = False
        self._origin = time.monotonic()
        self._last_timestamp = -1

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> "Landmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _next_timestamp(self, supplied: int | None) -> int:
        """A strictly increasing millisecond stamp for MediaPipe's video mode."""
        if supplied is None:
            supplied = int((time.monotonic() - self._origin) * 1000)
        stamp = max(int(supplied), self._last_timestamp + 1)
        self._last_timestamp = stamp
        return stamp

    def process(self, bgr: np.ndarray,
                timestamp_ms: int | None = None) -> FaceFrame | None:
        """Detect on one BGR frame. Returns None when no face is present.

        Timestamps are managed here rather than by callers. MediaPipe's video
        mode raises if a timestamp does not exceed the previous one, and every
        caller inventing its own clock is a reliable way to hit that: one tool
        mixed absolute monotonic time with time-since-start and crashed part
        way through. Pass nothing and the clock is handled; pass a value and it
        is still clamped to stay strictly increasing.
        """
        timestamp_ms = self._next_timestamp(timestamp_ms)
        # SRGB (3 channel) rather than SRGBA: it works on the CPU delegate and
        # moves a third less data per frame. See the module docstring for why
        # the delegate is CPU.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        h, w = bgr.shape[:2]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )

        shapes: dict[str, float] = {}
        if result.face_blendshapes:
            shapes = {c.category_name: float(c.score)
                      for c in result.face_blendshapes[0]}

        if result.facial_transformation_matrixes:
            yaw, pitch, roll = _euler_from_matrix(
                np.array(result.facial_transformation_matrixes[0]))
        else:
            yaw = pitch = 0.0
            roll = _roll_from_landmarks(pts)

        return FaceFrame(
            right=self._eye(pts, "right", shapes),
            left=self._eye(pts, "left", shapes),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            blendshapes=shapes,
            landmarks=pts,
        )

    @staticmethod
    def _eye(pts: np.ndarray, side: str, shapes: dict[str, float]) -> EyeGeometry:
        if side == "right":
            ring, upper, lower = EYE_RING_RIGHT, UPPER_LID_RIGHT, LOWER_LID_RIGHT
            lateral, medial = CORNER_LATERAL_RIGHT, CORNER_MEDIAL_RIGHT
            upper_mid, lower_mid = UPPER_MID_RIGHT, LOWER_MID_RIGHT
            iris_c, iris_r, brow = IRIS_CENTER_RIGHT, IRIS_RING_RIGHT, BROW_RIGHT
            blink = shapes.get("eyeBlinkRight", 0.0)
        else:
            ring, upper, lower = EYE_RING_LEFT, UPPER_LID_LEFT, LOWER_LID_LEFT
            lateral, medial = CORNER_LATERAL_LEFT, CORNER_MEDIAL_LEFT
            upper_mid, lower_mid = UPPER_MID_LEFT, LOWER_MID_LEFT
            iris_c, iris_r, brow = IRIS_CENTER_LEFT, IRIS_RING_LEFT, BROW_LEFT
            blink = shapes.get("eyeBlinkLeft", 0.0)

        iris_pts = pts[iris_r]
        iris_center = pts[iris_c]
        radius = float(np.linalg.norm(iris_pts - iris_center, axis=1).mean())

        return EyeGeometry(
            side=side,
            ring=pts[ring].copy(),
            upper_lid=pts[upper].copy(),
            lower_lid=pts[lower].copy(),
            lateral_corner=pts[lateral].copy(),
            medial_corner=pts[medial].copy(),
            iris_center=iris_center.copy(),
            iris_radius=radius,
            brow=pts[brow].copy(),
            ear=_eye_aspect_ratio(pts, upper_mid, lower_mid, lateral, medial),
            blink=float(blink),
        )
