"""Verify the head-pose sign conventions against real head movement.

This exists because the signs of yaw, pitch and roll coming out of MediaPipe's
facial transformation matrix are a convention, not a guarantee, and one of them
has a real functional consequence: the renderer decides which eye is
self-occluded from the sign of yaw. Get it backwards and the effect fades the
eye you can see while continuing to paint the one that is hidden.

Nothing else in the project can catch that. It needs a person turning their
head, so it lives in its own tool rather than the test suite.

Usage:
    .venv/bin/python tools/check_pose.py
    .venv/bin/python tools/check_pose.py --source clip.mov   # from a recording
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blinkcam.capture import CameraConfig
from blinkcam.landmarks import Landmarker
from blinkcam.render import EyeRenderer, RenderConfig
from blinkcam.bank import PatchBank
from blinkcam.source import open_source


@dataclass
class Probe:
    """One pose to hold, and the sign we expect to see while holding it."""

    axis: str
    instruction: str
    expected_sign: int
    documented: str
    samples: list[float] = field(default_factory=list)

    def extreme(self) -> float:
        """The most extreme value seen, which is the pose actually held."""
        if not self.samples:
            return 0.0
        arr = np.array(self.samples)
        return float(arr[np.argmax(np.abs(arr))])

    def verdict(self) -> tuple[bool, str]:
        value = self.extreme()
        if abs(value) < 8.0:
            return False, (f"inconclusive, only reached {value:+.1f} degrees. "
                           "Move further and re-run.")
        actual = 1 if value > 0 else -1
        if actual == self.expected_sign:
            return True, f"correct, reached {value:+.1f} degrees"
        return False, (f"INVERTED: reached {value:+.1f} degrees while the code "
                       f"documents {self.documented}")


PROBES = [
    Probe("yaw", "Turn your head to YOUR LEFT and hold", +1,
          "positive turning to the subject's left"),
    Probe("pitch", "Tip your head so you look UP and hold", +1,
          "positive looking up"),
    Probe("roll", "Tilt your head toward YOUR RIGHT shoulder and hold", +1,
          "positive tilting to the subject's right"),
]


def _draw(frame: np.ndarray, probe: Probe, remaining: float,
          value: float) -> np.ndarray:
    vis = frame.copy()
    vis[:130] = (vis[:130] * 0.3).astype(np.uint8)

    def text(s: str, y: int, scale: float, colour) -> None:
        cv2.putText(vis, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 2, cv2.LINE_AA)

    text(probe.instruction, 52, 0.95, (255, 255, 255))
    text(f"{probe.axis} = {value:+6.1f} deg     peak {probe.extreme():+6.1f}"
         f"     {remaining:.0f}s", 100, 0.7, (170, 220, 255))
    return vis


def run_probe(source, landmarker: Landmarker, probe: Probe, seconds: float,
              preview: bool, window: str, t0: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        frame = source.read_new()
        if frame is None:
            if getattr(source, "exhausted", False):
                return
            time.sleep(0.002)
            continue
        face = landmarker.process(frame)
        value = 0.0
        if face is not None:
            value = {"yaw": face.yaw, "pitch": face.pitch,
                     "roll": face.roll}[probe.axis]
            probe.samples.append(value)
        if preview:
            cv2.imshow(window, _draw(frame, probe,
                                     seconds - (time.monotonic() - start),
                                     value))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                raise KeyboardInterrupt


def check_eye_sides(landmarker: Landmarker, source) -> str:
    """Confirm the subject's right eye lands on the image left.

    Everything in the project is computed in unmirrored source coordinates, so
    this is the assumption the patch canonicalisation rests on.
    """
    for _ in range(200):
        frame = source.read_new()
        if frame is None:
            time.sleep(0.002)
            continue
        face = landmarker.process(frame)
        if face is None:
            continue
        if face.right.center[0] < face.left.center[0]:
            return "correct, subject's right eye is on the image left"
        return ("INVERTED: subject's right eye is on the image right. The "
                "frame is mirrored; patch canonicalisation assumes it is not.")
    return "inconclusive, no face detected"


def check_pose_gate(yaw_sign: int) -> str:
    """Given the measured yaw sign, does the gate fade the occluded eye?"""
    renderer = EyeRenderer(PatchBank(), RenderConfig())
    # Turn hard toward the subject's left. Their right eye rotates away from
    # the camera, so it is the one that must fade.
    yaw = 60.0 * yaw_sign
    dummy = _dummy_face(yaw)
    right = renderer._pose_gate(dummy, dummy.right)
    left = renderer._pose_gate(dummy, dummy.left)
    if right < left:
        return (f"correct, at yaw {yaw:+.0f} the subject's right eye fades "
                f"({right:.2f}) and the left stays ({left:.2f})")
    return (f"WRONG EYE: at yaw {yaw:+.0f} the right eye gate is {right:.2f} "
            f"and the left is {left:.2f}. The visible eye is being faded and "
            f"the occluded one painted. Flip the comparison in "
            f"EyeRenderer._pose_gate.")


def _dummy_face(yaw: float):
    from blinkcam.landmarks import EyeGeometry, FaceFrame

    def eye(side: str, cx: float) -> EyeGeometry:
        ring = np.array([[cx - 20, 100], [cx, 106], [cx + 20, 100],
                         [cx, 94]], np.float32)
        return EyeGeometry(
            side=side, ring=ring, upper_lid=ring[3:4], lower_lid=ring[1:2],
            lateral_corner=np.array([cx - 20, 100], np.float32),
            medial_corner=np.array([cx + 20, 100], np.float32),
            iris_center=np.array([cx, 100], np.float32), iris_radius=6.0,
            brow=ring, ear=0.3, blink=0.0)

    # Unmirrored: subject's right eye sits at the smaller image x.
    return FaceFrame(right=eye("right", 80.0), left=eye("left", 160.0),
                     yaw=yaw, pitch=0.0, roll=0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="video file instead of the live camera")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="hold time per pose (default 4)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    args = ap.parse_args()

    window = "BlinkCam pose check"
    print(__doc__.split("Usage:")[0].strip())
    print(f"\nThree poses, {args.seconds:.0f} seconds each. Esc quits.\n")

    config = CameraConfig(index=args.camera, fps=30)
    try:
        with open_source(args.source, config) as source, \
                Landmarker(args.model) as landmarker:
            if args.preview:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            print("Checking eye sides...")
            sides = check_eye_sides(landmarker, source)
            print(f"  eye sides: {sides}\n")

            t0 = time.monotonic()
            for probe in PROBES:
                print(f"  {probe.instruction}")
                run_probe(source, landmarker, probe, args.seconds,
                          args.preview, window, t0)
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nAborted.")
        return 130

    print("\nResults\n")
    all_ok = True
    for probe in PROBES:
        ok, message = probe.verdict()
        all_ok &= ok
        print(f"  {probe.axis:6s} {'OK  ' if ok else 'FAIL'}  {message}")

    yaw = PROBES[0]
    if abs(yaw.extreme()) >= 8.0:
        sign = 1 if yaw.extreme() > 0 else -1
        gate = check_pose_gate(sign)
        ok = gate.startswith("correct")
        all_ok &= ok
        print(f"\n  pose gate {'OK  ' if ok else 'FAIL'}  {gate}")
    else:
        all_ok = False
        print("\n  pose gate  SKIPPED, yaw was inconclusive")

    print("\nAll pose conventions verified."
          if all_ok else
          "\nSomething is off. The messages above say exactly what to change.")
    print("\nWhatever the result, record it in the docstring of "
          "_euler_from_matrix in blinkcam/landmarks.py.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
