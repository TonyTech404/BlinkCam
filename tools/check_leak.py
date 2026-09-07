"""Measure BlinkCam's memory growth over a sustained run.

This exists because BlinkCam once consumed 48 GB of RAM in about eight minutes
and wedged the machine it was running on. The cause was not in this codebase:
MediaPipe's GPU delegate never releases the CoreVideo pixel buffer behind each
input frame, so the graph retains one full-size image per call, about 3.4 MB at
1080p. Nothing in the pipeline reports it, the process just grows until macOS
runs out of application memory.

The project therefore pins mediapipe==0.10.35 and runs inference on CPU, which
is the only combination tested that does not leak. Both settings look like
things a future reader would "fix". Run this before and after touching either.

Usage:
    .venv/bin/python tools/check_leak.py                # 60s, synthetic frames
    .venv/bin/python tools/check_leak.py --seconds 300  # a call-length run
    .venv/bin/python tools/check_leak.py --camera       # from the real webcam
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blinkcam.bank import PatchBank
from blinkcam.capture import Camera, CameraConfig, default_camera_index
from blinkcam.geometry import StabilizedFace
from blinkcam.landmarks import Landmarker
from blinkcam.render import EyeRenderer, RenderConfig

# Anything above this is a leak worth investigating. The GPU path manages
# roughly 3400 KB per frame, so the gap between pass and fail is enormous;
# this threshold is deliberately loose to tolerate allocator warm-up.
BUDGET_KB_PER_FRAME = 40.0


def current_rss_mb() -> float:
    """Resident set size, not the high-water mark.

    resource.getrusage reports ru_maxrss, which never decreases and so cannot
    show memory being reclaimed. Asking ps for the live figure can.
    """
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0


def synthetic_frame(width: int, height: int) -> np.ndarray:
    """A face-free frame is fine: the leak is in handing the image over, and
    it happens whether or not a face is found."""
    path = "data/test/portrait.jpg"
    if os.path.exists(path):
        frame = cv2.imread(path)
        if frame is not None:
            return cv2.resize(frame, (width, height))
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--camera", action="store_true",
                    help="use the real webcam instead of a synthetic frame")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bank", default="data/bank.npz")
    ap.add_argument("--model", default="models/face_landmarker.task")
    args = ap.parse_args()

    import mediapipe
    print(f"mediapipe {mediapipe.__version__}   {args.width}x{args.height}   "
          f"{args.seconds:.0f}s\n")

    bank = PatchBank.load(args.bank) if os.path.exists(args.bank) else PatchBank()
    renderer = EyeRenderer(bank, RenderConfig())
    stabilizer = StabilizedFace(30.0)

    camera = None
    if args.camera:
        camera = Camera(CameraConfig(default_camera_index(), args.width,
                                     args.height, 30)).start()
    else:
        still = synthetic_frame(args.width, args.height)

    print("   t(s)   RSS(MB)    frames   KB/frame")
    samples: list[tuple[float, float, int]] = []
    frames = 0
    try:
        with Landmarker(args.model) as landmarker:
            gc.collect()
            start = time.monotonic()
            baseline = current_rss_mb()
            next_report = 10.0
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= args.seconds:
                    break

                frame = camera.read() if camera else still
                if frame is None:
                    time.sleep(0.003)
                    continue

                face = landmarker.process(frame)
                if face is not None:
                    face = stabilizer(face, time.monotonic())
                    renderer.render(frame, face, 1.0)
                frames += 1

                if elapsed >= next_report:
                    gc.collect()
                    rss = current_rss_mb()
                    per = ((rss - baseline) * 1024 / frames) if frames else 0.0
                    print(f"  {elapsed:5.0f}   {rss:8.1f}   {frames:7d}   "
                          f"{per:+8.1f}")
                    samples.append((elapsed, rss, frames))
                    next_report += 10.0
    finally:
        if camera is not None:
            camera.stop()

    if len(samples) < 2:
        print("\nRun for longer; not enough samples to judge.")
        return 2

    # Compare the second half against the first, so warm-up does not dominate.
    mid = len(samples) // 2
    (t0, r0, f0), (t1, r1, f1) = samples[mid], samples[-1]
    late_frames = max(f1 - f0, 1)
    late_kb = (r1 - r0) * 1024 / late_frames
    growth_per_min = (r1 - r0) / max((t1 - t0) / 60.0, 1e-6)

    print(f"\n  steady state: {late_kb:+.1f} KB/frame over the last "
          f"{late_frames} frames")
    print(f"                {growth_per_min:+.1f} MB/min")
    print(f"                projected after an hour: "
          f"{r1 + growth_per_min * 60:.0f} MB")

    if late_kb <= BUDGET_KB_PER_FRAME:
        print(f"\nPASS. Under the {BUDGET_KB_PER_FRAME:.0f} KB/frame budget.")
        return 0
    print(f"\nFAIL. Over the {BUDGET_KB_PER_FRAME:.0f} KB/frame budget.")
    print("Check the delegate and mediapipe version first: the GPU path leaks")
    print("about 3400 KB/frame and CPU on mediapipe 1.0.x crashes outright.")
    print("See the compatibility matrix in blinkcam/landmarks.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
