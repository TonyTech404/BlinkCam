"""M0 gate: prove we can push frames into the OBS virtual camera.

Run this BEFORE writing any vision code. It pushes a moving test pattern to the
virtual camera so we can confirm the device shows up and updates in QuickTime,
Chrome/Meet, Zoom and Teams.

The pattern deliberately includes a sweeping bar and a frame counter so you can
eyeball latency by putting this window next to the consuming app's self-view.

Usage:
    .venv/bin/python tools/smoke_virtualcam.py [--seconds 120]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

WIDTH, HEIGHT, FPS = 1280, 720, 30


def obs_is_running() -> bool:
    """OBS provides a single camera instance. If OBS is running and using its
    own virtual camera, our frames will not get through. Fail loudly instead of
    showing the user a mysteriously frozen image."""
    try:
        out = subprocess.run(
            ["pgrep", "-x", "OBS"], capture_output=True, text=True, timeout=5
        )
        return out.returncode == 0
    except Exception:
        return False


def make_frame(buf: np.ndarray, i: int) -> np.ndarray:
    """Colour bars, a sweeping vertical bar, and a coarse frame counter."""
    buf[:] = 0

    # Static colour bars across the top third, so a frozen feed is obvious.
    bars = [
        (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
        (255, 0, 255), (255, 0, 0), (0, 0, 255), (30, 30, 30),
    ]
    bw = WIDTH // len(bars)
    for k, colour in enumerate(bars):
        buf[: HEIGHT // 3, k * bw : (k + 1) * bw] = colour

    # Sweeping bar: a 3 second cycle. Any stutter or lag is visible here.
    x = int((i % (FPS * 3)) / (FPS * 3) * WIDTH)
    buf[HEIGHT // 3 :, max(0, x - 8) : x + 8] = (255, 140, 0)

    # Frame counter as a row of blocks, 1 block per 10 frames, wrapping at 100.
    blocks = (i // 10) % 100
    for b in range(blocks):
        y0 = HEIGHT - 60 + (b // 50) * 25
        x0 = 10 + (b % 50) * 25
        buf[y0 : y0 + 20, x0 : x0 + 20] = (0, 255, 0)

    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    if obs_is_running():
        print(
            "ERROR: OBS is running. OBS provides a single camera instance, so it\n"
            "       will compete with us for the sink. Quit OBS and re-run.",
            file=sys.stderr,
        )
        return 2

    try:
        import pyvirtualcam
    except ImportError:
        print("ERROR: pyvirtualcam not installed. pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    try:
        cam_ctx = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS)
    except RuntimeError as exc:
        print(
            f"ERROR: could not open the virtual camera: {exc}\n\n"
            "One-time setup, in order:\n"
            "  1. brew install --cask obs\n"
            "  2. Launch OBS.\n"
            "  3. Click 'Start Virtual Camera' (bottom right), then 'Stop Virtual Camera'.\n"
            "  4. Approve the extension in System Settings > Privacy & Security >\n"
            "     Extensions, where it appears as a 'Media Extension'. On macOS 26\n"
            "     this moved out of General > Login Items & Extensions.\n"
            "  5. Quit OBS. It never needs to run again.\n",
            file=sys.stderr,
        )
        return 1

    with cam_ctx as cam:
        print(f"Pushing to virtual camera: {cam.device}")
        print(f"{WIDTH}x{HEIGHT} @ {FPS}fps for {args.seconds:.0f}s. Ctrl-C to stop.\n")
        print("Now open QuickTime > File > New Movie Recording and pick this device.")
        print("Then try Chrome/Meet, Zoom and Teams. Note which ones work.\n")

        buf = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        start, i, last_report = time.monotonic(), 0, 0.0
        try:
            while (elapsed := time.monotonic() - start) < args.seconds:
                cam.send(make_frame(buf, i))
                cam.sleep_until_next_frame()
                i += 1
                if elapsed - last_report >= 5.0:
                    print(f"  {elapsed:5.1f}s  {i} frames  {i / elapsed:5.1f} fps sent")
                    last_report = elapsed
        except KeyboardInterrupt:
            print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
