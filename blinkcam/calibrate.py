"""Guided calibration: harvest the user's real closed eyelids.

The whole quality argument for BlinkCam rests on this step. Every published
system in this space draws exemplars from a different photograph, taken in
different light with a different camera, and then spends most of its
complexity correcting for that mismatch. We take exemplars from the same
person, same camera, same room, minutes earlier, so identity, sensor and gross
lighting mismatch are eliminated at the source instead of compensated for.

The recording is a scripted pose and expression sweep, because natural blinking
is a poor harvesting strategy: it yields only a few fully-closed frames per
blink and they all sit at whatever pose the user happened to hold.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .bank import Descriptor, PatchBank
from .geometry import extract_patch
from .landmarks import Landmarker
from .transition import BlinkDetector


@dataclass
class Step:
    seconds: float
    instruction: str
    detail: str
    spoken: str = ""

    def announce(self) -> str:
        return self.spoken or self.instruction


def speak(text: str) -> None:
    """Say an instruction out loud, without blocking.

    This is not a nicety. The whole recording depends on the user's eyes being
    shut, and printing instructions on screen requires them to open their eyes
    to read the next one. The first version did exactly that and harvested
    usable frames from under a fifth of the recording. macOS ships `say`, so
    the prompts are spoken and the user never needs to look.
    """
    try:
        subprocess.Popen(["say", "-r", "175", text],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass  # no speech available; the on-screen text still works


# Expression is a first-class index dimension, not an afterthought: smiling,
# talking and brow-raising all reshape the periorbital region and change the
# brow-to-lash distance. A bank indexed on pose alone retrieves patches whose
# lid shape does not match the live face.
SCRIPT: list[Step] = [
    Step(4.0, "Close your eyes and hold still", "facing the camera",
         "Close your eyes now, and keep them closed until I say open. "
         "Face the camera and hold still."),
    Step(6.0, "Slowly turn your head left, then right", "eyes stay closed",
         "Now slowly turn your head to the left, and then all the way to "
         "the right."),
    Step(5.0, "Slowly look up, then down", "eyes stay closed",
         "Keeping your eyes closed, tip your head up, then down."),
    Step(5.0, "Slowly tilt your head side to side", "eyes stay closed",
         "Now tilt your head towards one shoulder, then the other."),
    Step(4.0, "Smile, eyes still closed", "hold the smile",
         "Now smile, and hold the smile."),
    Step(4.0, "Raise your eyebrows, eyes still closed", "hold it",
         "Stop smiling, and raise your eyebrows. Hold it."),
    Step(4.0, "Turn left again, slowly", "eyes stay closed",
         "Relax your face, and slowly turn your head left and right once "
         "more."),
    Step(3.0, "Relax, facing forward", "nearly done",
         "Almost done. Relax, and face the camera."),
]

TOTAL_SECONDS = sum(s.seconds for s in SCRIPT)


def _overlay(frame: np.ndarray, step: Step, remaining: float,
             elapsed_total: float, harvested: int, eyes_closed: bool
             ) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    panel = vis[: 150].copy()
    vis[: 150] = (panel * 0.35).astype(np.uint8)

    def text(s: str, y: int, scale: float, colour: tuple[int, int, int]) -> None:
        cv2.putText(vis, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 2, cv2.LINE_AA)

    text(step.instruction, 52, 1.0, (255, 255, 255))
    text(step.detail, 88, 0.6, (180, 220, 255))
    text(f"{remaining:.0f}s   captured {harvested}", 124, 0.6, (170, 170, 170))

    # Progress bar across the top.
    done = int(w * min(elapsed_total / TOTAL_SECONDS, 1.0))
    cv2.rectangle(vis, (0, 0), (done, 8), (80, 220, 120), -1)

    if not eyes_closed:
        warn = "eyes appear OPEN - not capturing"
        cv2.putText(vis, warn, (24, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, warn, (24, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (60, 60, 255), 2, cv2.LINE_AA)
    return vis


def _should_keep(bank_cells: dict, descriptor: Descriptor,
                 per_cell: int = 5) -> bool:
    """Cap patches per pose cell.

    Without this the bank fills up with hundreds of near-identical frontal
    patches from the parts of the script where the head is still, and the
    genuinely useful off-axis poses are a rounding error in retrieval.
    """
    key = (int(round(descriptor.yaw / 8.0)), int(round(descriptor.pitch / 8.0)),
           int(round(descriptor.brow_height * 8.0)),
           int(round(descriptor.cheek_squint * 4.0)))
    count = bank_cells.get(key, 0)
    if count >= per_cell:
        return False
    bank_cells[key] = count + 1
    return True


def run_calibration(camera, landmarker: Landmarker, *, preview: bool = True,
                    window: str = "BlinkCam calibration",
                    per_cell: int = 5) -> PatchBank:
    """Walk the user through the script and return a populated patch bank.

    Only frames where the eyes are genuinely closed are harvested, verified
    against the learned blink blendshape rather than trusting the user to
    follow instructions.
    """
    bank = PatchBank()
    cells: dict = {}
    detector = BlinkDetector()

    start = time.monotonic()
    step_index = 0
    step_start = start
    skipped_open = 0
    frames = 0
    peak_blink = 0.0
    spoken_for = -1

    if preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while step_index < len(SCRIPT):
        frame = camera.read_new()
        if frame is None:
            time.sleep(0.002)
            continue

        now = time.monotonic()
        frames += 1
        step = SCRIPT[step_index]
        remaining = step.seconds - (now - step_start)

        if step_index != spoken_for:
            speak(step.announce())
            spoken_for = step_index

        face = landmarker.process(frame)
        closed = detector.update(face, now)
        if face is not None:
            peak_blink = max(peak_blink, face.blink)

        if face is not None and closed:
            for eye in face.eyes():
                descriptor = Descriptor.build(face, eye)
                if _should_keep(cells, descriptor, per_cell):
                    bank.add(extract_patch(frame, eye), descriptor, eye.side)
        elif face is not None:
            skipped_open += 1

        if preview:
            vis = _overlay(frame, step, max(remaining, 0.0), now - start,
                           len(bank), closed)
            cv2.imshow(window, vis)
            if (cv2.waitKey(1) & 0xFF) == 27:  # Esc aborts
                break

        if remaining <= 0.0:
            step_index += 1
            step_start = now

    if preview:
        cv2.destroyWindow(window)
    speak("Done. You can open your eyes.")

    captured = frames - skipped_open
    print(f"Calibration: {len(bank)} patches from {frames} frames "
          f"({captured} with eyes closed, {skipped_open} skipped).")
    if frames and captured < frames * 0.5:
        print(f"WARNING: only {100 * captured / frames:.0f}% of frames had your "
              "eyes closed, so the bank is thin.")
        if peak_blink < 0.6:
            # Distinguish "did not close them" from "we failed to notice".
            print(f"         The blink signal peaked at {peak_blink:.2f}, below "
                  f"the {0.55:.2f} threshold, so detection may be the problem "
                  "rather than you.")
            print("         Try again with more light on your face.")
        else:
            print("         The prompts are spoken aloud, so keep your eyes "
                  "shut the whole way through and just listen.")
    return bank


def report_coverage(bank: PatchBank) -> str:
    """Text heat map of pose coverage. Shows exactly where the bank has holes,
    which is the most actionable diagnostic in the whole system."""
    cells = bank.coverage(yaw_step=10.0, pitch_step=10.0)
    if not cells:
        return "empty bank"

    yaws = sorted({k[0] for k in cells})
    pitches = sorted({k[1] for k in cells})
    lines = ["", "pose coverage (rows = pitch, cols = yaw, in 10 degree cells)",
             "        " + "".join(f"{y * 10:+5d}" for y in yaws)]
    for p in pitches:
        row = "".join(f"{cells.get((y, p), 0):5d}" for y in yaws)
        lines.append(f"  {p * 10:+4d}  {row}")
    lines.append(f"  total {len(bank)} patches across {len(cells)} cells")
    return "\n".join(lines)
