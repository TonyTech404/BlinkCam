"""BlinkCam main loop.

    blinkcam calibrate     record your closed eyes and build the patch bank
    blinkcam run           capture, render, publish to the virtual camera
    blinkcam doctor        check the environment and report what is missing

Run with --help on any subcommand for options.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from .bank import PatchBank
from .calibrate import report_coverage, run_calibration
from .capture import (Camera, CameraConfig, default_camera_index,
                      list_devices)
from .geometry import StabilizedFace
from .hotkey import DEFAULT_COMBO, PERMISSION_HELP, GlobalHotkey
from .landmarks import Landmarker
from .output import SETUP_HELP, VirtualCamera, obs_is_running
from .procedural import synthesize_closed_patch
from .render import EyeRenderer, RenderConfig, draw_debug
from .source import open_source
from .transition import EffectTransition, TransitionConfig
from .transition import State as TransitionState

DEFAULT_BANK = "data/bank.npz"
DEFAULT_MODEL = "models/face_landmarker.task"


class _NullOutput:
    """Stand-in for the virtual camera in --preview-only mode.

    Lets the effect be tuned before the one-time OBS extension approval is
    done, and lets the main loop be exercised without a virtual camera.
    """

    device = "preview only (no virtual camera)"

    def send(self, frame) -> None:
        pass

    def sleep_until_next_frame(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


class Stats:
    """Rolling per-stage timings. Latency claims should come from measurement,
    not from estimates, so the loop always instruments itself."""

    def __init__(self, window: int = 120) -> None:
        self.stages: dict[str, deque] = {}
        self.window = window

    def add(self, stage: str, ms: float) -> None:
        self.stages.setdefault(stage, deque(maxlen=self.window)).append(ms)

    def summary(self) -> str:
        parts = []
        for stage, values in self.stages.items():
            if values:
                ordered = sorted(values)
                p50 = ordered[len(ordered) // 2]
                p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
                parts.append(f"{stage} {p50:.1f}/{p95:.1f}")
        return "  ".join(parts)


def _camera_index(args: argparse.Namespace) -> int:
    """Chosen camera, or the first real one when unspecified."""
    if getattr(args, "camera", None) is not None:
        return int(args.camera)
    return default_camera_index()


def _load_bank(path: str) -> PatchBank | None:
    if not os.path.exists(path):
        return None
    try:
        return PatchBank.load(path)
    except Exception as exc:
        print(f"Could not load patch bank at {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print("BlinkCam environment check\n")

    print(f"  python            {sys.version.split()[0]}")
    try:
        import mediapipe
        print(f"  mediapipe         {mediapipe.__version__}")
    except Exception as exc:
        print(f"  mediapipe         MISSING ({exc})")
        ok = False
    try:
        import pyvirtualcam
        print(f"  pyvirtualcam      {pyvirtualcam.__version__}")
    except Exception as exc:
        print(f"  pyvirtualcam      MISSING ({exc})")
        ok = False
    print(f"  opencv            {cv2.__version__}")

    print(f"\n  model             "
          f"{'found' if os.path.exists(args.model) else 'MISSING ' + args.model}")
    if not os.path.exists(args.model):
        ok = False
        print("                    scripts/setup.sh downloads it")

    bank = _load_bank(args.bank)
    if bank is None:
        print(f"  patch bank        none at {args.bank}")
        print("                    run: blinkcam calibrate")
    else:
        print(f"  patch bank        {len(bank)} patches")

    print("\n  cameras")
    devices = list_devices()
    if not devices:
        print("                    NONE VISIBLE")
        print("                    grant Camera access in System Settings >")
        print("                    Privacy & Security > Camera, then restart"
              " your terminal")
        ok = False

    landmarker = None
    try:
        if os.path.exists(args.model) and devices:
            landmarker = Landmarker(args.model)
    except Exception:
        landmarker = None

    chosen = None
    for device in devices:
        tag = "VIRTUAL, not usable as input" if device.is_virtual else ""
        if device.is_virtual:
            print(f"                    [{device.index}] {device.name}  {tag}")
            continue
        try:
            with Camera(CameraConfig(index=device.index)) as camera:
                measured = camera.measure_fps(1.2)
                frame = camera.read()
                luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                face = (landmarker is not None
                        and landmarker.process(frame) is not None)
                print(f"                    [{device.index}] {device.name}  "
                      f"{frame.shape[1]}x{frame.shape[0]}  {measured:4.1f} fps  "
                      f"brightness {luma:3.0f}  "
                      f"{'face visible' if face else 'no face right now'}")
                if chosen is None:
                    chosen = (device, measured, luma)
        except Exception as exc:
            print(f"                    [{device.index}] {device.name}  "
                  f"could not open: {exc}")
    if landmarker is not None:
        landmarker.close()

    if chosen is not None:
        device, measured, luma = chosen
        default = default_camera_index()
        print(f"\n                    capturing from [{device.index}] "
              f"{device.name}"
              f"{' by default' if device.index == default else ''}")
        if luma < 60:
            print("                    LOW LIGHT: more light on your face will")
            print("                    reduce tracking jitter.")
        if measured < 24:
            # Do not blame lighting when the frame is well exposed: this camera
            # delivers the same rate at every resolution regardless of exposure.
            cause = ("the camera itself, which delivers this rate at every "
                     "resolution" if luma >= 60 else "possibly the low light above")
            print(f"                    {measured:.0f} fps is below 30. "
                  f"Cause: {cause}.")
            print("                    Usable, just not perfectly smooth.")
    elif devices:
        print("\n                    No real camera could be opened.")
        ok = False

    print("\n  virtual camera")
    if obs_is_running():
        print("                    OBS IS RUNNING - quit it, it competes for"
              " the sink")
        ok = False
    try:
        with VirtualCamera(640, 480, 30) as vcam:
            print(f"                    ready: {vcam.device}")
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        print(f"                    NOT READY: {first}")
        print(SETUP_HELP)
        ok = False

    print("\nAll checks passed." if ok else "\nSome checks failed, see above.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    config = CameraConfig(index=_camera_index(args), width=args.width,
                          height=args.height, fps=args.fps)
    print("Calibration records your eyes genuinely closed across a range of")
    print("head poses and expressions. It takes about 30 seconds.")
    print("Keep your eyes closed the whole way through. Esc aborts.\n")

    with open_source(args.source, config) as camera:
        actual = camera.actual()
        print(f"camera: {actual.get('width', 0):.0f}x{actual.get('height', 0):.0f}"
              f"  measured {camera.measure_fps(1.0):.1f} fps")
        locked = camera.lock_exposure_and_white_balance()
        print(f"exposure/white balance lock: {locked}\n")

        with Landmarker(args.model) as landmarker:
            bank = run_calibration(camera, landmarker, preview=not args.no_preview,
                                   per_cell=args.per_cell)

    if len(bank) == 0:
        print("\nNo patches captured. Nothing saved.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.bank) or ".", exist_ok=True)
    bank.save(args.bank)
    print(report_coverage(bank))
    print(f"\nSaved {len(bank)} patches to {args.bank}")
    print("Now run: blinkcam run")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    bank = _load_bank(args.bank)
    if bank is None:
        if not args.procedural:
            print(f"No patch bank at {args.bank}.\n"
                  "Run 'blinkcam calibrate' first, or pass --procedural to use\n"
                  "the synthetic fallback lid without calibrating.",
                  file=sys.stderr)
            return 1
        bank = PatchBank()
        print("No patch bank; using the procedural fallback lid.")

    config = CameraConfig(index=_camera_index(args), width=args.width,
                          height=args.height, fps=args.fps)
    renderer = EyeRenderer(bank, RenderConfig(rise=args.rise))
    transition = EffectTransition(TransitionConfig(
        ride_real_blink=not args.no_blink_ride))
    stabilizer = StabilizedFace(freq=float(args.fps))
    stats = Stats()

    if args.start_on:
        # Skip the arming wait: there is nothing to hide on a cold start.
        transition.state = TransitionState.ON

    hotkey = GlobalHotkey(lambda: transition.toggle(), args.hotkey)
    if hotkey.start():
        print(f"Global hotkey: {args.hotkey}")
    else:
        print(PERMISSION_HELP)

    window = "BlinkCam preview"
    show_debug = args.debug
    seeded_procedural = len(bank) > 0

    try:
        with open_source(args.source, config) as camera:
            actual = camera.actual()
            width = int(actual.get("width") or args.width)
            height = int(actual.get("height") or args.height)
            print(f"camera: {width}x{height}  "
                  f"measured {camera.measure_fps(1.0):.1f} fps")
            camera.lock_exposure_and_white_balance()

            # The published size is fixed and independent of the source.
            # Inheriting the source size breaks the sink on unusual
            # dimensions, and meeting apps read the device format once.
            output = (_NullOutput() if args.preview_only
                      else VirtualCamera(args.out_width, args.out_height,
                                         args.fps))
            with Landmarker(args.model) as landmarker, output as vcam:
                print(f"output: {vcam.device}")
                if not args.preview_only:
                    print(f"\nSelect '{vcam.device}' as your camera in Meet, "
                          "Zoom or Teams.")
                if args.no_preview:
                    print("No preview window, so use --start-on or grant the "
                          "hotkey permission to toggle.\n")
                else:
                    print("CLICK the preview window to toggle the effect.")
                    print("Keys also work when the window has focus: 'b' "
                          "toggle, 'd' debug overlay, 'q' quit.\n")

                if not args.no_preview:
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window, min(width, 960),
                                     int(min(width, 960) * height / width))

                    # Clicking is the reliable toggle. cv2.waitKey only sees
                    # keys while the window holds OS keyboard focus, which
                    # clicking into it does not always grant on macOS, so
                    # pressing 'b' can silently do nothing. A mouse callback
                    # has no such requirement and needs no permissions.
                    def _on_mouse(event, x, y, flags, userdata):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            transition.toggle(time.monotonic())

                    cv2.setMouseCallback(window, _on_mouse)

                start = time.monotonic()
                last_face = None
                last_report = 0.0
                harvested = 0

                while True:
                    frame = camera.read_new()
                    if frame is None:
                        if getattr(camera, "exhausted", False):
                            break
                        time.sleep(0.001)
                        continue

                    now = time.monotonic()
                    t0 = time.perf_counter()
                    face = landmarker.process(frame)
                    stats.add("track", (time.perf_counter() - t0) * 1000)

                    if face is not None:
                        face = stabilizer(face, now)
                        # Holding the last good geometry through a genuine blink
                        # keeps the composite from jumping when the user's real
                        # lids close over our synthetic ones.
                        if transition.should_freeze_geometry() and last_face:
                            face = last_face
                        else:
                            last_face = face
                    else:
                        stabilizer.reset()

                    opacity = transition.update(face, now)

                    # Seed the bank from the procedural lid on first use, so
                    # --procedural has something to retrieve.
                    if (not seeded_procedural and face is not None
                            and opacity > 0.0):
                        from .bank import Descriptor
                        from .geometry import extract_patch
                        for eye in face.eyes():
                            patch = synthesize_closed_patch(
                                extract_patch(frame, eye), eye).astype(np.uint8)
                            bank.add(patch, Descriptor.build(face, eye),
                                     eye.side, include_mirror=False)
                        seeded_procedural = True

                    t0 = time.perf_counter()
                    if face is not None and opacity > 0.0:
                        out = renderer.render(frame, face, opacity)
                    else:
                        out = frame
                    stats.add("render", (time.perf_counter() - t0) * 1000)

                    # Harvest real closed-eye patches as they occur. The bank
                    # self-improves over a session and tracks lighting drift.
                    if (args.harvest and face is not None
                            and not transition.enabled
                            and transition.detector.closed_for(now) > 0.15):
                        from .bank import Descriptor
                        from .geometry import extract_patch
                        for eye in face.eyes():
                            bank.add(extract_patch(frame, eye),
                                     Descriptor.build(face, eye), eye.side)
                        harvested += 1

                    t0 = time.perf_counter()
                    vcam.send(out)
                    stats.add("send", (time.perf_counter() - t0) * 1000)

                    if not args.no_preview:
                        vis = out
                        if show_debug and face is not None:
                            vis = draw_debug(out, face, renderer)
                        if transition.state is TransitionState.ARMING:
                            label = "WAITING FOR YOUR NEXT BLINK to engage"
                        elif transition.state is TransitionState.OFF:
                            label = "OFF   click the window to turn ON"
                        else:
                            label = (f"{transition.state.value.upper()}   "
                                     f"opacity {opacity:.2f}   "
                                     f"click to turn OFF")
                        cv2.putText(vis, label, (12, height - 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(vis, label, (12, height - 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (120, 255, 160) if opacity > 0
                                    else (200, 200, 200), 2, cv2.LINE_AA)
                        cv2.imshow(window, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key == ord("b"):
                            transition.toggle(now)
                        if key == ord("d"):
                            show_debug = not show_debug

                    if args.seconds and now - start >= args.seconds:
                        break

                    if now - last_report >= 5.0:
                        extra = f"  harvested {harvested}" if args.harvest else ""
                        print(f"  {stats.summary()}  ms p50/p95   "
                              f"dropped {camera.dropped}{extra}")
                        last_report = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hotkey.stop()
        cv2.destroyAllWindows()

    if args.harvest and harvested and args.bank:
        bank.save(args.bank)
        print(f"Saved bank with {len(bank)} patches (harvested {harvested} "
              "new closures this session).")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blinkcam", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--camera", type=int, default=None,
                       help="capture device index; defaults to the "
                            "first real camera, skipping virtual ones")
        p.add_argument("--width", type=int, default=1920)
        p.add_argument("--height", type=int, default=1080)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--bank", default=DEFAULT_BANK)
        p.add_argument("--no-preview", action="store_true")
        p.add_argument("--source", default=None,
                       help="read from an image or video file instead of the "
                            "camera; useful for tuning and for replaying the "
                            "stress-test clips")

    d = sub.add_parser("doctor", help="check the environment")
    d.add_argument("--model", default=DEFAULT_MODEL)
    d.add_argument("--bank", default=DEFAULT_BANK)
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("calibrate", help="record closed eyes, build the bank")
    common(c)
    c.add_argument("--per-cell", type=int, default=5,
                   help="max patches per pose cell (default 5)")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="run the filter into the virtual camera")
    common(r)
    r.add_argument("--hotkey", default=DEFAULT_COMBO)
    r.add_argument("--rise", type=float, default=0.15,
                   help="closure line height as a fraction of the aperture")
    r.add_argument("--debug", action="store_true",
                   help="start with the tracking overlay visible")
    r.add_argument("--procedural", action="store_true",
                   help="use the synthetic lid instead of a calibrated bank")
    r.add_argument("--no-blink-ride", action="store_true",
                   help="always synthesise the closure instead of waiting for "
                        "a real blink")
    r.add_argument("--out-width", type=int, default=1920,
                   help="published virtual camera width (default 1920)")
    r.add_argument("--out-height", type=int, default=1080,
                   help="published virtual camera height (default 1080)")
    r.add_argument("--start-on", action="store_true",
                   help="begin with the effect already engaged")
    r.add_argument("--preview-only", action="store_true",
                   help="skip the virtual camera; show the preview only. "
                        "Useful for tuning before the OBS setup step.")
    r.add_argument("--seconds", type=float, default=0.0,
                   help="exit after this long; 0 means run until quit")
    r.add_argument("--harvest", action="store_true",
                   help="keep adding real closures to the bank while running")
    r.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
