"""Objective quality checks for the closed-eye render.

The headline question, "does this look real", is settled by a human A/B test.
These metrics exist to catch regressions between those tests and to show WHERE
quality fails, which whole-frame numbers hide.

Two rules shaped the choices here. Measure on the eye patch only, because
whole-frame metrics dilute a 100x60 pixel defect into nothing. And report
boundary metrics as a RATIO against genuine closed-eye footage rather than
against zero, because real anatomy has a gradient at the lid margin too, so
zero is the wrong target.

Usage:
    # Record ~60s of genuinely closed eyes, held out of the patch bank:
    .venv/bin/python tools/eval_patches.py record --out data/heldout.npz

    # Then score the renderer against it:
    .venv/bin/python tools/eval_patches.py score --heldout data/heldout.npz \
        --bank data/bank.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blinkcam.bank import Descriptor, PatchBank, annulus_mask
from blinkcam.capture import Camera, CameraConfig
from blinkcam.geometry import (PATCH_H, PATCH_W, extract_patch,
                               feathered_mask, lid_mask_polygon,
                               patch_transform, to_patch_coords)
from blinkcam.landmarks import Landmarker
from blinkcam.render import RenderConfig
from blinkcam.transition import BlinkDetector


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def gradient_energy(patch: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude on luma."""
    grey = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def boundary_ring(polygon: np.ndarray, width: int = 3) -> np.ndarray:
    """Thin annulus straddling the composite boundary."""
    inner = feathered_mask((PATCH_H, PATCH_W), polygon, 0.0)
    k = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    grown = cv2.dilate(inner, k)
    shrunk = cv2.erode(inner, k)
    return np.clip(grown - shrunk, 0.0, 1.0)


def seam_score(rendered: np.ndarray, reference: np.ndarray,
               polygon: np.ndarray) -> float:
    """Gradient discontinuity across the seam, relative to real footage.

    1.0 means our boundary is as smooth as a real lid margin. Above about 1.2
    there is a visible seam.
    """
    ring = boundary_ring(polygon)
    total = float(ring.sum())
    if total < 1.0:
        return float("nan")
    ours = float((gradient_energy(rendered) * ring).sum() / total)
    theirs = float((gradient_energy(reference) * ring).sum() / total)
    return ours / max(theirs, 1e-6)


def colour_step(rendered: np.ndarray, polygon: np.ndarray) -> float:
    """Mean CIEDE2000-ish colour step across the boundary, in Lab units."""
    lab = cv2.cvtColor(rendered.astype(np.uint8), cv2.COLOR_BGR2LAB
                       ).astype(np.float32)
    solid = feathered_mask((PATCH_H, PATCH_W), polygon, 0.0)
    k = np.ones((5, 5), np.uint8)
    just_in = np.clip(solid - cv2.erode(solid, k), 0, 1)
    just_out = np.clip(cv2.dilate(solid, k) - solid, 0, 1)
    if just_in.sum() < 1 or just_out.sum() < 1:
        return float("nan")
    a = (lab * just_in[..., None]).reshape(-1, 3).sum(0) / just_in.sum()
    b = (lab * just_out[..., None]).reshape(-1, 3).sum(0) / just_out.sum()
    return float(np.linalg.norm(a - b))


def patch_rmse(a: np.ndarray, b: np.ndarray,
               weight: np.ndarray | None = None) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if weight is None:
        return float(np.sqrt(d.mean()))
    w = weight[..., None]
    return float(np.sqrt((d * w).sum() / max(w.sum() * 3, 1e-6)))


def lash_band(closure_y: float, band: float = 6.0) -> np.ndarray:
    """Weight concentrated around the closure line, where failure lives.

    Whole-patch metrics look deceptively good on a low-texture eyelid while
    hiding a bad lash line, which is the highest-contrast feature and the one
    that carries "closed". So it gets measured on its own.
    """
    gy = np.mgrid[0:PATCH_H, 0:PATCH_W][0].astype(np.float32)
    return np.exp(-((gy - closure_y) ** 2) / (2.0 * band * band))


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    print("Recording a HELD-OUT set of genuinely closed eyes.")
    print("Keep this out of the patch bank: it is the ground truth we score")
    print("against. Sweep the same poses as calibration. Esc or q stops.\n")

    images: list[np.ndarray] = []
    raws: list[list[float]] = []
    sides: list[str] = []
    polygons: list[np.ndarray] = []
    detector = BlinkDetector()

    with Camera(CameraConfig(args.camera, args.width, args.height, args.fps)) as cam:
        with Landmarker(args.model) as lm:
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                frame = cam.read_new()
                if frame is None:
                    time.sleep(0.002)
                    continue
                now = time.monotonic()
                face = lm.process(frame)
                closed = detector.update(face, now)
                if face is not None and closed:
                    for eye in face.eyes():
                        d = Descriptor.build(face, eye)
                        m = patch_transform(eye)
                        polygons.append(
                            to_patch_coords(lid_mask_polygon(eye), m))
                        images.append(extract_patch(frame, eye))
                        raws.append([d.yaw, d.pitch, d.brow_height,
                                     d.cheek_squint, d.scale])
                        sides.append(eye.side)
                if args.preview:
                    vis = frame.copy()
                    msg = (f"{len(images)} captured   "
                           f"{args.seconds - (now - start):.0f}s   "
                           f"{'CLOSED' if closed else 'OPEN - not capturing'}")
                    cv2.putText(vis, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(vis, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("BlinkCam held-out recording", vis)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
            cv2.destroyAllWindows()

    if not images:
        print("Nothing captured. Keep your eyes closed.", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=np.stack(images),
                        raw=np.array(raws, np.float32),
                        sides=np.array(sides),
                        polygons=np.stack(polygons))
    print(f"Saved {len(images)} held-out patches to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    if not os.path.exists(args.heldout):
        print(f"No held-out set at {args.heldout}. Run 'record' first.",
              file=sys.stderr)
        return 1
    if not os.path.exists(args.bank):
        print(f"No patch bank at {args.bank}.", file=sys.stderr)
        return 1

    data = np.load(args.heldout, allow_pickle=False)
    bank = PatchBank.load(args.bank)
    print(f"bank {len(bank)} patches   held-out {len(data['images'])} patches\n")

    annulus = annulus_mask()
    polygons = data["polygons"] if "polygons" in data.files else None
    if polygons is None:
        print("note: held-out set predates polygon capture; boundary metrics"
              " unavailable. Re-record to enable them.\n")

    rows = []
    for i, (image, raw, side) in enumerate(
            zip(data["images"], data["raw"], data["sides"])):
        descriptor = Descriptor(*[float(v) for v in raw])
        results = bank.query(descriptor, image, k=args.top_k)
        if not results:
            continue
        blended = np.zeros(image.shape, np.float32)
        for patch, weight, _ in results:
            blended += patch.image.astype(np.float32) * weight

        from blinkcam.render import two_band_relight
        rendered = np.clip(two_band_relight(blended, image.astype(np.float32),
                                            RenderConfig().low_band_sigma),
                           0, 255)

        if polygons is not None:
            polygon = polygons[i]
            closure_y = float(polygon[len(polygon) // 2:, 1].mean())
            seam = seam_score(rendered, image, polygon)
            step = colour_step(rendered, polygon)
            # Score only what we actually composite. The illumination fit is
            # estimated from the periorbital ring and extrapolates poorly over
            # the brow, but the brow is outside the lid mask and is never
            # written, so including it inflates the error and hides real
            # regressions.
            composited = feathered_mask((PATCH_H, PATCH_W), polygon, 2.0)
        else:
            closure_y, seam, step = PATCH_H / 2 + 6, float("nan"), float("nan")
            composited = None

        rows.append((
            results[0][2],
            patch_rmse(rendered, image, composited),
            patch_rmse(rendered, image, lash_band(closure_y)),
            patch_rmse(rendered, image, annulus),
            seam, step,
            float(descriptor.yaw), float(descriptor.pitch),
        ))

    if not rows:
        print("No comparable patches.", file=sys.stderr)
        return 1

    arr = np.array(rows, np.float32)

    def line(name: str, col: int, note: str = "") -> None:
        v = arr[:, col]
        v = v[np.isfinite(v)]
        if not len(v):
            print(f"  {name:26s} unavailable")
            return
        print(f"  {name:26s} p50 {np.median(v):7.2f}   "
              f"p90 {np.percentile(v, 90):7.2f}   {note}")

    print("Reconstruction against real closed eyes (lower is better)")
    line("retrieval distance", 0, "descriptor units")
    line("composited-region RMSE", 1, "0-255, the pixels we write")
    line("lash-band RMSE", 2, "where failure concentrates")
    line("illumination fit error", 3, "lighting model accuracy")
    print("\nBoundary quality, as a ratio against real footage")
    line("seam gradient ratio", 4, "1.0 ideal, >1.2 visible seam")
    line("colour step (Lab)", 5, "across the composite edge")

    print("\nCoverage: retrieval distance by pose cell")
    print("  (high numbers mark holes in the bank; re-run calibration there)")
    cells: dict[tuple[int, int], list[float]] = {}
    for d, _, _, _, _, _, yaw, pitch in rows:
        cells.setdefault((int(round(yaw / 10)), int(round(pitch / 10))),
                         []).append(d)
    yaws = sorted({k[0] for k in cells})
    pitches = sorted({k[1] for k in cells})
    print("          " + "".join(f"{y * 10:+7d}" for y in yaws))
    for p in pitches:
        row = "".join(
            f"{np.median(cells[(y, p)]):7.2f}" if (y, p) in cells else "      ."
            for y in yaws)
        print(f"  {p * 10:+5d}   {row}")

    worst = arr[:, 0].max()
    print(f"\n  worst retrieval distance {worst:.2f} "
          f"(effect fades out above {RenderConfig().fade_distance})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record a held-out closed-eye set")
    r.add_argument("--out", default="data/heldout.npz")
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--model", default="models/face_landmarker.task")
    r.add_argument("--no-preview", dest="preview", action="store_false")
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("score", help="score the renderer against held-out data")
    s.add_argument("--heldout", default="data/heldout.npz")
    s.add_argument("--bank", default="data/bank.npz")
    s.add_argument("--top-k", type=int, default=3)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
