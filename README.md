# BlinkCam

A local macOS virtual-camera filter that makes your eyes appear naturally
closed on video calls. Capture from your USB webcam, redraw the eye region,
publish the result as a camera device that Google Meet, Zoom and Teams can
select. A hotkey toggles it. Nothing leaves the machine.

```
USB webcam → face + eye tracking → closed-eye render → virtual camera → Meet
```

## Setup

```bash
scripts/setup.sh
.venv/bin/python -m blinkcam.app doctor
```

## Everyday use

```bash
scripts/start.sh      # start it, effect on
scripts/blink-toggle.sh   # toggle the effect
```

Worth aliasing to `blinkcam` and `blink` in your shell. The launcher refuses to
start a second instance, refuses to run while OBS holds the camera, and points
you at calibration if there is no patch bank, because all three of those fail
confusingly rather than clearly.

## Setup

`setup.sh` installs Python 3.13, the dependencies, the face landmark model and
OBS. One step needs a human click: launch OBS once and press Start Virtual
Camera, allowing the extension if macOS asks. Then press Stop Virtual Camera
and **quit OBS**, which matters because OBS holds the single camera instance
while it runs. To inspect the extension later, look under **System Settings →
General → Login Items & Extensions → Camera Extensions**; some macOS 26 builds
list it under *Privacy & Security → Extensions* as a *Media Extension*
instead.

`doctor` checks every piece and tells you exactly what is missing.

## Use

```bash
.venv/bin/python -m blinkcam.app calibrate   # ~30 seconds, eyes closed
.venv/bin/python -m blinkcam.app run
```

Then pick **OBS Virtual Camera** as your camera in Meet, Zoom or Teams. Start
BlinkCam before joining, because several apps read the device list once at
launch.

In the preview window: `b` toggles, `d` shows the tracking overlay, `q` quits.
The global hotkey is `cmd+shift+b` and needs Input Monitoring permission; if
that is not granted the preview keypress still works.

## How it works

**Why not warp the eye shut.** A warp preserves content, so the sclera and iris
compress into a grey streak rather than disappearing, and it cannot invent the
lid crease because that detail is not in the open-eye frame. Apple ships this
technique in FaceTime Attention Correction, with a depth sensor, and it still
visibly smears.

**What we do instead.** Calibration records your eyes genuinely closed across a
sweep of head poses and expressions, and harvests a few hundred real eyelid
patches. At runtime we retrieve the nearest few by pose and expression, warp
them onto the live eye region, relight them and blend. This is the architecture
of EyeOpener (ACM TOG 2016), Bitouk's SIGGRAPH 2008 face swapping and
Photoshop Elements' "Open Closed Eyes", run backwards. Ours is easier than any
of them, because the exemplars come from the same person, camera, room and
session, so identity, sensor and gross lighting mismatch never arise.

Three details do most of the quality work.

- **The warp is recomputed from live landmarks every frame**, so the composited
  lid inherits your real head, brow and cheek motion. It is an animated lid,
  not a pasted still. This is what NVIDIA's Maxine Eye Contact gets criticised
  for missing.
- **Two-band relighting** takes the low-frequency shading from a quadratic
  fitted to the surrounding skin in the *current* frame, and only the
  high-frequency detail from the reference patch. Illumination is low-frequency
  and identity is high-frequency, so uneven light and exposure drift stop being
  a problem, with no solver involved.
- **Turning on rides your next real blink.** We watch for the genuine blink,
  engage during the closed phase and never open. That transition is perfect by
  construction. If no blink arrives within two seconds we synthesise one, at
  measured physiological timing: about 100ms to close and 220ms to open.
  Closing is roughly twice as fast as opening, which is the reverse of the
  standard animator's rule.

**When it cannot do a good job it fades out** rather than showing a bad
composite. Coverage is gated on head pose rather than landmark confidence,
because the mesh will confidently hallucinate landmarks for a fully occluded
eye. Past about 30 degrees of yaw the far eye fades, past 55 it is gone, and
past 70 the effect disables entirely.

## Notes and limits

- **Lighting drives everything.** In a dark room the camera lengthens its
  exposure and silently drops from 30fps to about 20, and the added sensor
  noise makes tracking jitter. `doctor` reports brightness and warns.
- **Auto-exposure lock is attempted but macOS often ignores it.** This matters
  less than it sounds: two-band relighting re-derives illumination from the
  live frame every frame, so exposure drift is compensated automatically.
- **Capture device indices are measured, not named.** OpenCV's
  `cv2.VideoCapture` indices do not correspond to AVFoundation's device
  enumeration. Measured on macOS 26.4: AVFoundation reported the virtual
  camera at index 0 and the webcam at 1, while OpenCV opened the webcam at 0
  and the virtual camera at 1. AVFoundation's own order also changed within a
  session. Choosing a device by name therefore put a live call into a feedback
  loop, capturing BlinkCam's own output. On first run after the camera set
  changes, BlinkCam publishes a marker pattern for a few seconds and finds
  which index returns it; the answer is cached. Override with `--camera N` if
  you know better.
- **MediaPipe is pinned to 0.10.35 and runs on CPU, deliberately.** The GPU
  delegate never releases the pixel buffer behind each input frame and leaks
  about 3.4 MB per frame, which consumed 48 GB of RAM in eight minutes and
  wedged the machine during development. On mediapipe 1.0.x there is no
  non-leaking option, because the CPU delegate crashes there outright. CPU
  inference costs 3.8 ms against 2.2 ms per frame, which is irrelevant against
  a 20 fps camera. Run `tools/check_leak.py` before changing either setting;
  `blinkcam/landmarks.py` carries the full compatibility matrix.
- **Glasses are not supported.** Frames and lens glare break the warp.
- **If the eyes look half-lidded rather than shut**, the bank has no close
  match for your pose. Run with `--debug` and read the `dist` values: below
  1.8 is full strength, above 3.2 the effect fades out entirely. Re-run
  calibration covering that pose.
- **The device is named "OBS Virtual Camera"** in meeting apps, until and
  unless we ship our own camera extension.
- **Teams desktop** sometimes does not list virtual cameras. Use Teams in
  Chrome if so.
- **Three ways to toggle**, in order of how much setup they need. Click the
  preview window: focus moves but your video keeps sending, so this is safe
  mid-call. Send a signal: `kill -USR1 <pid>`, and the app prints its own pid
  at startup, so this works from any shell, script or Stream Deck with no
  permission. Or the global hotkey `ctrl+alt+cmd+b`, which needs an
  Accessibility and Input Monitoring grant for the app that LAUNCHED BlinkCam,
  not for python and not for the terminal generally, followed by quitting and
  reopening that app. Note that an app will not appear in those Settings lists
  until you add it with the + button.
- **For a system-wide key with no permissions at all**, bind
  `scripts/blink-toggle.sh` to a macOS Shortcut and give the Shortcut a
  keyboard shortcut. The Shortcuts app already holds the right to capture keys,
  so BlinkCam needs no grant of its own. This is the route to prefer.
- **Frames take two to three seconds to start flowing** after launch. Start
  BlinkCam, wait for it, then join the call.
- **Restarting BlinkCam waits about five seconds.** After a client
  disconnects, the OBS sink needs that long before it accepts a new one.
  Reconnect sooner and it accepts the connection while consumers keep seeing
  OBS's "no signal" placeholder, with no error raised anywhere, so BlinkCam
  waits the cooldown out and tells you it is doing so.
- **The published resolution is fixed at 1920x1080**, independent of the
  capture source, and other frame sizes are letterboxed into it. Publishing at
  an unusual size such as 820x1024 makes the extension reject every frame and
  sends viewers solid green. Override with `--out-width` and `--out-height`,
  but stick to 1920x1080, 1280x720 or 640x480.
- Calibration recordings and the patch bank are personal data. They stay in
  `data/`, which is gitignored.

## Development

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python tools/smoke_virtualcam.py     # virtual camera alone
```

Four flags on `run` make the pipeline workable before setup is finished and
tunable without sitting in front of the camera:

| Flag | Effect |
| --- | --- |
| `--procedural` | Synthesise the eyelid instead of using a calibrated bank. Also the fallback for poses the bank does not cover. |
| `--preview-only` | Skip the virtual camera entirely, useful before the OBS approval step. |
| `--source PATH` | Read an image or video file instead of the camera. |
| `--start-on` | Begin with the effect already engaged. |

`--source` is what the stress suite needs: replaying one recorded clip every
build is the only way to compare quality across changes, and a fixed input
beats a live face when tuning, because it stops moving between attempts.

Measure quality against real footage rather than by eye:

```bash
.venv/bin/python tools/eval_patches.py record --out data/heldout.npz
.venv/bin/python tools/eval_patches.py score
```

`record` captures a held-out set of genuinely closed eyes, kept out of the
bank, so every synthetic patch has a real counterpart. `score` reports error in
the region actually composited, error concentrated in the lash band where
failure lives, and a seam gradient ratio against real footage where 1.0 is
ideal and above 1.2 means a visible seam. It also prints retrieval distance per
pose cell, which is the most actionable diagnostic here: high numbers mark
holes in the bank.

Two macOS specifics are worth knowing before touching `landmarks.py`. MediaPipe
requires the **GPU delegate** here: every CPU-delegate configuration crashes in
`TensorsToDetectionsCalculator` because the face detector reaches for a Metal
service the CPU path never registers. Widely-repeated advice says to force CPU
on macOS, and it will hard-crash the process. Frames must also be **SRGBA**,
not SRGB.
