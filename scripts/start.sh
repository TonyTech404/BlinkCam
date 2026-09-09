#!/bin/bash
# Start BlinkCam. This is the everyday entry point.
#
# Checks the few things that are easy to get wrong and give confusing symptoms
# rather than clear errors, then launches. Quit with 'q' in the preview window
# or Ctrl+C here.

set -u
cd "$(dirname "$0")/.." || exit 1

if pgrep -f "blinkcam.app run" >/dev/null; then
  PID=$(pgrep -f "blinkcam.app run" | head -1)
  echo "BlinkCam is already running (pid $PID)."
  echo "Toggle it with:  blink"
  echo "Or stop it with: kill $PID"
  exit 0
fi

# OBS holds the single camera instance while it runs, so the sink is
# unavailable to us and the failure looks like a dead virtual camera.
if pgrep -x OBS >/dev/null; then
  echo "OBS is running and competes for the virtual camera. Quit OBS first." >&2
  exit 1
fi

if [ ! -f data/bank.npz ]; then
  echo "No patch bank yet. Record one first (about 35 seconds):" >&2
  echo "  .venv/bin/python -m blinkcam.app calibrate" >&2
  exit 1
fi

# --start-on so the effect is applied immediately; drop it if you would rather
# begin with normal eyes and toggle when you want.
exec .venv/bin/python -u -m blinkcam.app run --start-on "$@"
