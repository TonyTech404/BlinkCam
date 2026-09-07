#!/usr/bin/env bash
# BlinkCam setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_VERSION=3.13

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

say "1/4  Python $PYTHON_VERSION"
# MediaPipe 1.0.1 ships a py3-none arm64 wheel that should work on 3.14, but no
# success report exists for it, so pin to the version we have actually tested.
if ! command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
  if [ -x "/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" ]; then
    PY="/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION"
  else
    echo "Installing python@$PYTHON_VERSION via Homebrew..."
    brew install "python@$PYTHON_VERSION"
    PY="/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION"
  fi
else
  PY="python$PYTHON_VERSION"
fi
echo "using $($PY --version) at $PY"

if [ "$($PY -c 'import platform; print(platform.machine())')" != "arm64" ]; then
  warn "That Python is not arm64. MediaPipe will fail to load."
  exit 1
fi

say "2/4  Virtual environment and dependencies"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "installed into .venv"

say "3/4  Face landmark model"
mkdir -p models data
if [ -s models/face_landmarker.task ]; then
  echo "already present"
else
  curl -fL --progress-bar -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
  echo "downloaded $(du -h models/face_landmarker.task | cut -f1)"
fi

say "4/4  Virtual camera (OBS)"
if [ ! -d /Applications/OBS.app ]; then
  echo "Installing OBS. We never run it after setup; we only borrow its"
  echo "signed camera extension, which accepts frames from any process."
  brew install --cask obs
fi

cat <<'EOF'

  One manual step remains, and it needs a human click:

    1. Open OBS.
    2. Click "Start Virtual Camera" (bottom right), then "Stop Virtual Camera".
    3. If macOS asks for approval, allow it. To check it later:
         System Settings > General > Login Items & Extensions
           > Camera Extensions
       Some macOS 26 builds list it under Privacy & Security > Extensions
       as a "Media Extension" instead, so look in both places.
    4. Click "Stop Virtual Camera", then QUIT OBS. While OBS runs it holds
       the single camera instance and BlinkCam cannot use the sink.

  You may need to restart the Mac once after approving.

EOF

say "Verify"
echo "  .venv/bin/python -m blinkcam.app doctor"
echo ""
echo "Then:"
echo "  .venv/bin/python -m blinkcam.app calibrate     # ~30s, eyes closed"
echo "  .venv/bin/python -m blinkcam.app run"
