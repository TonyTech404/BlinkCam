#!/bin/bash
# Toggle the BlinkCam effect. Needs no permissions of any kind.
#
# The global hotkey inside BlinkCam requires an Accessibility and Input
# Monitoring grant for whichever app launched it, which is awkward to set up
# and easy to get wrong. This script sidesteps that entirely: macOS Shortcuts
# can bind a system-wide keyboard shortcut to a shell script, and Shortcuts
# already has the right to capture keys.
#
# Bind it:
#   1. Open Shortcuts, File > New Shortcut
#   2. Add the "Run Shell Script" action
#   3. Paste:  bash ~/Developer/BlinkCam/scripts/blink-toggle.sh
#   4. Name it "Blink", then in the sidebar right-click it > Add Keyboard
#      Shortcut and pick whatever you like
#
# Or just run it from a terminal, or bind it to a Stream Deck button.

PID=$(pgrep -f "blinkcam.app run" | head -1)

if [ -z "$PID" ]; then
  echo "BlinkCam is not running." >&2
  # Make the failure visible even when launched with no terminal attached.
  osascript -e 'display notification "BlinkCam is not running" with title "Blink"' 2>/dev/null
  exit 1
fi

kill -USR1 "$PID" && echo "toggled BlinkCam (pid $PID)"
