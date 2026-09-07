"""Global hotkey for toggling the effect.

A global hotkey needs macOS Input Monitoring permission, which is a separate
approval from camera access. Because that is an extra hurdle and can be
refused, this degrades to nothing rather than failing: the preview window's
keypress handler always works and needs no permission at all.
"""

from __future__ import annotations

import threading
from typing import Callable

DEFAULT_COMBO = "<cmd>+<shift>+b"

PERMISSION_HELP = (
    "Global hotkey unavailable: this process is not trusted for input\n"
    "monitoring. Grant your terminal Accessibility and Input Monitoring in\n"
    "System Settings > Privacy & Security, then restart the terminal.\n"
    "Until then, focus the preview window and press 'b' to toggle."
)


def process_is_trusted() -> bool:
    """Whether macOS will deliver global key events to this process.

    Calls AXIsProcessTrusted directly rather than trusting the listener to
    fail loudly, because it does not: without the grant, pynput's thread runs
    happily and simply never sees a key.
    """
    import ctypes
    import ctypes.util

    for name in ("ApplicationServices", "HIServices"):
        path = ctypes.util.find_library(name)
        if not path:
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            fn = getattr(lib, "AXIsProcessTrusted", None)
            if fn is None:
                continue
            fn.restype = ctypes.c_bool
            return bool(fn())
        except Exception:
            continue
    return False  # cannot confirm, so do not promise it works


class GlobalHotkey:
    """Runs a callback when the combo is pressed, anywhere in the system."""

    def __init__(self, callback: Callable[[], None],
                 combo: str = DEFAULT_COMBO) -> None:
        self.callback = callback
        self.combo = combo
        self._listener = None
        self.available = False
        self.error = ""

    def start(self) -> bool:
        # Ask macOS directly. pynput's listener thread starts and stays alive
        # even when the process has no accessibility grant, so it reports
        # success while silently receiving no events. Checking liveness is not
        # enough; only AXIsProcessTrusted tells the truth.
        if not process_is_trusted():
            self.error = "process is not trusted for input monitoring"
            return False

        try:
            from pynput import keyboard
        except Exception as exc:
            self.error = f"pynput unavailable: {exc}"
            return False

        def on_activate() -> None:
            # Never let a hotkey callback kill the listener thread.
            try:
                self.callback()
            except Exception:
                pass

        try:
            self._listener = keyboard.GlobalHotKeys({self.combo: on_activate})
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:
            self.error = str(exc)
            self._listener = None
            return False

        # pynput fails lazily on macOS: construction succeeds and the thread
        # dies once it discovers it has no Input Monitoring grant. Give it a
        # moment, then check the thread is actually alive.
        alive = threading.Event()
        alive.wait(0.35)
        if not self._listener.is_alive():
            self.error = "listener stopped; Input Monitoring likely not granted"
            self._listener = None
            return False

        self.available = True
        return True

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self.available = False
