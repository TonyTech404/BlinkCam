"""Global hotkey for toggling the effect.

A global hotkey needs macOS Input Monitoring permission, which is a separate
approval from camera access. Because that is an extra hurdle and can be
refused, this degrades to nothing rather than failing: the preview window's
keypress handler always works and needs no permission at all.
"""

from __future__ import annotations

import threading
from typing import Callable

# Deliberately a triple modifier. cmd+shift+b, the obvious choice, toggles the
# bookmarks bar in Chrome and runs the build task in VS Code, so it fires those
# as well as us. Nothing standard binds ctrl+alt+cmd.
DEFAULT_COMBO = "<ctrl>+<alt>+<cmd>+b"

PERMISSION_HELP = (
    "Global hotkey unavailable: this process is not trusted for input\n"
    "monitoring.\n"
    "\n"
    "The grant belongs to the app that LAUNCHED this process, not to python\n"
    "and not to the terminal in general. If you started BlinkCam from the\n"
    "VS Code terminal, grant Visual Studio Code; from Terminal.app, grant\n"
    "Terminal. Add it under BOTH of these, then QUIT AND REOPEN that app:\n"
    "\n"
    "  System Settings > Privacy & Security > Accessibility\n"
    "  System Settings > Privacy & Security > Input Monitoring\n"
    "\n"
    "Accessibility alone is not enough, and the grant only applies to a\n"
    "freshly launched process, so reopening the app is required.\n"
    "\n"
    "Two alternatives that need no permission at all:\n"
    "  - click the preview window; focus moves but your video keeps sending\n"
    "  - send the process a signal from any shell:  kill -USR1 <pid>"
)


def request_trust() -> bool:
    """Ask macOS for the Accessibility grant, with the system prompt.

    This exists because finding the setting by hand is genuinely hard: an app
    does not appear in Privacy & Security > Accessibility at all until it has
    either requested the permission or been added by hand with the + button,
    and the grant belongs to whichever app launched us rather than to python.
    Telling a user to go and find "Visual Studio Code" in an empty list is a
    bad instruction.

    AXIsProcessTrustedWithOptions with the prompt option makes macOS show its
    own dialog AND add the launching app to the list, so all that is left is
    flipping a switch. Returns the trust state at the time of the call, which
    is still False immediately after prompting: the grant only takes effect
    for a freshly launched process.
    """
    import ctypes
    import ctypes.util

    for name in ("ApplicationServices", "HIServices"):
        path = ctypes.util.find_library(name)
        if not path:
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            fn = getattr(lib, "AXIsProcessTrustedWithOptions", None)
            if fn is None:
                continue

            # Build {kAXTrustedCheckOptionPrompt: kCFBooleanTrue} by hand.
            cf = ctypes.cdll.LoadLibrary(
                ctypes.util.find_library("CoreFoundation"))
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            cf.CFDictionaryCreate.restype = ctypes.c_void_p
            cf.CFDictionaryCreate.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_long,
                ctypes.c_void_p, ctypes.c_void_p]

            key = ctypes.c_void_p(cf.CFStringCreateWithCString(
                None, b"AXTrustedCheckOptionPrompt", 0x08000100))  # kCFStringEncodingUTF8
            true_ref = ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue")

            keys = (ctypes.c_void_p * 1)(key)
            values = (ctypes.c_void_p * 1)(true_ref)
            options = cf.CFDictionaryCreate(None, keys, values, 1, None, None)

            fn.restype = ctypes.c_bool
            fn.argtypes = [ctypes.c_void_p]
            return bool(fn(ctypes.c_void_p(options)))
        except Exception:
            continue
    return False


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
