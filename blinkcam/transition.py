"""Blink detection and the effect's on/off transition.

Snapping the effect on is the loudest possible tell: a hard cut in the eye
region is exactly the kind of temporal discontinuity human vision is built to
notice. So turning on plays a real closure instead.

The default is to ride the user's next genuine blink. We watch for the real
eye-aspect-ratio drop, engage during the closed phase, and simply never open.
That transition is perfect by construction and costs nothing to render. Only if
no blink arrives within a couple of seconds do we synthesise one.

Blink kinematics follow measured physiology, not animation convention. A
spontaneous blink closes in about 100ms and opens in about 220ms, so closing is
roughly twice as fast as opening. The standard animator's rule is the reverse
(a slow close and a fast open), which looks wrong composited onto real video.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from .landmarks import FaceFrame

# Measured spontaneous blink phases, in seconds.
CLOSE_DURATION = 0.10
OPEN_DURATION = 0.22

# Hysteresis on the learned blink blendshape. Two thresholds, not one, so a
# signal hovering near the boundary cannot chatter.
BLINK_ENTER = 0.55
BLINK_EXIT = 0.35

# Eye aspect ratio fallback, used when blendshapes are unavailable.
EAR_CLOSED = 0.20


class State(Enum):
    OFF = "off"
    ARMING = "arming"        # effect requested, waiting for a real blink
    CLOSING = "closing"      # synthesising a closure
    ON = "on"
    OPENING = "opening"


def ease_in(t: float) -> float:
    """Accelerating. Matches a lid starting from rest and dropping fast."""
    return t * t


def ease_out(t: float) -> float:
    """Decelerating. The opening phase settles rather than snapping to a stop."""
    return 1.0 - (1.0 - t) * (1.0 - t)


class BlinkDetector:
    """Tracks whether the eyes are genuinely closed, with hysteresis.

    The blink signal is deliberately never low-pass filtered. A blink closes in
    about three frames at 30fps, and any 1-2Hz filter smears that into a slow
    droop, which is the signature failure of naive face-filter smoothing.
    """

    def __init__(self) -> None:
        self._closed = False
        self._closed_since: float | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def closed_for(self, now: float) -> float:
        """Seconds the eyes have been continuously closed."""
        if not self._closed or self._closed_since is None:
            return 0.0
        return now - self._closed_since

    def update(self, face: FaceFrame | None, now: float) -> bool:
        if face is None:
            self._closed = False
            self._closed_since = None
            return False

        signal = face.blink
        if signal <= 0.0:  # no blendshapes; fall back to geometry
            signal = 1.0 if face.ear < EAR_CLOSED else 0.0

        if self._closed:
            if signal < BLINK_EXIT:
                self._closed = False
                self._closed_since = None
        else:
            if signal > BLINK_ENTER:
                self._closed = True
                self._closed_since = now
        return self._closed


@dataclass
class TransitionConfig:
    ride_real_blink: bool = True
    # Two seconds of visibly nothing after a toggle reads as broken.
    # Long enough to catch a natural blink, short enough to feel
    # responsive when none arrives.
    arm_timeout: float = 1.5
    close_duration: float = CLOSE_DURATION
    open_duration: float = OPEN_DURATION


class EffectTransition:
    """Drives effect opacity from 0 to 1 and back with believable timing."""

    def __init__(self, config: TransitionConfig | None = None) -> None:
        self.config = config or TransitionConfig()
        self.state = State.OFF
        self.detector = BlinkDetector()
        self._opacity = 0.0
        self._phase_start = 0.0
        self._armed_at = 0.0

    @property
    def opacity(self) -> float:
        return self._opacity

    @property
    def enabled(self) -> bool:
        return self.state is not State.OFF

    def toggle(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state in (State.OFF, State.OPENING):
            if self.config.ride_real_blink:
                self.state = State.ARMING
                self._armed_at = now
            else:
                self.state = State.CLOSING
                self._phase_start = now
        else:
            self.state = State.OPENING
            self._phase_start = now

    def force_off(self, now: float | None = None) -> None:
        self.state = State.OFF
        self._opacity = 0.0

    def update(self, face: FaceFrame | None, now: float | None = None) -> float:
        """Advance the state machine and return the effect opacity."""
        now = time.monotonic() if now is None else now
        cfg = self.config
        really_closed = self.detector.update(face, now)

        if self.state is State.OFF:
            self._opacity = 0.0

        elif self.state is State.ARMING:
            self._opacity = 0.0
            if really_closed:
                # Ride it. The eyes are already shut, so we can come up at full
                # strength with nothing to see: a perfect transition, free.
                self.state = State.ON
                self._opacity = 1.0
            elif now - self._armed_at >= cfg.arm_timeout:
                self.state = State.CLOSING
                self._phase_start = now

        elif self.state is State.CLOSING:
            t = (now - self._phase_start) / max(cfg.close_duration, 1e-3)
            if t >= 1.0:
                self.state = State.ON
                self._opacity = 1.0
            else:
                self._opacity = ease_in(t)

        elif self.state is State.ON:
            self._opacity = 1.0

        elif self.state is State.OPENING:
            t = (now - self._phase_start) / max(cfg.open_duration, 1e-3)
            if t >= 1.0:
                self.state = State.OFF
                self._opacity = 0.0
            else:
                self._opacity = 1.0 - ease_out(t)

        return self._opacity

    def should_freeze_geometry(self) -> bool:
        """True when the user is genuinely blinking while the effect is on.

        Their real lids are closing over our synthetic ones, so the landmarks
        are describing a nearly-shut eye. Re-deriving the warp from degenerate
        geometry makes the composite jump. Hold the last good geometry instead
        and the double-blink is invisible.
        """
        return self.state is State.ON and self.detector.closed
