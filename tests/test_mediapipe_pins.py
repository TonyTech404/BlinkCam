"""Guards on the MediaPipe configuration.

BlinkCam once ate 48 GB of RAM in about eight minutes and wedged the machine.
The cause was MediaPipe's GPU delegate never releasing the CoreVideo pixel
buffer behind each input frame, retaining roughly 3.4 MB per call at 1080p.

Both the pinned version and the CPU delegate look like things a future reader
would tidy up, and neither choice is defensible from reading the code alone.
These tests make that tidying fail loudly instead of silently reintroducing the
leak. tools/check_leak.py measures the real thing.
"""

from __future__ import annotations

import inspect
import re



def test_mediapipe_is_pinned_to_the_non_leaking_line():
    """0.10.x is the only line where the CPU delegate works at all. On 1.0.x
    CPU crashes with 'graph_service.h: Check failed: service_ Service is
    unavailable', leaving the leaking GPU path as the only option."""
    import mediapipe

    version = mediapipe.__version__
    major, minor = (int(p) for p in version.split(".")[:2])
    assert (major, minor) == (0, 10), (
        f"mediapipe {version} is installed. On 1.0.x there is NO non-leaking "
        "configuration: CPU crashes and GPU leaks 3.4 MB per frame. "
        "See blinkcam/landmarks.py for the compatibility matrix.")


def test_requirements_pins_the_same_version():
    with open("requirements.txt") as handle:
        content = handle.read()
    assert re.search(r"^mediapipe==0\.10\.", content, re.M), (
        "requirements.txt must pin mediapipe to the 0.10.x line")


def _strip_comments(source: str) -> str:
    """Only the executable lines matter here.

    An earlier version of this test asserted the string "SRGBA" was absent and
    tripped on the comment explaining why SRGB is used instead of SRGBA.
    """
    return "\n".join(line.split("#")[0] for line in source.splitlines())


def _landmarker_source() -> str:
    from blinkcam.landmarks import Landmarker
    return inspect.getsource(Landmarker.__init__)


def test_landmarker_uses_the_cpu_delegate():
    """The GPU delegate is what leaks. This is the single most important line
    in the project from a stability standpoint."""
    code = _strip_comments(_landmarker_source())
    assert "Delegate.CPU" in code, (
        "the landmarker must use the CPU delegate; the GPU path leaks a full "
        "frame per call on every mediapipe version tested")
    assert "Delegate.GPU" not in code


def test_landmarker_feeds_srgb_not_srgba():
    """Three channels rather than four, which the CPU path accepts and which
    moves a third less data per frame."""
    from blinkcam.landmarks import Landmarker
    code = _strip_comments(inspect.getsource(Landmarker.process))
    assert "ImageFormat.SRGB" in code
    assert "ImageFormat.SRGBA" not in code
    assert "COLOR_BGR2RGB" in code and "COLOR_BGR2RGBA" not in code


def test_video_running_mode_is_kept():
    """VIDEO mode enables MediaPipe's internal tracking, which only applies at
    num_faces=1 and is why the detector does not run on every frame."""
    source = _landmarker_source()
    assert "RunningMode.VIDEO" in source
    assert "num_faces=1" in source


def test_the_leak_checker_exists():
    """The landmarks docstring tells the reader to run this before changing the
    delegate. A pointer to a file that does not exist is worse than none."""
    import os
    assert os.path.exists("tools/check_leak.py")


def test_landmarks_docstring_records_the_matrix():
    """The reasoning has to survive in the file, not just in a commit message:
    every one of these settings looks arbitrary without it."""
    import blinkcam.landmarks as L

    doc = L.__doc__ or ""
    for token in ("LEAKS", "0.10.35", "CPU", "SRGB", "check_leak.py"):
        assert token in doc, f"the compatibility matrix should mention {token}"
