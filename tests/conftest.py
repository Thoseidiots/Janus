"""
conftest.py — shared pytest fixtures for janus_computer_use tests.

All OS-level calls (pyautogui, pygetwindow, pytesseract, PIL.ImageGrab) are
mocked here so the test suite runs without a physical display.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
settings.register_profile("ci", max_examples=100, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ci")

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Stub out heavy third-party packages so tests can import janus_computer_use
# without the real libraries installed.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stubs() -> None:
    """Install lightweight stubs for packages that may not be present."""
    stubs = {
        "pyautogui": {"size": lambda: (1920, 1080), "FAILSAFE": True},
        "pytesseract": {},
        "pygetwindow": {},
        "cv2": {},
    }
    for pkg, attrs in stubs.items():
        if pkg not in sys.modules:
            mod = _make_stub(pkg)
            for attr, val in attrs.items():
                setattr(mod, attr, val)

    # Only stub PIL if the real package is not installed
    try:
        import PIL as _real_pil  # noqa: F401
    except ImportError:
        for pkg in ("PIL", "PIL.Image", "PIL.ImageGrab"):
            if pkg not in sys.modules:
                _make_stub(pkg)

    # Only stub imagehash if the real package is not installed
    try:
        import imagehash as _real_ih  # noqa: F401
    except ImportError:
        if "imagehash" not in sys.modules:
            _make_stub("imagehash")


_ensure_stubs()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_screen_size():
    """Return a standard mock screen size of 1920×1080."""
    return (1920, 1080)


@pytest.fixture
def mock_pyautogui(mock_screen_size):
    """Patch pyautogui with a MagicMock that reports a 1920×1080 screen."""
    mock = MagicMock()
    mock.size.return_value = mock_screen_size
    mock.FAILSAFE = True
    with patch.dict(sys.modules, {"pyautogui": mock}):
        yield mock


@pytest.fixture
def mock_pil_imagegrab():
    """Patch PIL.ImageGrab.grab to return a blank 1920×1080 image."""
    try:
        from PIL import Image
        blank = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
    except ImportError:
        blank = MagicMock()
        blank.width = 1920
        blank.height = 1080

    mock_grab = MagicMock(return_value=blank)
    with patch("PIL.ImageGrab.grab", mock_grab):
        yield mock_grab


@pytest.fixture
def mock_pytesseract():
    """Patch pytesseract to return an empty OCR result."""
    empty_result = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
    }
    mock = MagicMock()
    mock.image_to_data.return_value = empty_result
    with patch.dict(sys.modules, {"pytesseract": mock}):
        yield mock


# ---------------------------------------------------------------------------
# Screen Recorder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mss():
    """Stub out mss so tests run without a display."""
    mock_sct = MagicMock()
    mock_sct.__enter__ = MagicMock(return_value=mock_sct)
    mock_sct.__exit__ = MagicMock(return_value=False)
    mock_sct.monitors = [None, {"left": 0, "top": 0, "width": 1920, "height": 1080}]

    try:
        from PIL import Image
        blank = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
    except ImportError:
        blank = MagicMock()
        blank.size = (1920, 1080)
        blank.bgra = b"\x00" * (1920 * 1080 * 4)

    mock_grab_result = MagicMock()
    mock_grab_result.size = (1920, 1080)
    mock_grab_result.bgra = b"\x00" * (1920 * 1080 * 4)
    mock_sct.grab = MagicMock(return_value=mock_grab_result)

    mock_mss_module = MagicMock()
    mock_mss_module.mss = MagicMock(return_value=mock_sct)

    with patch.dict(sys.modules, {"mss": mock_mss_module}):
        yield mock_mss_module


@pytest.fixture
def mock_imagehash():
    """Patch imagehash.phash to return a mock hash with controllable distance."""
    mock_hash = MagicMock()
    mock_hash.__sub__ = MagicMock(return_value=0)

    mock_ih = MagicMock()
    mock_ih.phash = MagicMock(return_value=mock_hash)

    with patch.dict(sys.modules, {"imagehash": mock_ih}):
        yield mock_ih


@pytest.fixture
def mock_cv2_writer():
    """Patch cv2.VideoWriter for encoder tests."""
    mock_writer = MagicMock()
    mock_writer.write = MagicMock()
    mock_writer.release = MagicMock()

    mock_cv2 = MagicMock()
    mock_cv2.VideoWriter = MagicMock(return_value=mock_writer)
    mock_cv2.VideoWriter_fourcc = MagicMock(return_value=0x7634706D)
    mock_cv2.COLOR_RGB2BGR = 4
    mock_cv2.cvtColor = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        yield mock_cv2, mock_writer


