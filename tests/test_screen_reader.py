"""
test_screen_reader.py — property and unit tests for ScreenReader.

Properties covered:
  - Property 6: Region capture returns correctly sized image
  - Property 7: OCR confidence scores are always in [0.0, 1.0]

All OS calls are mocked so the suite runs without a physical display.
"""
from __future__ import annotations

import asyncio
import sys
import os
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing the module under test.
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUBS = [
    "pyautogui", "pytesseract",
    "pygetwindow", "cv2", "imagehash",
]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        _stub(_pkg)

# Remove any PIL stubs so the real PIL (Pillow) is used
for _pil_pkg in ["PIL", "PIL.Image", "PIL.ImageGrab"]:
    if _pil_pkg in sys.modules and not hasattr(sys.modules[_pil_pkg], "__file__"):
        del sys.modules[_pil_pkg]

_pag = sys.modules["pyautogui"]
_pag.size = lambda: (1920, 1080)  # type: ignore[attr-defined]
_pag.FAILSAFE = True  # type: ignore[attr-defined]

import janus_computer_use as _jcu
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import ScreenReader, ScreenRegion, OCRWord  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_fake_grab(expected_w=None, expected_h=None):
    """Return a fake ImageGrab.grab function that returns a PIL Image of the right size."""
    from PIL import Image as PILImage

    def _fake_grab(bbox=None):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
        else:
            w = expected_w or 1920
            h = expected_h or 1080
        return PILImage.new("RGB", (w, h), color=(128, 128, 128))

    return _fake_grab


def _make_mock_imagegrab(fake_grab_fn):
    """Create a mock PIL.ImageGrab module with the given grab function."""
    mock_ig = MagicMock()
    mock_ig.grab = fake_grab_fn
    return mock_ig


def _make_mock_pytesseract(fake_data):
    """Create a mock pytesseract module with the given image_to_data return value."""
    mock_tess = MagicMock()
    mock_tess.image_to_data.return_value = fake_data
    mock_output = MagicMock()
    mock_output.DICT = "dict"
    mock_tess.Output = mock_output
    return mock_tess


# ===========================================================================
# Property 6: Region capture returns correctly sized image
# ===========================================================================

# Feature: janus-computer-use, Property 6: Region capture returns correctly sized image
@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=1900),
    st.integers(min_value=0, max_value=1060),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
)
def test_region_capture_returns_correct_size(rx, ry, rw, rh):
    """
    **Validates: Requirements 3.2**

    For any ScreenRegion within display bounds, capture(region) must return
    a PIL.Image whose width == region.width and height == region.height.
    """
    region = ScreenRegion(x=rx, y=ry, width=rw, height=rh)
    fake_grab = _make_fake_grab()
    mock_ig = _make_mock_imagegrab(fake_grab)

    with patch.dict(sys.modules, {"PIL.ImageGrab": mock_ig}):
        reader = ScreenReader()
        img = _run(reader.capture(region))

    assert img.width == rw, (
        f"Expected width={rw}, got {img.width} for region ({rx},{ry},{rw},{rh})"
    )
    assert img.height == rh, (
        f"Expected height={rh}, got {img.height} for region ({rx},{ry},{rw},{rh})"
    )


# ===========================================================================
# Property 7: OCR confidence scores are always in [0.0, 1.0]
# ===========================================================================

# Feature: janus-computer-use, Property 7: OCR confidence scores are always in [0.0, 1.0]
@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.integers(min_value=-200, max_value=200), min_size=1, max_size=20)
)
def test_ocr_confidence_always_in_range(raw_confidences):
    """
    **Validates: Requirements 3.7**

    For any raw tesseract confidence values (including out-of-range values),
    every OCRWord.confidence in the result must be in [0.0, 1.0].
    """
    from PIL import Image

    n = len(raw_confidences)
    fake_data = {
        "text": [f"word{i}" for i in range(n)],
        "conf": raw_confidences,
        "left": [10 * i for i in range(n)],
        "top": [0] * n,
        "width": [50] * n,
        "height": [20] * n,
    }

    mock_tess = _make_mock_pytesseract(fake_data)

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        reader = ScreenReader()
        blank_image = Image.new("RGB", (200, 50))
        words = _run(reader.ocr(blank_image))

    for word in words:
        assert 0.0 <= word.confidence <= 1.0, (
            f"OCRWord confidence {word.confidence} is outside [0.0, 1.0]"
        )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_capture_no_region_calls_grab_with_none():
    """capture() with no region calls ImageGrab.grab(None)."""
    captured_bbox = []
    fake_grab = _make_fake_grab()

    original_grab = fake_grab

    def _tracking_grab(bbox=None):
        captured_bbox.append(bbox)
        return original_grab(bbox)

    mock_ig = _make_mock_imagegrab(_tracking_grab)

    with patch.dict(sys.modules, {"PIL.ImageGrab": mock_ig}):
        # Also patch the attribute on the real PIL package if it's loaded
        try:
            import PIL as _pil_pkg
            _pil_pkg.ImageGrab = mock_ig
        except ImportError:
            pass
        reader = ScreenReader()
        img = _run(reader.capture())

    assert captured_bbox == [None]
    assert img.width == 1920
    assert img.height == 1080


def test_capture_with_region_passes_correct_bbox():
    """capture(region) passes (x, y, x+w, y+h) as bbox to ImageGrab.grab."""
    captured_bbox = []
    fake_grab = _make_fake_grab()

    def _tracking_grab(bbox=None):
        captured_bbox.append(bbox)
        return fake_grab(bbox)

    mock_ig = _make_mock_imagegrab(_tracking_grab)

    region = ScreenRegion(x=100, y=200, width=300, height=150)
    with patch.dict(sys.modules, {"PIL.ImageGrab": mock_ig}):
        # Also patch the attribute on the real PIL package if it's loaded
        try:
            import PIL as _pil_pkg
            _pil_pkg.ImageGrab = mock_ig
        except ImportError:
            pass
        reader = ScreenReader()
        img = _run(reader.capture(region))

    assert captured_bbox == [(100, 200, 400, 350)]
    assert img.width == 300
    assert img.height == 150


def test_ocr_filters_empty_text():
    """ocr() filters out words with empty or whitespace-only text."""
    from PIL import Image

    fake_data = {
        "text": ["hello", "", "  ", "world"],
        "conf": [90, 80, 70, 85],
        "left": [0, 60, 120, 180],
        "top": [0, 0, 0, 0],
        "width": [50, 50, 50, 50],
        "height": [20, 20, 20, 20],
    }

    mock_tess = _make_mock_pytesseract(fake_data)

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        reader = ScreenReader()
        words = _run(reader.ocr(Image.new("RGB", (300, 50))))

    texts = [w.text for w in words]
    assert "hello" in texts
    assert "world" in texts
    assert "" not in texts
    assert "  " not in texts
    assert len(words) == 2


def test_ocr_confidence_clamped_above_100():
    """Confidence values > 100 are clamped to 1.0."""
    from PIL import Image

    fake_data = {
        "text": ["test"],
        "conf": [150],
        "left": [0],
        "top": [0],
        "width": [50],
        "height": [20],
    }

    mock_tess = _make_mock_pytesseract(fake_data)

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        reader = ScreenReader()
        words = _run(reader.ocr(Image.new("RGB", (100, 30))))

    assert len(words) == 1
    assert words[0].confidence == 1.0


def test_ocr_confidence_clamped_below_zero():
    """Negative confidence values are clamped to 0.0."""
    from PIL import Image

    fake_data = {
        "text": ["test"],
        "conf": [-50],
        "left": [0],
        "top": [0],
        "width": [50],
        "height": [20],
    }

    mock_tess = _make_mock_pytesseract(fake_data)

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        reader = ScreenReader()
        words = _run(reader.ocr(Image.new("RGB", (100, 30))))

    assert len(words) == 1
    assert words[0].confidence == 0.0


def test_capture_and_ocr_composes_both():
    """capture_and_ocr() returns OCR words from the captured image."""
    from PIL import Image

    fake_data = {
        "text": ["hello"],
        "conf": [90],
        "left": [0],
        "top": [0],
        "width": [50],
        "height": [20],
    }

    mock_tess = _make_mock_pytesseract(fake_data)
    mock_ig = _make_mock_imagegrab(_make_fake_grab())

    with patch.dict(sys.modules, {"PIL.ImageGrab": mock_ig, "pytesseract": mock_tess}):
        reader = ScreenReader()
        words = _run(reader.capture_and_ocr())

    assert len(words) == 1
    assert words[0].text == "hello"

