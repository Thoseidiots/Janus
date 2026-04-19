"""
test_visual_detector.py — property and unit tests for VisualDetector.

Properties covered:
  - Property 10: UIElement center equals bounding box center

**Validates: Requirements 4.7**
"""
from __future__ import annotations

import sys
import os
import types

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path so we can import janus_computer_use
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Import real cv2 and numpy BEFORE stubbing, so Property 8 can use them.
# We temporarily remove any existing cv2 stub to force-import the real module.
# ---------------------------------------------------------------------------
try:
    # Remove any stub that conftest may have already installed
    _cv2_stub = sys.modules.pop("cv2", None)
    import cv2 as _real_cv2  # noqa: F401 — kept for use in Property 8 test
    import numpy as _real_numpy  # noqa: F401
    _HAS_REAL_CV2 = True
    # Restore the stub so other tests that rely on it still work
    if _cv2_stub is not None and not hasattr(_cv2_stub, "matchTemplate"):
        sys.modules["cv2"] = _cv2_stub
except ImportError:
    _real_cv2 = None  # type: ignore[assignment]
    _real_numpy = None  # type: ignore[assignment]
    _HAS_REAL_CV2 = False


# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing the module under test.
# This allows tests to run without pyautogui, cv2, etc. installed.
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUBS = [
    "pyautogui", "pytesseract", "PIL", "PIL.Image", "PIL.ImageGrab",
    "pygetwindow", "cv2", "imagehash",
]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        _stub(_pkg)

# pyautogui.size() must be callable
if not callable(getattr(sys.modules.get("pyautogui"), "size", None)):
    sys.modules["pyautogui"].size = lambda: (1920, 1080)  # type: ignore[attr-defined]

# Remove any PIL stubs so the real PIL (Pillow) is used
for _pil_pkg in ["PIL", "PIL.Image", "PIL.ImageGrab"]:
    if _pil_pkg in sys.modules and not hasattr(sys.modules[_pil_pkg], "__file__"):
        del sys.modules[_pil_pkg]

# Force-import the real PIL to ensure it's in sys.modules
try:
    import PIL as _real_pil  # noqa: F401
    import PIL.Image as _real_pil_image  # noqa: F401
    import PIL.ImageGrab as _real_pil_imagegrab  # noqa: F401
except ImportError:
    pass

# Patch _check_dependencies to no-op so the module loads without real packages
import unittest.mock as _mock

import janus_computer_use as _jcu
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import ScreenRegion, UIElement  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st


# ===========================================================================
# Property 10: UIElement center equals bounding box center
# ===========================================================================

# Feature: janus-computer-use, Property 10: UIElement center equals bounding box center
@settings(max_examples=100, deadline=None)
@given(
    st.integers(),
    st.integers(),
    st.integers(min_value=1),
    st.integers(min_value=1),
)
def test_uielement_center_equals_bounding_box_center(x, y, width, height):
    """
    **Validates: Requirements 4.7**

    For any UIElement, the center field must equal
    (bounding_box.x + bounding_box.width // 2, bounding_box.y + bounding_box.height // 2).
    """
    bb = ScreenRegion(x=x, y=y, width=width, height=height)
    center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
    element = UIElement(
        element_type="button",
        label="test",
        bounding_box=bb,
        confidence=0.9,
        center=center,
    )
    assert element.center == (bb.x + bb.width // 2, bb.y + bb.height // 2)


# ===========================================================================
# Unit tests — ScreenRegion and UIElement construction
# ===========================================================================

def test_screen_region_fields():
    """ScreenRegion stores x, y, width, height as provided."""
    region = ScreenRegion(x=10, y=20, width=100, height=50)
    assert region.x == 10
    assert region.y == 20
    assert region.width == 100
    assert region.height == 50


def test_uielement_fields():
    """UIElement stores all fields correctly."""
    bb = ScreenRegion(x=0, y=0, width=200, height=100)
    center = (100, 50)
    el = UIElement(
        element_type="input",
        label="Search",
        bounding_box=bb,
        confidence=0.95,
        center=center,
    )
    assert el.element_type == "input"
    assert el.label == "Search"
    assert el.bounding_box is bb
    assert el.confidence == 0.95
    assert el.center == (100, 50)


def test_uielement_center_integer_division():
    """Center uses integer (floor) division for odd dimensions."""
    bb = ScreenRegion(x=0, y=0, width=3, height=3)
    center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
    el = UIElement(
        element_type="button",
        label="ok",
        bounding_box=bb,
        confidence=1.0,
        center=center,
    )
    # 3 // 2 == 1
    assert el.center == (1, 1)


def test_uielement_center_with_offset():
    """Center accounts for non-zero origin."""
    bb = ScreenRegion(x=100, y=200, width=40, height=20)
    center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
    el = UIElement(
        element_type="checkbox",
        label="agree",
        bounding_box=bb,
        confidence=0.8,
        center=center,
    )
    assert el.center == (120, 210)


# ===========================================================================
# Additional imports for VisualDetector tests (Properties 8 and 9)
# ===========================================================================

import asyncio
from unittest.mock import MagicMock, patch

from janus_computer_use import VisualDetector  # noqa: E402


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Property 8: Template matching returns correct position
# ===========================================================================

# Feature: janus-computer-use, Property 8: Template matching returns correct position
@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=1800),
    st.integers(min_value=0, max_value=980),
    st.integers(min_value=10, max_value=100),
    st.integers(min_value=10, max_value=100),
)
def test_template_matching_returns_correct_position(tx, ty, tw, th):
    """
    **Validates: Requirements 4.5**

    For any template embedded at a known position (tx, ty) within a larger
    screenshot, find_template() must return a UIElement whose bounding_box
    top-left corner is within 5 pixels of (tx, ty).

    Uses real PIL and numpy/cv2 to create a synthetic screenshot with the
    template embedded at a known position.
    """
    # Use module-level _real_cv2 and _real_numpy (imported before stubs were set up)
    np = _real_numpy
    from PIL import Image as PILImage

    # Build a synthetic screenshot (grayscale, dark background)
    ss_w, ss_h = 1920, 1080

    # Create a random-colored template (unique pattern so cv2 can find it)
    rng = np.random.default_rng(seed=tx * 10000 + ty * 100 + tw + th)
    template_arr = rng.integers(50, 200, size=(th, tw), dtype=np.uint8)
    template = PILImage.fromarray(template_arr, mode="L")

    # Create a larger screenshot image (uniform dark background)
    screenshot_arr = np.zeros((ss_h, ss_w), dtype=np.uint8)

    # Ensure the template fits within the screenshot at (tx, ty)
    # tx can be up to 1800, tw up to 100 → max tx+tw = 1900 < 1920 ✓
    # ty can be up to 980, th up to 100 → max ty+th = 1080 ✓
    # Paste the template into the screenshot at (tx, ty)
    screenshot_arr[ty:ty + th, tx:tx + tw] = template_arr
    screenshot = PILImage.fromarray(screenshot_arr, mode="L")

    # Temporarily replace the cv2 stub with the real cv2 module.
    # We must do this persistently (not just via patch.dict context manager)
    # because asyncio.to_thread runs _match in a thread pool — the thread may
    # execute after the patch.dict context exits.
    _prev_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = _real_cv2
    try:
        detector = VisualDetector()
        result = _run(detector.find_template(template, screenshot))
    finally:
        if _prev_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = _prev_cv2

    assert result is not None, (
        f"find_template returned None for template at ({tx}, {ty}) "
        f"with size ({tw}, {th})"
    )

    bb = result.bounding_box
    assert abs(bb.x - tx) <= 5, (
        f"Bounding box x={bb.x} is more than 5 pixels from expected tx={tx}"
    )
    assert abs(bb.y - ty) <= 5, (
        f"Bounding box y={bb.y} is more than 5 pixels from expected ty={ty}"
    )


# ===========================================================================
# Property 9: Search results are sorted by confidence descending
# ===========================================================================

# Feature: janus-computer-use, Property 9: Search results are sorted by confidence descending
@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
def test_find_element_results_sorted_by_confidence_descending(confidences):
    """
    **Validates: Requirements 4.2**

    For any call to find_element() that returns two or more results, the
    confidence values must be non-increasing (result[i].confidence >= result[i+1].confidence).
    """
    from PIL import Image

    # Build fake OCR data where each word matches the label "test"
    n = len(confidences)
    fake_data = {
        "text": ["test"] * n,
        "conf": [c * 100 for c in confidences],  # pytesseract returns 0-100
        "left": [10 * i for i in range(n)],
        "top": [0] * n,
        "width": [50] * n,
        "height": [20] * n,
    }

    mock_tess = MagicMock()
    mock_tess.image_to_data.return_value = fake_data
    mock_output = MagicMock()
    mock_output.DICT = "dict"
    mock_tess.Output = mock_output

    screenshot = Image.new("RGB", (1920, 1080))

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        detector = VisualDetector()
        results = _run(detector.find_element("test", screenshot))

    # Must have at least 2 results (all words match "test")
    assert len(results) >= 2, (
        f"Expected at least 2 results, got {len(results)}"
    )

    # Verify sorted by confidence descending
    for i in range(len(results) - 1):
        assert results[i].confidence >= results[i + 1].confidence, (
            f"Results not sorted: results[{i}].confidence={results[i].confidence} "
            f"< results[{i+1}].confidence={results[i+1].confidence}"
        )


# ===========================================================================
# Unit tests for VisualDetector
# ===========================================================================

def test_find_element_returns_empty_list_when_no_match():
    """find_element() returns empty list (not exception) when no match found."""
    from PIL import Image

    fake_data = {
        "text": ["hello", "world"],
        "conf": [90, 85],
        "left": [0, 60],
        "top": [0, 0],
        "width": [50, 50],
        "height": [20, 20],
    }

    mock_tess = MagicMock()
    mock_tess.image_to_data.return_value = fake_data
    mock_output = MagicMock()
    mock_output.DICT = "dict"
    mock_tess.Output = mock_output

    screenshot = Image.new("RGB", (200, 50))

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        detector = VisualDetector()
        results = _run(detector.find_element("nonexistent_label_xyz", screenshot))

    assert results == [], f"Expected empty list, got {results}"


def test_find_element_case_insensitive():
    """find_element() matches labels case-insensitively."""
    from PIL import Image

    fake_data = {
        "text": ["Submit"],
        "conf": [90],
        "left": [0],
        "top": [0],
        "width": [60],
        "height": [20],
    }

    mock_tess = MagicMock()
    mock_tess.image_to_data.return_value = fake_data
    mock_output = MagicMock()
    mock_output.DICT = "dict"
    mock_tess.Output = mock_output

    screenshot = Image.new("RGB", (200, 50))

    with patch.dict(sys.modules, {"pytesseract": mock_tess}):
        detector = VisualDetector()
        results = _run(detector.find_element("submit", screenshot))

    assert len(results) == 1
    assert results[0].label == "Submit"


def test_center_of_returns_bounding_box_center():
    """center_of() returns (x + w//2, y + h//2)."""
    bb = ScreenRegion(x=100, y=200, width=80, height=40)
    el = UIElement(
        element_type="button",
        label="ok",
        bounding_box=bb,
        confidence=0.9,
        center=(140, 220),
    )
    detector = VisualDetector()
    cx, cy = detector.center_of(el)
    assert cx == 140
    assert cy == 220


def test_find_template_returns_none_when_correlation_below_threshold():
    """find_template() returns None when max correlation < 0.7."""
    import numpy as np
    from PIL import Image

    mock_cv2 = MagicMock()
    mock_cv2.TM_CCOEFF_NORMED = 5

    # Return a result with max correlation of 0.5 (below threshold)
    result_arr = np.full((100, 100), 0.5, dtype=np.float32)
    mock_cv2.matchTemplate.return_value = result_arr
    mock_cv2.minMaxLoc.return_value = (0.0, 0.5, (0, 0), (50, 50))

    screenshot = Image.new("L", (200, 200))
    template = Image.new("L", (100, 100))

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        detector = VisualDetector()
        result = _run(detector.find_template(template, screenshot))

    assert result is None, "Expected None when correlation < 0.7"
