"""
test_mouse_controller.py — property and unit tests for MouseController.

Properties covered:
  - Property 1:  Out-of-bounds coordinates are always rejected
  - Property 2:  Human-like movement visits intermediate points (duration > 0)
  - Property 11: Drag destination clamping stays within bounds

All OS calls are mocked so the suite runs without a physical display.
"""
from __future__ import annotations

import asyncio
import sys
import os
import types
from unittest.mock import MagicMock, patch, call

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
    "pyautogui", "pytesseract", "PIL", "PIL.Image", "PIL.ImageGrab",
    "pygetwindow", "cv2", "imagehash",
]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        _stub(_pkg)

# pyautogui.size() must return (1920, 1080) for all tests
_pag = sys.modules["pyautogui"]
_pag.size = lambda: (1920, 1080)  # type: ignore[attr-defined]
_pag.FAILSAFE = True  # type: ignore[attr-defined]
_pag.easeInOutQuad = "easeInOutQuad"  # type: ignore[attr-defined]

# Patch _check_dependencies to no-op so the module loads without real packages
import janus_computer_use as _jcu
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import (  # noqa: E402
    ActionType,
    MouseController,
    ScrollDirection,
)

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st

# Screen dimensions used throughout
_W, _H = 1920, 1080


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously (works in pytest without asyncio plugin)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_controller() -> MouseController:
    """Return a MouseController whose pyautogui.size() returns (1920, 1080)."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.easeInOutQuad = "easeInOutQuad"
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
    return ctrl


# ===========================================================================
# Property 1: Out-of-bounds coordinates are always rejected
# ===========================================================================

# Feature: janus-computer-use, Property 1: Out-of-bounds coordinates are always rejected
@settings(max_examples=100, deadline=None)
@given(
    st.one_of(
        # x out of range, y in range
        st.tuples(
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_W)),
            st.integers(min_value=0, max_value=_H - 1),
        ),
        # x in range, y out of range
        st.tuples(
            st.integers(min_value=0, max_value=_W - 1),
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_H)),
        ),
        # both out of range
        st.tuples(
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_W)),
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_H)),
        ),
    )
)
def test_out_of_bounds_coords_rejected(coords):
    """
    **Validates: Requirements 1.5**

    For any (x, y) outside [0, 1920) × [0, 1080), click() and move() must
    return a failed ActionResult and must NOT call any pyautogui mouse function.
    """
    x, y = coords
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.easeInOutQuad = "easeInOutQuad"

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result_click = _run(ctrl.click(x, y))
        result_move = _run(ctrl.move(x, y))

    # Both operations must fail
    assert result_click.success is False, (
        f"click({x}, {y}) should fail but returned success=True"
    )
    assert result_move.success is False, (
        f"move({x}, {y}) should fail but returned success=True"
    )

    # No pyautogui mouse call should have been made
    mock_pag.click.assert_not_called()
    mock_pag.moveTo.assert_not_called()
    mock_pag.doubleClick.assert_not_called()
    mock_pag.rightClick.assert_not_called()


# ===========================================================================
# Property 2: Human-like movement uses duration > 0
# ===========================================================================

# Feature: janus-computer-use, Property 2: Human-like movement visits intermediate points
@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=_W - 1),
    st.integers(min_value=0, max_value=_H - 1),
)
def test_human_like_movement_uses_duration(dst_x, dst_y):
    """
    **Validates: Requirements 1.7**

    When human_like=True, moveTo must be called with duration > 0.
    """
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.easeInOutQuad = "easeInOutQuad"

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.move(dst_x, dst_y, human_like=True))

    assert result.success is True, (
        f"move({dst_x}, {dst_y}, human_like=True) failed: {result.error_message}"
    )

    # moveTo must have been called with a duration keyword argument > 0
    mock_pag.moveTo.assert_called_once()
    _, kwargs = mock_pag.moveTo.call_args
    assert "duration" in kwargs, "moveTo was not called with a 'duration' keyword argument"
    assert kwargs["duration"] > 0, (
        f"Expected duration > 0 for human-like movement, got {kwargs['duration']}"
    )


# ===========================================================================
# Property 11: Drag destination clamping stays within bounds
# ===========================================================================

# Feature: janus-computer-use, Property 11: Drag destination clamping stays within bounds
@settings(max_examples=100, deadline=None)
@given(
    st.one_of(
        # dst_x out of range, dst_y in range
        st.tuples(
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_W)),
            st.integers(min_value=0, max_value=_H - 1),
        ),
        # dst_x in range, dst_y out of range
        st.tuples(
            st.integers(min_value=0, max_value=_W - 1),
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_H)),
        ),
        # both out of range
        st.tuples(
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_W)),
            st.one_of(st.integers(max_value=-1), st.integers(min_value=_H)),
        ),
    )
)
def test_drag_destination_clamped(dst_coords):
    """
    **Validates: Requirements 6.4**

    When the drag destination is out of bounds, the clamped coordinates passed
    to pyautogui.dragTo must satisfy 0 <= x < 1920 and 0 <= y < 1080.
    """
    dst_x, dst_y = dst_coords
    # Use a valid source coordinate (centre of screen)
    src_x, src_y = _W // 2, _H // 2

    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.easeInOutQuad = "easeInOutQuad"

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.drag(src_x, src_y, dst_x, dst_y))

    assert result.success is True, (
        f"drag to out-of-bounds ({dst_x}, {dst_y}) should succeed after clamping, "
        f"got error: {result.error_message}"
    )

    # dragTo must have been called with clamped coordinates
    mock_pag.dragTo.assert_called_once()
    args, _ = mock_pag.dragTo.call_args
    clamped_x, clamped_y = args[0], args[1]

    assert 0 <= clamped_x < _W, (
        f"Clamped x={clamped_x} is outside [0, {_W})"
    )
    assert 0 <= clamped_y < _H, (
        f"Clamped y={clamped_y} is outside [0, {_H})"
    )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_click_valid_coords_succeeds():
    """click() with valid coordinates returns success."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.click(100, 200))
    assert result.success is True
    assert result.action_type == ActionType.CLICK
    mock_pag.click.assert_called_once_with(100, 200, button="left")


def test_move_valid_coords_succeeds():
    """move() with valid coordinates returns success."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.move(500, 300))
    assert result.success is True
    assert result.action_type == ActionType.MOVE


def test_double_click_valid_coords_succeeds():
    """double_click() with valid coordinates returns success."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.double_click(400, 400))
    assert result.success is True
    assert result.action_type == ActionType.DOUBLE_CLICK
    mock_pag.doubleClick.assert_called_once_with(400, 400)


def test_right_click_valid_coords_succeeds():
    """right_click() with valid coordinates returns success."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.right_click(300, 300))
    assert result.success is True
    assert result.action_type == ActionType.RIGHT_CLICK
    mock_pag.rightClick.assert_called_once_with(300, 300)


def test_scroll_up_calls_scroll_positive():
    """scroll() UP calls pyautogui.scroll with a positive amount."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.scroll(500, 500, ScrollDirection.UP, amount=5))
    assert result.success is True
    mock_pag.scroll.assert_called_once_with(5, x=500, y=500)


def test_scroll_down_calls_scroll_negative():
    """scroll() DOWN calls pyautogui.scroll with a negative amount."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.scroll(500, 500, ScrollDirection.DOWN, amount=3))
    assert result.success is True
    mock_pag.scroll.assert_called_once_with(-3, x=500, y=500)


def test_scroll_left_calls_hscroll_negative():
    """scroll() LEFT calls pyautogui.hscroll with a negative amount."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.scroll(500, 500, ScrollDirection.LEFT, amount=2))
    assert result.success is True
    mock_pag.hscroll.assert_called_once_with(-2, x=500, y=500)


def test_scroll_right_calls_hscroll_positive():
    """scroll() RIGHT calls pyautogui.hscroll with a positive amount."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.scroll(500, 500, ScrollDirection.RIGHT, amount=4))
    assert result.success is True
    mock_pag.hscroll.assert_called_once_with(4, x=500, y=500)


def test_drag_valid_coords_succeeds():
    """drag() with valid source and destination returns success."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.drag(100, 100, 500, 500))
    assert result.success is True
    assert result.action_type == ActionType.DRAG
    mock_pag.dragTo.assert_called_once_with(500, 500, duration=0.5, button="left")


def test_drag_out_of_bounds_source_fails():
    """drag() with an out-of-bounds source coordinate returns failure."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.drag(-1, 100, 500, 500))
    assert result.success is False
    mock_pag.dragTo.assert_not_called()


def test_failsafe_exception_returns_failed_result():
    """FailSafeException from pyautogui is caught and returns a failed ActionResult."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.FailSafeException = Exception  # use base Exception as stand-in
    mock_pag.click.side_effect = Exception("FailSafe triggered")
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.click(100, 100))
    assert result.success is False
    assert "FailSafe triggered" in (result.error_message or "")


def test_move_without_human_like_does_not_use_duration():
    """move() without human_like=True calls moveTo without a duration argument."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    mock_pag.easeInOutQuad = "easeInOutQuad"
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.move(200, 200, human_like=False))
    assert result.success is True
    mock_pag.moveTo.assert_called_once_with(200, 200)


def test_click_boundary_coords_succeed():
    """Coordinates at the exact boundary (0,0) and (W-1, H-1) are valid."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        r1 = _run(ctrl.click(0, 0))
        r2 = _run(ctrl.click(_W - 1, _H - 1))
    assert r1.success is True
    assert r2.success is True


def test_click_exactly_at_width_fails():
    """x == screen_width is out of bounds."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (_W, _H)
    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = MouseController()
        result = _run(ctrl.click(_W, 0))
    assert result.success is False
    mock_pag.click.assert_not_called()

