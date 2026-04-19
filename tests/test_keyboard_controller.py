"""
test_keyboard_controller.py — property and unit tests for KeyboardController.

Properties covered:
  - Property 3: Key combination ordering invariant
  - Property 4: Typing delivers all characters
  - Property 5: Configurable typing speed sets inter-key delay

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

_pag = sys.modules["pyautogui"]
_pag.size = lambda: (1920, 1080)  # type: ignore[attr-defined]
_pag.FAILSAFE = True  # type: ignore[attr-defined]

import janus_computer_use as _jcu
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import ActionType, KeyboardController  # noqa: E402

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


def _make_controller(speed: float = 30.0) -> KeyboardController:
    return KeyboardController(typing_speed_cps=speed)


# ===========================================================================
# Property 4: Typing delivers all characters
# ===========================================================================

# Feature: janus-computer-use, Property 4: Typing delivers all characters
@settings(max_examples=100, deadline=None)
@given(st.text(min_size=1))
def test_typing_delivers_all_characters(text):
    """
    **Validates: Requirements 2.1, 2.4, 2.7**

    For any non-empty string (including Unicode), type_text() must return
    ActionResult with chars_delivered == len(text) and success == True.
    """
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    # Mock pyperclip for non-ASCII characters
    mock_pyperclip = MagicMock()

    with patch.dict(sys.modules, {"pyautogui": mock_pag, "pyperclip": mock_pyperclip}):
        ctrl = _make_controller()
        result = _run(ctrl.type_text(text))

    assert result.success is True, (
        f"type_text({text!r}) failed: {result.error_message}"
    )
    assert result.chars_delivered == len(text), (
        f"Expected chars_delivered={len(text)}, got {result.chars_delivered}"
    )


# ===========================================================================
# Property 5: Configurable typing speed sets inter-key delay
# ===========================================================================

# Feature: janus-computer-use, Property 5: Configurable typing speed sets inter-key delay
@settings(max_examples=100, deadline=None)
@given(st.floats(min_value=0.1, max_value=200.0))
def test_typing_speed_sets_interval(speed):
    """
    **Validates: Requirements 2.6**

    For any typing speed s (chars/sec), the inter-key interval must equal
    1/s within 10% tolerance. Also verifies that the interval passed to
    pyautogui.typewrite matches 1/speed within 10% tolerance.
    """
    expected_interval = 1.0 / speed
    tolerance = expected_interval * 0.10

    # 1. Check _interval attribute
    ctrl = KeyboardController(typing_speed_cps=speed)
    assert abs(ctrl._interval - expected_interval) <= tolerance, (
        f"For speed={speed}, expected interval={expected_interval:.6f} ± {tolerance:.6f}, "
        f"got {ctrl._interval:.6f}"
    )

    # 2. Check interval passed to pyautogui.typewrite when type_text is called
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)
    captured_intervals = []

    def _capture_typewrite(chars, interval=0.0):
        captured_intervals.append(interval)

    mock_pag.typewrite.side_effect = _capture_typewrite

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl2 = KeyboardController(typing_speed_cps=speed)
        _run(ctrl2.type_text("a"))  # single ASCII char

    assert len(captured_intervals) == 1, (
        f"Expected typewrite to be called once, got {len(captured_intervals)} calls"
    )
    actual_interval = captured_intervals[0]
    assert abs(actual_interval - expected_interval) <= tolerance, (
        f"For speed={speed}, expected typewrite interval={expected_interval:.6f} ± {tolerance:.6f}, "
        f"got {actual_interval:.6f}"
    )


# ===========================================================================
# Property 3: Key combination ordering invariant
# ===========================================================================

# Feature: janus-computer-use, Property 3: Key combination ordering invariant
@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.sampled_from(["ctrl", "alt", "shift", "win"]),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    st.sampled_from(["a", "b", "c", "z", "enter", "tab", "f5"]),
)
def test_key_combination_ordering_invariant(modifiers, key):
    """
    **Validates: Requirements 2.2**

    For any key combination, all modifier keys must be pressed (keyDown) before
    the primary key, and released (keyUp) after the primary key in reverse order.
    """
    call_log = []

    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    def _key_down(k):
        call_log.append(("down", k))

    def _key_up(k):
        call_log.append(("up", k))

    mock_pag.keyDown.side_effect = _key_down
    mock_pag.keyUp.side_effect = _key_up

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.key_combination(modifiers, key))

    assert result.success is True, (
        f"key_combination({modifiers}, {key!r}) failed: {result.error_message}"
    )

    # Build expected sequence:
    # 1. keyDown each modifier in order
    # 2. keyDown primary key
    # 3. keyUp primary key
    # 4. keyUp each modifier in reverse order
    expected = (
        [("down", m) for m in modifiers]
        + [("down", key), ("up", key)]
        + [("up", m) for m in reversed(modifiers)]
    )

    assert call_log == expected, (
        f"Key event sequence mismatch.\n"
        f"Expected: {expected}\n"
        f"Got:      {call_log}"
    )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_type_ascii_text_calls_typewrite():
    """type_text() with ASCII text calls pyautogui.typewrite for each character."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller(speed=10.0)
        result = _run(ctrl.type_text("hi"))

    assert result.success is True
    assert result.chars_delivered == 2
    assert mock_pag.typewrite.call_count == 2


def test_type_empty_string_returns_zero_chars():
    """type_text('') returns success with chars_delivered=0."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.type_text(""))

    assert result.success is True
    assert result.chars_delivered == 0


def test_press_key_calls_pyautogui_press():
    """press_key() calls pyautogui.press with the given key."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.press_key("enter"))

    assert result.success is True
    mock_pag.press.assert_called_once_with("enter")


def test_hotkey_calls_pyautogui_hotkey():
    """hotkey() calls pyautogui.hotkey with the given keys."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.hotkey("ctrl", "c"))

    assert result.success is True
    mock_pag.hotkey.assert_called_once_with("ctrl", "c")


def test_type_text_exception_returns_failed_result():
    """type_text() returns a failed ActionResult when pyautogui raises."""
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)
    mock_pag.typewrite.side_effect = RuntimeError("keyboard error")

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.type_text("a"))

    assert result.success is False
    assert "keyboard error" in (result.error_message or "")


def test_default_typing_speed_interval():
    """Default typing speed of 30 cps gives interval of ~0.0333 s."""
    ctrl = KeyboardController()
    expected = 1.0 / 30.0
    assert abs(ctrl._interval - expected) < 1e-9


def test_key_combination_single_modifier():
    """key_combination with one modifier presses modifier before key and releases after."""
    call_log = []
    mock_pag = MagicMock()
    mock_pag.size.return_value = (1920, 1080)
    mock_pag.keyDown.side_effect = lambda k: call_log.append(("down", k))
    mock_pag.keyUp.side_effect = lambda k: call_log.append(("up", k))

    with patch.dict(sys.modules, {"pyautogui": mock_pag}):
        ctrl = _make_controller()
        result = _run(ctrl.key_combination(["ctrl"], "z"))

    assert result.success is True
    assert call_log == [("down", "ctrl"), ("down", "z"), ("up", "z"), ("up", "ctrl")]

