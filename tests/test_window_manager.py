"""
test_window_manager.py — property and unit tests for WindowManager.

Properties covered:
  - Property 12: Window list contains all required fields
  - Property 13: Invalid window handles return failed ActionResult without exception
  - Property 14: Title pattern matching is case-insensitive substring

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

from janus_computer_use import WindowManager, WindowInfo, ScreenRegion, ActionType  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_window(handle: int, title: str, left=0, top=0, width=800, height=600):
    """Create a mock pygetwindow window object."""
    win = MagicMock()
    win._hWnd = handle
    win.title = title
    win.left = left
    win.top = top
    win.width = width
    win.height = height
    win.isMinimized = False
    return win


# ===========================================================================
# Property 12: Window list contains all required fields
# ===========================================================================

# Feature: janus-computer-use, Property 12: Window list contains all required fields
@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.fixed_dictionaries({
            "handle": st.integers(min_value=1, max_value=99999),
            "title": st.text(min_size=1, max_size=50),
            "left": st.integers(min_value=0, max_value=1000),
            "top": st.integers(min_value=0, max_value=800),
            "width": st.integers(min_value=1, max_value=1920),
            "height": st.integers(min_value=1, max_value=1080),
        }),
        min_size=1,
        max_size=10,
    )
)
def test_window_list_contains_all_required_fields(window_specs):
    """
    **Validates: Requirements 7.1**

    Every WindowInfo returned by list_windows() must have non-None values
    for handle, title, process_name, and bounding_box.
    """
    mock_windows = [
        _make_mock_window(
            handle=spec["handle"],
            title=spec["title"],
            left=spec["left"],
            top=spec["top"],
            width=spec["width"],
            height=spec["height"],
        )
        for spec in window_specs
    ]

    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = mock_windows

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        windows = _run(wm.list_windows())

    assert len(windows) == len(window_specs), (
        f"Expected {len(window_specs)} windows, got {len(windows)}"
    )

    for info in windows:
        assert info.handle is not None, "WindowInfo.handle must not be None"
        assert info.title is not None, "WindowInfo.title must not be None"
        assert info.process_name is not None, "WindowInfo.process_name must not be None"
        assert info.bounding_box is not None, "WindowInfo.bounding_box must not be None"
        assert isinstance(info.bounding_box, ScreenRegion), (
            "WindowInfo.bounding_box must be a ScreenRegion"
        )


# ===========================================================================
# Property 13: Invalid window handles return failed ActionResult without exception
# ===========================================================================

# Feature: janus-computer-use, Property 13: Invalid window handles return failed ActionResult without exception
@settings(max_examples=100, deadline=None)
@given(st.integers())
def test_invalid_handle_returns_failed_result_no_exception(handle):
    """
    **Validates: Requirements 7.7**

    For any integer handle that does not correspond to an open window,
    all WindowManager operations must return ActionResult(success=False)
    and must NOT raise an unhandled exception.
    """
    # Empty window list — no window will match any handle
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()

        # All five operations must return failed ActionResult without raising
        try:
            r_focus = _run(wm.focus(handle))
            r_resize = _run(wm.resize(handle, 800, 600))
            r_move = _run(wm.move(handle, 0, 0))
            r_min = _run(wm.minimise(handle))
            r_max = _run(wm.maximise(handle))
        except Exception as exc:
            raise AssertionError(
                f"WindowManager raised an exception for invalid handle {handle}: {exc}"
            ) from exc

    for name, result in [
        ("focus", r_focus),
        ("resize", r_resize),
        ("move", r_move),
        ("minimise", r_min),
        ("maximise", r_max),
    ]:
        assert result.success is False, (
            f"{name}({handle}) should return success=False for invalid handle"
        )


# ===========================================================================
# Property 14: Title pattern matching is case-insensitive substring
# ===========================================================================

# Feature: janus-computer-use, Property 14: Title pattern matching is case-insensitive substring
@settings(max_examples=100, deadline=None)
@given(
    st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))),
    st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))),
)
def test_title_matching_case_insensitive_substring(title, pattern):
    """
    **Validates: Requirements 7.8**

    WindowManager._resolve_window(pattern) must find a window with the given
    title if and only if pattern.lower() is a substring of title.lower().
    """
    mock_win = _make_mock_window(handle=42, title=title)
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = [mock_win]

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = wm._resolve_window(pattern)

    should_match = pattern.lower() in title.lower()

    if should_match:
        assert result is not None, (
            f"Expected to find window with title={title!r} using pattern={pattern!r}"
        )
    else:
        assert result is None, (
            f"Expected no match for title={title!r} with pattern={pattern!r}"
        )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_list_windows_returns_window_info_objects():
    """list_windows() returns a list of WindowInfo objects."""
    mock_win = _make_mock_window(handle=1001, title="Notepad", width=800, height=600)
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = [mock_win]

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        windows = _run(wm.list_windows())

    assert len(windows) == 1
    assert isinstance(windows[0], WindowInfo)
    assert windows[0].handle == 1001
    assert windows[0].title == "Notepad"
    assert windows[0].bounding_box.width == 800
    assert windows[0].bounding_box.height == 600


def test_focus_valid_window_succeeds():
    """focus() with a valid title returns success."""
    mock_win = _make_mock_window(handle=2001, title="My App")
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = [mock_win]

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.focus("My App"))

    assert result.success is True
    mock_win.activate.assert_called_once()


def test_focus_invalid_title_returns_failed():
    """focus() with a title that matches no window returns failed ActionResult."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.focus("NonExistentWindow"))

    assert result.success is False
    assert "not found" in (result.error_message or "").lower()


def test_resolve_window_by_handle():
    """_resolve_window() finds a window by integer handle."""
    mock_win = _make_mock_window(handle=9999, title="Test")
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = [mock_win]

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = wm._resolve_window(9999)

    assert result is mock_win


def test_resolve_window_by_handle_not_found():
    """_resolve_window() returns None when handle is not found."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = wm._resolve_window(12345)

    assert result is None


def test_resolve_window_case_insensitive():
    """_resolve_window() matches title case-insensitively."""
    mock_win = _make_mock_window(handle=100, title="Google Chrome")
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = [mock_win]

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        # All of these should match
        assert wm._resolve_window("google chrome") is mock_win
        assert wm._resolve_window("GOOGLE CHROME") is mock_win
        assert wm._resolve_window("chrome") is mock_win
        assert wm._resolve_window("Google") is mock_win


def test_resize_invalid_window_returns_failed():
    """resize() with invalid handle returns failed ActionResult."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.resize(99999, 800, 600))

    assert result.success is False


def test_move_invalid_window_returns_failed():
    """move() with invalid handle returns failed ActionResult."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.move(99999, 0, 0))

    assert result.success is False


def test_minimise_invalid_window_returns_failed():
    """minimise() with invalid handle returns failed ActionResult."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.minimise(99999))

    assert result.success is False


def test_maximise_invalid_window_returns_failed():
    """maximise() with invalid handle returns failed ActionResult."""
    mock_pgw = MagicMock()
    mock_pgw.getAllWindows.return_value = []

    with patch.dict(sys.modules, {"pygetwindow": mock_pgw}):
        wm = WindowManager()
        result = _run(wm.maximise(99999))

    assert result.success is False

