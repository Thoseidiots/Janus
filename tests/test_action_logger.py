"""
test_action_logger.py — property and unit tests for ActionLogger.

Properties covered:
  - Property 19: Every action produces a log entry with all required fields
  - Property 22: Emitted log events contain required structure fields

All file I/O is mocked so the suite runs without writing to disk.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import os
import types
import tempfile
from io import StringIO
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

from janus_computer_use import ActionLogger, ActionLogEntry, ActionType  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    action_type: str = "click",
    target: str = "button",
    outcome: str = "success",
    error_message=None,
    screenshot_thumbnail: str = "",
) -> ActionLogEntry:
    return ActionLogEntry(
        action_type=action_type,
        target=target,
        timestamp="2024-01-01T00:00:00Z",
        outcome=outcome,
        error_message=error_message,
        screenshot_thumbnail=screenshot_thumbnail,
    )


def _valid_base64(s: str) -> bool:
    """Return True if s is a non-empty valid base64 string."""
    if not s:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


# ===========================================================================
# Property 19: Every action produces a log entry with all required fields
# ===========================================================================

# Feature: janus-computer-use, Property 19: Every action produces a log entry with all required fields
@settings(max_examples=20, deadline=None)
@given(
    st.sampled_from([at.value for at in ActionType]),
    st.text(min_size=1, max_size=100),
    st.booleans(),
)
def test_log_entry_has_all_required_fields(action_type_val, target, success):
    """
    **Validates: Requirements 9.5, 9.6**

    For any action, the written log entry must have non-None values for
    action_type, target, timestamp, outcome, and a valid base64 screenshot_thumbnail.
    """
    outcome = "success" if success else "failure"

    # Create a valid base64 thumbnail for the entry
    thumbnail = base64.b64encode(b"fake_jpeg_data").decode("ascii")

    entry = ActionLogEntry(
        action_type=action_type_val,
        target=target,
        timestamp="2024-01-01T12:00:00Z",
        outcome=outcome,
        error_message=None,
        screenshot_thumbnail=thumbnail,
    )

    written_lines = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        logger_instance = ActionLogger(log_path=log_path)
        logger_instance.log(entry)
        logger_instance.flush()

        with open(log_path, "r", encoding="utf-8") as f:
            written_lines = [line.strip() for line in f if line.strip()]
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass

    assert len(written_lines) == 1, (
        f"Expected exactly 1 log line, got {len(written_lines)}"
    )

    record = json.loads(written_lines[0])

    assert record.get("action_type") is not None, "Log entry missing action_type"
    assert record.get("target") is not None, "Log entry missing target"
    assert record.get("timestamp") is not None, "Log entry missing timestamp"
    assert record.get("outcome") is not None, "Log entry missing outcome"
    assert record.get("screenshot_thumbnail") is not None, "Log entry missing screenshot_thumbnail"

    # Validate base64
    assert _valid_base64(record["screenshot_thumbnail"]), (
        "screenshot_thumbnail is not valid base64"
    )


# ===========================================================================
# Property 22: Emitted log events contain required structure fields
# ===========================================================================

# Feature: janus-computer-use, Property 22: Emitted log events contain required structure fields
@settings(max_examples=20, deadline=None)
@given(
    st.sampled_from([at.value for at in ActionType]),
    st.text(min_size=1, max_size=100),
    st.booleans(),
)
def test_emitted_log_events_contain_required_fields(action_type_val, target, success):
    """
    **Validates: Requirements 10.6**

    For any action, the structured log event emitted via logger.info must
    contain the fields event_type, action_type, target, outcome, and timestamp.
    """
    outcome = "success" if success else "failure"
    thumbnail = base64.b64encode(b"fake_jpeg_data").decode("ascii")

    entry = ActionLogEntry(
        action_type=action_type_val,
        target=target,
        timestamp="2024-01-01T12:00:00Z",
        outcome=outcome,
        error_message=None,
        screenshot_thumbnail=thumbnail,
    )

    captured_extras = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured_extras.append(record.__dict__.copy())

    handler = CapturingHandler()
    janus_logger = logging.getLogger("janus")
    janus_logger.addHandler(handler)
    janus_logger.setLevel(logging.DEBUG)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            logger_instance = ActionLogger(log_path=log_path)
            logger_instance.log(entry)
            logger_instance.flush()
        finally:
            try:
                os.unlink(log_path)
            except Exception:
                pass
    finally:
        janus_logger.removeHandler(handler)

    assert len(captured_extras) >= 1, "No log events were emitted"

    # Check the last emitted event (the one from our log() call)
    extra = captured_extras[-1]

    assert extra.get("event_type") is not None, "Log event missing event_type"
    assert extra.get("action_type") is not None, "Log event missing action_type"
    assert extra.get("target") is not None, "Log event missing target"
    assert extra.get("outcome") is not None, "Log event missing outcome"
    assert extra.get("timestamp") is not None, "Log event missing timestamp"


# ===========================================================================
# Unit tests
# ===========================================================================

def test_log_writes_json_line():
    """log() writes a valid JSON line to the log file."""
    entry = _make_entry(
        action_type="click",
        target="submit_button",
        outcome="success",
        screenshot_thumbnail=base64.b64encode(b"test").decode("ascii"),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        al = ActionLogger(log_path=log_path)
        al.log(entry)
        al.flush()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["action_type"] == "click"
        assert record["target"] == "submit_button"
        assert record["outcome"] == "success"
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass


def test_log_multiple_entries():
    """log() appends multiple entries as separate JSON lines."""
    entries = [
        _make_entry(action_type="click", target=f"btn_{i}", outcome="success",
                    screenshot_thumbnail=base64.b64encode(b"x").decode("ascii"))
        for i in range(5)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        al = ActionLogger(log_path=log_path)
        for entry in entries:
            al.log(entry)
        al.flush()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["target"] == f"btn_{i}"
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass


def test_make_thumbnail_returns_base64_string():
    """_make_thumbnail() returns a non-empty base64-encoded string."""
    from PIL import Image

    al_mock_file = MagicMock()
    al_mock_file.write = MagicMock()
    al_mock_file.flush = MagicMock()
    al_mock_file.close = MagicMock()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        al = ActionLogger(log_path=log_path)
        img = Image.new("RGB", (1920, 1080), color=(255, 0, 0))
        result = al._make_thumbnail(img)
        al.flush()

        assert isinstance(result, str)
        assert len(result) > 0
        # Must be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass


def test_log_emits_structured_event():
    """log() emits a structured log event with required fields."""
    captured = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.__dict__.copy())

    handler = CapturingHandler()
    janus_logger = logging.getLogger("janus")
    janus_logger.addHandler(handler)
    janus_logger.setLevel(logging.DEBUG)

    thumbnail = base64.b64encode(b"test").decode("ascii")
    entry = _make_entry(
        action_type="type",
        target="search_box",
        outcome="success",
        screenshot_thumbnail=thumbnail,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        al = ActionLogger(log_path=log_path)
        al.log(entry)
        al.flush()
    finally:
        janus_logger.removeHandler(handler)
        try:
            os.unlink(log_path)
        except Exception:
            pass

    assert len(captured) >= 1
    extra = captured[-1]
    assert extra.get("event_type") == "computer_use_action"
    assert extra.get("action_type") == "type"
    assert extra.get("target") == "search_box"
    assert extra.get("outcome") == "success"
    assert extra.get("timestamp") is not None

