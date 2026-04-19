"""
test_computer_use_engine.py — property tests for ComputerUseEngine / ActionLogger.

Properties covered:
  - Property 19: Every action produces a log entry with all required fields
  - Property 22: Emitted log events contain required structure fields
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
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


_STUBS = ["pyautogui", "pytesseract", "pygetwindow", "cv2", "imagehash"]
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

import janus_computer_use as _jcu  # noqa: E402

# ---------------------------------------------------------------------------
# Save the *original* _check_dependencies before any test file can overwrite
# it.  Multiple test files all do `_jcu._check_dependencies = lambda: None`
# at module-import time; whichever file is imported first wins.  We therefore
# retrieve the function directly from the module source via importlib so we
# always test the real implementation regardless of import order.
# ---------------------------------------------------------------------------
def _get_real_check_dependencies():
    """Return the original _check_dependencies function from janus_computer_use source."""
    import importlib.util
    import importlib.machinery

    # Load the source file into a fresh temporary module (without executing
    # the module-level _check_dependencies() call, which would fail without
    # real packages).  We patch builtins.__import__ to suppress the call.
    loader = importlib.machinery.SourceFileLoader("_jcu_real_tmp", _jcu.__file__)
    spec = importlib.util.spec_from_loader("_jcu_real_tmp", loader)
    tmp_mod = importlib.util.module_from_spec(spec)

    # Pre-populate sys.modules entries so the module-level imports succeed
    import sys as _sys
    # Temporarily suppress the module-level _check_dependencies() call by
    # injecting a no-op before exec_module runs.
    import builtins as _bt
    _orig_import = _bt.__import__

    def _safe_import(name, *args, **kwargs):
        try:
            return _orig_import(name, *args, **kwargs)
        except ImportError:
            import types as _types
            stub = _types.ModuleType(name)
            _sys.modules[name] = stub
            return stub

    _bt.__import__ = _safe_import
    try:
        spec.loader.exec_module(tmp_mod)
    except Exception:
        pass
    finally:
        _bt.__import__ = _orig_import

    return tmp_mod.__dict__.get("_check_dependencies")

_real_check_dependencies = _get_real_check_dependencies()
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import ActionLogger, ActionLogEntry, ActionType  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    st.sampled_from(ActionType),
    st.text(),
    st.booleans(),
)
def test_every_action_produces_log_entry_with_required_fields(action_type, target, success):
    """
    **Validates: Requirements 9.5, 9.6**

    For any action, the written log entry must have non-None values for
    action_type, target, timestamp, outcome, and a valid base64 screenshot_thumbnail.
    """
    outcome = "success" if success else "failure"
    screenshot_thumbnail = base64.b64encode(b"fake_image").decode()

    entry = ActionLogEntry(
        action_type=action_type.value,
        target=target,
        timestamp="2024-01-01T00:00:00Z",
        outcome=outcome,
        error_message=None,
        screenshot_thumbnail=screenshot_thumbnail,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        logger_instance = ActionLogger(log_path=log_path)
        logger_instance.log(entry)
        logger_instance.flush()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass

    assert len(lines) == 1, f"Expected exactly 1 log line, got {len(lines)}"

    record = json.loads(lines[0])

    assert record.get("action_type") is not None, "Log entry missing action_type"
    assert record.get("target") is not None, "Log entry missing target"
    assert record.get("timestamp") is not None, "Log entry missing timestamp"
    assert record.get("outcome") is not None, "Log entry missing outcome"
    assert record.get("screenshot_thumbnail") is not None, "Log entry missing screenshot_thumbnail"
    assert _valid_base64(record["screenshot_thumbnail"]), (
        "screenshot_thumbnail is not valid base64"
    )


# ===========================================================================
# Property 22: Emitted log events contain required structure fields
# ===========================================================================

# Feature: janus-computer-use, Property 22: Emitted log events contain required structure fields
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    st.sampled_from(ActionType),
    st.text(),
    st.booleans(),
)
def test_emitted_log_events_contain_required_structure_fields(action_type, target, success):
    """
    **Validates: Requirements 10.6**

    For any action, the structured log event emitted via logger.info must
    contain the fields event_type, action_type, target, outcome, and timestamp
    in the extra dict.
    """
    outcome = "success" if success else "failure"
    screenshot_thumbnail = base64.b64encode(b"fake_image").decode()

    entry = ActionLogEntry(
        action_type=action_type.value,
        target=target,
        timestamp="2024-01-01T00:00:00Z",
        outcome=outcome,
        error_message=None,
        screenshot_thumbnail=screenshot_thumbnail,
    )

    captured_records: list = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record.__dict__.copy())

    handler = CapturingHandler()
    janus_logger = logging.getLogger("janus")
    janus_logger.addHandler(handler)
    janus_logger.setLevel(logging.DEBUG)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        logger_instance = ActionLogger(log_path=log_path)
        logger_instance.log(entry)
        logger_instance.flush()
    finally:
        janus_logger.removeHandler(handler)
        try:
            os.unlink(log_path)
        except Exception:
            pass

    assert len(captured_records) >= 1, "No log events were emitted"

    extra = captured_records[-1]

    assert extra.get("event_type") is not None, "Log event missing event_type"
    assert extra.get("action_type") is not None, "Log event missing action_type"
    assert extra.get("target") is not None, "Log event missing target"
    assert extra.get("outcome") is not None, "Log event missing outcome"
    assert extra.get("timestamp") is not None, "Log event missing timestamp"


# ===========================================================================
# Property 18: Stuck state detected after exactly 3 consecutive no-change screens
# ===========================================================================

# Feature: janus-computer-use, Property 18: Stuck state detected after exactly 3 consecutive no-change screens
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.integers(min_value=3, max_value=10))
def test_stuck_state_detected_after_exactly_3_consecutive_no_change_screens(n):
    """
    **Validates: Requirements 9.4**

    When the screen hash is identical for N consecutive calls (N >= 3),
    _check_stuck_state() must return False for the first 2 calls and True
    starting from the 3rd call onward.
    """
    import asyncio
    import types
    from unittest.mock import AsyncMock, MagicMock, patch
    from PIL import Image

    # Create a dummy PIL image
    dummy_image = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Create a fixed hash object that always compares equal (distance 0)
    class FixedHash:
        def __sub__(self, other):
            return 0  # distance is always 0

    fixed_hash = FixedHash()

    # Ensure imagehash stub is in sys.modules with a phash that returns fixed_hash
    imagehash_stub = types.ModuleType("imagehash")
    imagehash_stub.phash = MagicMock(return_value=fixed_hash)
    sys.modules["imagehash"] = imagehash_stub

    engine = _jcu.ComputerUseEngine(context={})
    # Reset the hash buffer so each test run starts fresh
    engine._hash_buffer.clear()

    async def run():
        results = []
        with patch.object(engine._screen, "capture", new_callable=AsyncMock, return_value=dummy_image):
            for i in range(n):
                result = await engine._check_stuck_state()
                results.append(result)
        return results

    results = asyncio.get_event_loop().run_until_complete(run())

    # First two calls must return False (buffer not yet full)
    assert results[0] is False, f"Call 1 should return False, got {results[0]}"
    assert results[1] is False, f"Call 2 should return False, got {results[1]}"
    # 3rd call and beyond must return True
    for i in range(2, n):
        assert results[i] is True, f"Call {i+1} should return True, got {results[i]}"


# ===========================================================================
# Property 20: Missing dependency ImportError lists all missing packages
# ===========================================================================

# Feature: janus-computer-use, Property 20: Missing dependency ImportError lists all missing packages
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    st.lists(
        st.sampled_from(["pyautogui", "pytesseract", "PIL", "pygetwindow", "cv2"]),
        min_size=1,
        unique=True,
    )
)
def test_missing_dependency_import_error_lists_all_missing_packages(missing_packages):
    """
    **Validates: Requirements 10.2**

    When some required packages are missing, _check_dependencies() must raise
    an ImportError whose message contains every missing package name.
    """
    # _check_dependencies iterates _REQUIRED_PACKAGES and calls __import__.
    # We patch _REQUIRED_PACKAGES to only contain the missing packages (so
    # the function always tries to import exactly those), and we remove them
    # from sys.modules so __import__ actually invokes the import machinery,
    # then intercept with a fake that raises ImportError for those names.
    import builtins

    # Build a fake _REQUIRED_PACKAGES list containing only the missing packages
    # (mapping import_name -> pip_name using the original mapping)
    original_required = _jcu._REQUIRED_PACKAGES
    pip_map = dict(original_required)
    fake_required = [(pkg, pip_map.get(pkg, pkg)) for pkg in missing_packages]

    # Save and remove the packages from sys.modules so __import__ doesn't
    # short-circuit via the cache.
    saved = {}
    for pkg in missing_packages:
        if pkg in sys.modules:
            saved[pkg] = sys.modules.pop(pkg)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in missing_packages:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    try:
        with patch.object(_jcu, "_REQUIRED_PACKAGES", fake_required):
            with patch("builtins.__import__", side_effect=fake_import):
                try:
                    _real_check_dependencies()
                    assert False, "_check_dependencies() should have raised ImportError"
                except ImportError as exc:
                    error_message = str(exc)
                    for pkg in missing_packages:
                        assert pkg in error_message, (
                            f"ImportError message does not contain missing package '{pkg}': {error_message!r}"
                        )
    finally:
        # Restore sys.modules to avoid polluting other tests
        for pkg, mod in saved.items():
            sys.modules[pkg] = mod


# ===========================================================================
# Property 21: Session context is accessible to ActionPlanner
# ===========================================================================

# Feature: janus-computer-use, Property 21: Session context is accessible to ActionPlanner
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.dictionaries(st.text(), st.text()))
def test_session_context_is_accessible_to_action_planner(ctx):
    """
    **Validates: Requirements 10.4**

    The context dict passed to ComputerUseEngine.__init__ must be stored as
    engine._context and equal to the original dict.
    """
    engine = _jcu.ComputerUseEngine(context=ctx)
    assert engine._context == ctx, (
        f"engine._context {engine._context!r} does not equal passed context {ctx!r}"
    )
