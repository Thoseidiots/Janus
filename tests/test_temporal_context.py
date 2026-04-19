"""
tests/test_temporal_context.py
================================
Property-based and unit tests for ActionPlanner temporal context injection
and _build_temporal_context_section.

Properties covered:
  Property 8: Temporal context thumbnails are valid base64
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import time
import types
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stubs() -> None:
    stubs = {
        "pyautogui": {"size": lambda: (1920, 1080), "FAILSAFE": True},
        "pytesseract": {},
        "pygetwindow": {},
        "cv2": {},
        "mss": {},
    }
    for pkg, attrs in stubs.items():
        if pkg not in sys.modules:
            mod = _make_stub(pkg)
            for attr, val in attrs.items():
                setattr(mod, attr, val)
    try:
        import PIL  # noqa: F401
    except ImportError:
        for pkg in ("PIL", "PIL.Image", "PIL.ImageGrab"):
            if pkg not in sys.modules:
                _make_stub(pkg)
    try:
        import imagehash  # noqa: F401
    except ImportError:
        if "imagehash" not in sys.modules:
            _make_stub("imagehash")


_ensure_stubs()

from janus_computer_use import (  # noqa: E402
    ActionPlanner,
    ComputerUseEngine,
    ScreenRecorder,
    RecordedFrame,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_pil_image(width: int, height: int):
    """Create a real PIL image if available, else a mock."""
    try:
        from PIL import Image as _PILImage
        return _PILImage.new("RGB", (width, height), color=(100, 150, 200))
    except ImportError:
        mock_img = MagicMock()
        mock_img.size = (width, height)
        mock_img.copy.return_value = mock_img
        mock_img.thumbnail = MagicMock()
        return mock_img


def _make_mock_frame(timestamp: float = 0.0, width: int = 100, height: int = 100) -> RecordedFrame:
    phash = MagicMock()
    phash.__sub__ = MagicMock(return_value=0)
    img = _make_pil_image(width, height)
    return RecordedFrame(image=img, timestamp=timestamp, phash=phash)


def _make_engine_with_recorder(running: bool = False) -> ComputerUseEngine:
    """Create a ComputerUseEngine with a mocked ScreenRecorder."""
    engine = ComputerUseEngine.__new__(ComputerUseEngine)
    engine._context = {}
    engine._enable_temporal_context = True
    recorder = MagicMock(spec=ScreenRecorder)
    recorder.is_running = running
    recorder.last_frame_diff = 5
    recorder.motion_score = 12.5
    recorder._config = MagicMock()
    recorder._config.temporal_context_frames = 3
    engine._screen_recorder = recorder
    engine._logger = MagicMock()
    engine._logger._make_thumbnail = MagicMock(return_value="")
    return engine


# ===========================================================================
# Property 8: Temporal context thumbnails are valid base64
# ===========================================================================

class TestTemporalContextBase64:
    """Feature: janus-screen-recorder, Property 8: Temporal context thumbnails are valid base64"""

    @given(st.integers(1, 640), st.integers(1, 480))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_thumbnail_is_valid_base64(self, width, height):
        """Feature: janus-screen-recorder, Property 8: Temporal context thumbnails are valid base64"""
        try:
            from PIL import Image as _PILImage
        except ImportError:
            pytest.skip("PIL not available for this test")

        engine = _make_engine_with_recorder(running=True)
        brain = MagicMock()
        planner = ActionPlanner(engine=engine, brain=brain)

        frame = _make_mock_frame(timestamp=time.monotonic() - 1.0, width=width, height=height)
        recorder = MagicMock(spec=ScreenRecorder)
        recorder.last_frame_diff = 3
        recorder.motion_score = 7.0

        now = time.monotonic()
        section = planner._build_temporal_context_section([frame], recorder, now)

        # Extract base64 strings from the section
        lines = section.splitlines()
        frame_lines = [l for l in lines if l.strip().startswith("Frame -")]
        assert len(frame_lines) >= 1

        for line in frame_lines:
            # Format: "  Frame -X.Xs: <base64>"
            parts = line.strip().split(": ", 1)
            assert len(parts) == 2, f"Unexpected line format: {line!r}"
            encoded = parts[1].strip()
            # Must be valid base64
            decoded = base64.b64decode(encoded)
            assert len(decoded) > 0, "Decoded base64 should be non-empty"


# ===========================================================================
# Unit tests — _build_temporal_context_section
# ===========================================================================

class TestBuildTemporalContextSection:
    def test_section_contains_motion_summary(self):
        try:
            from PIL import Image as _PILImage
        except ImportError:
            pytest.skip("PIL not available")

        engine = _make_engine_with_recorder(running=True)
        brain = MagicMock()
        planner = ActionPlanner(engine=engine, brain=brain)

        recorder = MagicMock(spec=ScreenRecorder)
        recorder.last_frame_diff = 10
        recorder.motion_score = 25.5

        frame = _make_mock_frame(timestamp=time.monotonic() - 2.0)
        now = time.monotonic()
        section = planner._build_temporal_context_section([frame], recorder, now)

        assert "MOTION SUMMARY" in section
        assert "10" in section  # last_frame_diff
        assert "25.5" in section  # motion_score

    def test_section_contains_relative_timestamp(self):
        try:
            from PIL import Image as _PILImage
        except ImportError:
            pytest.skip("PIL not available")

        engine = _make_engine_with_recorder(running=True)
        brain = MagicMock()
        planner = ActionPlanner(engine=engine, brain=brain)

        recorder = MagicMock(spec=ScreenRecorder)
        recorder.last_frame_diff = 0
        recorder.motion_score = 0.0

        now = time.monotonic()
        frame = _make_mock_frame(timestamp=now - 3.0)
        section = planner._build_temporal_context_section([frame], recorder, now)

        assert "Frame -3." in section  # relative timestamp ~3.0s

    def test_empty_frames_list_produces_motion_summary_only(self):
        engine = _make_engine_with_recorder(running=True)
        brain = MagicMock()
        planner = ActionPlanner(engine=engine, brain=brain)

        recorder = MagicMock(spec=ScreenRecorder)
        recorder.last_frame_diff = 0
        recorder.motion_score = 0.0

        section = planner._build_temporal_context_section([], recorder, time.monotonic())
        assert "MOTION SUMMARY" in section


# ===========================================================================
# Unit tests — plan_next temporal context injection
# ===========================================================================

class TestPlanNextTemporalInjection:
    def _make_planner_with_recorder(self, recorder_running: bool = True):
        engine = ComputerUseEngine.__new__(ComputerUseEngine)
        engine._context = {}
        engine._enable_temporal_context = True

        recorder = MagicMock(spec=ScreenRecorder)
        recorder.is_running = recorder_running
        recorder.last_frame_diff = 5
        recorder.motion_score = 10.0
        recorder._config = MagicMock()
        recorder._config.temporal_context_frames = 2
        engine._screen_recorder = recorder

        engine._logger = MagicMock()
        engine._logger._make_thumbnail = MagicMock(return_value="")
        engine._screen = MagicMock()
        engine._screen.ocr = AsyncMock(return_value=[])
        engine._vision = MagicMock()
        engine._vision.find_element = AsyncMock(return_value=[])

        brain = MagicMock()
        brain.ask = MagicMock(return_value='[{"action_type": "screenshot", "params": {}, "confidence": 0.9, "rationale": "test"}]')

        planner = ActionPlanner(engine=engine, brain=brain)
        return planner, recorder

    def test_temporal_context_injected_when_recorder_running(self):
        try:
            from PIL import Image as _PILImage
        except ImportError:
            pytest.skip("PIL not available")

        planner, recorder = self._make_planner_with_recorder(recorder_running=True)

        frame = _make_mock_frame(timestamp=time.monotonic() - 1.0)
        recorder.get_recent_frames = AsyncMock(return_value=[frame])

        screenshot = MagicMock()
        _run(planner.plan_next("test goal", screenshot, []))

        # brain.ask should have been called with a prompt containing temporal context
        call_args = planner._brain.ask.call_args[0][0]
        assert "TEMPORAL CONTEXT" in call_args or "MOTION SUMMARY" in call_args

    def test_no_temporal_context_when_recorder_not_running(self):
        planner, recorder = self._make_planner_with_recorder(recorder_running=False)
        recorder.get_recent_frames = AsyncMock(return_value=[])

        screenshot = MagicMock()
        _run(planner.plan_next("test goal", screenshot, []))

        call_args = planner._brain.ask.call_args[0][0]
        assert "TEMPORAL CONTEXT" not in call_args

    def test_no_temporal_context_when_recorder_is_none(self):
        engine = ComputerUseEngine.__new__(ComputerUseEngine)
        engine._context = {}
        engine._enable_temporal_context = False
        engine._screen_recorder = None
        engine._logger = MagicMock()
        engine._logger._make_thumbnail = MagicMock(return_value="")
        engine._screen = MagicMock()
        engine._screen.ocr = AsyncMock(return_value=[])
        engine._vision = MagicMock()
        engine._vision.find_element = AsyncMock(return_value=[])

        brain = MagicMock()
        brain.ask = MagicMock(return_value='[{"action_type": "screenshot", "params": {}, "confidence": 0.9, "rationale": "test"}]')

        planner = ActionPlanner(engine=engine, brain=brain)
        screenshot = MagicMock()
        # Should not raise
        _run(planner.plan_next("test goal", screenshot, []))

        call_args = planner._brain.ask.call_args[0][0]
        assert "TEMPORAL CONTEXT" not in call_args

    def test_no_exception_when_buffer_empty(self):
        planner, recorder = self._make_planner_with_recorder(recorder_running=True)
        recorder.get_recent_frames = AsyncMock(return_value=[])

        screenshot = MagicMock()
        # Should not raise even with empty buffer
        result = _run(planner.plan_next("test goal", screenshot, []))
        assert result is not None
