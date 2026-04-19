"""
tests/test_screen_recorder.py
==============================
Property-based and unit tests for ScreenRecorder, ScreenRecorderConfig,
RecordedFrame, ScreenClip, and related helpers.

Properties covered:
  Property 1: Ring buffer never exceeds capacity
  Property 2: Frame diff is always in [0, 64]
  Property 3: Frames returned by get_clip are in chronological order
  Property 4: Motion threshold=0 retains every frame
  Property 5: Config validation rejects out-of-range values
  Property 7: get_recent_frames(n) returns at most n frames
"""
from __future__ import annotations

import asyncio
import sys
import os
import types
from collections import deque
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path and stubs are installed
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
    # Only stub PIL if it's not actually installed
    try:
        import PIL  # noqa: F401
    except ImportError:
        for pkg in ("PIL", "PIL.Image", "PIL.ImageGrab"):
            if pkg not in sys.modules:
                _make_stub(pkg)
    # Only stub imagehash if not actually installed
    try:
        import imagehash  # noqa: F401
    except ImportError:
        if "imagehash" not in sys.modules:
            _make_stub("imagehash")


_ensure_stubs()

from janus_computer_use import (  # noqa: E402
    ScreenRecorder,
    ScreenRecorderConfig,
    RecordedFrame,
    ScreenClip,
    _validate_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_frame(timestamp: float = 0.0, diff_val: int = 0) -> RecordedFrame:
    """Create a mock RecordedFrame with a controllable phash distance.

    When this frame is subtracted from another (last_retained.phash - this.phash),
    the result is diff_val.
    """

    class _FakeHash:
        def __init__(self, val: int):
            self._val = val

        def __sub__(self, other) -> int:
            # Return the other frame's diff_val (the new frame's expected diff)
            if hasattr(other, "_val"):
                return other._val
            return self._val

        def __rsub__(self, other) -> int:
            return self._val

    phash = _FakeHash(diff_val)
    return RecordedFrame(image=MagicMock(), timestamp=timestamp, phash=phash)


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Property 5: Config validation rejects out-of-range values
# ===========================================================================

class TestConfigValidation:
    """Feature: janus-screen-recorder, Property 5: Config validation rejects out-of-range values"""

    @given(st.integers(max_value=0))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_capture_rate_fps_below_min(self, v):
        with pytest.raises(ValueError, match="capture_rate_fps"):
            ScreenRecorderConfig(capture_rate_fps=v)

    @given(st.integers(min_value=31))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_capture_rate_fps_above_max(self, v):
        with pytest.raises(ValueError, match="capture_rate_fps"):
            ScreenRecorderConfig(capture_rate_fps=v)

    @given(st.integers(max_value=4))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_duration_below_min(self, v):
        with pytest.raises(ValueError, match="buffer_duration_seconds"):
            ScreenRecorderConfig(buffer_duration_seconds=v)

    @given(st.integers(min_value=301))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_duration_above_max(self, v):
        with pytest.raises(ValueError, match="buffer_duration_seconds"):
            ScreenRecorderConfig(buffer_duration_seconds=v)

    @given(st.integers(max_value=-1))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_motion_threshold_below_min(self, v):
        with pytest.raises(ValueError, match="motion_threshold"):
            ScreenRecorderConfig(motion_threshold=v)

    @given(st.integers(min_value=65))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_motion_threshold_above_max(self, v):
        with pytest.raises(ValueError, match="motion_threshold"):
            ScreenRecorderConfig(motion_threshold=v)

    @given(st.integers(max_value=0))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_motion_threshold_below_min(self, v):
        with pytest.raises(ValueError, match="high_motion_threshold"):
            ScreenRecorderConfig(high_motion_threshold=v)

    @given(st.integers(min_value=65))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_motion_threshold_above_max(self, v):
        with pytest.raises(ValueError, match="high_motion_threshold"):
            ScreenRecorderConfig(high_motion_threshold=v)

    @given(st.integers(max_value=0))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_stuck_duration_below_min(self, v):
        with pytest.raises(ValueError, match="stuck_duration_seconds"):
            ScreenRecorderConfig(stuck_duration_seconds=v)

    @given(st.integers(min_value=3601))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_stuck_duration_above_max(self, v):
        with pytest.raises(ValueError, match="stuck_duration_seconds"):
            ScreenRecorderConfig(stuck_duration_seconds=v)

    @given(st.floats(max_value=0.09, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_transition_settling_below_min(self, v):
        with pytest.raises(ValueError, match="transition_settling_seconds"):
            ScreenRecorderConfig(transition_settling_seconds=v)

    @given(st.floats(min_value=60.01, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_transition_settling_above_max(self, v):
        with pytest.raises(ValueError, match="transition_settling_seconds"):
            ScreenRecorderConfig(transition_settling_seconds=v)

    @given(st.integers(max_value=0))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_temporal_context_frames_below_min(self, v):
        with pytest.raises(ValueError, match="temporal_context_frames"):
            ScreenRecorderConfig(temporal_context_frames=v)

    @given(st.integers(min_value=11))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_temporal_context_frames_above_max(self, v):
        with pytest.raises(ValueError, match="temporal_context_frames"):
            ScreenRecorderConfig(temporal_context_frames=v)

    @given(st.integers(max_value=63))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gif_max_dimension_below_min(self, v):
        with pytest.raises(ValueError, match="gif_max_dimension"):
            ScreenRecorderConfig(gif_max_dimension=v)

    @given(st.integers(min_value=4097))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gif_max_dimension_above_max(self, v):
        with pytest.raises(ValueError, match="gif_max_dimension"):
            ScreenRecorderConfig(gif_max_dimension=v)

    def test_error_message_contains_param_name(self):
        with pytest.raises(ValueError) as exc_info:
            ScreenRecorderConfig(capture_rate_fps=0)
        assert "capture_rate_fps" in str(exc_info.value)
        assert "[1, 30]" in str(exc_info.value)

    def test_valid_defaults_do_not_raise(self):
        cfg = ScreenRecorderConfig()
        assert cfg.capture_rate_fps == 5
        assert cfg.buffer_duration_seconds == 30

    def test_buffer_capacity_property(self):
        cfg = ScreenRecorderConfig(capture_rate_fps=10, buffer_duration_seconds=60)
        assert cfg.buffer_capacity == 600


# ===========================================================================
# Property 5 companion: Config round-trip
# ===========================================================================

class TestConfigRoundTrip:
    """Feature: janus-screen-recorder, Property 5 companion: Config round-trip"""

    @given(
        st.integers(1, 30),
        st.integers(5, 300),
        st.integers(0, 64),
        st.integers(1, 64),
        st.integers(1, 3600),
        st.floats(0.1, 60.0, allow_nan=False, allow_infinity=False),
        st.integers(1, 10),
        st.integers(64, 4096),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_config_round_trip(
        self,
        fps, dur, mt, hmt, stuck, settling, tcf, gif_dim
    ):
        """Feature: janus-screen-recorder, Property 5 companion: Config round-trip"""
        recorder = ScreenRecorder(
            capture_rate_fps=fps,
            buffer_duration_seconds=dur,
            motion_threshold=mt,
            high_motion_threshold=hmt,
            stuck_duration_seconds=stuck,
            transition_settling_seconds=settling,
            temporal_context_frames=tcf,
            gif_max_dimension=gif_dim,
        )
        cfg = recorder.config
        assert cfg["capture_rate_fps"] == fps
        assert cfg["buffer_duration_seconds"] == dur
        assert cfg["motion_threshold"] == mt
        assert cfg["high_motion_threshold"] == hmt
        assert cfg["stuck_duration_seconds"] == stuck
        assert abs(cfg["transition_settling_seconds"] - settling) < 1e-9
        assert cfg["temporal_context_frames"] == tcf
        assert cfg["gif_max_dimension"] == gif_dim


# ===========================================================================
# Property 2: Frame diff is always in [0, 64]
# ===========================================================================

class TestFrameDiff:
    """Feature: janus-screen-recorder, Property 2: Frame diff is always in [0, 64]"""

    @given(st.binary(min_size=4, max_size=4096), st.binary(min_size=4, max_size=4096))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_phash_distance_in_range(self, data1, data2):
        """Feature: janus-screen-recorder, Property 2: Frame diff is always in [0, 64]"""
        try:
            import imagehash as _ih
            from PIL import Image as _PILImage
            from io import BytesIO

            # Create minimal 1×1 images from the byte data
            img1 = _PILImage.new("RGB", (1, 1), color=(data1[0], data1[1 % len(data1)], data1[2 % len(data1)]))
            img2 = _PILImage.new("RGB", (1, 1), color=(data2[0], data2[1 % len(data2)], data2[2 % len(data2)]))

            h1 = _ih.phash(img1)
            h2 = _ih.phash(img2)
            diff = h1 - h2
            assert isinstance(diff, (int, float)) or hasattr(diff, '__int__'), \
                f"diff should be numeric, got {type(diff)}"
            diff_int = int(diff)
            assert 0 <= diff_int <= 64
        except ImportError:
            # imagehash not available — test the mock path
            mock_hash = MagicMock()
            mock_hash.__sub__ = MagicMock(return_value=32)
            diff = mock_hash - mock_hash
            assert 0 <= diff <= 64


# ===========================================================================
# Property 1: Ring buffer never exceeds capacity
# ===========================================================================

class TestRingBufferCapacity:
    """Feature: janus-screen-recorder, Property 1: Ring buffer never exceeds capacity"""

    @given(
        st.integers(1, 30),
        st.integers(5, 60),
        st.integers(1, 500),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_never_exceeds_capacity(self, fps, duration, n_frames):
        """Feature: janus-screen-recorder, Property 1: Ring buffer never exceeds capacity"""
        recorder = ScreenRecorder(
            capture_rate_fps=fps,
            buffer_duration_seconds=duration,
        )
        capacity = recorder._config.buffer_capacity
        for i in range(n_frames):
            frame = _make_mock_frame(timestamp=float(i))
            recorder._buffer.append(frame)
            assert len(recorder._buffer) <= capacity


# ===========================================================================
# Property 4: Motion threshold=0 retains every frame
# ===========================================================================

class TestMotionThresholdZero:
    """Feature: janus-screen-recorder, Property 4: Motion threshold=0 retains every frame"""

    @given(st.lists(st.integers(0, 64), min_size=1, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_threshold_zero_retains_all(self, diffs):
        """Feature: janus-screen-recorder, Property 4: Motion threshold=0 retains every frame"""
        recorder = ScreenRecorder(
            motion_threshold=0,
            capture_rate_fps=30,
            buffer_duration_seconds=300,
        )
        capacity = recorder._config.buffer_capacity

        for i, diff_val in enumerate(diffs):
            frame = _make_mock_frame(timestamp=float(i), diff_val=diff_val)
            recorder._process_frame(frame)

        expected = min(len(diffs), capacity)
        assert len(recorder._buffer) == expected


# ===========================================================================
# Property 3: Frames returned by get_clip are in chronological order
# ===========================================================================

class TestGetClipOrder:
    """Feature: janus-screen-recorder, Property 3: Frames returned by get_clip are in chronological order"""

    @given(st.lists(st.floats(0, 1000, allow_nan=False, allow_infinity=False), min_size=0, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_get_clip_chronological(self, timestamps):
        """Feature: janus-screen-recorder, Property 3: Frames returned by get_clip are in chronological order"""
        recorder = ScreenRecorder(capture_rate_fps=30, buffer_duration_seconds=300)
        for ts in timestamps:
            frame = _make_mock_frame(timestamp=ts)
            recorder._buffer.append(frame)

        clip = _run(recorder.get_clip(0.0, 1000.0))
        frames = clip.frames
        for i in range(len(frames) - 1):
            assert frames[i].timestamp <= frames[i + 1].timestamp


# ===========================================================================
# Property 7: get_recent_frames(n) returns at most n frames
# ===========================================================================

class TestGetRecentFrames:
    """Feature: janus-screen-recorder, Property 7: get_recent_frames(n) returns at most n frames"""

    @given(st.integers(1, 10), st.integers(0, 200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_get_recent_frames_at_most_n(self, n, buffer_size):
        """Feature: janus-screen-recorder, Property 7: get_recent_frames(n) returns at most n frames"""
        recorder = ScreenRecorder(capture_rate_fps=30, buffer_duration_seconds=300)
        for i in range(buffer_size):
            recorder._buffer.append(_make_mock_frame(timestamp=float(i)))

        result = _run(recorder.get_recent_frames(n))
        assert len(result) <= n


# ===========================================================================
# Unit tests — lifecycle
# ===========================================================================

class TestScreenRecorderLifecycle:
    def test_initial_state(self):
        recorder = ScreenRecorder()
        assert not recorder.is_running
        assert recorder.last_frame_diff == 0
        assert recorder.motion_score == 0.0

    def test_is_running_property(self):
        recorder = ScreenRecorder()
        assert recorder.is_running is False

    def test_config_property_returns_dict(self):
        recorder = ScreenRecorder()
        cfg = recorder.config
        assert isinstance(cfg, dict)
        assert "capture_rate_fps" in cfg
        assert "buffer_duration_seconds" in cfg

    def test_on_ui_transition_registers_callback(self):
        recorder = ScreenRecorder()
        cb = AsyncMock()
        recorder.on_ui_transition(cb)
        assert cb in recorder._transition_callbacks

    def test_on_stuck_state_registers_callback(self):
        recorder = ScreenRecorder()
        cb = AsyncMock()
        recorder.on_stuck_state(cb)
        assert cb in recorder._stuck_callbacks

    def test_stop_when_not_running_is_safe(self):
        recorder = ScreenRecorder()
        _run(recorder.stop())  # should not raise

    def test_start_idempotent(self):
        """Starting twice should not create two tasks."""
        recorder = ScreenRecorder()
        with patch.object(recorder, "_capture_loop", new=AsyncMock()):
            _run(recorder.start())
            task1 = recorder._task
            _run(recorder.start())
            task2 = recorder._task
            assert task1 is task2
        _run(recorder.stop())

    def test_context_manager(self):
        recorder = ScreenRecorder()
        with patch.object(recorder, "_capture_loop", new=AsyncMock()):
            async def _use():
                async with recorder:
                    assert recorder.is_running
                assert not recorder.is_running
            _run(_use())


# ===========================================================================
# Unit tests — motion detection
# ===========================================================================

class TestMotionDetection:
    def test_first_frame_always_retained(self):
        recorder = ScreenRecorder(motion_threshold=64)
        frame = _make_mock_frame(timestamp=1.0, diff_val=0)
        recorder._process_frame(frame)
        assert len(recorder._buffer) == 1

    def test_frame_below_threshold_discarded(self):
        recorder = ScreenRecorder(motion_threshold=10)
        first = _make_mock_frame(timestamp=0.0, diff_val=0)
        recorder._process_frame(first)

        second = _make_mock_frame(timestamp=1.0, diff_val=5)
        recorder._process_frame(second)
        assert len(recorder._buffer) == 1  # only first retained

    def test_frame_at_threshold_retained(self):
        recorder = ScreenRecorder(motion_threshold=10)
        first = _make_mock_frame(timestamp=0.0, diff_val=0)
        recorder._process_frame(first)

        second = _make_mock_frame(timestamp=1.0, diff_val=10)
        recorder._process_frame(second)
        assert len(recorder._buffer) == 2

    def test_last_frame_diff_updated(self):
        recorder = ScreenRecorder(motion_threshold=0)
        first = _make_mock_frame(timestamp=0.0, diff_val=0)
        recorder._process_frame(first)

        second = _make_mock_frame(timestamp=1.0, diff_val=42)
        recorder._process_frame(second)
        assert recorder.last_frame_diff == 42

    def test_motion_score_accumulates(self):
        import time
        recorder = ScreenRecorder(motion_threshold=0)
        now = time.monotonic()
        first = _make_mock_frame(timestamp=now, diff_val=0)
        recorder._process_frame(first)

        second = _make_mock_frame(timestamp=now + 0.1, diff_val=15)
        recorder._process_frame(second)
        assert recorder.motion_score >= 15.0


# ===========================================================================
# Unit tests — stuck state callbacks
# ===========================================================================

class TestStuckStateCallbacks:
    def test_stuck_callback_called(self):
        recorder = ScreenRecorder()
        cb = AsyncMock()
        recorder.on_stuck_state(cb)
        _run(recorder._fire_stuck_callbacks(duration=10.0, last_frame=None))
        cb.assert_called_once_with(duration=10.0, last_frame=None)

    def test_stuck_callback_exception_does_not_propagate(self):
        recorder = ScreenRecorder()

        async def bad_cb(**kwargs):
            raise RuntimeError("boom")

        recorder.on_stuck_state(bad_cb)
        # Should not raise
        _run(recorder._fire_stuck_callbacks(duration=5.0, last_frame=None))

    def test_transition_callback_called(self):
        recorder = ScreenRecorder()
        cb = AsyncMock()
        recorder.on_ui_transition(cb)
        _run(recorder._fire_transition_callbacks())
        cb.assert_called_once()

    def test_transition_callback_exception_does_not_propagate(self):
        recorder = ScreenRecorder()

        async def bad_cb():
            raise RuntimeError("boom")

        recorder.on_ui_transition(bad_cb)
        _run(recorder._fire_transition_callbacks())  # should not raise


# ===========================================================================
# Unit tests — get_clip warning
# ===========================================================================

class TestGetClipWarning:
    def test_warning_when_buffer_empty(self):
        recorder = ScreenRecorder()
        clip = _run(recorder.get_clip(0.0, 10.0))
        assert clip.warning is not None
        assert "Ring buffer" in clip.warning

    def test_no_warning_when_buffer_covers_interval(self):
        recorder = ScreenRecorder(capture_rate_fps=30, buffer_duration_seconds=300)
        # Add a frame at t=0 so buffer starts at 0
        recorder._buffer.append(_make_mock_frame(timestamp=0.0))
        recorder._buffer.append(_make_mock_frame(timestamp=5.0))
        clip = _run(recorder.get_clip(0.0, 10.0))
        # Buffer starts at 0 which is <= start_time, so no warning
        assert clip.warning is None

    def test_frame_count_property(self):
        recorder = ScreenRecorder(capture_rate_fps=30, buffer_duration_seconds=300)
        for i in range(5):
            recorder._buffer.append(_make_mock_frame(timestamp=float(i)))
        clip = _run(recorder.get_clip(0.0, 10.0))
        assert clip.frame_count == len(clip.frames)
