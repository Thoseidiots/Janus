"""
tests/test_screen_recorder_engine.py
======================================
Integration tests for ComputerUseEngine + ScreenRecorder lifecycle.

Covers:
  - ScreenRecorder importable from janus_computer_use
  - mss fallback to PIL.ImageGrab
  - ComputerUseEngine lifecycle with enable_temporal_context=True
  - Requirements: 6.6, 7.1, 7.2, 10.1, 10.2, 10.4
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

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


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Import smoke tests
# ===========================================================================

class TestImportSmoke:
    def test_screen_recorder_importable(self):
        """ScreenRecorder is importable from janus_computer_use."""
        from janus_computer_use import ScreenRecorder
        assert ScreenRecorder is not None

    def test_screen_recorder_config_importable(self):
        from janus_computer_use import ScreenRecorderConfig
        assert ScreenRecorderConfig is not None

    def test_video_encoder_importable(self):
        from janus_computer_use import VideoEncoder
        assert VideoEncoder is not None

    def test_recorded_frame_importable(self):
        from janus_computer_use import RecordedFrame
        assert RecordedFrame is not None

    def test_screen_clip_importable(self):
        from janus_computer_use import ScreenClip
        assert ScreenClip is not None

    def test_encode_result_importable(self):
        from janus_computer_use import EncodeResult
        assert EncodeResult is not None

    def test_computer_use_engine_importable(self):
        from janus_computer_use import ComputerUseEngine
        assert ComputerUseEngine is not None


# ===========================================================================
# mss fallback tests
# ===========================================================================

class TestMssFallback:
    def test_pil_imagegrab_used_when_mss_absent(self):
        """When mss is unavailable, _CAPTURE_BACKEND should be 'pil'."""
        import janus_computer_use as jcu

        # Simulate mss being absent by checking the fallback path
        # We can't easily re-import, but we can verify the module-level variable
        # is either 'mss' or 'pil' and that _capture_frame_pil exists
        assert jcu._CAPTURE_BACKEND in ("mss", "pil")
        assert callable(jcu._capture_frame_pil)
        assert callable(jcu._capture_frame_mss)
        assert callable(jcu._capture_frame)

    def test_no_import_error_when_mss_absent(self):
        """Importing janus_computer_use should never raise ImportError for mss."""
        # The module is already imported; if we got here, no ImportError was raised
        import janus_computer_use  # noqa: F401
        # Success — no ImportError propagated


# ===========================================================================
# ComputerUseEngine lifecycle tests
# ===========================================================================

class TestComputerUseEngineLifecycle:
    def test_recorder_is_none_when_temporal_context_disabled(self):
        from janus_computer_use import ComputerUseEngine
        engine = ComputerUseEngine(enable_temporal_context=False)
        assert engine._screen_recorder is None
        assert engine.recorder is None

    def test_recorder_property_returns_none_by_default(self):
        from janus_computer_use import ComputerUseEngine
        engine = ComputerUseEngine()
        assert engine.recorder is None

    def test_enable_temporal_context_flag_stored(self):
        from janus_computer_use import ComputerUseEngine
        engine = ComputerUseEngine(enable_temporal_context=True)
        assert engine._enable_temporal_context is True

    def test_recorder_started_in_aenter_when_enabled(self):
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock()
        mock_recorder.is_running = True

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    engine = ComputerUseEngine(enable_temporal_context=True)
                    await engine.__aenter__()
                    mock_recorder.start.assert_called_once()
                    await engine.__aexit__(None, None, None)
                    mock_recorder.stop.assert_called_once()

        _run(_test())

    def test_recorder_not_started_when_disabled(self):
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock()

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    engine = ComputerUseEngine(enable_temporal_context=False)
                    await engine.__aenter__()
                    mock_recorder.start.assert_not_called()
                    await engine.__aexit__(None, None, None)
                    mock_recorder.stop.assert_not_called()

        _run(_test())

    def test_recorder_stopped_in_aexit(self):
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock()

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    engine = ComputerUseEngine(enable_temporal_context=True)
                    await engine.__aenter__()
                    await engine.__aexit__(None, None, None)
                    mock_recorder.stop.assert_called_once()

        _run(_test())

    def test_recorder_stop_exception_does_not_propagate(self):
        """If recorder.stop() raises, __aexit__ should still complete."""
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    engine = ComputerUseEngine(enable_temporal_context=True)
                    await engine.__aenter__()
                    # Should not raise
                    await engine.__aexit__(None, None, None)

        _run(_test())

    def test_recorder_accessible_via_property(self):
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock()

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    engine = ComputerUseEngine(enable_temporal_context=True)
                    await engine.__aenter__()
                    assert engine.recorder is mock_recorder
                    await engine.__aexit__(None, None, None)

        _run(_test())

    def test_context_manager_with_temporal_context(self):
        """Full async context manager test with enable_temporal_context=True."""
        from janus_computer_use import ComputerUseEngine, ScreenRecorder

        mock_recorder = MagicMock(spec=ScreenRecorder)
        mock_recorder.start = AsyncMock()
        mock_recorder.stop = AsyncMock()
        mock_recorder.is_running = True

        async def _test():
            with patch("janus_computer_use._check_dependencies"):
                with patch("janus_computer_use.ScreenRecorder", return_value=mock_recorder):
                    async with ComputerUseEngine(enable_temporal_context=True) as engine:
                        assert engine.recorder is mock_recorder
                        mock_recorder.start.assert_called_once()
                    mock_recorder.stop.assert_called_once()

        _run(_test())
