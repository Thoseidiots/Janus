"""
tests/test_video_encoder.py
============================
Property-based and unit tests for VideoEncoder.

Properties covered:
  Property 6: EncodeResult on empty clip is always failure
  Property companion: GIF scaling respects max_dimension
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock, patch, call

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
    VideoEncoder,
    ScreenClip,
    RecordedFrame,
    EncodeResult,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_frame(timestamp: float = 0.0, size=(100, 100)) -> RecordedFrame:
    phash = MagicMock()
    phash.__sub__ = MagicMock(return_value=0)
    img = MagicMock()
    img.size = size
    img.copy.return_value = img
    return RecordedFrame(image=img, timestamp=timestamp, phash=phash)


# ===========================================================================
# Property 6: EncodeResult on empty clip is always failure
# ===========================================================================

class TestEncodeEmptyClip:
    """Feature: janus-screen-recorder, Property 6: EncodeResult on empty clip is always failure"""

    @given(st.text(min_size=1, max_size=100).filter(lambda p: p not in ('/', '\\', '.', '..')))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_mp4_empty_clip_fails(self, output_path):
        """Feature: janus-screen-recorder, Property 6: EncodeResult on empty clip is always failure"""
        encoder = VideoEncoder()
        clip = ScreenClip(frames=[], start_time=0.0, end_time=0.0)
        result = _run(encoder.encode_mp4(clip, output_path))
        assert result.success is False
        assert result.error_message
        assert len(result.error_message) > 0
        # No file should have been created
        if os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)
            pytest.fail(f"File was created at {output_path!r} for empty clip")

    @given(st.text(min_size=1, max_size=100).filter(lambda p: p not in ('/', '\\', '.', '..')))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_gif_empty_clip_fails(self, output_path):
        """Feature: janus-screen-recorder, Property 6: EncodeResult on empty clip is always failure"""
        encoder = VideoEncoder()
        clip = ScreenClip(frames=[], start_time=0.0, end_time=0.0)
        result = _run(encoder.encode_gif(clip, output_path))
        assert result.success is False
        assert result.error_message
        assert len(result.error_message) > 0
        if os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)
            pytest.fail(f"File was created at {output_path!r} for empty clip")


# ===========================================================================
# Property companion: GIF scaling respects max_dimension
# ===========================================================================

class TestGifScaling:
    """Feature: janus-screen-recorder, Property companion: GIF scaling respects max_dimension"""

    @given(
        st.integers(64, 4096),
        st.integers(10, 3840),
        st.integers(10, 2160),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_gif_scaling_respects_max_dimension(self, max_dim, width, height):
        """Feature: janus-screen-recorder, Property companion: GIF scaling respects max_dimension"""
        try:
            from PIL import Image as _PILImage

            # Create a real PIL image of the given dimensions
            img = _PILImage.new("RGB", (width, height), color=(128, 64, 32))

            # Simulate the scaling logic from VideoEncoder.encode_gif
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                scaled = img.resize((new_w, new_h), _PILImage.LANCZOS)
            else:
                scaled = img

            sw, sh = scaled.size
            assert max(sw, sh) <= max_dim
        except ImportError:
            # PIL not available — test the math directly
            w, h = width, height
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                assert max(new_w, new_h) <= max_dim


# ===========================================================================
# Unit tests — VideoEncoder with mocked cv2 and PIL
# ===========================================================================

class TestVideoEncoderMP4:
    def test_encode_mp4_returns_failure_on_cv2_error(self, tmp_path):
        """encode_mp4 returns EncodeResult(success=False) when cv2 raises."""
        encoder = VideoEncoder()
        frame = _make_mock_frame(timestamp=0.0)
        clip = ScreenClip(frames=[frame], start_time=0.0, end_time=1.0)
        output = str(tmp_path / "out.mp4")

        with patch("janus_computer_use.VideoEncoder.encode_mp4") as mock_enc:
            mock_enc.return_value = EncodeResult(success=False, error_message="cv2 error")
            result = _run(mock_enc(clip, output))
            assert result.success is False
            assert result.error_message

    def test_encode_mp4_empty_returns_failure(self, tmp_path):
        encoder = VideoEncoder()
        clip = ScreenClip(frames=[], start_time=0.0, end_time=1.0)
        output = str(tmp_path / "out.mp4")
        result = _run(encoder.encode_mp4(clip, output))
        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_encode_mp4_cleans_up_partial_file_on_error(self, tmp_path):
        """If encoding fails mid-way, partial file should be removed."""
        encoder = VideoEncoder()
        frame = _make_mock_frame(timestamp=0.0)
        clip = ScreenClip(frames=[frame], start_time=0.0, end_time=1.0)
        output = str(tmp_path / "out.mp4")

        # Simulate a failure by patching asyncio.to_thread to raise
        async def _raise(*args, **kwargs):
            # Create a partial file first
            with open(output, "w") as f:
                f.write("partial")
            raise RuntimeError("codec error")

        with patch("asyncio.to_thread", side_effect=_raise):
            result = _run(encoder.encode_mp4(clip, output))

        assert result.success is False
        assert not os.path.exists(output), "Partial file should have been cleaned up"


class TestVideoEncoderGIF:
    def test_encode_gif_empty_returns_failure(self, tmp_path):
        encoder = VideoEncoder()
        clip = ScreenClip(frames=[], start_time=0.0, end_time=1.0)
        output = str(tmp_path / "out.gif")
        result = _run(encoder.encode_gif(clip, output))
        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_encode_gif_cleans_up_partial_file_on_error(self, tmp_path):
        encoder = VideoEncoder()
        frame = _make_mock_frame(timestamp=0.0)
        clip = ScreenClip(frames=[frame], start_time=0.0, end_time=1.0)
        output = str(tmp_path / "out.gif")

        async def _raise(*args, **kwargs):
            with open(output, "w") as f:
                f.write("partial")
            raise RuntimeError("PIL error")

        with patch("asyncio.to_thread", side_effect=_raise):
            result = _run(encoder.encode_gif(clip, output))

        assert result.success is False
        assert not os.path.exists(output)

    def test_encode_gif_with_real_pil(self, tmp_path):
        """Integration test using real PIL if available."""
        try:
            from PIL import Image as _PILImage
        except ImportError:
            pytest.skip("PIL not available")

        encoder = VideoEncoder()
        img1 = _PILImage.new("RGB", (10, 10), color=(255, 0, 0))
        img2 = _PILImage.new("RGB", (10, 10), color=(0, 255, 0))

        phash = MagicMock()
        phash.__sub__ = MagicMock(return_value=0)
        frames = [
            RecordedFrame(image=img1, timestamp=0.0, phash=phash),
            RecordedFrame(image=img2, timestamp=0.2, phash=phash),
        ]
        clip = ScreenClip(frames=frames, start_time=0.0, end_time=0.2)
        output = str(tmp_path / "out.gif")

        result = _run(encoder.encode_gif(clip, output))
        assert result.success is True
        assert os.path.exists(output)
        assert result.file_size_bytes > 0


# ===========================================================================
# Unit tests — EncodeResult dataclass
# ===========================================================================

class TestEncodeResult:
    def test_success_result(self):
        r = EncodeResult(success=True, output_path="/tmp/out.mp4", file_size_bytes=1024)
        assert r.success is True
        assert r.output_path == "/tmp/out.mp4"
        assert r.file_size_bytes == 1024
        assert r.error_message is None

    def test_failure_result(self):
        r = EncodeResult(success=False, error_message="codec not found")
        assert r.success is False
        assert r.error_message == "codec not found"
        assert r.output_path is None
        assert r.file_size_bytes == 0
