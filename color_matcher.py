"""
ColorMatcher: LAB histogram matching and sharpness adjustment for generated frames.
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColorMatchRequest:
    frame: np.ndarray                    # (H, W, 3) float32 RGB [0,1]
    reference: Optional[np.ndarray] = None  # (H, W, 3) float32 RGB [0,1]
    sharpness_target: Optional[float] = None  # Laplacian variance
    frame_index: Optional[int] = None    # for logging on exception


class ColorMatcher:
    DELTA_E_THRESHOLD = 10.0
    SHARPNESS_TOLERANCE = 0.20

    def match(self, request: ColorMatchRequest) -> np.ndarray:
        """Returns color-matched frame. If reference is None, returns frame unchanged."""
        # Requirement 6.5: no reference → return unmodified
        if request.reference is None:
            return request.frame

        try:
            result = self._apply_color_match(request.frame, request.reference)

            if request.sharpness_target is not None:
                result = self._adjust_sharpness(result, request.sharpness_target)

            return result

        except Exception as exc:
            idx = request.frame_index
            label = f"frame {idx}" if idx is not None else "frame"
            logger.error("ColorMatcher.match failed for %s: %s", label, exc)
            return request.frame

    def _apply_color_match(
        self, frame: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Apply LAB histogram matching per channel. Returns float32 RGB [0,1]."""
        from skimage.color import rgb2lab, lab2rgb
        from skimage.exposure import match_histograms

        # Validate shapes before conversion
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        if reference.ndim != 3 or reference.shape[2] != 3:
            raise ValueError(f"Unexpected reference shape: {reference.shape}")

        frame_lab = rgb2lab(frame.astype(np.float32))
        ref_lab = rgb2lab(reference.astype(np.float32))

        # match_histograms with channel_axis=-1 matches each channel independently
        matched_lab = match_histograms(frame_lab, ref_lab, channel_axis=-1)

        result = lab2rgb(matched_lab).astype(np.float32)
        return np.clip(result, 0.0, 1.0)

    def _adjust_sharpness(self, frame: np.ndarray, target: float) -> np.ndarray:
        """Single-pass sharpness correction using PIL filters."""
        from PIL import Image, ImageFilter

        current = self.compute_sharpness(frame)

        if abs(current - target) <= target * self.SHARPNESS_TOLERANCE:
            return frame  # already within tolerance

        # Convert to uint8 PIL image for filtering
        pil_img = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))

        if current < target:
            # Need more sharpness → unsharp mask
            pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        else:
            # Need less sharpness → gaussian blur
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))

        return np.array(pil_img, dtype=np.float32) / 255.0

    def compute_delta_e(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """Mean CIE76 delta-E in LAB space."""
        from skimage.color import rgb2lab

        frame_lab = rgb2lab(frame.astype(np.float32))
        ref_lab = rgb2lab(reference.astype(np.float32))

        # CIE76: sqrt(sum of squared differences across L, a, b channels)
        diff = frame_lab - ref_lab
        delta_e = np.sqrt(np.sum(diff ** 2, axis=-1))
        return float(np.mean(delta_e))

    def compute_sharpness(self, frame: np.ndarray) -> float:
        """Laplacian variance of the grayscale frame."""
        from PIL import Image, ImageFilter

        pil_img = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))
        gray = pil_img.convert("L")
        lap = gray.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1,
            offset=0,
        ))
        lap_arr = np.array(lap, dtype=np.float32)
        return float(np.var(lap_arr))
