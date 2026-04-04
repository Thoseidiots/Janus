"""
Temporal Consistency Guard for the Synthetic Media Pipeline.

Enforces frame-to-frame coherence in generated video sequences by detecting
and correcting freeze, drift, and flash violations.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import center_of_mass
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class ConsistencyViolation:
    frame_index: int
    constraint: Literal['freeze', 'drift', 'flash', 'exhausted']
    value: float
    threshold: float


def _compute_centroid(frame: np.ndarray) -> tuple[float, float]:
    """Compute the center of mass of the grayscale image."""
    gray = frame.mean(axis=2)  # (H, W)
    h, w = gray.shape
    if _SCIPY_AVAILABLE:
        cy, cx = center_of_mass(gray)
        # scipy returns NaN for all-zero images; fall back to frame center
        if cy != cy or cx != cx:  # NaN check
            return h / 2.0, w / 2.0
    else:
        # Fallback: weighted centroid
        total = gray.sum()
        if total == 0:
            return h / 2.0, w / 2.0
        ys = np.arange(gray.shape[0])
        xs = np.arange(gray.shape[1])
        cy = (gray * ys[:, None]).sum() / total
        cx = (gray * xs[None, :]).sum() / total
    return float(cy), float(cx)


def _check_violation(
    prev: np.ndarray,
    curr: np.ndarray,
    frame_index: int,
    drift_threshold: float,
    flash_threshold: float,
) -> ConsistencyViolation | None:
    """Check a consecutive frame pair for any consistency violation.

    Returns the first violation found, or None if the pair is clean.
    """
    # 1. Freeze check
    if np.array_equal(prev, curr):
        return ConsistencyViolation(
            frame_index=frame_index,
            constraint='freeze',
            value=1.0,
            threshold=0.0,
        )

    h, w = prev.shape[:2]
    max_dim = max(h, w)

    # 2. Drift check
    cy_prev, cx_prev = _compute_centroid(prev)
    cy_curr, cx_curr = _compute_centroid(curr)
    shift = np.sqrt((cy_curr - cy_prev) ** 2 + (cx_curr - cx_prev) ** 2)
    drift_value = shift / max_dim
    if drift_value > drift_threshold:
        return ConsistencyViolation(
            frame_index=frame_index,
            constraint='drift',
            value=float(drift_value),
            threshold=drift_threshold,
        )

    # 3. Flash check
    lum_delta = abs(float(prev.mean()) - float(curr.mean()))
    if lum_delta > flash_threshold:
        return ConsistencyViolation(
            frame_index=frame_index,
            constraint='flash',
            value=lum_delta,
            threshold=flash_threshold,
        )

    return None


class TemporalConsistencyGuard:
    """Enforces temporal consistency constraints across a video frame sequence."""

    DRIFT_THRESHOLD = 0.20   # fraction of frame dimension
    FLASH_THRESHOLD = 0.25   # normalized luminance delta
    MAX_RETRIES = 3

    def check_sequence(
        self,
        frames: list[np.ndarray],
        regenerate_fn: Callable[[int, np.ndarray], np.ndarray],
    ) -> tuple[list[np.ndarray], list[ConsistencyViolation]]:
        """Check and correct a sequence of frames for temporal consistency.

        Args:
            frames: List of (H, W, 3) float32 arrays in [0, 1].
            regenerate_fn: Callable(frame_index, prev_valid_frame) -> np.ndarray.
                           Called to regenerate a violating frame.

        Returns:
            (corrected_frames, violations): The corrected frame list and a list
            of all ConsistencyViolation objects encountered.
        """
        if not frames:
            return [], []

        corrected: list[np.ndarray] = [frames[0]]
        violations: list[ConsistencyViolation] = []

        for i in range(1, len(frames)):
            prev_valid = corrected[-1]
            candidate = frames[i]

            violation = _check_violation(
                prev_valid, candidate, i,
                self.DRIFT_THRESHOLD, self.FLASH_THRESHOLD,
            )

            if violation is None:
                corrected.append(candidate)
                continue

            # --- Violation detected: attempt up to MAX_RETRIES regenerations ---
            violations.append(violation)
            resolved = False

            for attempt in range(self.MAX_RETRIES):
                try:
                    candidate = regenerate_fn(i, prev_valid)
                except Exception as exc:
                    logger.warning(
                        "regenerate_fn raised on frame %d attempt %d: %s",
                        i, attempt + 1, exc,
                    )
                    # Count this attempt and continue
                    continue

                retry_violation = _check_violation(
                    prev_valid, candidate, i,
                    self.DRIFT_THRESHOLD, self.FLASH_THRESHOLD,
                )
                if retry_violation is None:
                    resolved = True
                    break

            if resolved:
                corrected.append(candidate)
            else:
                # All retries exhausted — substitute last valid frame
                exhausted_violation = ConsistencyViolation(
                    frame_index=i,
                    constraint='exhausted',
                    value=violation.value,
                    threshold=violation.threshold,
                )
                violations.append(exhausted_violation)
                logger.warning(
                    "Frame %d: all %d retries exhausted for constraint '%s' "
                    "(value=%.4f, threshold=%.4f); substituting last valid frame.",
                    i, self.MAX_RETRIES, violation.constraint,
                    violation.value, violation.threshold,
                )
                corrected.append(prev_valid)

        return corrected, violations
