"""
Skill Gap Detector for the Janus Reasoning Engine.

Compares opportunity requirements to the current skill inventory,
estimates learning time, and calculates ROI to decide whether
learning a skill is worth the investment.

**Validates: Requirements REQ-3.1, REQ-5.3**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default heuristic: ~2 hours per unknown skill
DEFAULT_LEARNING_HOURS_PER_SKILL = 2.0
# Default hourly rate used for ROI calculation when none is provided
DEFAULT_HOURLY_RATE = 50.0
# Minimum ROI ratio to consider learning worthwhile
DEFAULT_ROI_THRESHOLD = 1.5


@dataclass
class SkillGap:
    """Represents a gap between a required skill and current capability."""

    skill_name: str
    required_confidence: float  # 0.0–1.0 confidence needed
    current_confidence: float   # 0.0–1.0 confidence currently held
    gap_size: float             # required - current (clamped to [0, 1])
    estimated_learning_hours: float
    roi: float                  # opportunity_value / (learning_hours * hourly_rate)
    is_learnable: bool = True   # False if gap is too large to bridge quickly
    notes: str = ""

    @property
    def is_known(self) -> bool:
        """True when current confidence meets or exceeds the requirement."""
        return self.current_confidence >= self.required_confidence


class SkillGapDetector:
    """
    Detects skill gaps between opportunity requirements and current inventory.

    Uses JanusGPT to estimate learning time when available; falls back to a
    simple heuristic of DEFAULT_LEARNING_HOURS_PER_SKILL per unknown skill.
    """

    def __init__(
        self,
        janus_gpt=None,
        hourly_rate: float = DEFAULT_HOURLY_RATE,
        roi_threshold: float = DEFAULT_ROI_THRESHOLD,
    ) -> None:
        """
        Args:
            janus_gpt: Optional JanusGPT instance for LLM-based estimates.
            hourly_rate: Owner's effective hourly rate (used in ROI calc).
            roi_threshold: Minimum ROI ratio to recommend learning.
        """
        self.janus_gpt = janus_gpt
        self.hourly_rate = hourly_rate
        self.roi_threshold = roi_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_gaps(
        self,
        opportunity: Dict[str, Any],
        skill_inventory: Dict[str, float],
    ) -> List[SkillGap]:
        """
        Compare opportunity requirements to the current skill inventory.

        Args:
            opportunity: Dict with at least:
                - "required_skills": List[str] or Dict[str, float]
                  (if dict, values are required confidence levels 0–1)
                - "value": float  (estimated monetary value of the opportunity)
            skill_inventory: Mapping of skill_name -> current confidence (0–1).

        Returns:
            List of SkillGap objects for every skill that has a gap.
        """
        required_skills = opportunity.get("required_skills", [])
        opportunity_value = float(opportunity.get("value", 0.0))

        # Normalise required_skills to {name: required_confidence}
        if isinstance(required_skills, list):
            requirements: Dict[str, float] = {s: 0.7 for s in required_skills}
        elif isinstance(required_skills, dict):
            requirements = {k: float(v) for k, v in required_skills.items()}
        else:
            requirements = {}

        gaps: List[SkillGap] = []
        for skill_name, required_conf in requirements.items():
            current_conf = skill_inventory.get(skill_name, 0.0)
            gap_size = max(0.0, required_conf - current_conf)

            if gap_size <= 0.0:
                # No gap — skill is already sufficient
                continue

            learning_hours = self._estimate_learning_time(skill_name, gap_size)
            roi = self._calculate_roi(opportunity_value, learning_hours)

            gaps.append(
                SkillGap(
                    skill_name=skill_name,
                    required_confidence=required_conf,
                    current_confidence=current_conf,
                    gap_size=gap_size,
                    estimated_learning_hours=learning_hours,
                    roi=roi,
                )
            )

        return gaps

    def should_learn(
        self,
        skill_gap: SkillGap,
        opportunity_value: float,
    ) -> bool:
        """
        Decide whether learning a skill is worth the investment.

        Returns True when the ROI exceeds the configured threshold.

        Args:
            skill_gap: The SkillGap to evaluate.
            opportunity_value: Monetary value of the opportunity.

        Returns:
            True if learning is recommended.
        """
        if not skill_gap.is_learnable:
            return False

        # Recalculate ROI with the provided opportunity value (may differ from
        # the value used when the gap was first detected).
        roi = self._calculate_roi(opportunity_value, skill_gap.estimated_learning_hours)
        return roi >= self.roi_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_learning_time(self, skill_name: str, gap_size: float) -> float:
        """
        Estimate hours needed to close a skill gap.

        Tries JanusGPT first; falls back to heuristic.
        """
        if self.janus_gpt is not None:
            try:
                prompt = (
                    f"Estimate the number of hours a motivated adult would need "
                    f"to learn '{skill_name}' well enough to use it professionally. "
                    f"The learner already has {(1.0 - gap_size) * 100:.0f}% of the "
                    f"required knowledge. Reply with a single number only."
                )
                response = self.janus_gpt.generate(prompt, max_new=20)
                # Extract first number from response
                import re
                numbers = re.findall(r"\d+(?:\.\d+)?", response)
                if numbers:
                    hours = float(numbers[0])
                    # Sanity-clamp: 0.5 h – 200 h
                    return max(0.5, min(200.0, hours))
            except Exception as exc:
                logger.debug("JanusGPT learning-time estimate failed: %s", exc)

        # Heuristic fallback: scale by gap size
        return DEFAULT_LEARNING_HOURS_PER_SKILL * max(gap_size, 0.1) / 0.5

    def _calculate_roi(self, opportunity_value: float, learning_hours: float) -> float:
        """
        ROI = opportunity_value / (learning_hours * hourly_rate).

        Returns 0.0 when learning_hours or hourly_rate is zero.
        """
        cost = learning_hours * self.hourly_rate
        if cost <= 0.0:
            return 0.0
        return opportunity_value / cost
