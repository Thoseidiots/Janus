"""
Metacognition module for the Janus Autonomous Reasoning Engine.

Enables Janus to reflect on its own performance, identify patterns in
successes and failures, and maintain calibrated uncertainty about topics.

**Validates: Requirements REQ-5.4, REQ-1.3**
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """
    The result of reflecting on a completed task.

    Attributes:
        task_id: Identifier of the task that was reflected upon.
        success: Whether the task was considered successful.
        patterns_identified: Recurring patterns noticed in this task.
        lessons: Actionable lessons learned.
        confidence_delta: Change in overall confidence (+ve = more confident).
    """
    task_id: str
    success: bool
    patterns_identified: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    confidence_delta: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Reflection(task_id={self.task_id!r}, success={self.success}, "
            f"patterns={self.patterns_identified}, delta={self.confidence_delta:+.2f})"
        )


class Metacognition:
    """
    Metacognitive layer that lets Janus reflect on its own performance,
    identify patterns, and track uncertainty about topics.

    Usage::

        meta = Metacognition()
        reflection = meta.reflect({"task_id": "t1", "success": True, "notes": "..."})
        patterns = meta.identify_patterns(history)
        uncertainty = meta.get_uncertainty("python")
        meta.update_uncertainty("python", {"outcome": "success", "difficulty": 0.3})
    """

    def __init__(self) -> None:
        # topic → uncertainty score (0.0 = certain, 1.0 = completely unknown)
        self._uncertainty: Dict[str, float] = {}
        # history of reflections
        self._reflections: List[Reflection] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(self, task_result: Dict[str, Any]) -> Reflection:
        """
        Analyse a completed task result and produce a Reflection.

        The ``task_result`` dict may contain:
        - ``task_id`` (str): identifier (auto-generated if absent)
        - ``success`` (bool): whether the task succeeded
        - ``notes`` (str): free-text notes about what happened
        - ``errors`` (List[str]): error messages encountered
        - ``skills_used`` (List[str]): skills that were exercised
        - ``duration_minutes`` (float): how long the task took

        Args:
            task_result: Dict describing the task outcome.

        Returns:
            Reflection with patterns and lessons.
        """
        task_id = str(task_result.get("task_id", uuid.uuid4()))
        success = bool(task_result.get("success", False))
        notes = str(task_result.get("notes", ""))
        errors = task_result.get("errors", [])
        skills_used = task_result.get("skills_used", [])
        duration = float(task_result.get("duration_minutes", 0.0))

        patterns: List[str] = []
        lessons: List[str] = []

        # --- Pattern: repeated errors ---
        if errors:
            patterns.append(f"encountered_errors:{len(errors)}")
            for err in errors[:3]:  # cap to avoid noise
                lessons.append(f"Avoid: {err}")

        # --- Pattern: skill usage ---
        for skill in skills_used:
            patterns.append(f"used_skill:{skill}")

        # --- Pattern: duration ---
        if duration > 60:
            patterns.append("long_running_task")
            lessons.append("Consider breaking long tasks into smaller steps.")

        # --- Lessons from success/failure ---
        if success:
            lessons.append("Strategy worked — reinforce this approach.")
            if skills_used:
                lessons.append(f"Skills {skills_used} contributed to success.")
        else:
            lessons.append("Strategy failed — consider alternative approaches.")
            if notes:
                lessons.append(f"Context: {notes[:200]}")

        # --- Confidence delta ---
        confidence_delta = 0.05 if success else -0.05
        if errors:
            confidence_delta -= 0.02 * len(errors)
        confidence_delta = max(-0.5, min(0.5, confidence_delta))

        reflection = Reflection(
            task_id=task_id,
            success=success,
            patterns_identified=patterns,
            lessons=lessons,
            confidence_delta=round(confidence_delta, 4),
        )

        self._reflections.append(reflection)
        logger.info(f"Reflected on task {task_id}: success={success}, delta={confidence_delta:+.2f}")
        return reflection

    def identify_patterns(self, history: List[Dict[str, Any]]) -> List[str]:
        """
        Find recurring success/failure patterns across a history of task results.

        Args:
            history: List of task result dicts (same format as :meth:`reflect`).

        Returns:
            List of pattern strings sorted by frequency (most common first).
        """
        pattern_counter: Counter = Counter()

        for task_result in history:
            reflection = self.reflect(task_result)
            for pattern in reflection.patterns_identified:
                pattern_counter[pattern] += 1

        # Return patterns that appear more than once, sorted by frequency
        recurring = [
            pattern
            for pattern, count in pattern_counter.most_common()
            if count > 1
        ]

        # If nothing recurs, return all patterns sorted by frequency
        if not recurring:
            recurring = [p for p, _ in pattern_counter.most_common()]

        return recurring

    def get_uncertainty(self, topic: str) -> float:
        """
        Return the current uncertainty about a topic.

        Args:
            topic: The topic to query (case-insensitive).

        Returns:
            Float in [0.0, 1.0] where 1.0 = completely unknown.
        """
        key = topic.lower().strip()
        return self._uncertainty.get(key, 1.0)  # unknown topics = max uncertainty

    def update_uncertainty(
        self,
        topic: str,
        new_evidence: Dict[str, Any],
    ) -> None:
        """
        Reduce (or increase) uncertainty about a topic based on new evidence.

        The ``new_evidence`` dict may contain:
        - ``outcome`` (str): "success" or "failure"
        - ``difficulty`` (float 0–1): how hard the task was
        - ``confidence`` (float 0–1): explicit confidence signal

        Args:
            topic: The topic to update.
            new_evidence: Evidence dict.
        """
        key = topic.lower().strip()
        current = self._uncertainty.get(key, 1.0)

        outcome = new_evidence.get("outcome", "")
        difficulty = float(new_evidence.get("difficulty", 0.5))
        explicit_confidence = new_evidence.get("confidence")

        if explicit_confidence is not None:
            # Direct confidence signal → uncertainty = 1 - confidence
            new_uncertainty = 1.0 - float(explicit_confidence)
        elif outcome == "success":
            # Success reduces uncertainty proportional to difficulty
            reduction = 0.1 + 0.1 * (1.0 - difficulty)
            new_uncertainty = current - reduction
        elif outcome == "failure":
            # Failure slightly increases uncertainty
            new_uncertainty = current + 0.05
        else:
            # Generic evidence — small reduction
            new_uncertainty = current - 0.05

        self._uncertainty[key] = round(max(0.0, min(1.0, new_uncertainty)), 4)
        logger.debug(
            f"Uncertainty for {topic!r}: {current:.3f} → {self._uncertainty[key]:.3f}"
        )

    def get_all_uncertainties(self) -> Dict[str, float]:
        """Return a copy of all tracked uncertainty scores."""
        return dict(self._uncertainty)

    def get_reflections(self) -> List[Reflection]:
        """Return all stored reflections."""
        return list(self._reflections)
