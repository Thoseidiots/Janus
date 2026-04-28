"""
Progress tracking and adaptation for goals and strategies.

Monitors metrics, detects failure patterns, implements pivot vs persist
logic, and records success/failure learnings.
Satisfies REQ-1.3 and REQ-5.4.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from janus_reasoning_engine.core.interfaces import (
    Goal,
    GoalStatus,
    Strategy,
    StrategyStatus,
)
from janus_reasoning_engine.goals.goal_manager import GoalManagerImpl

logger = logging.getLogger(__name__)


class AdaptationDecision(Enum):
    """Decision about whether to pivot or persist with a strategy."""
    PERSIST = "persist"
    PIVOT = "pivot"
    ABANDON = "abandon"
    COMPLETE = "complete"


@dataclass
class ProgressSnapshot:
    """A point-in-time snapshot of goal progress."""
    goal_id: str
    progress: float          # 0.0–1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Detected failure pattern for a strategy."""
    strategy_id: str
    goal_id: str
    pattern_type: str        # "stalled", "regressing", "timeout", "repeated_failure"
    detected_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningRecord:
    """Records a success or failure for future reference."""
    goal_id: str
    strategy_id: Optional[str]
    outcome: str             # "success" or "failure"
    lessons: List[str]
    recorded_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Monitors goal progress and drives adaptive strategy decisions.

    Tracks progress snapshots over time, detects failure patterns
    (stalling, regression, timeout), and decides whether to pivot
    to a new strategy or persist with the current one.
    """

    # Thresholds for failure detection
    STALL_THRESHOLD_HOURS: float = 2.0       # No progress for this long → stalled
    REGRESSION_THRESHOLD: float = 0.05       # Progress dropped by this much → regressing
    MIN_PROGRESS_RATE: float = 0.02          # Minimum progress per hour to be considered healthy
    PIVOT_FAILURE_COUNT: int = 2             # Failures before recommending pivot
    ABANDON_FAILURE_COUNT: int = 4           # Failures before recommending abandon

    def __init__(self, goal_manager: GoalManagerImpl):
        """
        Initialize the progress tracker.

        Args:
            goal_manager: GoalManagerImpl for reading/updating goals.
        """
        self.goal_manager = goal_manager

        # In-memory stores (could be persisted in a future iteration)
        self._snapshots: Dict[str, List[ProgressSnapshot]] = {}   # goal_id → snapshots
        self._failure_counts: Dict[str, int] = {}                  # strategy_id → count
        self._failure_patterns: List[FailurePattern] = []
        self._learning_records: List[LearningRecord] = []

    # ------------------------------------------------------------------
    # Progress recording
    # ------------------------------------------------------------------

    def record_progress(
        self,
        goal_id: str,
        progress: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a progress update for a goal.

        Also persists the update to the goal store.

        Args:
            goal_id: Goal identifier.
            progress: Current progress (0.0–1.0).
            metadata: Optional context about this progress update.
        """
        progress = max(0.0, min(1.0, progress))
        snapshot = ProgressSnapshot(
            goal_id=goal_id,
            progress=progress,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        if goal_id not in self._snapshots:
            self._snapshots[goal_id] = []
        self._snapshots[goal_id].append(snapshot)

        # Persist to goal store
        self.goal_manager.update_goal_progress(goal_id, progress)
        logger.debug(f"Progress recorded for goal {goal_id}: {progress:.2%}")

    def get_progress_history(self, goal_id: str) -> List[ProgressSnapshot]:
        """Return all recorded progress snapshots for a goal."""
        return list(self._snapshots.get(goal_id, []))

    def get_current_progress(self, goal_id: str) -> float:
        """Return the most recent progress value for a goal (0.0 if none)."""
        snapshots = self._snapshots.get(goal_id, [])
        if not snapshots:
            goal = self.goal_manager.get_goal(goal_id)
            return goal.metadata.get("progress", 0.0) if goal else 0.0
        return snapshots[-1].progress

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    def detect_failure_patterns(
        self,
        goal_id: str,
        strategy_id: Optional[str] = None,
        max_duration_hours: float = 24.0,
    ) -> List[FailurePattern]:
        """
        Analyse progress history and detect failure patterns.

        Checks for:
        - Stalled progress (no change for STALL_THRESHOLD_HOURS)
        - Regressing progress (progress decreased)
        - Timeout (exceeded max_duration_hours without completion)
        - Low progress rate

        Args:
            goal_id: Goal to analyse.
            strategy_id: Optional strategy context.
            max_duration_hours: Maximum allowed duration before timeout.

        Returns:
            List of detected FailurePattern objects.
        """
        snapshots = self._snapshots.get(goal_id, [])
        patterns: List[FailurePattern] = []
        now = datetime.utcnow()

        if len(snapshots) < 2:
            return patterns

        latest = snapshots[-1]
        earliest = snapshots[0]

        # --- Stall detection ---
        recent_snapshots = [
            s for s in snapshots
            if (now - s.timestamp).total_seconds() / 3600 <= self.STALL_THRESHOLD_HOURS
        ]
        if recent_snapshots:
            recent_progress_values = [s.progress for s in recent_snapshots]
            if max(recent_progress_values) - min(recent_progress_values) < 0.001:
                patterns.append(FailurePattern(
                    strategy_id=strategy_id or "",
                    goal_id=goal_id,
                    pattern_type="stalled",
                    detected_at=now,
                    details={
                        "stall_duration_hours": self.STALL_THRESHOLD_HOURS,
                        "current_progress": latest.progress,
                    },
                ))

        # --- Regression detection ---
        if latest.progress < earliest.progress - self.REGRESSION_THRESHOLD:
            patterns.append(FailurePattern(
                strategy_id=strategy_id or "",
                goal_id=goal_id,
                pattern_type="regressing",
                detected_at=now,
                details={
                    "start_progress": earliest.progress,
                    "current_progress": latest.progress,
                    "regression": earliest.progress - latest.progress,
                },
            ))

        # --- Timeout detection ---
        elapsed_hours = (now - earliest.timestamp).total_seconds() / 3600
        if elapsed_hours > max_duration_hours and latest.progress < 1.0:
            patterns.append(FailurePattern(
                strategy_id=strategy_id or "",
                goal_id=goal_id,
                pattern_type="timeout",
                detected_at=now,
                details={
                    "elapsed_hours": elapsed_hours,
                    "max_duration_hours": max_duration_hours,
                    "current_progress": latest.progress,
                },
            ))

        # --- Low progress rate ---
        elapsed_hours = max((latest.timestamp - earliest.timestamp).total_seconds() / 3600, 0.01)
        progress_rate = (latest.progress - earliest.progress) / elapsed_hours
        if progress_rate < self.MIN_PROGRESS_RATE and latest.progress < 0.9:
            patterns.append(FailurePattern(
                strategy_id=strategy_id or "",
                goal_id=goal_id,
                pattern_type="low_progress_rate",
                detected_at=now,
                details={
                    "progress_rate_per_hour": progress_rate,
                    "min_expected": self.MIN_PROGRESS_RATE,
                },
            ))

        self._failure_patterns.extend(patterns)
        return patterns

    def record_strategy_failure(self, strategy_id: str) -> int:
        """
        Increment the failure count for a strategy.

        Args:
            strategy_id: Strategy that failed.

        Returns:
            Updated failure count.
        """
        self._failure_counts[strategy_id] = self._failure_counts.get(strategy_id, 0) + 1
        count = self._failure_counts[strategy_id]
        logger.info(f"Strategy {strategy_id} failure count: {count}")
        return count

    def get_strategy_failure_count(self, strategy_id: str) -> int:
        """Return the number of recorded failures for a strategy."""
        return self._failure_counts.get(strategy_id, 0)

    # ------------------------------------------------------------------
    # Pivot vs persist decision
    # ------------------------------------------------------------------

    def decide_adaptation(
        self,
        goal_id: str,
        strategy_id: str,
        failure_patterns: Optional[List[FailurePattern]] = None,
    ) -> Tuple[AdaptationDecision, str]:
        """
        Decide whether to pivot, persist, or abandon a strategy.

        Decision logic:
        - COMPLETE  → progress >= 1.0
        - ABANDON   → failure count >= ABANDON_FAILURE_COUNT
        - PIVOT     → failure count >= PIVOT_FAILURE_COUNT or critical patterns
        - PERSIST   → otherwise

        Args:
            goal_id: Goal being tracked.
            strategy_id: Current strategy.
            failure_patterns: Pre-computed patterns (computed if None).

        Returns:
            Tuple of (AdaptationDecision, rationale string).
        """
        current_progress = self.get_current_progress(goal_id)

        if current_progress >= 1.0:
            return AdaptationDecision.COMPLETE, "Goal progress reached 100%"

        failure_count = self.get_strategy_failure_count(strategy_id)

        if failure_count >= self.ABANDON_FAILURE_COUNT:
            return (
                AdaptationDecision.ABANDON,
                f"Strategy failed {failure_count} times (threshold={self.ABANDON_FAILURE_COUNT})",
            )

        if failure_patterns is None:
            failure_patterns = self.detect_failure_patterns(goal_id, strategy_id)

        critical_patterns = {p.pattern_type for p in failure_patterns}
        has_critical = bool(critical_patterns & {"regressing", "timeout"})

        if failure_count >= self.PIVOT_FAILURE_COUNT or has_critical:
            reason_parts = []
            if failure_count >= self.PIVOT_FAILURE_COUNT:
                reason_parts.append(f"failure count={failure_count}")
            if has_critical:
                reason_parts.append(f"critical patterns={critical_patterns & {'regressing', 'timeout'}}")
            return (
                AdaptationDecision.PIVOT,
                "Pivoting: " + ", ".join(reason_parts),
            )

        return AdaptationDecision.PERSIST, f"Strategy progressing normally (progress={current_progress:.2%})"

    # ------------------------------------------------------------------
    # Learning and reflection
    # ------------------------------------------------------------------

    def record_success(
        self,
        goal_id: str,
        strategy_id: Optional[str] = None,
        lessons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningRecord:
        """
        Record a successful goal/strategy outcome.

        Args:
            goal_id: Completed goal.
            strategy_id: Strategy that succeeded.
            lessons: Key takeaways from the success.
            metadata: Additional context.

        Returns:
            LearningRecord.
        """
        record = LearningRecord(
            goal_id=goal_id,
            strategy_id=strategy_id,
            outcome="success",
            lessons=lessons or ["Goal achieved successfully"],
            recorded_at=datetime.utcnow(),
            metadata=metadata or {},
        )
        self._learning_records.append(record)
        self.goal_manager.update_goal_status(goal_id, GoalStatus.COMPLETED)
        logger.info(f"Success recorded for goal {goal_id}")
        return record

    def record_failure(
        self,
        goal_id: str,
        strategy_id: Optional[str] = None,
        lessons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningRecord:
        """
        Record a failed goal/strategy outcome.

        Args:
            goal_id: Failed goal.
            strategy_id: Strategy that failed.
            lessons: Key takeaways from the failure.
            metadata: Additional context.

        Returns:
            LearningRecord.
        """
        if strategy_id:
            self.record_strategy_failure(strategy_id)

        record = LearningRecord(
            goal_id=goal_id,
            strategy_id=strategy_id,
            outcome="failure",
            lessons=lessons or ["Strategy did not achieve the goal"],
            recorded_at=datetime.utcnow(),
            metadata=metadata or {},
        )
        self._learning_records.append(record)
        self.goal_manager.update_goal_status(goal_id, GoalStatus.FAILED)
        logger.info(f"Failure recorded for goal {goal_id}")
        return record

    def get_learning_records(
        self,
        goal_id: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[LearningRecord]:
        """
        Retrieve learning records, optionally filtered.

        Args:
            goal_id: Filter by goal (None = all goals).
            outcome: Filter by "success" or "failure" (None = both).

        Returns:
            Matching LearningRecord list.
        """
        records = self._learning_records
        if goal_id is not None:
            records = [r for r in records if r.goal_id == goal_id]
        if outcome is not None:
            records = [r for r in records if r.outcome == outcome]
        return records

    def reflect(self, goal_id: str) -> Dict[str, Any]:
        """
        Generate a reflection summary for a goal.

        Returns statistics about progress, failures, and lessons learned.

        Args:
            goal_id: Goal to reflect on.

        Returns:
            Dictionary with reflection data.
        """
        snapshots = self._snapshots.get(goal_id, [])
        records = self.get_learning_records(goal_id=goal_id)
        patterns = [p for p in self._failure_patterns if p.goal_id == goal_id]

        current_progress = self.get_current_progress(goal_id)
        goal = self.goal_manager.get_goal(goal_id)

        successes = [r for r in records if r.outcome == "success"]
        failures = [r for r in records if r.outcome == "failure"]

        all_lessons: List[str] = []
        for r in records:
            all_lessons.extend(r.lessons)

        return {
            "goal_id": goal_id,
            "goal_description": goal.description if goal else "unknown",
            "current_progress": current_progress,
            "goal_status": goal.status.value if goal else "unknown",
            "progress_snapshots": len(snapshots),
            "failure_patterns_detected": len(patterns),
            "pattern_types": list({p.pattern_type for p in patterns}),
            "success_count": len(successes),
            "failure_count": len(failures),
            "lessons_learned": all_lessons,
            "reflected_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics across all tracked goals."""
        total_goals = len(self._snapshots)
        total_snapshots = sum(len(v) for v in self._snapshots.values())
        total_failures = sum(self._failure_counts.values())

        return {
            "tracked_goals": total_goals,
            "total_progress_snapshots": total_snapshots,
            "total_strategy_failures": total_failures,
            "failure_patterns_detected": len(self._failure_patterns),
            "learning_records": len(self._learning_records),
        }
