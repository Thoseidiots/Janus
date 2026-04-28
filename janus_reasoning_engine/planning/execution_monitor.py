"""
execution_monitor.py
====================
Execution monitoring for the Janus Reasoning Engine.

Tracks plan execution, detects stuck/blocked states, supports dynamic plan
adjustment, and optionally persists state via janus_checkpoint.py.

Requirements: REQ-4.3, REQ-12.3
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from janus_reasoning_engine.planning.multi_step_planner import (
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from janus_reasoning_engine.planning.tool_orchestrator import StepResult, ToolOrchestrator

logger = logging.getLogger(__name__)

# Optional checkpoint integration
try:
    from janus_checkpoint import get_checkpointer as _get_checkpointer  # type: ignore
    _HAS_CHECKPOINT = True
except Exception:
    _HAS_CHECKPOINT = False
    _get_checkpointer = None


# ── ExecutionSession ──────────────────────────────────────────────────────────

@dataclass
class ExecutionSession:
    """Tracks the live execution state of a Plan."""
    id: str
    plan: Plan
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_progress_at: datetime = field(default_factory=datetime.utcnow)
    step_results: List[StepResult] = field(default_factory=list)
    is_stuck: bool = False
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_results is None:
            self.step_results = []
        if self.metadata is None:
            self.metadata = {}

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.finished_at is None

    @property
    def completed_steps(self) -> int:
        return sum(
            1 for s in self.plan.steps
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )

    @property
    def total_steps(self) -> int:
        return len(self.plan.steps)

    @property
    def progress(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialise session state for checkpointing."""
        return {
            "session_id": self.id,
            "plan_id": self.plan.id,
            "goal": self.plan.goal_description,
            "started_at": self.started_at.isoformat(),
            "last_progress_at": self.last_progress_at.isoformat(),
            "is_stuck": self.is_stuck,
            "progress": self.progress,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "plan_status": self.plan.status.value,
        }


# ── ExecutionMonitor ──────────────────────────────────────────────────────────

class ExecutionMonitor:
    """
    Monitors plan execution, detects stuck states, and supports dynamic
    plan adjustment.

    Args:
        orchestrator:    ToolOrchestrator used to execute individual steps.
        timeout_minutes: Minutes a step can run without progress before it is
                         considered stuck (default: 30).
    """

    DEFAULT_TIMEOUT_MINUTES = 30.0

    def __init__(
        self,
        orchestrator: Optional[ToolOrchestrator] = None,
        timeout_minutes: float = DEFAULT_TIMEOUT_MINUTES,
    ) -> None:
        self.orchestrator = orchestrator or ToolOrchestrator()
        self.timeout_minutes = timeout_minutes
        self._sessions: Dict[str, ExecutionSession] = {}

        # Lazy-load checkpointer
        self._checkpointer = None
        if _HAS_CHECKPOINT and _get_checkpointer is not None:
            try:
                self._checkpointer = _get_checkpointer()
            except Exception as exc:
                logger.warning("Checkpoint integration unavailable: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def start_plan(self, plan: Plan) -> ExecutionSession:
        """
        Begin executing *plan* and return a tracking session.

        Args:
            plan: The Plan to execute.

        Returns:
            A new ExecutionSession.
        """
        session = ExecutionSession(
            id=str(uuid.uuid4()),
            plan=plan,
            started_at=datetime.utcnow(),
            last_progress_at=datetime.utcnow(),
        )
        plan.status = PlanStatus.IN_PROGRESS
        self._sessions[session.id] = session
        self._checkpoint(session)
        logger.info(
            "Started execution session %s for plan %s (%d steps)",
            session.id, plan.id, len(plan.steps),
        )
        return session

    def advance(self, session: ExecutionSession) -> StepResult:
        """
        Execute the next pending step in *session*'s plan.

        Detects stuck states before executing.  Updates plan/step status and
        persists a checkpoint after each step.

        Args:
            session: Active ExecutionSession.

        Returns:
            StepResult for the executed step, or a failed result if no step
            is available or the session is stuck.
        """
        if not session.is_active:
            return StepResult(
                step_id="",
                success=False,
                error="Session is already finished",
            )

        # Stuck detection
        if self._is_stuck(session):
            session.is_stuck = True
            logger.warning("Session %s is stuck (no progress for %.1f min)", session.id, self.timeout_minutes)
            return StepResult(
                step_id="",
                success=False,
                error=f"Execution stuck: no progress for {self.timeout_minutes} minutes",
                metadata={"stuck": True},
            )

        next_step = session.plan.next_step()
        if next_step is None:
            # No more steps — finalise the plan
            self._finalise(session)
            return StepResult(
                step_id="",
                success=True,
                output="Plan complete — no more steps",
                metadata={"plan_complete": True},
            )

        next_step.status = StepStatus.IN_PROGRESS
        result = self.orchestrator.execute_step(next_step)
        session.step_results.append(result)

        if result.success:
            next_step.status = StepStatus.COMPLETED
            session.last_progress_at = datetime.utcnow()
            session.is_stuck = False
            logger.info("Step %s completed successfully", next_step.id)
        else:
            next_step.status = StepStatus.FAILED
            logger.warning("Step %s failed: %s", next_step.id, result.error)
            # Try contingency if available
            if next_step.contingency:
                logger.info("Applying contingency for step %s: %s", next_step.id, next_step.contingency)
                result.metadata["contingency_applied"] = next_step.contingency

        # Check if plan is now complete or failed
        if session.plan.is_complete():
            self._finalise(session)
        elif session.plan.has_failed() and not session.plan.next_step():
            session.plan.status = PlanStatus.FAILED
            session.finished_at = datetime.utcnow()

        self._checkpoint(session)
        return result

    def adjust_plan(
        self,
        session: ExecutionSession,
        new_steps: List[PlanStep],
    ) -> None:
        """
        Dynamically add *new_steps* to the session's plan.

        New steps are appended after the last existing step.  The session's
        stuck flag is cleared so execution can resume.

        Args:
            session:   Active ExecutionSession.
            new_steps: Additional PlanStep objects to append.
        """
        if not new_steps:
            return

        # Wire dependencies: first new step depends on last existing step
        if session.plan.steps:
            last_id = session.plan.steps[-1].id
            if new_steps[0].dependencies is None:
                new_steps[0].dependencies = []
            if last_id not in new_steps[0].dependencies:
                new_steps[0].dependencies.append(last_id)

        session.plan.steps.extend(new_steps)
        session.is_stuck = False
        session.last_progress_at = datetime.utcnow()

        # Reopen plan if it was marked complete/failed
        if session.plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            session.plan.status = PlanStatus.IN_PROGRESS
            session.finished_at = None

        self._checkpoint(session)
        logger.info(
            "Adjusted plan %s: added %d new steps (total %d)",
            session.plan.id, len(new_steps), len(session.plan.steps),
        )

    def get_session(self, session_id: str) -> Optional[ExecutionSession]:
        """Return the session with *session_id*, or None."""
        return self._sessions.get(session_id)

    def get_status(self, session: ExecutionSession) -> Dict[str, Any]:
        """Return a status snapshot for *session*."""
        return {
            "session_id": session.id,
            "plan_id": session.plan.id,
            "goal": session.plan.goal_description,
            "plan_status": session.plan.status.value,
            "progress": session.progress,
            "completed_steps": session.completed_steps,
            "total_steps": session.total_steps,
            "is_stuck": session.is_stuck,
            "is_active": session.is_active,
            "started_at": session.started_at.isoformat(),
            "last_progress_at": session.last_progress_at.isoformat(),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_stuck(self, session: ExecutionSession) -> bool:
        """Return True if the session has had no progress within the timeout."""
        if not session.is_active:
            return False
        elapsed = datetime.utcnow() - session.last_progress_at
        return elapsed > timedelta(minutes=self.timeout_minutes)

    def _finalise(self, session: ExecutionSession) -> None:
        """Mark the session and plan as completed."""
        session.plan.status = PlanStatus.COMPLETED
        session.finished_at = datetime.utcnow()
        logger.info(
            "Session %s finished. Progress: %d/%d steps",
            session.id, session.completed_steps, session.total_steps,
        )

    def _checkpoint(self, session: ExecutionSession) -> None:
        """Persist session state via janus_checkpoint if available."""
        if self._checkpointer is None:
            return
        try:
            self._checkpointer.save(
                job_id=session.id,
                stage=session.plan.status.value,
                data=session.to_dict(),
            )
        except Exception as exc:
            logger.debug("Checkpoint save failed (non-fatal): %s", exc)
