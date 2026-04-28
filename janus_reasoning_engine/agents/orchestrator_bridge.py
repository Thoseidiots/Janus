"""System orchestration bridge — master coordinator for all Janus subsystems.

Wires JanusOrchestrator (OBSERVE→PLAN→ACT→EXECUTE→REVIEW) as the execution
backend for goal-directed tasks.

Requirements: REQ-14.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("janus.agents.orchestrator_bridge")

# Optional import of JanusOrchestrator
try:
    from janus_system_orchestrator import JanusOrchestrator as _JanusOrchestrator  # type: ignore
    _HAS_ORCHESTRATOR = True
    logger.info("JanusOrchestrator available — real OBSERVE→PLAN→ACT loop enabled")
except (ImportError, SyntaxError, Exception) as _e:
    _JanusOrchestrator = None  # type: ignore
    _HAS_ORCHESTRATOR = False
    logger.debug("JanusOrchestrator not available: %s", _e)


@dataclass
class CycleResult:
    """Result of a single OBSERVE → PLAN → ACT → EXECUTE → REVIEW cycle."""
    cycle_num: int
    goal: str
    success: bool
    output: str = ""
    error: Optional[str] = None


class OrchestratorBridge:
    """
    Bridge to JanusOrchestrator for OBSERVE→PLAN→ACT→EXECUTE→REVIEW cycles.

    When JanusOrchestrator is available, creates a real instance, loads all
    subsystems (Avus brain, ScreenInterpreter, SkillExecutor, CEOAgent), and
    delegates goal execution to it.

    Falls back to a lightweight stub when the orchestrator is unavailable.
    """

    def __init__(self) -> None:
        self._current_goal: str = ""
        self._cycles_run: int = 0
        self._orchestrator: Optional[Any] = None
        self._components: Dict[str, Any] = {}
        self._init_orchestrator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self, goal: str) -> CycleResult:
        """
        Execute one full OBSERVE→PLAN→ACT→EXECUTE→REVIEW cycle for *goal*.

        If JanusOrchestrator is loaded, sets the goal and calls cycle().
        Otherwise returns a stub result.
        """
        self._cycles_run += 1
        cycle_num = self._cycles_run

        if self._orchestrator is not None:
            try:
                self._orchestrator.set_goal(goal)
                raw = self._orchestrator.cycle()
                return CycleResult(
                    cycle_num=cycle_num,
                    goal=goal,
                    success=raw.success,
                    output=str(raw.avus_output or raw.action_taken or ""),
                    error=raw.error,
                )
            except Exception as exc:
                logger.warning("OrchestratorBridge.run_cycle failed: %s", exc)
                return CycleResult(
                    cycle_num=cycle_num,
                    goal=goal,
                    success=False,
                    error=str(exc),
                )

        # Stub
        return CycleResult(
            cycle_num=cycle_num,
            goal=goal,
            success=True,
            output=f"[Stub cycle {cycle_num}] No orchestrator loaded — goal: {goal}",
        )

    def run_goal(self, goal: str, max_cycles: int = 10) -> CycleResult:
        """
        Run up to *max_cycles* cycles until the goal is achieved or the limit
        is reached.

        Returns the last CycleResult.
        """
        last: Optional[CycleResult] = None
        for _ in range(max_cycles):
            last = self.run_cycle(goal)
            if last.success:
                break
        return last or CycleResult(
            cycle_num=self._cycles_run,
            goal=goal,
            success=False,
            error="No cycles executed",
        )

    def get_status(self) -> Dict[str, Any]:
        """Return current orchestrator status."""
        if self._orchestrator is not None:
            try:
                return {
                    "current_goal": self._current_goal,
                    "cycles_run": self._cycles_run,
                    "components": self._orchestrator._status(),
                }
            except Exception:
                pass

        return {
            "current_goal": self._current_goal,
            "cycles_run": self._cycles_run,
            "components": self._components,
        }

    def set_goal(self, goal: str) -> None:
        """Set the current high-level goal."""
        self._current_goal = goal
        if self._orchestrator is not None:
            try:
                self._orchestrator.set_goal(goal)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_orchestrator(self) -> None:
        """Instantiate and load JanusOrchestrator if available."""
        if not _HAS_ORCHESTRATOR:
            return
        try:
            self._orchestrator = _JanusOrchestrator("Janus")
            status = self._orchestrator.load()
            self._components = status
            logger.info(
                "JanusOrchestrator loaded — components: %s",
                {k: ("✅" if v else "❌") for k, v in status.items()},
            )
        except Exception as exc:
            logger.warning("JanusOrchestrator init failed: %s", exc)
            self._orchestrator = None
