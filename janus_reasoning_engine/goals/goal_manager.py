"""
Concrete GoalManager implementation.

Handles goal creation, hierarchy, state tracking, and decomposition.
Satisfies REQ-1.1 and REQ-1.3.
"""

import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from janus_reasoning_engine.core.interfaces import (
    Goal,
    GoalManager,
    GoalStatus,
    Strategy,
    StrategyStatus,
)
from janus_reasoning_engine.goals.goal_store import GoalStore

logger = logging.getLogger(__name__)


class GoalManagerImpl(GoalManager):
    """
    Concrete implementation of GoalManager.

    Persists goals in SQLite via GoalStore and supports a full goal
    hierarchy: high-level goals → strategies → sub-goals.
    """

    def __init__(self, goal_store: Optional[GoalStore] = None, db_path: str = "janus_goals.db"):
        """
        Initialize the goal manager.

        Args:
            goal_store: Optional pre-configured GoalStore. If None, one is
                        created using db_path.
            db_path: Path to the SQLite database (used when goal_store is None).
        """
        self.store = goal_store or GoalStore(db_path=db_path)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the goal manager and its storage backend."""
        if self._initialized:
            return
        self.store.initialize()
        self._initialized = True
        logger.info("GoalManagerImpl initialized")

    def shutdown(self) -> None:
        """Shutdown the goal manager."""
        if not self._initialized:
            return
        self.store.shutdown()
        self._initialized = False
        logger.info("GoalManagerImpl shutdown")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # GoalManager interface
    # ------------------------------------------------------------------

    def create_goal(
        self,
        description: str,
        priority: float,
        expected_value: float,
        parent_goal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feasibility: float = 0.5,
    ) -> Goal:
        """
        Create and persist a new goal.

        Args:
            description: Human-readable goal description.
            priority: Priority level (0.0–1.0).
            expected_value: Expected value of achieving the goal.
            parent_goal_id: Optional parent goal for hierarchical goals.
            metadata: Optional additional metadata.
            feasibility: Estimated feasibility (0.0–1.0).

        Returns:
            Created Goal object.
        """
        self._ensure_initialized()

        now = datetime.utcnow()
        goal = Goal(
            id=str(uuid.uuid4()),
            description=description,
            priority=max(0.0, min(1.0, priority)),
            expected_value=expected_value,
            feasibility=max(0.0, min(1.0, feasibility)),
            status=GoalStatus.ACTIVE,
            parent_goal_id=parent_goal_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self.store.save_goal(goal)
        logger.info(f"Created goal '{description}' (id={goal.id}, priority={priority:.2f})")
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a goal by ID."""
        self._ensure_initialized()
        return self.store.get_goal(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Return all active goals sorted by priority (highest first)."""
        self._ensure_initialized()
        goals = self.store.get_goals_by_status(GoalStatus.ACTIVE)
        return sorted(goals, key=lambda g: g.priority, reverse=True)

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Return all goals with the given status."""
        self._ensure_initialized()
        return self.store.get_goals_by_status(status)

    def get_child_goals(self, parent_goal_id: str) -> List[Goal]:
        """Return all direct sub-goals of a goal."""
        self._ensure_initialized()
        return self.store.get_child_goals(parent_goal_id)

    def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        """
        Update the status of a goal.

        Automatically marks child goals as cancelled when a parent is
        cancelled or failed.
        """
        self._ensure_initialized()
        self.store.update_goal_status(goal_id, status)
        logger.info(f"Goal {goal_id} status → {status.value}")

        # Cascade cancellation to children
        if status in (GoalStatus.CANCELLED, GoalStatus.FAILED):
            children = self.store.get_child_goals(goal_id)
            for child in children:
                if child.status == GoalStatus.ACTIVE:
                    self.update_goal_status(child.id, GoalStatus.CANCELLED)

    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """
        Update progress toward a goal (0.0–1.0).

        Automatically marks the goal as completed when progress reaches 1.0.
        """
        self._ensure_initialized()
        progress = max(0.0, min(1.0, progress))
        self.store.update_goal_progress(goal_id, progress)

        if progress >= 1.0:
            self.store.update_goal_status(goal_id, GoalStatus.COMPLETED)
            logger.info(f"Goal {goal_id} completed (progress=1.0)")
        else:
            logger.debug(f"Goal {goal_id} progress → {progress:.2%}")

    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """
        Break a high-level goal into sub-goals.

        This base implementation returns existing child goals. The
        GoalDecomposer (Task 3.2) provides LLM-powered decomposition.

        Args:
            goal_id: Goal to decompose.

        Returns:
            List of sub-goals (existing children).
        """
        self._ensure_initialized()
        return self.store.get_child_goals(goal_id)

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def save_strategy(self, strategy: Strategy) -> None:
        """Persist a strategy."""
        self._ensure_initialized()
        self.store.save_strategy(strategy)

    def get_strategies_for_goal(self, goal_id: str) -> List[Strategy]:
        """Return all strategies for a goal."""
        self._ensure_initialized()
        return self.store.get_strategies_for_goal(goal_id)

    def update_strategy_status(self, strategy_id: str, status: StrategyStatus) -> None:
        """Update the status of a strategy."""
        self._ensure_initialized()
        self.store.update_strategy_status(strategy_id, status)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics."""
        self._ensure_initialized()
        return self.store.get_statistics()
