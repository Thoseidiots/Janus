"""
Goal persistence layer using SQLite.

Handles CRUD operations for goals and strategies with full state tracking.
"""

import sqlite3
import json
import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from janus_reasoning_engine.core.interfaces import Goal, GoalStatus, Strategy, StrategyStatus

logger = logging.getLogger(__name__)


class GoalStore:
    """
    SQLite-backed persistence for goals and strategies.

    Stores goal hierarchy, state, and progress. Supports parent/child
    relationships for high-level → strategy → sub-goal decomposition.
    """

    def __init__(self, db_path: str = "janus_goals.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the database and create tables if needed."""
        if self.initialized:
            return

        db_dir = Path(self.db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.initialized = True
        logger.info(f"GoalStore initialized at {self.db_path}")

    def shutdown(self) -> None:
        """Close the database connection."""
        if not self.initialized:
            return
        if self.conn:
            self.conn.close()
            self.conn = None
        self.initialized = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                goal_id         TEXT PRIMARY KEY,
                description     TEXT NOT NULL,
                priority        REAL NOT NULL DEFAULT 0.5,
                expected_value  REAL NOT NULL DEFAULT 0.0,
                feasibility     REAL NOT NULL DEFAULT 0.5,
                status          TEXT NOT NULL DEFAULT 'active',
                parent_goal_id  TEXT,
                progress        REAL NOT NULL DEFAULT 0.0,
                metadata        TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY (parent_goal_id) REFERENCES goals(goal_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_status
            ON goals(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_parent
            ON goals(parent_goal_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_priority
            ON goals(priority DESC)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id         TEXT PRIMARY KEY,
                goal_id             TEXT NOT NULL,
                description         TEXT NOT NULL,
                expected_value      REAL NOT NULL DEFAULT 0.0,
                time_estimate       REAL NOT NULL DEFAULT 1.0,
                success_probability REAL NOT NULL DEFAULT 0.5,
                resource_requirements TEXT,
                status              TEXT NOT NULL DEFAULT 'proposed',
                steps               TEXT,
                metadata            TEXT,
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategies_goal
            ON strategies(goal_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategies_status
            ON strategies(status)
        """)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Goal CRUD
    # ------------------------------------------------------------------

    def save_goal(self, goal: Goal) -> None:
        """Insert or replace a goal record."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO goals
                (goal_id, description, priority, expected_value, feasibility,
                 status, parent_goal_id, progress, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            goal.id,
            goal.description,
            goal.priority,
            goal.expected_value,
            goal.feasibility,
            goal.status.value,
            goal.parent_goal_id,
            goal.metadata.get("progress", 0.0),
            json.dumps(goal.metadata),
            goal.created_at.isoformat(),
            goal.updated_at.isoformat(),
        ))
        self.conn.commit()

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a goal by ID."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
        row = cursor.fetchone()
        return self._row_to_goal(row) if row else None

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Return all goals with the given status, sorted by priority desc."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM goals WHERE status = ? ORDER BY priority DESC",
            (status.value,)
        )
        return [self._row_to_goal(r) for r in cursor.fetchall()]

    def get_child_goals(self, parent_goal_id: str) -> List[Goal]:
        """Return all direct children of a goal."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM goals WHERE parent_goal_id = ? ORDER BY priority DESC",
            (parent_goal_id,)
        )
        return [self._row_to_goal(r) for r in cursor.fetchall()]

    def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        """Update the status of a goal."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE goal_id = ?",
            (status.value, datetime.utcnow().isoformat(), goal_id)
        )
        self.conn.commit()

    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """Update progress (0.0–1.0) for a goal."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        progress = max(0.0, min(1.0, progress))
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE goals SET progress = ?, updated_at = ? WHERE goal_id = ?",
            (progress, datetime.utcnow().isoformat(), goal_id)
        )
        self.conn.commit()

    def update_goal_metadata(self, goal_id: str, metadata: Dict[str, Any]) -> None:
        """Merge new metadata into an existing goal."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        goal = self.get_goal(goal_id)
        if goal is None:
            raise ValueError(f"Goal {goal_id} not found")

        merged = {**goal.metadata, **metadata}
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE goals SET metadata = ?, updated_at = ? WHERE goal_id = ?",
            (json.dumps(merged), datetime.utcnow().isoformat(), goal_id)
        )
        self.conn.commit()

    def delete_goal(self, goal_id: str) -> None:
        """Delete a goal and its strategies."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM strategies WHERE goal_id = ?", (goal_id,))
        cursor.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Strategy CRUD
    # ------------------------------------------------------------------

    def save_strategy(self, strategy: Strategy) -> None:
        """Insert or replace a strategy record."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO strategies
                (strategy_id, goal_id, description, expected_value, time_estimate,
                 success_probability, resource_requirements, status, steps, metadata,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.id,
            strategy.goal_id,
            strategy.description,
            strategy.expected_value,
            strategy.time_estimate,
            strategy.success_probability,
            json.dumps(strategy.resource_requirements),
            strategy.status.value,
            json.dumps(strategy.steps),
            json.dumps(strategy.metadata),
            strategy.created_at.isoformat(),
            strategy.created_at.isoformat(),
        ))
        self.conn.commit()

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Retrieve a strategy by ID."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
        row = cursor.fetchone()
        return self._row_to_strategy(row) if row else None

    def get_strategies_for_goal(self, goal_id: str) -> List[Strategy]:
        """Return all strategies for a goal."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM strategies WHERE goal_id = ? ORDER BY expected_value DESC",
            (goal_id,)
        )
        return [self._row_to_strategy(r) for r in cursor.fetchall()]

    def update_strategy_status(self, strategy_id: str, status: StrategyStatus) -> None:
        """Update the status of a strategy."""
        if not self.initialized:
            raise RuntimeError("GoalStore not initialized")

        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE strategies SET status = ?, updated_at = ? WHERE strategy_id = ?",
            (status.value, datetime.utcnow().isoformat(), strategy_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about stored goals."""
        if not self.initialized:
            return {"initialized": False}

        cursor = self.conn.cursor()
        cursor.execute("SELECT status, COUNT(*) as cnt FROM goals GROUP BY status")
        goal_counts = {row["status"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) as cnt FROM strategies")
        strategy_count = cursor.fetchone()["cnt"]

        return {
            "goals_by_status": goal_counts,
            "total_strategies": strategy_count,
            "db_path": self.db_path,
        }

    # ------------------------------------------------------------------
    # Row converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        metadata["progress"] = row["progress"]
        return Goal(
            id=row["goal_id"],
            description=row["description"],
            priority=row["priority"],
            expected_value=row["expected_value"],
            feasibility=row["feasibility"],
            status=GoalStatus(row["status"]),
            parent_goal_id=row["parent_goal_id"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_strategy(row: sqlite3.Row) -> Strategy:
        now = datetime.utcnow()
        return Strategy(
            id=row["strategy_id"],
            goal_id=row["goal_id"],
            description=row["description"],
            expected_value=row["expected_value"],
            time_estimate=row["time_estimate"],
            success_probability=row["success_probability"],
            resource_requirements=json.loads(row["resource_requirements"]) if row["resource_requirements"] else {},
            status=StrategyStatus(row["status"]),
            steps=json.loads(row["steps"]) if row["steps"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )
