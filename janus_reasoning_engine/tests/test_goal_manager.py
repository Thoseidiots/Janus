"""
Tests for the goal management system (Tasks 3.1, 3.2, 3.3).

Covers:
- Goal creation, persistence, hierarchy, and state tracking (3.1)
- Goal decomposition and strategy generation (3.2)
- Progress tracking, failure detection, and adaptation decisions (3.3)
"""

import os
import uuid
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from janus_reasoning_engine.core.interfaces import (
    Goal,
    GoalStatus,
    Strategy,
    StrategyStatus,
)
from janus_reasoning_engine.goals.goal_store import GoalStore
from janus_reasoning_engine.goals.goal_manager import GoalManagerImpl
from janus_reasoning_engine.goals.goal_decomposer import GoalDecomposer
from janus_reasoning_engine.goals.progress_tracker import (
    ProgressTracker,
    AdaptationDecision,
    FailurePattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite database path."""
    return str(tmp_path / "test_goals.db")


@pytest.fixture
def goal_store(tmp_db):
    store = GoalStore(db_path=tmp_db)
    store.initialize()
    yield store
    store.shutdown()


@pytest.fixture
def goal_manager(tmp_db):
    manager = GoalManagerImpl(db_path=tmp_db)
    manager.initialize()
    yield manager
    manager.shutdown()


@pytest.fixture
def decomposer(goal_manager):
    return GoalDecomposer(goal_manager=goal_manager, janus_gpt=None, num_strategies=3)


@pytest.fixture
def tracker(goal_manager):
    return ProgressTracker(goal_manager=goal_manager)


# ---------------------------------------------------------------------------
# Task 3.1 – Goal representation and storage
# ---------------------------------------------------------------------------

class TestGoalStore:
    """Unit tests for GoalStore persistence."""

    def test_save_and_retrieve_goal(self, goal_store):
        now = datetime.utcnow()
        goal = Goal(
            id=str(uuid.uuid4()),
            description="Earn $1000",
            priority=0.9,
            expected_value=1000.0,
            feasibility=0.7,
            status=GoalStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(goal)
        retrieved = goal_store.get_goal(goal.id)

        assert retrieved is not None
        assert retrieved.id == goal.id
        assert retrieved.description == "Earn $1000"
        assert retrieved.priority == pytest.approx(0.9)
        assert retrieved.status == GoalStatus.ACTIVE

    def test_goal_hierarchy_parent_child(self, goal_store):
        now = datetime.utcnow()
        parent = Goal(
            id=str(uuid.uuid4()),
            description="High-level goal",
            priority=1.0,
            expected_value=5000.0,
            feasibility=0.6,
            status=GoalStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        child = Goal(
            id=str(uuid.uuid4()),
            description="Sub-goal",
            priority=0.8,
            expected_value=2000.0,
            feasibility=0.7,
            status=GoalStatus.ACTIVE,
            parent_goal_id=parent.id,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(parent)
        goal_store.save_goal(child)

        children = goal_store.get_child_goals(parent.id)
        assert len(children) == 1
        assert children[0].id == child.id
        assert children[0].parent_goal_id == parent.id

    def test_update_goal_status(self, goal_store):
        now = datetime.utcnow()
        goal = Goal(
            id=str(uuid.uuid4()),
            description="Test goal",
            priority=0.5,
            expected_value=100.0,
            feasibility=0.5,
            status=GoalStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(goal)
        goal_store.update_goal_status(goal.id, GoalStatus.COMPLETED)

        updated = goal_store.get_goal(goal.id)
        assert updated.status == GoalStatus.COMPLETED

    def test_update_goal_progress(self, goal_store):
        now = datetime.utcnow()
        goal = Goal(
            id=str(uuid.uuid4()),
            description="Progress test",
            priority=0.5,
            expected_value=100.0,
            feasibility=0.5,
            status=GoalStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(goal)
        goal_store.update_goal_progress(goal.id, 0.75)

        updated = goal_store.get_goal(goal.id)
        assert updated.metadata["progress"] == pytest.approx(0.75)

    def test_get_goals_by_status(self, goal_store):
        now = datetime.utcnow()
        for i in range(3):
            g = Goal(
                id=str(uuid.uuid4()),
                description=f"Active goal {i}",
                priority=float(i) / 3,
                expected_value=100.0,
                feasibility=0.5,
                status=GoalStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
            goal_store.save_goal(g)

        failed = Goal(
            id=str(uuid.uuid4()),
            description="Failed goal",
            priority=0.5,
            expected_value=100.0,
            feasibility=0.5,
            status=GoalStatus.FAILED,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(failed)

        active_goals = goal_store.get_goals_by_status(GoalStatus.ACTIVE)
        assert len(active_goals) == 3

        failed_goals = goal_store.get_goals_by_status(GoalStatus.FAILED)
        assert len(failed_goals) == 1

    def test_save_and_retrieve_strategy(self, goal_store):
        now = datetime.utcnow()
        goal = Goal(
            id=str(uuid.uuid4()),
            description="Goal with strategy",
            priority=0.8,
            expected_value=500.0,
            feasibility=0.6,
            status=GoalStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        goal_store.save_goal(goal)

        strategy = Strategy(
            id=str(uuid.uuid4()),
            goal_id=goal.id,
            description="Direct approach",
            expected_value=400.0,
            time_estimate=5.0,
            success_probability=0.7,
            resource_requirements={"tools": ["browser"]},
            status=StrategyStatus.PROPOSED,
            steps=["Step 1", "Step 2"],
            created_at=now,
        )
        goal_store.save_strategy(strategy)

        strategies = goal_store.get_strategies_for_goal(goal.id)
        assert len(strategies) == 1
        assert strategies[0].description == "Direct approach"
        assert strategies[0].steps == ["Step 1", "Step 2"]

    def test_statistics(self, goal_store):
        now = datetime.utcnow()
        for status in [GoalStatus.ACTIVE, GoalStatus.COMPLETED, GoalStatus.FAILED]:
            g = Goal(
                id=str(uuid.uuid4()),
                description=f"Goal {status.value}",
                priority=0.5,
                expected_value=100.0,
                feasibility=0.5,
                status=status,
                created_at=now,
                updated_at=now,
            )
            goal_store.save_goal(g)

        stats = goal_store.get_statistics()
        assert stats["goals_by_status"]["active"] == 1
        assert stats["goals_by_status"]["completed"] == 1
        assert stats["goals_by_status"]["failed"] == 1


class TestGoalManagerImpl:
    """Integration tests for GoalManagerImpl."""

    def test_create_goal(self, goal_manager):
        goal = goal_manager.create_goal(
            description="Build a portfolio",
            priority=0.8,
            expected_value=2000.0,
        )
        assert goal.id is not None
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == pytest.approx(0.8)

    def test_get_active_goals_sorted_by_priority(self, goal_manager):
        goal_manager.create_goal("Low priority", priority=0.2, expected_value=100.0)
        goal_manager.create_goal("High priority", priority=0.9, expected_value=500.0)
        goal_manager.create_goal("Medium priority", priority=0.5, expected_value=300.0)

        active = goal_manager.get_active_goals()
        priorities = [g.priority for g in active]
        assert priorities == sorted(priorities, reverse=True)

    def test_goal_hierarchy(self, goal_manager):
        parent = goal_manager.create_goal("Earn $10k", priority=1.0, expected_value=10000.0)
        child1 = goal_manager.create_goal(
            "Freelance work", priority=0.8, expected_value=5000.0,
            parent_goal_id=parent.id
        )
        child2 = goal_manager.create_goal(
            "Sell products", priority=0.6, expected_value=5000.0,
            parent_goal_id=parent.id
        )

        children = goal_manager.get_child_goals(parent.id)
        child_ids = {c.id for c in children}
        assert child1.id in child_ids
        assert child2.id in child_ids

    def test_update_status_cascades_to_children(self, goal_manager):
        parent = goal_manager.create_goal("Parent", priority=1.0, expected_value=1000.0)
        child = goal_manager.create_goal(
            "Child", priority=0.8, expected_value=500.0,
            parent_goal_id=parent.id
        )

        goal_manager.update_goal_status(parent.id, GoalStatus.CANCELLED)

        updated_child = goal_manager.get_goal(child.id)
        assert updated_child.status == GoalStatus.CANCELLED

    def test_update_progress_auto_completes(self, goal_manager):
        goal = goal_manager.create_goal("Auto-complete test", priority=0.5, expected_value=100.0)
        goal_manager.update_goal_progress(goal.id, 1.0)

        updated = goal_manager.get_goal(goal.id)
        assert updated.status == GoalStatus.COMPLETED

    def test_priority_clamped_to_range(self, goal_manager):
        goal = goal_manager.create_goal("Clamped", priority=2.5, expected_value=100.0)
        assert goal.priority <= 1.0

        goal2 = goal_manager.create_goal("Clamped low", priority=-0.5, expected_value=100.0)
        assert goal2.priority >= 0.0

    def test_decompose_goal_returns_children(self, goal_manager):
        parent = goal_manager.create_goal("Parent", priority=1.0, expected_value=1000.0)
        goal_manager.create_goal("Child 1", priority=0.8, expected_value=500.0, parent_goal_id=parent.id)
        goal_manager.create_goal("Child 2", priority=0.6, expected_value=500.0, parent_goal_id=parent.id)

        sub_goals = goal_manager.decompose_goal(parent.id)
        assert len(sub_goals) == 2


# ---------------------------------------------------------------------------
# Task 3.2 – Goal decomposition
# ---------------------------------------------------------------------------

class TestGoalDecomposer:
    """Tests for GoalDecomposer strategy generation and selection."""

    def _make_goal(self, goal_manager, description="Test goal", priority=0.8, value=1000.0):
        return goal_manager.create_goal(description, priority=priority, expected_value=value)

    def test_heuristic_strategies_generated(self, decomposer, goal_manager):
        goal = self._make_goal(goal_manager)
        strategies = decomposer.generate_strategies(goal)

        assert len(strategies) == 3
        for s in strategies:
            assert s.goal_id == goal.id
            assert s.description
            assert s.success_probability > 0
            assert s.time_estimate > 0
            assert len(s.steps) > 0

    def test_strategies_persisted(self, decomposer, goal_manager):
        goal = self._make_goal(goal_manager)
        strategies = decomposer.generate_strategies(goal)

        persisted = goal_manager.get_strategies_for_goal(goal.id)
        assert len(persisted) == len(strategies)

    def test_decompose_goal_creates_sub_goals(self, decomposer, goal_manager):
        goal = self._make_goal(goal_manager)
        sub_goals = decomposer.decompose_goal(goal)

        assert len(sub_goals) == 3
        for sg in sub_goals:
            assert sg.parent_goal_id == goal.id
            assert sg.status == GoalStatus.ACTIVE

    def test_evaluate_strategies_sorted_by_utility(self, decomposer, goal_manager):
        goal = self._make_goal(goal_manager)
        strategies = decomposer.generate_strategies(goal)
        scored = decomposer.evaluate_strategies(strategies)

        scores = [score for _, score in scored]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_strategy(self, decomposer, goal_manager):
        goal = self._make_goal(goal_manager)
        strategies = decomposer.generate_strategies(goal)
        best = decomposer.select_best_strategy(strategies)

        assert best is not None
        assert best in strategies

    def test_select_best_strategy_empty_list(self, decomposer):
        result = decomposer.select_best_strategy([])
        assert result is None

    def test_llm_fallback_on_exception(self, goal_manager):
        """When JanusGPT raises, heuristics are used."""
        mock_gpt = MagicMock()
        mock_gpt.generate.side_effect = RuntimeError("Model unavailable")

        decomposer = GoalDecomposer(goal_manager=goal_manager, janus_gpt=mock_gpt, num_strategies=3)
        goal = goal_manager.create_goal("LLM fallback test", priority=0.7, expected_value=500.0)
        strategies = decomposer.generate_strategies(goal)

        # Should fall back to heuristics
        assert len(strategies) == 3
        for s in strategies:
            assert s.metadata.get("source") == "heuristic"

    def test_llm_strategy_parsing(self, goal_manager):
        """When JanusGPT returns valid JSON, strategies are parsed."""
        llm_output = (
            '{"description": "LLM strategy 1", "expected_value": 800, '
            '"time_estimate_hours": 3, "success_probability": 0.75, '
            '"steps": ["Step A", "Step B"]}\n'
            '{"description": "LLM strategy 2", "expected_value": 600, '
            '"time_estimate_hours": 5, "success_probability": 0.65, '
            '"steps": ["Step X"]}\n'
        )
        mock_gpt = MagicMock()
        mock_gpt.generate.return_value = llm_output

        decomposer = GoalDecomposer(goal_manager=goal_manager, janus_gpt=mock_gpt, num_strategies=3)
        goal = goal_manager.create_goal("LLM parse test", priority=0.8, expected_value=1000.0)
        strategies = decomposer.generate_strategies(goal)

        assert len(strategies) >= 2
        descriptions = [s.description for s in strategies]
        assert "LLM strategy 1" in descriptions

    def test_utility_score_formula(self, decomposer, goal_manager):
        """Utility = (expected_value * success_probability) / time_estimate."""
        goal = self._make_goal(goal_manager)
        from datetime import datetime
        s = Strategy(
            id="test-id",
            goal_id=goal.id,
            description="Test",
            expected_value=100.0,
            time_estimate=2.0,
            success_probability=0.5,
            resource_requirements={},
            status=StrategyStatus.PROPOSED,
            created_at=datetime.utcnow(),
        )
        score = decomposer._utility_score(s)
        assert score == pytest.approx(100.0 * 0.5 / 2.0)


# ---------------------------------------------------------------------------
# Task 3.3 – Progress tracking and adaptation
# ---------------------------------------------------------------------------

class TestProgressTracker:
    """Tests for ProgressTracker monitoring and adaptation."""

    def _make_goal(self, goal_manager, description="Tracked goal"):
        return goal_manager.create_goal(description, priority=0.7, expected_value=500.0)

    def test_record_and_retrieve_progress(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        tracker.record_progress(goal.id, 0.3)
        tracker.record_progress(goal.id, 0.6)

        history = tracker.get_progress_history(goal.id)
        assert len(history) == 2
        assert history[-1].progress == pytest.approx(0.6)

    def test_current_progress(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        tracker.record_progress(goal.id, 0.45)
        assert tracker.get_current_progress(goal.id) == pytest.approx(0.45)

    def test_progress_persisted_to_goal_store(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        tracker.record_progress(goal.id, 0.5)

        updated = goal_manager.get_goal(goal.id)
        assert updated.metadata["progress"] == pytest.approx(0.5)

    def test_detect_stall_pattern(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        # Inject snapshots with no progress change over time
        from janus_reasoning_engine.goals.progress_tracker import ProgressSnapshot
        now = datetime.utcnow()
        for i in range(5):
            snapshot = ProgressSnapshot(
                goal_id=goal.id,
                progress=0.3,
                timestamp=now - timedelta(hours=tracker.STALL_THRESHOLD_HOURS - 0.1 + i * 0.01),
            )
            tracker._snapshots.setdefault(goal.id, []).append(snapshot)

        patterns = tracker.detect_failure_patterns(goal.id)
        pattern_types = {p.pattern_type for p in patterns}
        assert "stalled" in pattern_types

    def test_detect_regression_pattern(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        from janus_reasoning_engine.goals.progress_tracker import ProgressSnapshot
        now = datetime.utcnow()
        tracker._snapshots[goal.id] = [
            ProgressSnapshot(goal_id=goal.id, progress=0.8, timestamp=now - timedelta(hours=1)),
            ProgressSnapshot(goal_id=goal.id, progress=0.5, timestamp=now),
        ]

        patterns = tracker.detect_failure_patterns(goal.id)
        pattern_types = {p.pattern_type for p in patterns}
        assert "regressing" in pattern_types

    def test_detect_timeout_pattern(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        from janus_reasoning_engine.goals.progress_tracker import ProgressSnapshot
        now = datetime.utcnow()
        tracker._snapshots[goal.id] = [
            ProgressSnapshot(goal_id=goal.id, progress=0.1, timestamp=now - timedelta(hours=30)),
            ProgressSnapshot(goal_id=goal.id, progress=0.2, timestamp=now),
        ]

        patterns = tracker.detect_failure_patterns(goal.id, max_duration_hours=24.0)
        pattern_types = {p.pattern_type for p in patterns}
        assert "timeout" in pattern_types

    def test_no_patterns_on_healthy_progress(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        from janus_reasoning_engine.goals.progress_tracker import ProgressSnapshot
        now = datetime.utcnow()
        # Healthy: steady progress over 1 hour
        tracker._snapshots[goal.id] = [
            ProgressSnapshot(goal_id=goal.id, progress=0.0, timestamp=now - timedelta(hours=1)),
            ProgressSnapshot(goal_id=goal.id, progress=0.5, timestamp=now - timedelta(minutes=30)),
            ProgressSnapshot(goal_id=goal.id, progress=0.9, timestamp=now),
        ]

        patterns = tracker.detect_failure_patterns(goal.id, max_duration_hours=24.0)
        critical = {p.pattern_type for p in patterns} & {"regressing", "timeout"}
        assert not critical

    def test_decide_persist_on_healthy_goal(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())
        tracker.record_progress(goal.id, 0.4)

        decision, rationale = tracker.decide_adaptation(goal.id, strategy_id, failure_patterns=[])
        assert decision == AdaptationDecision.PERSIST

    def test_decide_complete_when_progress_full(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())
        tracker.record_progress(goal.id, 1.0)

        decision, _ = tracker.decide_adaptation(goal.id, strategy_id)
        assert decision == AdaptationDecision.COMPLETE

    def test_decide_pivot_after_failures(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())
        tracker.record_progress(goal.id, 0.2)

        for _ in range(tracker.PIVOT_FAILURE_COUNT):
            tracker.record_strategy_failure(strategy_id)

        decision, _ = tracker.decide_adaptation(goal.id, strategy_id, failure_patterns=[])
        assert decision == AdaptationDecision.PIVOT

    def test_decide_abandon_after_many_failures(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())
        tracker.record_progress(goal.id, 0.1)

        for _ in range(tracker.ABANDON_FAILURE_COUNT):
            tracker.record_strategy_failure(strategy_id)

        decision, _ = tracker.decide_adaptation(goal.id, strategy_id, failure_patterns=[])
        assert decision == AdaptationDecision.ABANDON

    def test_decide_pivot_on_critical_pattern(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())
        tracker.record_progress(goal.id, 0.2)

        critical_pattern = FailurePattern(
            strategy_id=strategy_id,
            goal_id=goal.id,
            pattern_type="regressing",
            detected_at=datetime.utcnow(),
        )

        decision, _ = tracker.decide_adaptation(goal.id, strategy_id, failure_patterns=[critical_pattern])
        assert decision == AdaptationDecision.PIVOT

    def test_record_success(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())

        record = tracker.record_success(
            goal.id, strategy_id=strategy_id, lessons=["Worked well"]
        )

        assert record.outcome == "success"
        assert "Worked well" in record.lessons

        updated = goal_manager.get_goal(goal.id)
        assert updated.status == GoalStatus.COMPLETED

    def test_record_failure(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        strategy_id = str(uuid.uuid4())

        record = tracker.record_failure(
            goal.id, strategy_id=strategy_id, lessons=["Approach was wrong"]
        )

        assert record.outcome == "failure"
        updated = goal_manager.get_goal(goal.id)
        assert updated.status == GoalStatus.FAILED

        # Failure count should be incremented
        assert tracker.get_strategy_failure_count(strategy_id) == 1

    def test_get_learning_records_filtered(self, tracker, goal_manager):
        goal1 = self._make_goal(goal_manager, "Goal 1")
        goal2 = self._make_goal(goal_manager, "Goal 2")

        tracker.record_success(goal1.id, lessons=["Win"])
        tracker.record_failure(goal2.id, lessons=["Loss"])

        successes = tracker.get_learning_records(outcome="success")
        assert all(r.outcome == "success" for r in successes)

        goal1_records = tracker.get_learning_records(goal_id=goal1.id)
        assert all(r.goal_id == goal1.id for r in goal1_records)

    def test_reflect_returns_summary(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        tracker.record_progress(goal.id, 0.3)
        tracker.record_progress(goal.id, 0.6)
        tracker.record_success(goal.id, lessons=["Lesson A"])

        reflection = tracker.reflect(goal.id)

        assert reflection["goal_id"] == goal.id
        assert reflection["current_progress"] == pytest.approx(0.6)
        assert reflection["success_count"] == 1
        assert "Lesson A" in reflection["lessons_learned"]

    def test_statistics(self, tracker, goal_manager):
        goal = self._make_goal(goal_manager)
        tracker.record_progress(goal.id, 0.5)
        tracker.record_strategy_failure("strat-1")

        stats = tracker.get_statistics()
        assert stats["tracked_goals"] >= 1
        assert stats["total_strategy_failures"] >= 1
