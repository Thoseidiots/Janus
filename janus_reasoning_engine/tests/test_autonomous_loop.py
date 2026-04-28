"""
Tests for janus_reasoning_engine.core.autonomous_loop

Covers tasks 17.1, 17.2, and 17.3:
- LoopState / CycleOutcome / ExecutionResult dataclasses
- AutonomousLoop.run_cycle() with all subsystems None (graceful degradation)
- AutonomousLoop._discover_and_select() with mock monitor/scorer/strategy
- AutonomousLoop._execute_opportunity() with mock planner/monitor
- AutonomousLoop._reflect_and_learn() with mock metacognition/progress_tracker
- AutonomousLoop._update_knowledge() with mock skill_inventory/episodic_memory
- AutonomousLoop._manage_resources() balance threshold logic

All tests are synchronous.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from janus_reasoning_engine.core.autonomous_loop import (
    AutonomousLoop,
    CycleOutcome,
    ExecutionResult,
    LoopState,
)


# ── Helpers / stubs ───────────────────────────────────────────────────────────

def _make_opportunity(
    opp_id: str = "opp-1",
    title: str = "Test Job",
    platform: str = "upwork",
    earning_potential: float = 100.0,
    required_skills: Optional[List[str]] = None,
    score: float = 0.8,
):
    """Return a minimal Opportunity-like object."""
    from janus_reasoning_engine.discovery.opportunity_scorer import Opportunity
    opp = Opportunity(
        id=opp_id,
        title=title,
        description="A test opportunity",
        platform=platform,
        url="https://example.com/job/1",
        earning_potential=earning_potential,
        required_skills=required_skills or ["python"],
        score=score,
    )
    return opp


def _make_plan(num_steps: int = 2):
    """Return a minimal Plan with num_steps steps."""
    from janus_reasoning_engine.planning.multi_step_planner import (
        Plan,
        PlanStatus,
        PlanStep,
        StepStatus,
    )
    import uuid

    steps = []
    for i in range(num_steps):
        step = PlanStep(
            id=str(uuid.uuid4()),
            description=f"Step {i + 1}",
            tool_type="browser",
            estimated_minutes=5.0,
            dependencies=[steps[i - 1].id] if i > 0 else [],
        )
        steps.append(step)

    return Plan(
        id=str(uuid.uuid4()),
        goal_description="Test goal",
        steps=steps,
        status=PlanStatus.PENDING,
    )


# ── Dataclass tests ───────────────────────────────────────────────────────────

class TestLoopState:
    def test_defaults(self):
        state = LoopState()
        assert state.cycle_count == 0
        assert state.current_goal is None
        assert state.active_work is None
        assert state.balance == 0.0
        assert state.last_reflection_at is None

    def test_mutation(self):
        state = LoopState()
        state.cycle_count = 5
        state.balance = 123.45
        assert state.cycle_count == 5
        assert state.balance == 123.45


class TestCycleOutcome:
    def test_defaults(self):
        outcome = CycleOutcome(cycle_num=1, action_taken="idle", success=True)
        assert outcome.earnings == 0.0
        assert outcome.notes == ""

    def test_with_values(self):
        outcome = CycleOutcome(
            cycle_num=3,
            action_taken="execute_opportunity:opp-1",
            success=True,
            earnings=50.0,
            notes="Completed successfully",
        )
        assert outcome.cycle_num == 3
        assert outcome.earnings == 50.0


class TestExecutionResult:
    def test_defaults(self):
        result = ExecutionResult(opportunity_id="opp-1", success=True)
        assert result.earnings == 0.0
        assert result.duration_seconds == 0.0
        assert result.notes == ""

    def test_failure(self):
        result = ExecutionResult(
            opportunity_id="opp-2",
            success=False,
            notes="Timed out",
        )
        assert not result.success
        assert result.notes == "Timed out"


# ── AutonomousLoop — no subsystems ────────────────────────────────────────────

class TestAutonomousLoopNoSubsystems:
    """Loop should work gracefully when all subsystems are None."""

    def test_run_cycle_returns_outcome(self):
        loop = AutonomousLoop()
        outcome = loop.run_cycle()
        assert isinstance(outcome, CycleOutcome)
        assert outcome.cycle_num == 1

    def test_cycle_count_increments(self):
        loop = AutonomousLoop()
        loop.run_cycle()
        loop.run_cycle()
        assert loop.state.cycle_count == 2

    def test_no_opportunity_action(self):
        loop = AutonomousLoop()
        outcome = loop.run_cycle()
        assert outcome.action_taken == "no_opportunity_found"

    def test_active_work_continues(self):
        loop = AutonomousLoop()
        loop.state.active_work = "opp-existing"
        outcome = loop.run_cycle()
        assert "continue_work" in outcome.action_taken
        assert "opp-existing" in outcome.action_taken

    def test_multiple_cycles_stable(self):
        loop = AutonomousLoop()
        for _ in range(10):
            outcome = loop.run_cycle()
            assert isinstance(outcome, CycleOutcome)


# ── AutonomousLoop._assess_state ──────────────────────────────────────────────

class TestAssessState:
    def test_reads_balance_from_financial_manager(self):
        fm = MagicMock()
        budget = MagicMock()
        budget.current_balance = 250.0
        fm.get_budget_status.return_value = budget

        loop = AutonomousLoop(financial_manager=fm)
        loop._assess_state()
        assert loop.state.balance == 250.0

    def test_balance_fallback_on_error(self):
        fm = MagicMock()
        fm.get_budget_status.side_effect = RuntimeError("wallet down")

        loop = AutonomousLoop(financial_manager=fm)
        loop._assess_state()
        # Should not raise; balance stays at default
        assert loop.state.balance == 0.0

    def test_working_memory_context_called(self):
        wm = MagicMock()
        wm.get_recent_context.return_value = []

        loop = AutonomousLoop(working_memory=wm)
        loop._assess_state()
        wm.get_recent_context.assert_called_once()


# ── AutonomousLoop._discover_and_select ───────────────────────────────────────

class TestDiscoverAndSelect:
    def _make_monitor_with_opportunities(self, opps):
        """Return a mock OpportunityMonitor that yields raw dicts."""
        monitor = MagicMock()

        async def _scan(**kwargs):
            return [
                {
                    "id": o.id,
                    "title": o.title,
                    "description": o.description,
                    "platform": o.platform,
                    "url": o.url,
                    "budget": o.earning_potential,
                    "required_skills": o.required_skills,
                }
                for o in opps
            ]

        monitor.scan = _scan
        return monitor

    def test_returns_none_without_monitor(self):
        loop = AutonomousLoop()
        result = loop._discover_and_select()
        assert result is None

    def test_returns_opportunity_with_monitor(self):
        opp = _make_opportunity()
        monitor = self._make_monitor_with_opportunities([opp])

        loop = AutonomousLoop(opportunity_monitor=monitor)
        result = loop._discover_and_select()
        assert result is not None
        assert result.id == "opp-1"

    def test_scorer_is_called(self):
        opp = _make_opportunity()
        monitor = self._make_monitor_with_opportunities([opp])

        scorer = MagicMock()
        scorer.score_and_rank.side_effect = lambda opps: opps

        loop = AutonomousLoop(
            opportunity_monitor=monitor,
            opportunity_scorer=scorer,
        )
        loop._discover_and_select()
        scorer.score_and_rank.assert_called_once()

    def test_exploration_strategy_is_called(self):
        opp = _make_opportunity()
        monitor = self._make_monitor_with_opportunities([opp])

        strategy = MagicMock()
        strategy.select_opportunity.return_value = opp

        loop = AutonomousLoop(
            opportunity_monitor=monitor,
            exploration_strategy=strategy,
        )
        result = loop._discover_and_select()
        strategy.select_opportunity.assert_called_once()
        assert result is opp

    def test_returns_none_when_scan_empty(self):
        monitor = MagicMock()

        async def _scan(**kwargs):
            return []

        monitor.scan = _scan

        loop = AutonomousLoop(opportunity_monitor=monitor)
        result = loop._discover_and_select()
        assert result is None

    def test_uses_skill_inventory_for_search(self):
        opp = _make_opportunity()
        monitor = self._make_monitor_with_opportunities([opp])

        skill_inv = MagicMock()
        skill_inv.as_dict.return_value = {"python": 0.9, "javascript": 0.7}

        loop = AutonomousLoop(
            opportunity_monitor=monitor,
            skill_inventory=skill_inv,
        )
        loop._discover_and_select()
        skill_inv.as_dict.assert_called_once()


# ── AutonomousLoop._execute_opportunity ───────────────────────────────────────

class TestExecuteOpportunity:
    def test_returns_execution_result(self):
        opp = _make_opportunity()
        loop = AutonomousLoop()
        result = loop._execute_opportunity(opp)
        assert isinstance(result, ExecutionResult)
        assert result.opportunity_id == opp.id

    def test_simulated_success_without_subsystems(self):
        opp = _make_opportunity()
        loop = AutonomousLoop()
        result = loop._execute_opportunity(opp)
        assert result.success is True

    def test_uses_planner_to_create_plan(self):
        opp = _make_opportunity()
        plan = _make_plan()

        planner = MagicMock()
        planner.create_plan.return_value = plan

        loop = AutonomousLoop(multi_step_planner=planner)
        loop._execute_opportunity(opp)
        planner.create_plan.assert_called_once()

    def test_uses_execution_monitor(self):
        opp = _make_opportunity()
        plan = _make_plan(num_steps=1)

        planner = MagicMock()
        planner.create_plan.return_value = plan

        session = MagicMock()
        session.is_active = False

        exec_mon = MagicMock()
        exec_mon.start_plan.return_value = session
        exec_mon.advance.return_value = MagicMock(
            success=True, metadata={"plan_complete": True}
        )
        exec_mon.get_status.return_value = {
            "plan_status": "completed",
            "plan_id": plan.id,
        }

        loop = AutonomousLoop(
            multi_step_planner=planner,
            execution_monitor=exec_mon,
        )
        result = loop._execute_opportunity(opp)
        exec_mon.start_plan.assert_called_once_with(plan)
        assert result.success is True

    def test_execution_failure_handled(self):
        opp = _make_opportunity()
        plan = _make_plan()

        planner = MagicMock()
        planner.create_plan.return_value = plan

        exec_mon = MagicMock()
        exec_mon.start_plan.side_effect = RuntimeError("monitor crashed")

        loop = AutonomousLoop(
            multi_step_planner=planner,
            execution_monitor=exec_mon,
        )
        result = loop._execute_opportunity(opp)
        assert result.success is False
        assert "monitor crashed" in result.notes

    def test_duration_is_recorded(self):
        opp = _make_opportunity()
        loop = AutonomousLoop()
        result = loop._execute_opportunity(opp)
        assert result.duration_seconds >= 0.0


# ── AutonomousLoop._reflect_and_learn ─────────────────────────────────────────

class TestReflectAndLearn:
    def test_no_reflection_before_threshold(self):
        meta = MagicMock()
        loop = AutonomousLoop(metacognition=meta)
        loop.state.cycle_count = 3  # not a multiple of REFLECT_EVERY_N_CYCLES

        outcome = CycleOutcome(cycle_num=3, action_taken="idle", success=True)
        loop._reflect_and_learn(outcome)
        meta.reflect.assert_not_called()

    def test_reflection_at_threshold(self):
        meta = MagicMock()
        reflection = MagicMock()
        reflection.success = True
        reflection.confidence_delta = 0.05
        reflection.patterns_identified = []
        meta.reflect.return_value = reflection

        loop = AutonomousLoop(metacognition=meta)
        loop.state.cycle_count = AutonomousLoop.REFLECT_EVERY_N_CYCLES

        outcome = CycleOutcome(
            cycle_num=AutonomousLoop.REFLECT_EVERY_N_CYCLES,
            action_taken="execute_opportunity:opp-1",
            success=True,
        )
        loop._reflect_and_learn(outcome)
        meta.reflect.assert_called_once()

    def test_last_reflection_at_updated(self):
        meta = MagicMock()
        meta.reflect.return_value = MagicMock(
            success=True, confidence_delta=0.0, patterns_identified=[]
        )

        loop = AutonomousLoop(metacognition=meta)
        loop.state.cycle_count = AutonomousLoop.REFLECT_EVERY_N_CYCLES

        outcome = CycleOutcome(
            cycle_num=AutonomousLoop.REFLECT_EVERY_N_CYCLES,
            action_taken="idle",
            success=True,
        )
        loop._reflect_and_learn(outcome)
        assert loop.state.last_reflection_at is not None

    def test_progress_tracker_reflect_called(self):
        tracker = MagicMock()
        tracker.reflect.return_value = {}

        loop = AutonomousLoop(progress_tracker=tracker)
        loop.state.cycle_count = AutonomousLoop.REFLECT_EVERY_N_CYCLES
        loop.state.current_goal = "goal-123"

        outcome = CycleOutcome(
            cycle_num=AutonomousLoop.REFLECT_EVERY_N_CYCLES,
            action_taken="idle",
            success=True,
        )
        loop._reflect_and_learn(outcome)
        tracker.reflect.assert_called_once_with("goal-123")

    def test_metacognition_error_does_not_crash(self):
        meta = MagicMock()
        meta.reflect.side_effect = RuntimeError("meta error")

        loop = AutonomousLoop(metacognition=meta)
        loop.state.cycle_count = AutonomousLoop.REFLECT_EVERY_N_CYCLES

        outcome = CycleOutcome(
            cycle_num=AutonomousLoop.REFLECT_EVERY_N_CYCLES,
            action_taken="idle",
            success=True,
        )
        # Should not raise
        loop._reflect_and_learn(outcome)


# ── AutonomousLoop._update_knowledge ─────────────────────────────────────────

class TestUpdateKnowledge:
    def test_skill_inventory_updated_on_success(self):
        inv = MagicMock()
        loop = AutonomousLoop(skill_inventory=inv)

        result = ExecutionResult(opportunity_id="opp-1", success=True)
        loop._update_knowledge(result)
        inv.update_confidence.assert_called_once()
        args = inv.update_confidence.call_args[0]
        assert args[1] > 0  # positive delta on success

    def test_skill_inventory_updated_on_failure(self):
        inv = MagicMock()
        loop = AutonomousLoop(skill_inventory=inv)

        result = ExecutionResult(opportunity_id="opp-1", success=False)
        loop._update_knowledge(result)
        inv.update_confidence.assert_called_once()
        args = inv.update_confidence.call_args[0]
        assert args[1] < 0  # negative delta on failure

    def test_episodic_memory_stores_experience(self):
        mem = MagicMock()
        loop = AutonomousLoop(episodic_memory=mem)

        result = ExecutionResult(
            opportunity_id="opp-2",
            success=True,
            earnings=75.0,
            duration_seconds=120.0,
        )
        loop._update_knowledge(result)
        mem.store_experience.assert_called_once()
        kwargs = mem.store_experience.call_args[1]
        assert kwargs["earnings"] == 75.0

    def test_episodic_memory_failure_outcome(self):
        mem = MagicMock()
        loop = AutonomousLoop(episodic_memory=mem)

        result = ExecutionResult(opportunity_id="opp-3", success=False)
        loop._update_knowledge(result)
        from janus_reasoning_engine.memory.episodic_memory import OutcomeType
        kwargs = mem.store_experience.call_args[1]
        assert kwargs["outcome_type"] == OutcomeType.FAILURE

    def test_update_knowledge_no_subsystems(self):
        loop = AutonomousLoop()
        result = ExecutionResult(opportunity_id="opp-1", success=True)
        # Should not raise
        loop._update_knowledge(result)

    def test_skill_inventory_error_does_not_crash(self):
        inv = MagicMock()
        inv.update_confidence.side_effect = RuntimeError("inv error")

        loop = AutonomousLoop(skill_inventory=inv)
        result = ExecutionResult(opportunity_id="opp-1", success=True)
        loop._update_knowledge(result)  # should not raise


# ── AutonomousLoop._manage_resources ─────────────────────────────────────────

class TestManageResources:
    def test_no_action_when_balance_healthy(self):
        loop = AutonomousLoop(low_balance_threshold=50.0)
        loop.state.balance = 200.0
        loop.state.active_work = "opp-1"

        loop._manage_resources()
        # active_work should be unchanged
        assert loop.state.active_work == "opp-1"

    def test_clears_active_work_when_balance_low(self):
        loop = AutonomousLoop(low_balance_threshold=50.0)
        loop.state.balance = 10.0
        loop.state.active_work = "opp-1"

        loop._manage_resources()
        assert loop.state.active_work is None

    def test_no_action_when_no_active_work_and_low_balance(self):
        loop = AutonomousLoop(low_balance_threshold=50.0)
        loop.state.balance = 5.0
        loop.state.active_work = None

        loop._manage_resources()  # should not raise
        assert loop.state.active_work is None

    def test_working_memory_urgency_set_on_low_balance(self):
        wm = MagicMock()
        loop = AutonomousLoop(working_memory=wm, low_balance_threshold=50.0)
        loop.state.balance = 20.0

        loop._manage_resources()
        wm.add_context.assert_called_once()
        kwargs = wm.add_context.call_args[1]
        assert kwargs["key"] == "resource_urgency"

    def test_working_memory_error_does_not_crash(self):
        wm = MagicMock()
        wm.add_context.side_effect = RuntimeError("wm error")

        loop = AutonomousLoop(working_memory=wm, low_balance_threshold=50.0)
        loop.state.balance = 5.0
        loop._manage_resources()  # should not raise

    def test_custom_threshold(self):
        loop = AutonomousLoop(low_balance_threshold=500.0)
        loop.state.balance = 100.0
        loop.state.active_work = "opp-1"

        loop._manage_resources()
        # 100 < 500 → should clear active_work
        assert loop.state.active_work is None


# ── Full cycle integration ────────────────────────────────────────────────────

class TestFullCycleIntegration:
    """Integration tests wiring multiple subsystems together."""

    def _make_monitor_with_raw(self, raw_list):
        monitor = MagicMock()

        async def _scan(**kwargs):
            return raw_list

        monitor.scan = _scan
        return monitor

    def test_full_cycle_with_opportunity(self):
        raw = [
            {
                "id": "opp-full-1",
                "title": "Full Test Job",
                "description": "desc",
                "platform": "upwork",
                "url": "https://example.com",
                "budget": 200.0,
                "required_skills": ["python"],
            }
        ]
        monitor = self._make_monitor_with_raw(raw)

        plan = _make_plan(num_steps=1)
        planner = MagicMock()
        planner.create_plan.return_value = plan

        session = MagicMock()
        session.is_active = False

        exec_mon = MagicMock()
        exec_mon.start_plan.return_value = session
        exec_mon.advance.return_value = MagicMock(
            success=True, metadata={"plan_complete": True}
        )
        exec_mon.get_status.return_value = {"plan_status": "completed"}

        loop = AutonomousLoop(
            opportunity_monitor=monitor,
            multi_step_planner=planner,
            execution_monitor=exec_mon,
        )
        outcome = loop.run_cycle()
        assert outcome.success is True
        assert "execute_opportunity" in outcome.action_taken

    def test_cycle_increments_even_on_error(self):
        monitor = MagicMock()

        async def _scan(**kwargs):
            raise RuntimeError("network error")

        monitor.scan = _scan

        loop = AutonomousLoop(opportunity_monitor=monitor)
        outcome = loop.run_cycle()
        assert loop.state.cycle_count == 1
        # Should handle gracefully
        assert isinstance(outcome, CycleOutcome)

    def test_reflection_triggered_at_correct_interval(self):
        meta = MagicMock()
        meta.reflect.return_value = MagicMock(
            success=True, confidence_delta=0.0, patterns_identified=[]
        )

        loop = AutonomousLoop(metacognition=meta)
        n = AutonomousLoop.REFLECT_EVERY_N_CYCLES

        for _ in range(n):
            loop.run_cycle()

        assert meta.reflect.call_count == 1

    def test_balance_prioritisation_in_full_cycle(self):
        fm = MagicMock()
        budget = MagicMock()
        budget.current_balance = 5.0
        fm.get_budget_status.return_value = budget

        loop = AutonomousLoop(
            financial_manager=fm,
            low_balance_threshold=50.0,
        )
        loop.state.active_work = "opp-existing"
        loop.run_cycle()

        # After manage_resources, active_work should be cleared
        assert loop.state.active_work is None
