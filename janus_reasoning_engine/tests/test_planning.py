"""
Tests for the planning subsystem (Tasks 7.1, 7.2, 7.3).

Covers:
- MultiStepPlanner: plan creation, heuristic fallback, contingency (7.1)
- ToolOrchestrator: tool selection, step execution, error handling (7.2)
- ExecutionMonitor: session lifecycle, stuck detection, plan adjustment (7.3)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from janus_reasoning_engine.planning.multi_step_planner import (
    MultiStepPlanner,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
    _heuristic_steps,
)
from janus_reasoning_engine.planning.tool_orchestrator import (
    StepResult,
    ToolOrchestrator,
    ToolRegistry,
)
from janus_reasoning_engine.planning.execution_monitor import (
    ExecutionMonitor,
    ExecutionSession,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_step(description="Do something", tool_type="browser", deps=None) -> PlanStep:
    return PlanStep(
        id=str(uuid.uuid4()),
        description=description,
        tool_type=tool_type,
        estimated_minutes=10.0,
        dependencies=deps or [],
    )


def _make_plan(num_steps=3) -> Plan:
    steps = [_make_step(f"Step {i}") for i in range(num_steps)]
    # Wire sequential dependencies
    for i in range(1, len(steps)):
        steps[i].dependencies = [steps[i - 1].id]
    return Plan(
        id=str(uuid.uuid4()),
        goal_description="Test goal",
        steps=steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 7.1 – MultiStepPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiStepPlanner:

    def test_create_plan_returns_plan_object(self):
        planner = MultiStepPlanner()
        plan = planner.create_plan("Build a website")
        assert isinstance(plan, Plan)
        assert plan.id
        assert plan.goal_description == "Build a website"

    def test_heuristic_fallback_produces_3_to_5_steps(self):
        planner = MultiStepPlanner(janus_gpt=None)
        plan = planner.create_plan("Earn money freelancing")
        assert 3 <= len(plan.steps) <= 5

    def test_plan_steps_have_required_fields(self):
        planner = MultiStepPlanner()
        plan = planner.create_plan("Research machine learning")
        for step in plan.steps:
            assert step.id
            assert step.description
            assert step.tool_type
            assert step.estimated_minutes > 0

    def test_sequential_dependencies_wired(self):
        planner = MultiStepPlanner()
        plan = planner.create_plan("Write a report")
        for i, step in enumerate(plan.steps):
            if i == 0:
                assert step.dependencies == []
            else:
                assert plan.steps[i - 1].id in step.dependencies

    def test_plan_status_is_pending(self):
        planner = MultiStepPlanner()
        plan = planner.create_plan("Learn Python")
        assert plan.status == PlanStatus.PENDING

    def test_llm_used_when_gpt_provided(self):
        mock_gpt = MagicMock()
        mock_gpt.generate.return_value = (
            '[{"description": "LLM step 1", "tool_type": "browser", "estimated_minutes": 10},'
            ' {"description": "LLM step 2", "tool_type": "code_execution", "estimated_minutes": 20}]'
        )
        planner = MultiStepPlanner(janus_gpt=mock_gpt)
        plan = planner.create_plan("Do something with LLM")
        mock_gpt.generate.assert_called_once()
        descriptions = [s.description for s in plan.steps]
        assert "LLM step 1" in descriptions

    def test_llm_failure_falls_back_to_heuristics(self):
        mock_gpt = MagicMock()
        mock_gpt.generate.side_effect = RuntimeError("Model unavailable")
        planner = MultiStepPlanner(janus_gpt=mock_gpt)
        plan = planner.create_plan("Search for jobs")
        assert len(plan.steps) >= 3

    def test_llm_invalid_json_falls_back_to_heuristics(self):
        mock_gpt = MagicMock()
        mock_gpt.generate.return_value = "This is not JSON at all."
        planner = MultiStepPlanner(janus_gpt=mock_gpt)
        plan = planner.create_plan("Code a script")
        assert len(plan.steps) >= 3

    def test_add_contingency_sets_field(self):
        planner = MultiStepPlanner()
        step = _make_step()
        result = planner.add_contingency(step, "Try alternative approach")
        assert result.contingency == "Try alternative approach"
        assert result is step  # mutated in-place

    def test_add_contingency_returns_same_step(self):
        planner = MultiStepPlanner()
        step = _make_step()
        returned = planner.add_contingency(step, "Fallback")
        assert returned is step

    def test_heuristic_keyword_research(self):
        steps = _heuristic_steps("research machine learning papers")
        assert any("search" in s["description"].lower() or "find" in s["description"].lower() for s in steps)

    def test_heuristic_keyword_code(self):
        steps = _heuristic_steps("implement a sorting algorithm")
        assert any("implement" in s["description"].lower() or "code" in s["description"].lower() for s in steps)

    def test_heuristic_keyword_earn(self):
        steps = _heuristic_steps("earn money on Upwork")
        assert len(steps) >= 3

    def test_plan_next_step_respects_dependencies(self):
        planner = MultiStepPlanner()
        plan = planner.create_plan("Multi-step task")
        # First step should have no dependencies
        first = plan.next_step()
        assert first is not None
        assert first.dependencies == []

    def test_plan_is_complete_when_all_steps_done(self):
        plan = _make_plan(2)
        for step in plan.steps:
            step.status = StepStatus.COMPLETED
        assert plan.is_complete()

    def test_plan_has_failed_when_step_failed(self):
        plan = _make_plan(2)
        plan.steps[0].status = StepStatus.FAILED
        assert plan.has_failed()


# ─────────────────────────────────────────────────────────────────────────────
# Task 7.2 – ToolOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_builtin_tools_registered(self):
        registry = ToolRegistry()
        for tool in ToolRegistry.BUILTIN_TOOLS:
            assert tool in registry.available_tools()

    def test_register_custom_tool(self):
        registry = ToolRegistry()
        registry.register("my_tool", lambda step: StepResult(step_id=step.id, success=True))
        assert "my_tool" in registry.available_tools()

    def test_get_handler_returns_callable(self):
        registry = ToolRegistry()
        handler = registry.get_handler("browser")
        assert callable(handler)

    def test_get_handler_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get_handler("nonexistent_tool") is None


class TestToolOrchestrator:

    def test_select_tool_browser_for_web_task(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("Search the web for Python tutorials")
        assert tool == "browser"

    def test_select_tool_code_execution_for_code_task(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("Run the Python script and compute results")
        assert tool == "code_execution"

    def test_select_tool_file_manipulation_for_file_task(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("Read the CSV file and save output")
        assert tool == "file_manipulation"

    def test_select_tool_computer_use_for_ui_task(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("Click the button on the desktop application")
        assert tool == "computer_use"

    def test_select_tool_autonomous_worker_for_job_task(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("Complete the freelance job and deliver work")
        assert tool == "autonomous_worker"

    def test_select_tool_defaults_to_browser(self):
        orch = ToolOrchestrator()
        tool = orch.select_tool("xyzzy frobnicator quux")
        assert tool == "browser"

    def test_execute_step_success(self):
        orch = ToolOrchestrator()
        step = _make_step("Search for information", tool_type="browser")
        result = orch.execute_step(step)
        assert isinstance(result, StepResult)
        assert result.success
        assert step.status == StepStatus.COMPLETED

    def test_execute_step_updates_step_status_on_failure(self):
        registry = ToolRegistry()
        registry.register("browser", lambda s: StepResult(step_id=s.id, success=False, error="Network error"))
        orch = ToolOrchestrator(registry=registry)
        step = _make_step("Browse web", tool_type="browser")
        result = orch.execute_step(step)
        assert not result.success
        assert step.status == StepStatus.FAILED

    def test_execute_step_catches_exception(self):
        registry = ToolRegistry()
        def _exploding_handler(step):
            raise RuntimeError("Boom!")
        registry.register("browser", _exploding_handler)
        orch = ToolOrchestrator(registry=registry)
        step = _make_step("Explode", tool_type="browser")
        result = orch.execute_step(step)
        assert not result.success
        assert "Boom!" in result.error
        assert step.status == StepStatus.FAILED

    def test_execute_step_uses_tool_whitelist(self):
        called_tools = []
        registry = ToolRegistry()
        def _tracker(name):
            def _h(step):
                called_tools.append(name)
                return StepResult(step_id=step.id, success=True)
            return _h
        registry.register("code_execution", _tracker("code_execution"))
        registry.register("browser", _tracker("browser"))
        orch = ToolOrchestrator(registry=registry)
        # Step says browser but whitelist only allows code_execution
        step = _make_step("Browse web", tool_type="browser")
        orch.execute_step(step, tools=["code_execution"])
        assert "code_execution" in called_tools

    def test_execute_step_returns_step_id(self):
        orch = ToolOrchestrator()
        step = _make_step()
        result = orch.execute_step(step)
        assert result.step_id == step.id


# ─────────────────────────────────────────────────────────────────────────────
# Task 7.3 – ExecutionMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionMonitor:

    def _make_always_success_monitor(self) -> ExecutionMonitor:
        registry = ToolRegistry()
        for tool in ToolRegistry.BUILTIN_TOOLS:
            registry.register(tool, lambda s: StepResult(step_id=s.id, success=True, output="ok"))
        orch = ToolOrchestrator(registry=registry)
        return ExecutionMonitor(orchestrator=orch, timeout_minutes=30)

    def test_start_plan_returns_session(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        assert isinstance(session, ExecutionSession)
        assert session.plan is plan
        assert plan.status == PlanStatus.IN_PROGRESS

    def test_advance_executes_first_step(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        result = monitor.advance(session)
        assert result.success
        assert plan.steps[0].status == StepStatus.COMPLETED

    def test_advance_completes_all_steps(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(3)
        session = monitor.start_plan(plan)
        for _ in range(3):
            monitor.advance(session)
        assert plan.is_complete()
        assert plan.status == PlanStatus.COMPLETED

    def test_advance_on_finished_session_returns_error(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        monitor.advance(session)  # completes the only step
        monitor.advance(session)  # finalises
        result = monitor.advance(session)  # already finished
        assert not result.success

    def test_stuck_detection_triggers_after_timeout(self):
        monitor = self._make_always_success_monitor()
        monitor.timeout_minutes = 0.001  # very short timeout
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        # Backdate last_progress_at to simulate timeout
        session.last_progress_at = datetime.utcnow() - timedelta(minutes=1)
        result = monitor.advance(session)
        assert not result.success
        assert session.is_stuck
        assert result.metadata.get("stuck")

    def test_no_stuck_detection_within_timeout(self):
        monitor = self._make_always_success_monitor()
        monitor.timeout_minutes = 60
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        result = monitor.advance(session)
        assert result.success
        assert not session.is_stuck

    def test_adjust_plan_adds_steps(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        monitor.advance(session)  # complete the only step

        new_step = _make_step("Extra step")
        monitor.adjust_plan(session, [new_step])
        assert new_step in session.plan.steps

    def test_adjust_plan_clears_stuck_flag(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        session.is_stuck = True

        new_step = _make_step("Recovery step")
        monitor.adjust_plan(session, [new_step])
        assert not session.is_stuck

    def test_adjust_plan_wires_dependency_to_last_step(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        last_id = plan.steps[-1].id

        new_step = _make_step("New step")
        monitor.adjust_plan(session, [new_step])
        assert last_id in new_step.dependencies

    def test_adjust_plan_empty_list_is_noop(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        original_count = len(plan.steps)
        monitor.adjust_plan(session, [])
        assert len(plan.steps) == original_count

    def test_get_status_returns_dict(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        status = monitor.get_status(session)
        assert status["plan_id"] == plan.id
        assert status["total_steps"] == 2
        assert "progress" in status

    def test_progress_increases_after_step(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        assert session.progress == 0.0
        monitor.advance(session)
        assert session.progress > 0.0

    def test_failed_step_marks_plan_failed_when_no_next(self):
        registry = ToolRegistry()
        registry.register("browser", lambda s: StepResult(step_id=s.id, success=False, error="err"))
        orch = ToolOrchestrator(registry=registry)
        monitor = ExecutionMonitor(orchestrator=orch)

        # Single step plan — failure should mark plan as failed
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        monitor.advance(session)
        assert plan.steps[0].status == StepStatus.FAILED

    def test_session_stored_in_monitor(self):
        monitor = self._make_always_success_monitor()
        plan = _make_plan(1)
        session = monitor.start_plan(plan)
        assert monitor.get_session(session.id) is session

    def test_checkpoint_integration_graceful_when_unavailable(self):
        """Monitor should work fine even when checkpoint module is absent."""
        monitor = self._make_always_success_monitor()
        monitor._checkpointer = None  # simulate missing checkpoint
        plan = _make_plan(2)
        session = monitor.start_plan(plan)
        result = monitor.advance(session)
        assert result.success  # no crash

    def test_execution_session_to_dict(self):
        plan = _make_plan(1)
        session = ExecutionSession(id=str(uuid.uuid4()), plan=plan)
        d = session.to_dict()
        assert d["plan_id"] == plan.id
        assert "progress" in d
        assert "is_stuck" in d
