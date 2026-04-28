"""
Tests for the main engine wiring (Tasks 20.1–20.3).

Covers:
- JanusReasoningEngine lifecycle (20.1)
- CLI commands (20.2)
- Example usage (20.3)
"""

from __future__ import annotations

import pytest

from janus_reasoning_engine.engine import JanusReasoningEngine
from janus_reasoning_engine.core.config import EngineConfig
from janus_reasoning_engine.core.autonomous_loop import CycleOutcome


# =============================================================================
# Task 20.1 – JanusReasoningEngine
# =============================================================================

class TestJanusReasoningEngineLifecycle:

    def test_create_engine_default_config(self):
        engine = JanusReasoningEngine()
        assert engine is not None
        assert engine.config is not None

    def test_create_engine_custom_config(self):
        config = EngineConfig()
        config.reasoning.exploration_rate = 0.2
        engine = JanusReasoningEngine(config=config)
        assert engine.config.reasoning.exploration_rate == 0.2

    def test_initialize_sets_flag(self):
        engine = JanusReasoningEngine()
        assert engine._initialized is False
        engine.initialize()
        assert engine._initialized is True
        engine.shutdown()

    def test_initialize_idempotent(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        engine.initialize()  # second call should not raise
        assert engine._initialized is True
        engine.shutdown()

    def test_shutdown_clears_flag(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        engine.shutdown()
        assert engine._initialized is False

    def test_shutdown_without_initialize_does_not_raise(self):
        engine = JanusReasoningEngine()
        engine.shutdown()  # should not raise

    def test_run_cycle_raises_if_not_initialized(self):
        engine = JanusReasoningEngine()
        with pytest.raises(RuntimeError):
            engine.run_cycle()

    def test_run_cycle_returns_cycle_outcome(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        outcome = engine.run_cycle()
        assert isinstance(outcome, CycleOutcome)
        engine.shutdown()

    def test_run_cycle_increments_count(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        engine.run_cycle()
        engine.run_cycle()
        assert engine._cycle_count == 2
        engine.shutdown()

    def test_run_cycle_outcome_has_fields(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        outcome = engine.run_cycle()
        assert hasattr(outcome, "cycle_num")
        assert hasattr(outcome, "action_taken")
        assert hasattr(outcome, "success")
        assert hasattr(outcome, "earnings")
        engine.shutdown()

    def test_run_cycle_cycle_num_increments(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        o1 = engine.run_cycle()
        o2 = engine.run_cycle()
        assert o2.cycle_num > o1.cycle_num
        engine.shutdown()


class TestJanusReasoningEngineGetStatus:

    def test_get_status_returns_dict(self):
        engine = JanusReasoningEngine()
        status = engine.get_status()
        assert isinstance(status, dict)

    def test_get_status_before_init(self):
        engine = JanusReasoningEngine()
        status = engine.get_status()
        assert status["initialized"] is False
        assert status["cycle_count"] == 0

    def test_get_status_after_init(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        status = engine.get_status()
        assert status["initialized"] is True
        engine.shutdown()

    def test_get_status_has_subsystems(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        status = engine.get_status()
        assert "subsystems" in status
        assert isinstance(status["subsystems"], dict)
        engine.shutdown()

    def test_get_status_has_config(self):
        engine = JanusReasoningEngine()
        status = engine.get_status()
        assert "config" in status
        assert "autonomous_mode" in status["config"]

    def test_get_status_uptime_increases(self):
        import time
        engine = JanusReasoningEngine()
        engine.initialize()
        time.sleep(0.05)
        status = engine.get_status()
        assert status["uptime_seconds"] > 0
        engine.shutdown()

    def test_get_status_start_time_set_after_init(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        status = engine.get_status()
        assert status["start_time"] is not None
        engine.shutdown()

    def test_get_status_start_time_none_before_init(self):
        engine = JanusReasoningEngine()
        status = engine.get_status()
        assert status["start_time"] is None


class TestJanusReasoningEngineSetGoal:

    def test_set_goal_before_init_returns_none_or_id(self):
        engine = JanusReasoningEngine()
        # Before init, goal manager may not be available
        result = engine.set_goal("test goal")
        # Should not raise; may return None or a string
        assert result is None or isinstance(result, str)

    def test_set_goal_after_init(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        result = engine.set_goal("Earn $1000 this week")
        # May return None if goal manager unavailable in test env
        assert result is None or isinstance(result, str)
        engine.shutdown()

    def test_set_goal_with_priority(self):
        engine = JanusReasoningEngine()
        engine.initialize()
        result = engine.set_goal("Learn Python", priority=0.5)
        assert result is None or isinstance(result, str)
        engine.shutdown()


# =============================================================================
# Task 20.2 – CLI
# =============================================================================

class TestCLI:

    def test_import_cli(self):
        from janus_reasoning_engine import cli
        assert cli is not None

    def test_build_parser(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        assert parser is not None

    def test_parser_start_command(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["start", "--cycles", "1"])
        assert args.command == "start"
        assert args.cycles == 1

    def test_parser_start_default_cycles(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["start"])
        assert args.cycles == 0

    def test_parser_start_with_goal(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["start", "--goal", "Make money"])
        assert args.goal == "Make money"

    def test_parser_status_command(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_parser_set_goal_command(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["set-goal", "Earn $10k"])
        assert args.command == "set-goal"
        assert args.goal == "Earn $10k"

    def test_parser_stop_command(self):
        from janus_reasoning_engine.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"

    def test_main_no_args_exits(self):
        from janus_reasoning_engine.cli import main
        with pytest.raises(SystemExit):
            main([])

    def test_cmd_status_runs(self, capsys):
        from janus_reasoning_engine.cli import cmd_status
        import argparse
        args = argparse.Namespace(command="status")
        ret = cmd_status(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_cmd_set_goal_runs(self, capsys):
        from janus_reasoning_engine.cli import cmd_set_goal
        import argparse
        args = argparse.Namespace(command="set-goal", goal="Test goal")
        ret = cmd_set_goal(args)
        assert ret == 0

    def test_cmd_stop_no_pid_file(self, tmp_path, monkeypatch):
        from janus_reasoning_engine import cli
        monkeypatch.setattr(cli, "_PID_FILE", str(tmp_path / "nonexistent.pid"))
        import argparse
        args = argparse.Namespace(command="stop")
        ret = cli.cmd_stop(args)
        assert ret == 1


# =============================================================================
# Task 20.3 – Example usage
# =============================================================================

class TestExampleUsage:

    def test_basic_usage_importable(self):
        from janus_reasoning_engine.examples import basic_usage
        assert basic_usage is not None

    def test_basic_usage_has_main(self):
        from janus_reasoning_engine.examples.basic_usage import main
        assert callable(main)

    def test_basic_usage_runs(self, capsys):
        from janus_reasoning_engine.examples.basic_usage import main
        main()
        captured = capsys.readouterr()
        assert "Janus Reasoning Engine" in captured.out
        assert "Example completed successfully" in captured.out
