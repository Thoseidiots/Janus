"""
Tests for the infrastructure module: ServiceGateway, SystemMonitor, ReliabilityManager.

Covers tasks 14.1, 14.2, 14.3.
"""

from __future__ import annotations

import os
import pytest

from janus_reasoning_engine.infrastructure.service_gateway import ServiceGateway
from janus_reasoning_engine.infrastructure.monitor import PerformanceSnapshot, SystemMonitor
from janus_reasoning_engine.infrastructure.reliability import ReliabilityManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gateway():
    return ServiceGateway()


@pytest.fixture
def monitor():
    return SystemMonitor()


@pytest.fixture
def reliability(tmp_path):
    return ReliabilityManager(checkpoint_dir=str(tmp_path / "checkpoints"))


# ---------------------------------------------------------------------------
# Task 14.1 — ServiceGateway
# ---------------------------------------------------------------------------

class TestServiceGateway:
    def test_submit_task_returns_string(self, gateway):
        task_id = gateway.submit_task("do something", priority=5)
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    def test_submit_task_ids_are_unique(self, gateway):
        id1 = gateway.submit_task("task A")
        id2 = gateway.submit_task("task B")
        assert id1 != id2

    def test_get_task_status_after_submit(self, gateway):
        task_id = gateway.submit_task("process data")
        status = gateway.get_task_status(task_id)
        assert status in {"pending", "running", "done", "unknown"}

    def test_get_task_status_submitted_is_pending(self, gateway):
        task_id = gateway.submit_task("pending task")
        assert gateway.get_task_status(task_id) == "pending"

    def test_get_task_status_unknown_id(self, gateway):
        assert gateway.get_task_status("nonexistent-id") == "unknown"

    def test_get_credits_balance_returns_float(self, gateway):
        balance = gateway.get_credits_balance()
        assert isinstance(balance, float)

    def test_get_credits_balance_non_negative(self, gateway):
        balance = gateway.get_credits_balance()
        assert balance >= 0.0

    def test_submit_task_with_priority(self, gateway):
        task_id = gateway.submit_task("high priority task", priority=1)
        assert isinstance(task_id, str)


# ---------------------------------------------------------------------------
# Task 14.2 — SystemMonitor
# ---------------------------------------------------------------------------

class TestSystemMonitor:
    def test_get_performance_snapshot_returns_snapshot(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert isinstance(snap, PerformanceSnapshot)

    def test_snapshot_has_timestamp(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert snap.timestamp > 0

    def test_snapshot_cpu_pct_range(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert 0.0 <= snap.cpu_pct <= 100.0

    def test_snapshot_memory_pct_range(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert 0.0 <= snap.memory_pct <= 100.0

    def test_snapshot_active_tasks_non_negative(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert snap.active_tasks >= 0

    def test_snapshot_metrics_is_dict(self, monitor):
        snap = monitor.get_performance_snapshot()
        assert isinstance(snap.metrics, dict)

    def test_record_metric_and_retrieve(self, monitor):
        monitor.record_metric("latency", 42.0)
        history = monitor.get_metric_history("latency")
        assert 42.0 in history

    def test_get_metric_history_empty_for_unknown(self, monitor):
        assert monitor.get_metric_history("nonexistent") == []

    def test_get_metric_history_respects_limit(self, monitor):
        for i in range(20):
            monitor.record_metric("cpu", float(i))
        history = monitor.get_metric_history("cpu", limit=5)
        assert len(history) == 5
        assert history[-1] == 19.0  # most recent

    def test_get_metric_history_returns_list(self, monitor):
        monitor.record_metric("mem", 55.0)
        result = monitor.get_metric_history("mem")
        assert isinstance(result, list)

    def test_alert_if_threshold_above_triggered(self, monitor):
        monitor.record_metric("cpu", 95.0)
        assert monitor.alert_if_threshold("cpu", threshold=90.0, direction="above") is True

    def test_alert_if_threshold_above_not_triggered(self, monitor):
        monitor.record_metric("cpu", 50.0)
        assert monitor.alert_if_threshold("cpu", threshold=90.0, direction="above") is False

    def test_alert_if_threshold_below_triggered(self, monitor):
        monitor.record_metric("balance", 5.0)
        assert monitor.alert_if_threshold("balance", threshold=10.0, direction="below") is True

    def test_alert_if_threshold_below_not_triggered(self, monitor):
        monitor.record_metric("balance", 50.0)
        assert monitor.alert_if_threshold("balance", threshold=10.0, direction="below") is False

    def test_alert_if_threshold_no_data_returns_false(self, monitor):
        assert monitor.alert_if_threshold("unknown_metric", 50.0) is False

    def test_snapshot_reflects_recorded_metrics(self, monitor):
        monitor.record_metric("requests", 100.0)
        snap = monitor.get_performance_snapshot()
        assert "requests" in snap.metrics
        assert snap.metrics["requests"] == 100.0


# ---------------------------------------------------------------------------
# Task 14.3 — ReliabilityManager
# ---------------------------------------------------------------------------

class TestReliabilityManager:
    def test_save_checkpoint_returns_string(self, reliability):
        cid = reliability.save_checkpoint({"step": 1, "data": "hello"})
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_save_and_load_checkpoint(self, reliability):
        state = {"step": 5, "value": 42, "items": [1, 2, 3]}
        cid = reliability.save_checkpoint(state)
        loaded = reliability.load_checkpoint(cid)
        assert loaded["step"] == 5
        assert loaded["value"] == 42
        assert loaded["items"] == [1, 2, 3]

    def test_load_nonexistent_checkpoint_returns_empty(self, reliability):
        result = reliability.load_checkpoint("nonexistent-id")
        assert result == {}

    def test_checkpoint_ids_are_unique(self, reliability):
        id1 = reliability.save_checkpoint({"a": 1})
        id2 = reliability.save_checkpoint({"b": 2})
        assert id1 != id2

    def test_checkpoint_file_created(self, reliability, tmp_path):
        cid = reliability.save_checkpoint({"x": 99})
        checkpoint_dir = str(tmp_path / "checkpoints")
        files = os.listdir(checkpoint_dir)
        assert any(cid in f for f in files)

    def test_detect_loop_no_loop(self, reliability):
        history = ["action_a", "action_b", "action_c", "action_a", "action_b"]
        assert reliability.detect_loop(history) is False

    def test_detect_loop_with_loop(self, reliability):
        history = ["action_a"] * 5
        assert reliability.detect_loop(history) is True

    def test_detect_loop_exactly_threshold(self, reliability):
        # 3 repeats should NOT trigger (> 3 required)
        history = ["action_x"] * 3
        assert reliability.detect_loop(history) is False

    def test_detect_loop_just_above_threshold(self, reliability):
        # 4 repeats should trigger (> 3)
        history = ["action_x"] * 4
        assert reliability.detect_loop(history) is True

    def test_detect_loop_respects_window(self, reliability):
        # Loop only in the last 10 entries
        history = ["other"] * 20 + ["loop_action"] * 4
        assert reliability.detect_loop(history, window=10) is True

    def test_detect_loop_empty_history(self, reliability):
        assert reliability.detect_loop([]) is False

    def test_trigger_self_heal_returns_bool(self, reliability):
        result = reliability.trigger_self_heal()
        assert isinstance(result, bool)

    def test_trigger_self_heal_stub_returns_true(self, reliability):
        # Without backend, stub returns True
        result = reliability.trigger_self_heal()
        assert result is True
