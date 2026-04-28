"""
Tests for the finance module: PaymentHub, RevenueTracker, FinancialManager.

Covers tasks 11.1, 11.2, 11.3, 11.4.
"""

from __future__ import annotations

import os
import tempfile
import pytest

from janus_reasoning_engine.finance.payment_hub import PaymentHub, PaymentResult
from janus_reasoning_engine.finance.revenue_tracker import RevenueTracker
from janus_reasoning_engine.finance.financial_manager import FinancialManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    db = str(tmp_path / "test_revenue.db")
    rt = RevenueTracker(db_path=db)
    yield rt
    rt.close()


@pytest.fixture
def hub():
    return PaymentHub()


@pytest.fixture
def manager():
    return FinancialManager()


# ---------------------------------------------------------------------------
# Task 11.1 — PaymentHub
# ---------------------------------------------------------------------------

class TestPaymentHub:
    def test_get_supported_methods_returns_list(self, hub):
        methods = hub.get_supported_methods()
        assert isinstance(methods, list)
        assert len(methods) >= 1

    def test_manual_always_supported(self, hub):
        assert "manual" in hub.get_supported_methods()

    def test_process_payment_manual_succeeds(self, hub):
        result = hub.process_payment(10.0, "manual", "test@example.com")
        assert isinstance(result, PaymentResult)
        assert result.success is True
        assert result.amount == 10.0
        assert result.recipient == "test@example.com"
        assert result.method == "manual"
        assert result.payment_id  # non-empty

    def test_process_payment_unknown_method_fails(self, hub):
        result = hub.process_payment(5.0, "nonexistent_method", "x@y.com")
        assert result.success is False
        assert result.error is not None

    def test_verify_payment_empty_id_returns_false(self, hub):
        assert hub.verify_payment("") is False

    def test_verify_payment_unknown_id_returns_false(self, hub):
        assert hub.verify_payment("00000000-0000-0000-0000-000000000000") is False

    def test_payment_result_has_payment_id(self, hub):
        result = hub.process_payment(1.0, "manual", "a@b.com")
        assert isinstance(result.payment_id, str)
        assert len(result.payment_id) > 0


# ---------------------------------------------------------------------------
# Task 11.2 — RevenueTracker core
# ---------------------------------------------------------------------------

class TestRevenueTrackerCore:
    def test_record_earning_returns_id(self, tracker):
        eid = tracker.record_earning(100.0, "upwork", "job-1")
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_get_total_earnings_empty(self, tracker):
        assert tracker.get_total_earnings() == 0.0

    def test_get_total_earnings_after_recording(self, tracker):
        tracker.record_earning(50.0, "fiverr")
        tracker.record_earning(75.0, "upwork")
        total = tracker.get_total_earnings(period_days=30)
        assert total == pytest.approx(125.0)

    def test_get_earnings_by_source(self, tracker):
        tracker.record_earning(100.0, "upwork")
        tracker.record_earning(50.0, "fiverr")
        tracker.record_earning(25.0, "upwork")
        by_source = tracker.get_earnings_by_source()
        assert by_source["upwork"] == pytest.approx(125.0)
        assert by_source["fiverr"] == pytest.approx(50.0)

    def test_calculate_profit(self, tracker):
        assert tracker.calculate_profit(500.0, 200.0) == pytest.approx(300.0)

    def test_calculate_profit_negative(self, tracker):
        assert tracker.calculate_profit(100.0, 150.0) == pytest.approx(-50.0)

    def test_forecast_monthly(self, tracker):
        # Record 300 over 30 days → daily avg 10 → monthly forecast 300
        tracker.record_earning(300.0, "upwork")
        forecast = tracker.forecast_monthly(history_days=30)
        assert forecast == pytest.approx(300.0)

    def test_forecast_monthly_zero_history(self, tracker):
        assert tracker.forecast_monthly(history_days=30) == pytest.approx(0.0)

    def test_record_earning_negative_raises(self, tracker):
        with pytest.raises(ValueError):
            tracker.record_earning(-10.0, "upwork")


# ---------------------------------------------------------------------------
# Task 11.4 — RevenueTracker optimisation
# ---------------------------------------------------------------------------

class TestRevenueTrackerOptimisation:
    def test_get_profit_per_hour_no_data(self, tracker):
        assert tracker.get_profit_per_hour("nonexistent-job") == 0.0

    def test_get_profit_per_hour_with_data(self, tracker):
        tracker.record_earning(200.0, "upwork", "job-abc")
        tracker.record_job_hours("job-abc", 4.0)
        pph = tracker.get_profit_per_hour("job-abc")
        assert pph == pytest.approx(50.0)

    def test_get_profit_per_hour_no_hours(self, tracker):
        tracker.record_earning(100.0, "fiverr", "job-xyz")
        assert tracker.get_profit_per_hour("job-xyz") == 0.0

    def test_record_job_hours_negative_raises(self, tracker):
        with pytest.raises(ValueError):
            tracker.record_job_hours("job-1", -1.0)

    def test_optimize_revenue_mix_sorts_by_pph(self, tracker):
        opps = [
            {"name": "low", "profit_per_hour": 10.0},
            {"name": "high", "profit_per_hour": 50.0},
            {"name": "mid", "profit_per_hour": 25.0},
        ]
        ranked = tracker.optimize_revenue_mix(opps)
        assert ranked[0]["name"] == "high"
        assert ranked[1]["name"] == "mid"
        assert ranked[2]["name"] == "low"

    def test_optimize_revenue_mix_derived_pph(self, tracker):
        opps = [
            {"name": "a", "estimated_revenue": 100.0, "estimated_hours": 10.0},
            {"name": "b", "estimated_revenue": 200.0, "estimated_hours": 5.0},
        ]
        ranked = tracker.optimize_revenue_mix(opps)
        assert ranked[0]["name"] == "b"  # 40/hr vs 10/hr

    def test_optimize_revenue_mix_empty(self, tracker):
        assert tracker.optimize_revenue_mix([]) == []

    def test_get_top_revenue_sources(self, tracker):
        tracker.record_earning(500.0, "upwork")
        tracker.record_earning(300.0, "fiverr")
        tracker.record_earning(100.0, "direct")
        top = tracker.get_top_revenue_sources(n=2)
        assert len(top) == 2
        assert top[0]["source"] == "upwork"
        assert top[0]["total_earnings"] == pytest.approx(500.0)

    def test_get_top_revenue_sources_percentage(self, tracker):
        tracker.record_earning(800.0, "upwork")
        tracker.record_earning(200.0, "fiverr")
        top = tracker.get_top_revenue_sources(n=2)
        assert top[0]["percentage"] == pytest.approx(80.0)
        assert top[1]["percentage"] == pytest.approx(20.0)

    def test_get_top_revenue_sources_empty(self, tracker):
        assert tracker.get_top_revenue_sources() == []


# ---------------------------------------------------------------------------
# Task 11.3 — FinancialManager
# ---------------------------------------------------------------------------

class TestFinancialManager:
    def test_get_budget_status_returns_something(self, manager):
        status = manager.get_budget_status()
        assert status is not None

    def test_allocate_budget_positive(self, manager):
        result = manager.allocate_budget("compute", 50.0)
        assert isinstance(result, bool)

    def test_allocate_budget_negative_rejected(self, manager):
        result = manager.allocate_budget("compute", -10.0)
        assert result is False

    def test_predict_cash_flow_returns_float(self, manager):
        result = manager.predict_cash_flow(days=30)
        assert isinstance(result, float)

    def test_check_subscription_health_returns_dict(self, manager):
        health = manager.check_subscription_health()
        assert isinstance(health, dict)
        assert "status" in health

    def test_check_subscription_health_unavailable(self, manager):
        # When subscription system is not installed, should return graceful dict
        health = manager.check_subscription_health()
        assert health.get("status") in ("unavailable", "ok", "error")
