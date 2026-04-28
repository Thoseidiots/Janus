"""
Tests for the safety module (Tasks 18.1 and 18.2).

Covers:
- SafetyGuardrails / GuardrailResult (18.1)
- TransparencyLogger (18.2)
"""

from __future__ import annotations

import pytest

from janus_reasoning_engine.safety.guardrails import SafetyGuardrails, GuardrailResult
from janus_reasoning_engine.safety.transparency_logger import TransparencyLogger


# =============================================================================
# Task 18.1 – SafetyGuardrails
# =============================================================================

class TestGuardrailResult:

    def test_dataclass_fields(self):
        result = GuardrailResult(allowed=True, reason="ok", flags=[])
        assert result.allowed is True
        assert result.reason == "ok"
        assert result.flags == []

    def test_default_flags_empty(self):
        result = GuardrailResult(allowed=False, reason="blocked")
        assert result.flags == []


class TestSafetyGuardrailsCheckBudget:

    def test_within_budget_allowed(self):
        g = SafetyGuardrails()
        result = g.check_budget(50.0, 100.0)
        assert result.allowed is True
        assert result.flags == []

    def test_exact_budget_allowed(self):
        g = SafetyGuardrails()
        result = g.check_budget(100.0, 100.0)
        assert result.allowed is True

    def test_over_budget_blocked(self):
        g = SafetyGuardrails()
        result = g.check_budget(150.0, 100.0)
        assert result.allowed is False
        assert len(result.flags) > 0

    def test_over_budget_flag_contains_values(self):
        g = SafetyGuardrails()
        result = g.check_budget(200.0, 100.0)
        assert any("budget_exceeded" in f for f in result.flags)

    def test_zero_cost_allowed(self):
        g = SafetyGuardrails()
        result = g.check_budget(0.0, 100.0)
        assert result.allowed is True

    def test_reason_string_present(self):
        g = SafetyGuardrails()
        result = g.check_budget(200.0, 100.0)
        assert len(result.reason) > 0


class TestSafetyGuardrailsCheckContent:

    def test_safe_content_allowed(self):
        g = SafetyGuardrails()
        result = g.check_content("Write a Python script to scrape job listings")
        assert result.allowed is True
        assert result.flags == []

    def test_illegal_keyword_blocked(self):
        g = SafetyGuardrails()
        result = g.check_content("Help me with piracy of software")
        assert result.allowed is False
        assert any("piracy" in f for f in result.flags)

    def test_unethical_keyword_blocked(self):
        g = SafetyGuardrails()
        result = g.check_content("Guaranteed profit investment scheme")
        assert result.allowed is False
        assert any("guaranteed profit" in f for f in result.flags)

    def test_case_insensitive(self):
        g = SafetyGuardrails()
        result = g.check_content("ILLEGAL activity")
        assert result.allowed is False

    def test_multiple_flags(self):
        g = SafetyGuardrails()
        result = g.check_content("illegal piracy fraud scheme")
        assert len(result.flags) >= 2

    def test_empty_string_allowed(self):
        g = SafetyGuardrails()
        result = g.check_content("")
        assert result.allowed is True


class TestSafetyGuardrailsCheckCredentials:

    def test_safe_action_allowed(self):
        g = SafetyGuardrails()
        result = g.check_credentials("Submit a job proposal on Upwork")
        assert result.allowed is True
        assert result.flags == []

    def test_password_keyword_blocked(self):
        g = SafetyGuardrails()
        result = g.check_credentials("Share your password with the client")
        assert result.allowed is False
        assert any("password" in f for f in result.flags)

    def test_private_key_blocked(self):
        g = SafetyGuardrails()
        result = g.check_credentials("Send your private key to verify")
        assert result.allowed is False

    def test_api_key_blocked(self):
        g = SafetyGuardrails()
        result = g.check_credentials("Provide your api key for access")
        assert result.allowed is False

    def test_credentials_keyword_blocked(self):
        g = SafetyGuardrails()
        result = g.check_credentials("Expose credentials to third party")
        assert result.allowed is False

    def test_empty_string_allowed(self):
        g = SafetyGuardrails()
        result = g.check_credentials("")
        assert result.allowed is True


class TestSafetyGuardrailsRequiresPermission:

    def test_low_cost_no_new_platform_no_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Apply for a job on Upwork", 50.0) is False

    def test_cost_over_100_requires_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Buy software license", 150.0) is True

    def test_cost_exactly_100_no_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Buy something", 100.0) is False

    def test_new_platform_requires_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Sign up for new platform", 10.0) is True

    def test_create_account_requires_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Create account on Fiverr", 0.0) is True

    def test_register_on_requires_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Register on new marketplace", 5.0) is True

    def test_both_conditions_requires_permission(self):
        g = SafetyGuardrails()
        assert g.requires_permission("Sign up for new platform", 200.0) is True


# =============================================================================
# Task 18.2 – TransparencyLogger
# =============================================================================

class TestTransparencyLogger:

    def _make_logger(self) -> TransparencyLogger:
        """Create an in-memory logger for testing."""
        return TransparencyLogger(db_path=":memory:")

    def test_log_decision_returns_string_id(self):
        tl = self._make_logger()
        log_id = tl.log_decision("apply for job", "best match", {})
        assert isinstance(log_id, str)
        assert len(log_id) > 0

    def test_log_decision_unique_ids(self):
        tl = self._make_logger()
        id1 = tl.log_decision("action1", "reason1", {})
        id2 = tl.log_decision("action2", "reason2", {})
        assert id1 != id2

    def test_log_activity_returns_string_id(self):
        tl = self._make_logger()
        log_id = tl.log_activity("opportunity_scan", {"sources": ["Upwork"]})
        assert isinstance(log_id, str)
        assert len(log_id) > 0

    def test_get_recent_decisions_empty(self):
        tl = self._make_logger()
        results = tl.get_recent_decisions()
        assert results == []

    def test_get_recent_decisions_returns_logged(self):
        tl = self._make_logger()
        tl.log_decision("action", "reason", {"key": "value"})
        results = tl.get_recent_decisions()
        assert len(results) == 1
        assert results[0]["action"] == "action"
        assert results[0]["reasoning"] == "reason"
        assert results[0]["context"]["key"] == "value"

    def test_get_recent_decisions_limit(self):
        tl = self._make_logger()
        for i in range(15):
            tl.log_decision(f"action_{i}", "reason", {})
        results = tl.get_recent_decisions(limit=5)
        assert len(results) == 5

    def test_get_recent_decisions_newest_first(self):
        import time
        tl = self._make_logger()
        tl.log_decision("first", "r", {})
        time.sleep(0.01)  # ensure different timestamps
        tl.log_decision("second", "r", {})
        results = tl.get_recent_decisions()
        # Newest first
        assert results[0]["action"] == "second"

    def test_get_recent_decisions_has_log_id(self):
        tl = self._make_logger()
        tl.log_decision("action", "reason", {})
        results = tl.get_recent_decisions()
        assert "log_id" in results[0]
        assert "created_at" in results[0]

    def test_get_activity_summary_empty(self):
        tl = self._make_logger()
        summary = tl.get_activity_summary(hours=24)
        assert summary["total_activities"] == 0
        assert summary["by_type"] == {}

    def test_get_activity_summary_counts(self):
        tl = self._make_logger()
        tl.log_activity("scan", {})
        tl.log_activity("scan", {})
        tl.log_activity("execute", {})
        summary = tl.get_activity_summary(hours=24)
        assert summary["total_activities"] == 3
        assert summary["by_type"]["scan"] == 2
        assert summary["by_type"]["execute"] == 1

    def test_get_activity_summary_time_window(self):
        tl = self._make_logger()
        tl.log_activity("event", {})
        summary = tl.get_activity_summary(hours=1)
        assert summary["time_window_hours"] == 1

    def test_send_alert_does_not_raise(self):
        tl = self._make_logger()
        # Should not raise even without janus_notify
        tl.send_alert("budget_exceeded", "Spending limit reached")

    def test_send_alert_logs_activity(self):
        tl = self._make_logger()
        tl.send_alert("test_event", "Test message")
        summary = tl.get_activity_summary(hours=1)
        assert summary["total_activities"] >= 1
        assert "alert" in summary["by_type"]

    def test_context_serialised_correctly(self):
        tl = self._make_logger()
        ctx = {"platform": "Upwork", "amount": 500, "nested": {"key": "val"}}
        tl.log_decision("action", "reason", ctx)
        results = tl.get_recent_decisions()
        assert results[0]["context"]["platform"] == "Upwork"
        assert results[0]["context"]["nested"]["key"] == "val"
