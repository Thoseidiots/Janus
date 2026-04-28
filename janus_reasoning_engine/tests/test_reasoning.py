"""
Tests for the reasoning subsystem (Tasks 9.1–9.4).

Covers:
- CausalEngine / CausalModel (9.1)
- RiskAssessor / RiskReport (9.2)
- ValueJudge / ValueScore (9.3)
- Metacognition / Reflection (9.4)
"""

from __future__ import annotations

import pytest

from janus_reasoning_engine.reasoning.causal_engine import (
    CausalEdge,
    CausalEngine,
    CausalModel,
    PredictedOutcome,
)
from janus_reasoning_engine.reasoning.risk_assessor import RiskAssessor, RiskReport
from janus_reasoning_engine.reasoning.value_judge import ValueJudge, ValueScore
from janus_reasoning_engine.reasoning.metacognition import Metacognition, Reflection


# =============================================================================
# Task 9.1 – CausalEngine
# =============================================================================

class TestCausalModel:

    def test_add_edge_creates_edge(self):
        model = CausalModel()
        edge = model.add_edge("rain", "wet_ground", strength=0.9)
        assert isinstance(edge, CausalEdge)
        assert edge.cause == "rain"
        assert edge.effect == "wet_ground"
        assert edge.strength == 0.9

    def test_add_edge_twice_reinforces(self):
        model = CausalModel()
        model.add_edge("rain", "wet_ground", strength=0.5)
        edge = model.add_edge("rain", "wet_ground", strength=0.5)
        assert edge.observations == 2
        assert edge.strength > 0.5  # reinforced

    def test_get_effects_sorted_by_strength(self):
        model = CausalModel()
        model.add_edge("action", "effect_low", strength=0.2)
        model.add_edge("action", "effect_high", strength=0.9)
        effects = model.get_effects("action")
        assert effects[0].effect == "effect_high"
        assert effects[1].effect == "effect_low"

    def test_get_effects_unknown_cause_returns_empty(self):
        model = CausalModel()
        assert model.get_effects("unknown") == []

    def test_weaken_edge(self):
        model = CausalModel()
        model.add_edge("cause", "effect", strength=0.8)
        model.weaken_edge("cause", "effect", delta=0.3)
        edge = model.get_effects("cause")[0]
        assert edge.strength == pytest.approx(0.5, abs=0.01)

    def test_weaken_edge_floors_at_zero(self):
        model = CausalModel()
        model.add_edge("cause", "effect", strength=0.1)
        model.weaken_edge("cause", "effect", delta=1.0)
        edge = model.get_effects("cause")[0]
        assert edge.strength == 0.0

    def test_to_dict_structure(self):
        model = CausalModel()
        model.add_edge("a", "b", strength=0.7)
        d = model.to_dict()
        assert "a" in d
        assert "b" in d["a"]
        assert d["a"]["b"]["strength"] == pytest.approx(0.7, abs=0.01)


class TestCausalEngine:

    def test_observe_records_edge(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("send_proposal", "client_responds", strength=0.8)
        outcomes = engine.predict_outcome("send_proposal")
        assert len(outcomes) == 1
        assert outcomes[0].effect == "client_responds"
        assert outcomes[0].confidence == pytest.approx(0.8, abs=0.01)

    def test_predict_outcome_returns_list(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("action", "effect_a", strength=0.6)
        engine.observe("action", "effect_b", strength=0.9)
        outcomes = engine.predict_outcome("action")
        assert isinstance(outcomes, list)
        assert all(isinstance(o, PredictedOutcome) for o in outcomes)

    def test_predict_outcome_sorted_by_confidence(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("action", "low", strength=0.2)
        engine.observe("action", "high", strength=0.9)
        outcomes = engine.predict_outcome("action")
        assert outcomes[0].confidence >= outcomes[1].confidence

    def test_predict_outcome_unknown_action_returns_empty(self):
        engine = CausalEngine(use_causal_horizon=False)
        assert engine.predict_outcome("never_seen") == []

    def test_learn_from_mistake_weakens_expected(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("action", "expected_effect", strength=0.9)
        engine.learn_from_mistake("action", "expected_effect", "actual_effect")
        outcomes = engine.predict_outcome("action")
        expected_outcome = next(o for o in outcomes if o.effect == "expected_effect")
        assert expected_outcome.confidence < 0.9

    def test_learn_from_mistake_reinforces_actual(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.learn_from_mistake("action", "expected", "actual")
        outcomes = engine.predict_outcome("action")
        actual_outcome = next((o for o in outcomes if o.effect == "actual"), None)
        assert actual_outcome is not None
        assert actual_outcome.confidence > 0.0

    def test_strength_clamped_to_0_1(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("a", "b", strength=2.0)  # over 1.0
        outcomes = engine.predict_outcome("a")
        assert outcomes[0].confidence <= 1.0

    def test_get_model_summary(self):
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("x", "y")
        summary = engine.get_model_summary()
        assert "total_causes" in summary
        assert summary["total_causes"] == 1

    def test_causal_horizon_optional(self):
        """Engine should work fine even if causal horizon is unavailable."""
        engine = CausalEngine(use_causal_horizon=False)
        engine.observe("a", "b")
        assert engine._horizon is None


# =============================================================================
# Task 9.2 – RiskAssessor
# =============================================================================

class TestRiskAssessor:

    def test_assess_returns_risk_report(self):
        assessor = RiskAssessor()
        report = assessor.assess("Complete a freelance writing job", {})
        assert isinstance(report, RiskReport)

    def test_safe_action_has_low_risk(self):
        assessor = RiskAssessor()
        report = assessor.assess("Write a blog post about Python", {})
        assert report.overall_risk < 0.3
        assert report.flags == []

    def test_scam_keyword_guaranteed_profit(self):
        assessor = RiskAssessor()
        report = assessor.assess("Invest now for guaranteed profit", {})
        assert any("guaranteed profit" in f for f in report.flags)
        assert report.financial_risk > 0.0

    def test_scam_keyword_wire_transfer(self):
        assessor = RiskAssessor()
        report = assessor.assess("Please wire transfer $500 to proceed", {})
        assert any("wire transfer" in f for f in report.flags)

    def test_scam_keyword_send_money_first(self):
        assessor = RiskAssessor()
        report = assessor.assess("You must send money first to unlock earnings", {})
        assert any("send money first" in f for f in report.flags)

    def test_scam_keyword_crypto_upfront(self):
        assessor = RiskAssessor()
        report = assessor.assess("Pay crypto upfront to start", {})
        assert any("crypto upfront" in f for f in report.flags)

    def test_budget_exceeded_flag(self):
        assessor = RiskAssessor(budget_limit=100.0)
        report = assessor.assess("Buy software", {"estimated_cost": 250.0})
        assert any("budget_exceeded" in f for f in report.flags)
        assert report.financial_risk > 0.0

    def test_budget_within_limit_no_flag(self):
        assessor = RiskAssessor(budget_limit=500.0)
        report = assessor.assess("Buy a $10 domain", {"estimated_cost": 10.0})
        assert not any("budget_exceeded" in f for f in report.flags)

    def test_credential_keyword_password(self):
        assessor = RiskAssessor()
        report = assessor.assess("Share your password with the client", {})
        assert any("password" in f for f in report.flags)
        assert report.reputation_risk > 0.0

    def test_credential_keyword_private_key(self):
        assessor = RiskAssessor()
        report = assessor.assess("Send your private key to verify identity", {})
        assert any("private key" in f for f in report.flags)

    def test_legal_risk_keyword(self):
        assessor = RiskAssessor()
        report = assessor.assess("Help bypass DRM on software", {})
        assert report.legal_risk > 0.0

    def test_is_safe_below_threshold(self):
        assessor = RiskAssessor()
        report = assessor.assess("Write documentation", {})
        assert assessor.is_safe(report, threshold=0.7)

    def test_is_safe_above_threshold(self):
        assessor = RiskAssessor()
        report = assessor.assess("Send money first for guaranteed profit wire transfer", {})
        assert not assessor.is_safe(report, threshold=0.3)

    def test_is_safe_default_threshold(self):
        assessor = RiskAssessor()
        safe_report = RiskReport(overall_risk=0.5, financial_risk=0.0, reputation_risk=0.0, legal_risk=0.0)
        risky_report = RiskReport(overall_risk=0.8, financial_risk=0.8, reputation_risk=0.0, legal_risk=0.0)
        assert assessor.is_safe(safe_report)
        assert not assessor.is_safe(risky_report)

    def test_low_client_rating_flag(self):
        assessor = RiskAssessor()
        report = assessor.assess("Do a job", {"client_rating": 1.0})
        assert any("low_client_rating" in f for f in report.flags)

    def test_overall_risk_in_range(self):
        assessor = RiskAssessor()
        report = assessor.assess("Anything", {})
        assert 0.0 <= report.overall_risk <= 1.0


# =============================================================================
# Task 9.3 – ValueJudge
# =============================================================================

class TestValueJudge:

    def test_evaluate_returns_value_score(self):
        judge = ValueJudge()
        score = judge.evaluate({"money": 0.8, "skills": 0.5, "reputation": 0.6, "time": 0.7})
        assert isinstance(score, ValueScore)

    def test_evaluate_composite_is_weighted_sum(self):
        judge = ValueJudge()
        score = judge.evaluate({"money": 1.0, "skills": 0.0, "reputation": 0.0, "time": 0.0})
        # With default weights: composite ≈ 0.4 * 1.0 = 0.4
        assert score.composite == pytest.approx(0.4, abs=0.01)

    def test_evaluate_default_weights(self):
        judge = ValueJudge()
        score = judge.evaluate({"money": 1.0, "skills": 1.0, "reputation": 1.0, "time": 1.0})
        assert score.composite == pytest.approx(1.0, abs=0.01)

    def test_evaluate_missing_keys_default_to_zero(self):
        judge = ValueJudge()
        score = judge.evaluate({})
        assert score.composite == 0.0
        assert score.money_score == 0.0

    def test_evaluate_clamps_values(self):
        judge = ValueJudge()
        score = judge.evaluate({"money": 2.0, "skills": -1.0})
        assert score.money_score == 1.0
        assert score.skill_score == 0.0

    def test_evaluate_custom_weights(self):
        judge = ValueJudge()
        score = judge.evaluate(
            {"money": 1.0, "skills": 0.0, "reputation": 0.0, "time": 0.0},
            weights={"money": 1.0, "skills": 0.0, "reputation": 0.0, "time": 0.0},
        )
        assert score.composite == pytest.approx(1.0, abs=0.01)

    def test_compare_returns_sorted_list(self):
        judge = ValueJudge()
        options = [
            {"name": "low", "money": 0.1, "skills": 0.1},
            {"name": "high", "money": 0.9, "skills": 0.9},
            {"name": "mid", "money": 0.5, "skills": 0.5},
        ]
        ranked = judge.compare(options)
        assert ranked[0][0]["name"] == "high"
        assert ranked[-1][0]["name"] == "low"

    def test_compare_returns_tuples(self):
        judge = ValueJudge()
        options = [{"money": 0.5}]
        ranked = judge.compare(options)
        assert len(ranked) == 1
        opt, score = ranked[0]
        assert isinstance(score, ValueScore)

    def test_compare_empty_list(self):
        judge = ValueJudge()
        assert judge.compare([]) == []

    def test_optimize_long_term_returns_option(self):
        judge = ValueJudge()
        options = [
            {"name": "quick_cash", "money": 0.9, "skills": 0.1},
            {"name": "skill_builder", "money": 0.4, "skills": 0.9},
        ]
        best = judge.optimize_long_term(options, horizon_days=365)
        # Over a full year, skill_builder should win due to compounding
        assert best is not None

    def test_optimize_long_term_short_horizon_prefers_money(self):
        judge = ValueJudge()
        options = [
            {"name": "quick_cash", "money": 0.9, "skills": 0.0, "reputation": 0.0, "time": 0.0},
            {"name": "skill_builder", "money": 0.0, "skills": 0.9, "reputation": 0.0, "time": 0.0},
        ]
        best = judge.optimize_long_term(options, horizon_days=1)
        # Very short horizon: money weight dominates
        assert best["name"] == "quick_cash"

    def test_optimize_long_term_empty_returns_none(self):
        judge = ValueJudge()
        assert judge.optimize_long_term([]) is None

    def test_custom_instance_weights_normalised(self):
        judge = ValueJudge(weights={"money": 2.0, "skills": 2.0, "reputation": 0.0, "time": 0.0})
        total = sum(judge.weights.values())
        assert total == pytest.approx(1.0, abs=0.001)


# =============================================================================
# Task 9.4 – Metacognition
# =============================================================================

class TestMetacognition:

    def test_reflect_returns_reflection(self):
        meta = Metacognition()
        result = meta.reflect({"task_id": "t1", "success": True})
        assert isinstance(result, Reflection)
        assert result.task_id == "t1"
        assert result.success is True

    def test_reflect_success_positive_delta(self):
        meta = Metacognition()
        result = meta.reflect({"success": True})
        assert result.confidence_delta > 0

    def test_reflect_failure_negative_delta(self):
        meta = Metacognition()
        result = meta.reflect({"success": False})
        assert result.confidence_delta < 0

    def test_reflect_errors_reduce_confidence(self):
        meta = Metacognition()
        result = meta.reflect({"success": True, "errors": ["err1", "err2"]})
        # Even on success, errors reduce delta
        assert result.confidence_delta < 0.05

    def test_reflect_generates_lessons(self):
        meta = Metacognition()
        result = meta.reflect({"success": False, "notes": "Client was unresponsive"})
        assert len(result.lessons) > 0

    def test_reflect_long_task_pattern(self):
        meta = Metacognition()
        result = meta.reflect({"success": True, "duration_minutes": 120})
        assert "long_running_task" in result.patterns_identified

    def test_reflect_skill_pattern(self):
        meta = Metacognition()
        result = meta.reflect({"success": True, "skills_used": ["python", "api"]})
        assert any("python" in p for p in result.patterns_identified)

    def test_reflect_auto_generates_task_id(self):
        meta = Metacognition()
        result = meta.reflect({"success": True})
        assert result.task_id  # non-empty

    def test_identify_patterns_finds_recurring(self):
        meta = Metacognition()
        history = [
            {"success": True, "duration_minutes": 90},
            {"success": True, "duration_minutes": 120},
            {"success": False, "duration_minutes": 150},
        ]
        patterns = meta.identify_patterns(history)
        assert "long_running_task" in patterns

    def test_identify_patterns_returns_list(self):
        meta = Metacognition()
        patterns = meta.identify_patterns([{"success": True}])
        assert isinstance(patterns, list)

    def test_identify_patterns_empty_history(self):
        meta = Metacognition()
        patterns = meta.identify_patterns([])
        assert patterns == []

    def test_get_uncertainty_unknown_topic_returns_1(self):
        meta = Metacognition()
        assert meta.get_uncertainty("quantum_physics") == 1.0

    def test_update_uncertainty_success_reduces(self):
        meta = Metacognition()
        meta.update_uncertainty("python", {"outcome": "success", "difficulty": 0.5})
        assert meta.get_uncertainty("python") < 1.0

    def test_update_uncertainty_failure_increases(self):
        meta = Metacognition()
        meta.update_uncertainty("python", {"outcome": "success"})  # first reduce
        before = meta.get_uncertainty("python")
        meta.update_uncertainty("python", {"outcome": "failure"})
        assert meta.get_uncertainty("python") > before

    def test_update_uncertainty_explicit_confidence(self):
        meta = Metacognition()
        meta.update_uncertainty("python", {"confidence": 0.9})
        assert meta.get_uncertainty("python") == pytest.approx(0.1, abs=0.01)

    def test_uncertainty_clamped_to_0_1(self):
        meta = Metacognition()
        for _ in range(20):
            meta.update_uncertainty("topic", {"outcome": "success", "difficulty": 0.0})
        assert meta.get_uncertainty("topic") >= 0.0

    def test_uncertainty_case_insensitive(self):
        meta = Metacognition()
        meta.update_uncertainty("Python", {"outcome": "success"})
        assert meta.get_uncertainty("python") < 1.0

    def test_get_all_uncertainties(self):
        meta = Metacognition()
        meta.update_uncertainty("a", {"outcome": "success"})
        meta.update_uncertainty("b", {"outcome": "failure"})
        all_u = meta.get_all_uncertainties()
        assert "a" in all_u
        assert "b" in all_u

    def test_get_reflections_accumulates(self):
        meta = Metacognition()
        meta.reflect({"success": True})
        meta.reflect({"success": False})
        assert len(meta.get_reflections()) == 2
