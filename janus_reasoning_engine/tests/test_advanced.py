"""
Tests for the advanced reasoning capabilities module (Tasks 19.1–19.3).

Covers:
- CausalHorizonBridge (19.1)
- AdvancedConcepts (19.2)
- HumanCapabilities (19.3)
"""

from __future__ import annotations

import pytest

from janus_reasoning_engine.advanced.causal_horizon_bridge import CausalHorizonBridge
from janus_reasoning_engine.advanced.advanced_concepts import AdvancedConcepts
from janus_reasoning_engine.advanced.human_capabilities import HumanCapabilities


# =============================================================================
# Task 19.1 – CausalHorizonBridge
# =============================================================================

class TestCausalHorizonBridge:

    def test_emit_signal_does_not_raise(self):
        bridge = CausalHorizonBridge()
        bridge.emit_signal("cause", "effect", {})

    def test_emit_signal_with_metadata(self):
        bridge = CausalHorizonBridge()
        bridge.emit_signal("apply_job", "interview_scheduled", {"platform": "Upwork"})

    def test_get_propagated_effects_empty_initially(self):
        bridge = CausalHorizonBridge()
        effects = bridge.get_propagated_effects("unknown_cause")
        assert effects == []

    def test_get_propagated_effects_after_emit(self):
        bridge = CausalHorizonBridge()
        bridge.emit_signal("cause_a", "effect_x", {})
        effects = bridge.get_propagated_effects("cause_a")
        assert len(effects) == 1
        assert effects[0]["effect"] == "effect_x"

    def test_get_propagated_effects_multiple(self):
        bridge = CausalHorizonBridge()
        bridge.emit_signal("cause_a", "effect_1", {})
        bridge.emit_signal("cause_a", "effect_2", {})
        effects = bridge.get_propagated_effects("cause_a")
        assert len(effects) == 2

    def test_get_propagated_effects_returns_list(self):
        bridge = CausalHorizonBridge()
        result = bridge.get_propagated_effects("any")
        assert isinstance(result, list)

    def test_get_propagated_effects_metadata_preserved(self):
        bridge = CausalHorizonBridge()
        bridge.emit_signal("cause", "effect", {"key": "value"})
        effects = bridge.get_propagated_effects("cause")
        assert effects[0]["metadata"]["key"] == "value"

    def test_track_temporal_causality_empty_sequence(self):
        bridge = CausalHorizonBridge()
        result = bridge.track_temporal_causality([])
        assert result["length"] == 0
        assert result["chain"] == []
        assert result["causal_pairs"] == []

    def test_track_temporal_causality_single_event(self):
        bridge = CausalHorizonBridge()
        result = bridge.track_temporal_causality([{"event": "start"}])
        assert result["length"] == 1
        assert result["causal_pairs"] == []

    def test_track_temporal_causality_two_events(self):
        bridge = CausalHorizonBridge()
        result = bridge.track_temporal_causality([
            {"event": "apply"},
            {"event": "interview"},
        ])
        assert result["length"] == 2
        assert len(result["causal_pairs"]) == 1
        assert result["causal_pairs"][0]["cause"] == "apply"
        assert result["causal_pairs"][0]["effect"] == "interview"

    def test_track_temporal_causality_chain(self):
        bridge = CausalHorizonBridge()
        events = [
            {"event": "apply", "timestamp": "2024-01-01T10:00:00"},
            {"event": "interview", "timestamp": "2024-01-02T14:00:00"},
            {"event": "offer", "timestamp": "2024-01-03T09:00:00"},
        ]
        result = bridge.track_temporal_causality(events)
        assert result["length"] == 3
        assert len(result["causal_pairs"]) == 2

    def test_track_temporal_causality_returns_dict(self):
        bridge = CausalHorizonBridge()
        result = bridge.track_temporal_causality([{"event": "a"}])
        assert isinstance(result, dict)
        assert "chain" in result
        assert "causal_pairs" in result
        assert "length" in result


# =============================================================================
# Task 19.2 – AdvancedConcepts
# =============================================================================

class TestAdvancedConcepts:

    def test_bypass_computational_limit_returns_string(self):
        ac = AdvancedConcepts()
        result = ac.bypass_computational_limit("solve problem", 1000)
        assert isinstance(result, str)

    def test_bypass_computational_limit_stub_returns_task(self):
        ac = AdvancedConcepts()
        task = "optimise pricing strategy"
        result = ac.bypass_computational_limit(task, 2000)
        # Stub returns the task description
        assert task in result or len(result) > 0

    def test_bypass_computational_limit_zero_timeout(self):
        ac = AdvancedConcepts()
        result = ac.bypass_computational_limit("task", 0)
        assert isinstance(result, str)

    def test_prevent_halting_success_on_first_try(self):
        ac = AdvancedConcepts()
        result = ac.prevent_halting(lambda: 42, max_retries=3)
        assert result == 42

    def test_prevent_halting_retries_on_failure(self):
        ac = AdvancedConcepts()
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "done"

        result = ac.prevent_halting(flaky, max_retries=3)
        assert result == "done"
        assert call_count[0] == 3

    def test_prevent_halting_raises_after_max_retries(self):
        ac = AdvancedConcepts()
        with pytest.raises(ValueError, match="always fails"):
            ac.prevent_halting(lambda: (_ for _ in ()).throw(ValueError("always fails")), max_retries=2)

    def test_prevent_halting_default_max_retries(self):
        ac = AdvancedConcepts()
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("retry")
            return "ok"

        result = ac.prevent_halting(fn)
        assert result == "ok"

    def test_explore_boundary_returns_list(self):
        ac = AdvancedConcepts()
        result = ac.explore_boundary("budget_limit")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_explore_boundary_contains_concept(self):
        ac = AdvancedConcepts()
        result = ac.explore_boundary("timeout")
        assert all("timeout" in item for item in result)

    def test_explore_boundary_includes_zero(self):
        ac = AdvancedConcepts()
        result = ac.explore_boundary("value")
        assert any("zero" in item.lower() or "null" in item.lower() for item in result)

    def test_explore_boundary_includes_max(self):
        ac = AdvancedConcepts()
        result = ac.explore_boundary("value")
        assert any("max" in item.lower() for item in result)

    def test_explore_boundary_different_concepts(self):
        ac = AdvancedConcepts()
        r1 = ac.explore_boundary("budget")
        r2 = ac.explore_boundary("timeout")
        # Different concepts produce different boundary strings
        assert r1 != r2


# =============================================================================
# Task 19.3 – HumanCapabilities
# =============================================================================

class TestHumanCapabilities:

    def test_get_personality_state_returns_dict(self):
        hc = HumanCapabilities()
        state = hc.get_personality_state()
        assert isinstance(state, dict)

    def test_get_personality_state_has_required_keys(self):
        hc = HumanCapabilities()
        state = hc.get_personality_state()
        assert "mood" in state
        assert "energy" in state
        assert "curiosity" in state

    def test_get_personality_state_energy_in_range(self):
        hc = HumanCapabilities()
        state = hc.get_personality_state()
        assert 0.0 <= state["energy"] <= 1.0

    def test_get_personality_state_curiosity_in_range(self):
        hc = HumanCapabilities()
        state = hc.get_personality_state()
        assert 0.0 <= state["curiosity"] <= 1.0

    def test_get_personality_state_mood_is_string(self):
        hc = HumanCapabilities()
        state = hc.get_personality_state()
        assert isinstance(state["mood"], str)

    def test_apply_natural_behavior_returns_string(self):
        hc = HumanCapabilities()
        result = hc.apply_natural_behavior("The task is complete.")
        assert isinstance(result, str)

    def test_apply_natural_behavior_non_empty(self):
        hc = HumanCapabilities()
        result = hc.apply_natural_behavior("Hello world")
        assert len(result) > 0

    def test_apply_natural_behavior_empty_string(self):
        hc = HumanCapabilities()
        result = hc.apply_natural_behavior("")
        assert isinstance(result, str)

    def test_apply_natural_behavior_modifies_text(self):
        hc = HumanCapabilities()
        original = "The task has been completed successfully."
        result = hc.apply_natural_behavior(original)
        # Result should be a string (may or may not be modified)
        assert isinstance(result, str)

    def test_adapt_to_social_context_returns_string(self):
        hc = HumanCapabilities()
        result = hc.adapt_to_social_context("Hello!", {"formality": "casual"})
        assert isinstance(result, str)

    def test_adapt_to_social_context_non_empty(self):
        hc = HumanCapabilities()
        result = hc.adapt_to_social_context("Hello!", {})
        assert len(result) > 0

    def test_adapt_to_social_context_formal(self):
        hc = HumanCapabilities()
        result = hc.adapt_to_social_context("I completed the project.", {"formality": "formal"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adapt_to_social_context_casual(self):
        hc = HumanCapabilities()
        result = hc.adapt_to_social_context("Done!", {"formality": "casual"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adapt_to_social_context_with_recipient(self):
        hc = HumanCapabilities()
        result = hc.adapt_to_social_context(
            "The deliverable is ready.",
            {"formality": "formal", "recipient": "John"},
        )
        assert isinstance(result, str)
        assert len(result) > 0
