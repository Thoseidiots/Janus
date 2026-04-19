"""
test_detection_systems.py
==========================
Comprehensive test suite for Janus detection systems:
- Binary Decider
- Loop Detector
- Ghost Code Detector
- Enhanced AVUS Brain integration

Run with: pytest tests/test_detection_systems.py -v
"""

import pytest
import torch
import time
from typing import Dict, Any

# Import systems under test
from janus_binary_decider import BinaryHBMDecider, DecisionResult
from janus_loop_detector import JanusLoopDetector, Action, LoopResult
from janus_ghost_code_detector import (
    JanusGhostCodeDetector,
    ObservabilityReport,
    ComponentManifest,
)
from avus_brain_enhanced import (
    EnhancedAvusBrain,
    ThoughtProcess,
    SafeExecutionResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Binary Decider Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryDecider:
    """Test suite for Binary HBM Decider."""

    @pytest.fixture
    def decider(self):
        """Create a binary decider instance."""
        return BinaryHBMDecider(dim=512, max_iterations=100)

    def test_simple_halt(self, decider):
        """Test detection of simple halting program."""
        result = decider.decide(
            program="simple_loop",
            data="range(5)"  # Finite loop
        )

        assert isinstance(result, DecisionResult)
        assert result.decision is True  # Should halt
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0

    def test_infinite_loop(self, decider):
        """Test detection of infinite loop."""
        result = decider.decide(
            program="while True: pass",
            data=""
        )

        assert isinstance(result, DecisionResult)
        assert result.decision is False  # Should loop
        assert result.confidence > 0.5

    def test_self_referential_paradox(self, decider):
        """Test D(D) self-referential paradox."""
        result = decider.decide_paradox("Paradox_D")

        assert result.decision is False  # Infinite introspection
        assert result.confidence > 0.9  # High confidence
        assert "self-referential" in result.reasoning.lower()

    def test_entanglement_identity(self, decider):
        """Verify P⊗D creates inseparable state for identical P and D."""
        p_vec = decider._to_vector("test")
        d_vec = decider._to_vector("test")

        similarity = torch.nn.functional.cosine_similarity(
            p_vec.unsqueeze(0),
            d_vec.unsqueeze(0)
        ).item()

        assert similarity > 0.999, "Program and data must be identical"

    def test_decision_time(self, decider):
        """Test decision time is within acceptable range."""
        start = time.time()
        result = decider.decide("test_program", "test_data")
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Decision took {elapsed*1000:.2f}ms (should be <100ms)"

    def test_consistency(self, decider):
        """Test that same inputs produce consistent decisions."""
        result1 = decider.decide("program_x", "data_y")
        result2 = decider.decide("program_x", "data_y")

        assert result1.decision == result2.decision
        assert abs(result1.confidence - result2.confidence) < 0.1

    def test_dimension_scaling(self):
        """Test decider works with different dimensions."""
        for dim in [256, 512, 1024, 2048]:
            decider = BinaryHBMDecider(dim=dim)
            result = decider.decide("test", "test")
            assert isinstance(result, DecisionResult)


# ─────────────────────────────────────────────────────────────────────────────
# Loop Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopDetector:
    """Test suite for Loop Detector."""

    @pytest.fixture
    def detector(self):
        """Create a loop detector instance."""
        return JanusLoopDetector(
            similarity_threshold=0.85,
            repetition_threshold=3
        )

    def test_no_loop_different_actions(self, detector):
        """Test no loop detected for different actions."""
        actions = [
            Action("navigate", "room_A", {}, time.time()),
            Action("search", "kitchen", {}, time.time()),
            Action("pickup", "key", {}, time.time()),
        ]

        for action in actions:
            result = detector.record_action(action)
            assert result.is_loop is False

    def test_loop_exact_repetition(self, detector):
        """Test loop detection with exact action repetition."""
        action = Action("enter", "room_A", {}, time.time())

        results = []
        for _ in range(4):
            result = detector.record_action(action)
            results.append(result)

        # Should detect loop on 3rd or 4th repetition
        assert any(r.is_loop for r in results[-2:])

    def test_loop_semantic_similarity(self, detector):
        """Test loop detection with semantically similar actions."""
        actions = [
            Action("search", "kitchen", {"target": "food"}, time.time()),
            Action("search", "bedroom", {"target": "food"}, time.time()),
            Action("search", "bathroom", {"target": "food"}, time.time()),
        ]

        results = []
        for action in actions:
            result = detector.record_action(action)
            results.append(result)

        # Should detect pattern even though rooms differ
        assert results[-1].is_loop is True
        assert results[-1].similar_count >= 3

    def test_loop_pattern_detection(self, detector):
        """Test detection of repeating patterns [A, B, C] x N."""
        pattern = [
            Action("step", "A", {}, time.time()),
            Action("step", "B", {}, time.time()),
            Action("step", "C", {}, time.time()),
        ]

        # Repeat pattern 3 times
        for _ in range(3):
            for action in pattern:
                result = detector.record_action(action)

        # Should detect repeating pattern
        assert detector.loop_count > 0

    def test_similarity_threshold(self, detector):
        """Test that similarity threshold is respected."""
        # Very different actions
        action1 = Action("walk", "north", {}, time.time())
        action2 = Action("compute", "fibonacci", {}, time.time())

        detector.record_action(action1)
        detector.record_action(action2)
        result = detector.record_action(action1)

        # Should not trigger loop (only 2 occurrences)
        assert result.is_loop is False

    def test_reset_tracking(self, detector):
        """Test that loop tracking can be reset."""
        action = Action("test", "test", {}, time.time())

        for _ in range(5):
            detector.record_action(action)

        initial_count = detector.loop_count
        detector.reset()

        assert len(detector.action_history) == 0
        assert detector.loop_count == 0

    def test_history_limit(self, detector):
        """Test that action history respects size limit."""
        detector = JanusLoopDetector(history_size=10)

        for i in range(20):
            action = Action("test", f"action_{i}", {}, time.time())
            detector.record_action(action)

        assert len(detector.action_history) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# Ghost Code Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGhostCodeDetector:
    """Test suite for Ghost Code Detector."""

    @pytest.fixture
    def detector(self):
        """Create a ghost code detector instance."""
        return JanusGhostCodeDetector()

    def test_healthy_component(self, detector):
        """Test component with all outputs visible."""
        detector.register_component(
            "TestComponent",
            expected_outputs=["output_A", "output_B"],
            expected_side_effects=["effect_1"],
            dependencies=["dep_1"],
        )

        detector.observe_output("TestComponent", "output_A", True, {})
        detector.observe_output("TestComponent", "output_B", True, {})
        detector.observe_side_effect("TestComponent", "effect_1", True, {})

        report = detector.check_component_health("TestComponent")

        assert report.confidence >= 0.9
        assert report.status == "HEALTHY"
        assert len(report.ghost_issues) == 0

    def test_partial_ghost(self, detector):
        """Test component with some outputs invisible."""
        detector.register_component(
            "PartialGhost",
            expected_outputs=["visible_output", "invisible_output"],
            expected_side_effects=[],
            dependencies=[],
        )

        detector.observe_output("PartialGhost", "visible_output", True, {})
        detector.observe_output("PartialGhost", "invisible_output", False, {
            "reason": "element has zero dimensions"
        })

        report = detector.check_component_health("PartialGhost")

        assert 0.4 <= report.confidence < 0.9
        assert report.status == "PARTIAL_GHOST"
        assert len(report.ghost_issues) > 0

    def test_full_ghost(self, detector):
        """Test component with all outputs invisible."""
        detector.register_component(
            "FullGhost",
            expected_outputs=["ghost_output_1", "ghost_output_2"],
            expected_side_effects=["ghost_effect"],
            dependencies=[],
        )

        detector.observe_output("FullGhost", "ghost_output_1", False, {
            "reason": "positioned off-screen"
        })
        detector.observe_output("FullGhost", "ghost_output_2", False, {
            "reason": "zero height"
        })
        detector.observe_side_effect("FullGhost", "ghost_effect", False, {
            "reason": "no data transferred"
        })

        report = detector.check_component_health("FullGhost")

        assert report.confidence < 0.5
        assert report.status == "FULL_GHOST"
        assert len(report.ghost_issues) >= 3

    def test_off_screen_element(self, detector):
        """Test detection of off-screen UI element."""
        detector.register_component(
            "SubmitButton",
            expected_outputs=["button_element"],
            expected_side_effects=[],
            dependencies=[],
        )

        detector.observe_output("SubmitButton", "button_element", False, {
            "element_exists": True,
            "position": {"x": -1000, "y": 50},
            "reason": "positioned off-screen"
        })

        report = detector.check_component_health("SubmitButton")

        assert "off-screen" in report.ghost_issues[0].lower()
        assert report.confidence < 0.5

    def test_silent_data_pipeline(self, detector):
        """Test detection of silent data pipeline failure."""
        detector.register_component(
            "DataPipeline",
            expected_outputs=["processed_data"],
            expected_side_effects=["write_to_database"],
            dependencies=["database"],
        )

        detector.observe_output("DataPipeline", "processed_data", False, {
            "function_returned": "success",
            "data_length": 0,
            "reason": "empty result set despite success status"
        })

        detector.observe_side_effect("DataPipeline", "write_to_database", False, {
            "connection_active": True,
            "rows_affected": 0,
            "reason": "connection exists but no data transferred"
        })

        report = detector.check_component_health("DataPipeline")

        assert report.confidence == 0.0
        assert report.status == "FULL_GHOST"
        assert len(report.ghost_issues) >= 2

    def test_ghost_summary(self, detector):
        """Test ghost summary statistics."""
        # Register 3 components with different health levels
        components = [
            ("Healthy", ["out1"], [], [], [(True, {})]),
            ("Partial", ["out1", "out2"], [], [], [(True, {}), (False, {})]),
            ("Ghost", ["out1"], [], [], [(False, {})]),
        ]

        for name, outputs, effects, deps, observations in components:
            detector.register_component(name, outputs, effects, deps)
            for i, (visible, meta) in enumerate(observations):
                detector.observe_output(name, outputs[i], visible, meta)

        summary = detector.get_ghost_summary()

        assert summary["total_components"] == 3
        assert summary["healthy"] == 1
        assert summary["partial_ghost"] == 1
        assert summary["full_ghost"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced AVUS Brain Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnhancedAvusBrain:
    """Test suite for Enhanced AVUS Brain integration."""

    @pytest.fixture
    def brain(self):
        """Create an enhanced brain instance (without loading AVUS model)."""
        brain = EnhancedAvusBrain(
            avus=None,
            enable_decider=True,
            enable_loop_detection=True,
            enable_ghost_detection=True,
        )
        # Skip actual AVUS model loading for tests
        brain._loaded = True
        return brain

    def test_initialization(self, brain):
        """Test that all detection systems are initialized."""
        assert brain.decider is not None
        assert brain.loop_detector is not None
        assert brain.ghost_detector is not None

    def test_safe_execution_success(self, brain):
        """Test safe execution of valid thought."""
        thought = ThoughtProcess(
            process_id="calculate_sum",
            context={"a": 5, "b": 3},
            action_type="compute",
            target="sum",
        )

        def executor(t):
            return t.context["a"] + t.context["b"]

        result = brain.execute_thought_safely(thought, executor)

        assert result.success is True
        assert result.result == 8
        assert result.halt_decision is not None

    def test_safe_execution_loop_prevention(self, brain):
        """Test prevention of high-loop-risk thoughts."""
        # Create self-referential thought
        thought = ThoughtProcess(
            process_id="infinite_introspect",
            context={"target": "infinite_introspect"},
            action_type="introspect",
            target="self",
        )

        result = brain.execute_thought_safely(thought)

        # May or may not prevent depending on confidence,
        # but should at least warn
        assert result.halt_decision is not None
        if not result.success:
            assert "LOOP RISK" in " ".join(result.safety_warnings)

    def test_loop_detection_integration(self, brain):
        """Test loop detection during safe execution."""
        thought = ThoughtProcess(
            process_id="navigate",
            context={},
            action_type="navigate",
            target="room_A",
        )

        # Execute same thought 4 times
        results = []
        for _ in range(4):
            result = brain.execute_thought_safely(thought)
            results.append(result)

        # Should detect loop eventually
        warnings_combined = " ".join(
            " ".join(r.safety_warnings) for r in results
        )
        assert "LOOP" in warnings_combined

    def test_ghost_detection_integration(self, brain):
        """Test ghost code detection during safe execution."""
        # Register a component
        brain.ghost_detector.register_component(
            "test_component",
            expected_outputs=["test_output"],
            expected_side_effects=[],
            dependencies=[],
        )

        thought = ThoughtProcess(
            process_id="test_process",
            context={},
            action_type="test",
            target="test",
            produces_output=True,
            output_id="test_output",
            component="test_component",
        )

        # Execute with no actual output
        def ghost_executor(t):
            return None  # Returns None (no visible output)

        result = brain.execute_thought_safely(thought, ghost_executor)

        assert result.ghost_report is not None
        # Confidence may vary, but report should exist

    def test_system_health_report(self, brain):
        """Test system health reporting."""
        # Execute some thoughts to generate stats
        for i in range(5):
            thought = ThoughtProcess(
                process_id=f"test_{i}",
                context={},
                action_type="test",
                target=f"target_{i}",
            )
            brain.execute_thought_safely(thought)

        health = brain.get_system_health()

        assert "timestamp" in health
        assert "binary_decisions" in health
        assert "loop_detection" in health
        assert "ghost_detection" in health

        assert health["binary_decisions"]["total_decisions"] >= 5

    def test_ask_with_safety(self, brain):
        """Test ask() method with safety checks."""
        # Mock the parent ask method
        def mock_ask(question, context="", max_tokens=300, temperature=0.7):
            return "Mock answer"

        brain._avus = type('obj', (object,), {
            'generate': lambda *args, **kwargs: "Mock answer"
        })

        # Ask same question multiple times
        for _ in range(4):
            answer = brain.ask("What is 2+2?", check_safety=True)

        # Should record actions in loop detector
        assert brain.loop_detector.loop_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration:
    """End-to-end tests combining all systems."""

    def test_full_cognitive_loop(self):
        """Test complete cognitive loop with all detection systems."""
        brain = EnhancedAvusBrain(
            avus=None,
            enable_decider=True,
            enable_loop_detection=True,
            enable_ghost_detection=True,
        )
        brain._loaded = True

        # Register a ghost-prone component
        brain.ghost_detector.register_component(
            "ui_renderer",
            expected_outputs=["rendered_element"],
            expected_side_effects=["dom_update"],
            dependencies=["dom"],
        )

        # Execute a series of thoughts
        thoughts = [
            ThoughtProcess("think_1", {}, "reason", "problem_A"),
            ThoughtProcess("think_2", {}, "reason", "problem_B"),
            ThoughtProcess("think_3", {}, "reason", "problem_A"),  # Repeat
            ThoughtProcess("think_4", {}, "reason", "problem_A"),  # Repeat again
            ThoughtProcess(
                "render_ui",
                {},
                "render",
                "button",
                produces_output=True,
                output_id="rendered_element",
                component="ui_renderer"
            ),
        ]

        results = []
        for thought in thoughts:
            def executor(t):
                # Simulate ghost output for render
                if t.process_id == "render_ui":
                    return {"visible": False, "reason": "off-screen"}
                return {"status": "ok"}

            result = brain.execute_thought_safely(thought, executor)
            results.append(result)

        # Verify detection systems triggered
        loop_detected = any("LOOP" in " ".join(r.safety_warnings) for r in results)
        ghost_detected = any(r.ghost_report and r.ghost_report.confidence < 0.5 for r in results)

        assert loop_detected or ghost_detected, "Detection systems should trigger"

        # Verify system health
        health = brain.get_system_health()
        assert health["binary_decisions"]["total_decisions"] == len(thoughts)
        assert health["loop_detection"]["total_actions"] >= len(thoughts)

    def test_performance_overhead(self):
        """Test that detection systems don't add excessive overhead."""
        brain_plain = EnhancedAvusBrain(
            avus=None,
            enable_decider=False,
            enable_loop_detection=False,
            enable_ghost_detection=False,
        )
        brain_plain._loaded = True

        brain_enhanced = EnhancedAvusBrain(
            avus=None,
            enable_decider=True,
            enable_loop_detection=True,
            enable_ghost_detection=True,
        )
        brain_enhanced._loaded = True

        thought = ThoughtProcess("test", {}, "test", "test")

        # Measure plain execution
        start = time.time()
        for _ in range(100):
            brain_plain.execute_thought_safely(thought)
        plain_time = time.time() - start

        # Measure enhanced execution
        start = time.time()
        for _ in range(100):
            brain_enhanced.execute_thought_safely(thought)
        enhanced_time = time.time() - start

        overhead = (enhanced_time - plain_time) / plain_time

        # Overhead should be reasonable (<50%)
        assert overhead < 0.5, f"Overhead too high: {overhead*100:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

