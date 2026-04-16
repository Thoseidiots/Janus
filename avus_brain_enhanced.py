"""
avus_brain_enhanced.py
======================
Enhanced AVUS Brain with integrated detection systems:
- Binary Decider for halt/loop prediction
- Loop Detector for repetitive action recognition
- Ghost Code Detector for silent failure detection

This is the production-ready cognitive loop that combines AVUS inference
with Janus's advanced detection capabilities.

Usage:
    from avus_brain_enhanced import get_enhanced_brain

    brain = get_enhanced_brain()

    # Standard AVUS operations
    answer = brain.ask("What is the capital of France?")

    # Enhanced operations with detection
    result = brain.execute_thought_safely(thought_process)
    loop_status = brain.check_for_loops()
    ghost_report = brain.scan_for_ghost_code()
"""

from __future__ import annotations

import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from avus_brain import AvusBrain, get_brain
from janus_binary_decider import BinaryHBMDecider, DecisionResult
from janus_loop_detector import JanusLoopDetector, Action, LoopResult
from janus_ghost_code_detector import (
    JanusGhostCodeDetector,
    ObservabilityReport,
    ComponentManifest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Brain with Detection Systems
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtProcess:
    """Represents a cognitive operation to be executed."""
    process_id: str
    context: Dict[str, Any]
    action_type: str
    target: str
    produces_output: bool = False
    output_id: Optional[str] = None
    component: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafeExecutionResult:
    """Result from safe thought execution with all detection checks."""
    success: bool
    result: Any
    halt_decision: Optional[DecisionResult] = None
    loop_status: Optional[LoopResult] = None
    ghost_report: Optional[ObservabilityReport] = None
    execution_time: float = 0.0
    safety_warnings: List[str] = field(default_factory=list)


class EnhancedAvusBrain(AvusBrain):
    """
    AVUS Brain enhanced with Janus detection systems.

    Adds three layers of intelligence:
    1. Binary Decider - Predicts if operations will halt or loop
    2. Loop Detector - Recognizes when repeating actions 3+ times
    3. Ghost Code Detector - Finds code that runs but produces no output
    """

    def __init__(
        self,
        avus: Optional[Any] = None,
        enable_decider: bool = True,
        enable_loop_detection: bool = True,
        enable_ghost_detection: bool = True,
        decider_dim: int = 2048,
        similarity_threshold: float = 0.85,
        repetition_threshold: int = 3,
    ):
        """
        Initialize enhanced brain with detection systems.

        Args:
            avus: Optional AvusInference instance
            enable_decider: Enable binary halt/loop prediction
            enable_loop_detection: Enable action loop detection
            enable_ghost_detection: Enable silent failure detection
            decider_dim: Holographic dimension for decider
            similarity_threshold: Threshold for loop detection similarity
            repetition_threshold: Number of repetitions to trigger loop alert
        """
        super().__init__(avus)

        # Initialize detection systems
        self.enable_decider = enable_decider
        self.enable_loop_detection = enable_loop_detection
        self.enable_ghost_detection = enable_ghost_detection

        if enable_decider:
            self.decider = BinaryHBMDecider(dim=decider_dim)
            print(f"[EnhancedBrain] Binary Decider enabled (dim={decider_dim})")
        else:
            self.decider = None

        if enable_loop_detection:
            self.loop_detector = JanusLoopDetector(
                similarity_threshold=similarity_threshold,
                repetition_threshold=repetition_threshold,
            )
            print(f"[EnhancedBrain] Loop Detector enabled (threshold={repetition_threshold})")
        else:
            self.loop_detector = None

        if enable_ghost_detection:
            self.ghost_detector = JanusGhostCodeDetector()
            print(f"[EnhancedBrain] Ghost Code Detector enabled")
            self._register_core_components()
        else:
            self.ghost_detector = None

    def _register_core_components(self):
        """Register AVUS core components for ghost detection."""
        if not self.ghost_detector:
            return

        # Register perception system
        self.ghost_detector.register_component(
            "avus_inference",
            expected_outputs=["generated_text", "logits"],
            expected_side_effects=["token_generation", "cache_update"],
            dependencies=["avus_model", "tokenizer"],
        )

        # Register memory system
        self.ghost_detector.register_component(
            "holographic_memory",
            expected_outputs=["memory_field", "retrieved_data"],
            expected_side_effects=["write_to_field", "field_normalization"],
            dependencies=["pytorch"],
        )

        # Register cognitive loop
        self.ghost_detector.register_component(
            "cognitive_loop",
            expected_outputs=["thought_result", "action_plan"],
            expected_side_effects=["execute_action", "update_state"],
            dependencies=["avus_inference", "holographic_memory"],
        )

    # ── Core Detection Methods ────────────────────────────────────────────────

    def predict_halt(
        self,
        process_id: str,
        context: Dict[str, Any],
    ) -> DecisionResult:
        """
        Predict if a thought process will halt or loop infinitely.

        Args:
            process_id: Unique identifier for the process
            context: Context data for the process

        Returns:
            DecisionResult with halt/loop prediction and confidence
        """
        if not self.decider:
            return DecisionResult(
                decision=True,  # Assume halts if decider disabled
                confidence=0.5,
                reasoning="Binary decider disabled",
            )

        return self.decider.decide(
            program=process_id,
            data=str(context),
        )

    def record_action(
        self,
        action_type: str,
        target: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopResult:
        """
        Record an action and check for loop patterns.

        Args:
            action_type: Type of action (e.g., "navigate", "search", "execute")
            target: Target of the action
            context: Additional context

        Returns:
            LoopResult indicating if loop detected
        """
        if not self.loop_detector:
            return LoopResult(
                is_loop=False,
                similar_count=0,
                pattern_detected=False,
            )

        action = Action(
            action_type=action_type,
            target=target,
            context=context or {},
            timestamp=time.time(),
        )

        return self.loop_detector.record_action(action)

    def observe_output(
        self,
        component_name: str,
        output_name: str,
        is_visible: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record observation of expected output.

        Args:
            component_name: Name of component producing output
            output_name: Name of the output
            is_visible: Whether output actually manifested
            metadata: Additional metadata about the observation
        """
        if not self.ghost_detector:
            return

        self.ghost_detector.observe_output(
            component_name=component_name,
            output_name=output_name,
            is_visible=is_visible,
            metadata=metadata or {},
        )

    def check_component_health(
        self,
        component_name: str,
    ) -> Optional[ObservabilityReport]:
        """
        Check health of a registered component.

        Args:
            component_name: Name of component to check

        Returns:
            ObservabilityReport or None if ghost detection disabled
        """
        if not self.ghost_detector:
            return None

        return self.ghost_detector.check_component_health(component_name)

    # ── High-Level Safe Execution ─────────────────────────────────────────────

    def execute_thought_safely(
        self,
        thought: ThoughtProcess,
        executor: Optional[callable] = None,
    ) -> SafeExecutionResult:
        """
        Execute a thought process with all detection systems active.

        This is the main entry point for safe cognitive operations.

        Args:
            thought: ThoughtProcess to execute
            executor: Optional callable to execute the thought

        Returns:
            SafeExecutionResult with execution outcome and detection reports
        """
        start_time = time.time()
        warnings = []

        # Stage 1: Binary Decision Check
        halt_decision = None
        if self.enable_decider:
            halt_decision = self.predict_halt(
                thought.process_id,
                thought.context,
            )

            if not halt_decision.decision and halt_decision.confidence > 0.8:
                return SafeExecutionResult(
                    success=False,
                    result=None,
                    halt_decision=halt_decision,
                    execution_time=time.time() - start_time,
                    safety_warnings=[
                        f"⚠️ HIGH LOOP RISK: {halt_decision.reasoning}",
                        f"Confidence: {halt_decision.confidence:.2%}",
                        "Execution prevented",
                    ],
                )
            elif not halt_decision.decision:
                warnings.append(
                    f"⚠️ Loop risk detected (confidence: {halt_decision.confidence:.2%})"
                )

        # Stage 2: Loop Detection
        loop_status = None
        if self.enable_loop_detection:
            loop_status = self.record_action(
                thought.action_type,
                thought.target,
                thought.context,
            )

            if loop_status.is_loop:
                warnings.append(
                    f"🔄 LOOP DETECTED: {loop_status.recommendation}"
                )
                if loop_status.pattern_detected:
                    warnings.append(
                        f"Pattern: {loop_status.loop_pattern} (x{loop_status.similar_count})"
                    )

        # Stage 3: Execute (if we got this far, it's safe enough)
        result = None
        if executor:
            try:
                result = executor(thought)
            except Exception as e:
                return SafeExecutionResult(
                    success=False,
                    result=None,
                    halt_decision=halt_decision,
                    loop_status=loop_status,
                    execution_time=time.time() - start_time,
                    safety_warnings=[f"❌ Execution failed: {str(e)}"],
                )

        # Stage 4: Ghost Code Detection
        ghost_report = None
        if self.enable_ghost_detection and thought.produces_output:
            # Record the output observation
            self.observe_output(
                component_name=thought.component or "unknown",
                output_name=thought.output_id or "output",
                is_visible=result is not None,
                metadata={
                    "process_id": thought.process_id,
                    "action_type": thought.action_type,
                    "result_type": type(result).__name__ if result else "None",
                },
            )

            # Check component health
            ghost_report = self.check_component_health(
                thought.component or "unknown"
            )

            if ghost_report and ghost_report.confidence < 0.5:
                warnings.append(
                    f"👻 GHOST CODE: {ghost_report.recommendation}"
                )
                warnings.extend([f"  • {issue}" for issue in ghost_report.ghost_issues])

        return SafeExecutionResult(
            success=True,
            result=result,
            halt_decision=halt_decision,
            loop_status=loop_status,
            ghost_report=ghost_report,
            execution_time=time.time() - start_time,
            safety_warnings=warnings,
        )

    # ── System Health Dashboard ───────────────────────────────────────────────

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health report.

        Returns:
            Dictionary with health metrics from all detection systems
        """
        health = {
            "timestamp": time.time(),
            "avus_loaded": self.is_loaded,
        }

        # Binary decider stats
        if self.decider:
            health["binary_decisions"] = {
                "total_decisions": self.decider.total_decisions,
                "halt_count": self.decider.halt_count,
                "loop_count": self.decider.loop_count,
                "halt_rate": (
                    self.decider.halt_count / max(self.decider.total_decisions, 1)
                ),
                "avg_confidence": (
                    sum(self.decider.confidence_history) /
                    max(len(self.decider.confidence_history), 1)
                ),
            }

        # Loop detection stats
        if self.loop_detector:
            health["loop_detection"] = {
                "total_actions": len(self.loop_detector.action_history),
                "detected_loops": self.loop_detector.loop_count,
                "loop_rate": (
                    self.loop_detector.loop_count /
                    max(len(self.loop_detector.action_history), 1)
                ),
                "similarity_threshold": self.loop_detector.similarity_threshold,
                "repetition_threshold": self.loop_detector.repetition_threshold,
            }

        # Ghost detection stats
        if self.ghost_detector:
            summary = self.ghost_detector.get_ghost_summary()
            health["ghost_detection"] = {
                "total_components": summary["total_components"],
                "healthy": summary["healthy"],
                "partial_ghost": summary["partial_ghost"],
                "full_ghost": summary["full_ghost"],
                "system_health": summary["system_health"],
            }

        return health

    def print_health_report(self):
        """Print formatted system health report."""
        health = self.get_system_health()

        print("\n" + "=" * 60)
        print("ENHANCED AVUS BRAIN - SYSTEM HEALTH REPORT")
        print("=" * 60)

        print(f"\nAVUS Status: {'✓ Loaded' if health['avus_loaded'] else '✗ Not Loaded'}")

        if "binary_decisions" in health:
            bd = health["binary_decisions"]
            print(f"\n📊 Binary Decider:")
            print(f"  Total Decisions: {bd['total_decisions']}")
            print(f"  Halt Rate: {bd['halt_rate']:.1%}")
            print(f"  Avg Confidence: {bd['avg_confidence']:.1%}")

        if "loop_detection" in health:
            ld = health["loop_detection"]
            print(f"\n🔄 Loop Detector:")
            print(f"  Total Actions: {ld['total_actions']}")
            print(f"  Detected Loops: {ld['detected_loops']}")
            print(f"  Loop Rate: {ld['loop_rate']:.1%}")

        if "ghost_detection" in health:
            gd = health["ghost_detection"]
            print(f"\n👻 Ghost Code Detector:")
            print(f"  Total Components: {gd['total_components']}")
            print(f"  Healthy: {gd['healthy']}")
            print(f"  Partial Ghost: {gd['partial_ghost']}")
            print(f"  Full Ghost: {gd['full_ghost']}")
            print(f"  System Health: {gd['system_health']:.1%}")

        print("\n" + "=" * 60)

    # ── Override ask() with loop detection ────────────────────────────────────

    def ask(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 300,
        temperature: float = 0.7,
        check_safety: bool = True,
    ) -> str:
        """
        Answer a question with optional safety checks.

        Args:
            question: Question to answer
            context: Optional context
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            check_safety: Enable detection systems

        Returns:
            Answer string
        """
        if check_safety and self.enable_loop_detection:
            # Record this as a "query" action
            loop_result = self.record_action(
                action_type="query",
                target=question[:50],  # First 50 chars as identifier
                context={"context_provided": bool(context)},
            )

            if loop_result.is_loop:
                print(f"⚠️ Loop detected: Asking similar questions repeatedly")
                print(f"   Recommendation: {loop_result.recommendation}")

        # Call parent ask method
        return super().ask(question, context, max_tokens, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_enhanced_brain: Optional[EnhancedAvusBrain] = None


def get_enhanced_brain(
    enable_decider: bool = True,
    enable_loop_detection: bool = True,
    enable_ghost_detection: bool = True,
) -> EnhancedAvusBrain:
    """
    Get the shared EnhancedAvusBrain instance.

    Args:
        enable_decider: Enable binary halt/loop prediction
        enable_loop_detection: Enable action loop detection
        enable_ghost_detection: Enable silent failure detection

    Returns:
        Singleton EnhancedAvusBrain instance
    """
    global _enhanced_brain
    if _enhanced_brain is None:
        _enhanced_brain = EnhancedAvusBrain(
            enable_decider=enable_decider,
            enable_loop_detection=enable_loop_detection,
            enable_ghost_detection=enable_ghost_detection,
        )
    return _enhanced_brain


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing Enhanced AVUS Brain...")
    brain = get_enhanced_brain()

    print("\n" + "=" * 60)
    print("DEMO: Safe Thought Execution")
    print("=" * 60)

    # Test 1: Safe thought
    print("\n[Test 1] Safe thought execution:")
    thought1 = ThoughtProcess(
        process_id="calculate_sum",
        context={"a": 5, "b": 3},
        action_type="compute",
        target="sum",
        produces_output=True,
        output_id="result",
        component="cognitive_loop",
    )

    def safe_executor(thought):
        return thought.context["a"] + thought.context["b"]

    result1 = brain.execute_thought_safely(thought1, safe_executor)
    print(f"Success: {result1.success}")
    print(f"Result: {result1.result}")
    print(f"Execution time: {result1.execution_time * 1000:.2f}ms")

    # Test 2: Repeated action (loop detection)
    print("\n[Test 2] Loop detection (repeating action 4 times):")
    for i in range(4):
        loop_result = brain.record_action(
            action_type="navigate",
            target="room_A",
            context={"attempt": i + 1},
        )
        print(f"  Attempt {i+1}: Loop={loop_result.is_loop}")

    # Test 3: Self-referential paradox
    print("\n[Test 3] Self-referential paradox detection:")
    thought3 = ThoughtProcess(
        process_id="analyze_self",
        context={"target": "analyze_self"},  # Self-reference
        action_type="introspect",
        target="self",
    )
    result3 = brain.execute_thought_safely(thought3)
    print(f"Success: {result3.success}")
    if result3.halt_decision:
        print(f"Decision: {'HALTS' if result3.halt_decision.decision else 'LOOPS'}")
        print(f"Reasoning: {result3.halt_decision.reasoning}")
    if result3.safety_warnings:
        for warning in result3.safety_warnings:
            print(f"  {warning}")

    # System health report
    print("\n")
    brain.print_health_report()
