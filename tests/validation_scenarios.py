"""
validation_scenarios.py
=======================
Real-world validation scenarios for Janus detection systems.

These scenarios test the systems against actual problems that would
occur in production cognitive architectures.

Run with: python tests/validation_scenarios.py
"""

import time
import sys
from typing import List, Dict, Any

from avus_brain_enhanced import (
    EnhancedAvusBrain,
    ThoughtProcess,
    get_enhanced_brain,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Infinite Loop Prevention
# ─────────────────────────────────────────────────────────────────────────────

def scenario_infinite_loop_prevention():
    """
    Scenario: AI enters infinite introspection loop.

    Problem: AI analyzing its own thought process repeatedly without
    halting, consuming resources indefinitely.

    Expected: Binary decider prevents execution with high confidence.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Infinite Loop Prevention")
    print("=" * 70)

    brain = get_enhanced_brain()

    # Create self-referential thought
    thought = ThoughtProcess(
        process_id="analyze_my_analysis",
        context={"analyzing": "analyze_my_analysis"},  # Self-reference
        action_type="introspect",
        target="self",
    )

    print("\n[Setup] Creating self-referential introspection task")
    print(f"Process: {thought.process_id}")
    print(f"Context: {thought.context}")

    # Attempt safe execution
    result = brain.execute_thought_safely(thought)

    print(f"\n[Result] Execution {'PREVENTED' if not result.success else 'ALLOWED'}")
    if result.halt_decision:
        print(f"Decision: {'HALTS' if result.halt_decision.decision else 'LOOPS'}")
        print(f"Confidence: {result.halt_decision.confidence:.1%}")
        print(f"Reasoning: {result.halt_decision.reasoning}")

    if result.safety_warnings:
        print("\n[Warnings]")
        for warning in result.safety_warnings:
            print(f"  {warning}")

    # Validation
    success = (
        not result.success and
        result.halt_decision is not None and
        not result.halt_decision.decision and
        result.halt_decision.confidence > 0.8
    )

    print(f"\n[Validation] {'✓ PASS' if success else '✗ FAIL'}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Repeated Action Recognition
# ─────────────────────────────────────────────────────────────────────────────

def scenario_repeated_action_recognition():
    """
    Scenario: AI enters same room 10 times thinking it's new.

    Problem: AI doesn't recognize it's repeating actions, wasting
    resources on identical operations.

    Expected: Loop detector triggers after 3 repetitions.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Repeated Action Recognition (Same Room Problem)")
    print("=" * 70)

    brain = get_enhanced_brain()

    print("\n[Setup] AI navigating through rooms")

    room_sequence = [
        "hallway", "room_A", "hallway", "room_B",
        "hallway", "room_A",  # 2nd time in room_A
        "hallway", "room_A",  # 3rd time in room_A (should trigger)
        "hallway", "room_A",  # 4th time (definitely triggered)
    ]

    results = []
    for i, room in enumerate(room_sequence):
        print(f"\n[Step {i+1}] Navigating to: {room}")

        thought = ThoughtProcess(
            process_id=f"navigate_{i}",
            context={"room": room, "step": i + 1},
            action_type="navigate",
            target=room,
        )

        result = brain.execute_thought_safely(thought)

        if result.loop_status and result.loop_status.is_loop:
            print(f"  🔄 LOOP DETECTED!")
            print(f"  Similar actions: {result.loop_status.similar_count}")
            print(f"  Recommendation: {result.loop_status.recommendation}")

        results.append(result)

    # Validation: Loop should be detected by step 8-10
    loop_detected = any(
        r.loop_status and r.loop_status.is_loop
        for r in results[6:]  # Check last 4 steps
    )

    print(f"\n[Validation] {'✓ PASS' if loop_detected else '✗ FAIL'}")
    print(f"Loop detected after entering room_A multiple times: {loop_detected}")

    return loop_detected


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Off-Screen UI Element Detection
# ─────────────────────────────────────────────────────────────────────────────

def scenario_offscreen_ui_element():
    """
    Scenario: Button renders off-screen at x=-1000.

    Problem: Button exists in DOM but positioned outside viewport.
    No errors, no exceptions, just silent failure.

    Expected: Ghost detector identifies invisible output.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Off-Screen UI Element Detection")
    print("=" * 70)

    brain = get_enhanced_brain()

    # Register UI component
    brain.ghost_detector.register_component(
        "LoginButton",
        expected_outputs=["button_element", "click_handler"],
        expected_side_effects=["dom_mount"],
        dependencies=["react", "dom"],
    )

    print("\n[Setup] Rendering login button")

    thought = ThoughtProcess(
        process_id="render_login_button",
        context={"component": "LoginButton"},
        action_type="render",
        target="button",
        produces_output=True,
        output_id="button_element",
        component="LoginButton",
    )

    # Simulate rendering with off-screen position
    def ghost_renderer(t):
        # Button renders, but at x=-1000 (off-screen)
        return {
            "element_created": True,
            "position": {"x": -1000, "y": 50},
            "width": 120,
            "height": 40,
            "visible_in_viewport": False,
        }

    result = brain.execute_thought_safely(thought, ghost_renderer)

    # Record observations
    brain.observe_output(
        "LoginButton",
        "button_element",
        False,  # Not visible
        {
            "element_exists": True,
            "position": {"x": -1000, "y": 50},
            "reason": "positioned off-screen"
        }
    )

    brain.observe_output(
        "LoginButton",
        "click_handler",
        True,  # Handler exists
        {"attached": True}
    )

    brain.observe_side_effect(
        "LoginButton",
        "dom_mount",
        True,  # Mounted successfully
        {"mounted": True}
    )

    # Check component health
    report = brain.check_component_health("LoginButton")

    print(f"\n[Component Health]")
    print(f"Status: {report.status}")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"Manifested outputs: {report.outputs_manifested}/{report.outputs_expected}")

    if report.ghost_issues:
        print(f"\n[Ghost Issues Detected]")
        for issue in report.ghost_issues:
            print(f"  {issue}")

    print(f"\n[Recommendation]")
    print(f"  {report.recommendation}")

    # Validation
    success = (
        report.confidence < 0.9 and  # Not fully healthy
        report.status in ["PARTIAL_GHOST", "FULL_GHOST"] and
        len(report.ghost_issues) > 0 and
        "off-screen" in " ".join(report.ghost_issues).lower()
    )

    print(f"\n[Validation] {'✓ PASS' if success else '✗ FAIL'}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Silent Data Pipeline Failure
# ─────────────────────────────────────────────────────────────────────────────

def scenario_silent_data_pipeline():
    """
    Scenario: Data pipeline runs successfully but writes 0 rows.

    Problem: Pipeline completes with "success" status but no data
    is actually transferred. No errors logged.

    Expected: Ghost detector identifies zero-output success.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Silent Data Pipeline Failure")
    print("=" * 70)

    brain = get_enhanced_brain()

    # Register pipeline component
    brain.ghost_detector.register_component(
        "UserDataPipeline",
        expected_outputs=["processed_records", "export_file"],
        expected_side_effects=["write_to_database", "log_completion"],
        dependencies=["database", "file_system"],
    )

    print("\n[Setup] Running user data export pipeline")

    thought = ThoughtProcess(
        process_id="export_users",
        context={"source": "users_table", "destination": "export.csv"},
        action_type="pipeline",
        target="data_export",
        produces_output=True,
        output_id="processed_records",
        component="UserDataPipeline",
    )

    # Simulate pipeline that "succeeds" but outputs nothing
    def silent_pipeline(t):
        return {
            "status": "SUCCESS",
            "records_processed": 0,  # Zero data!
            "errors": [],
            "completion_time": 1.5,
        }

    result = brain.execute_thought_safely(thought, silent_pipeline)

    # Record observations
    brain.observe_output(
        "UserDataPipeline",
        "processed_records",
        False,  # No visible output
        {
            "function_returned": "SUCCESS",
            "records_count": 0,
            "reason": "empty result set despite success status"
        }
    )

    brain.observe_output(
        "UserDataPipeline",
        "export_file",
        False,  # File empty or missing
        {
            "file_exists": True,
            "file_size": 0,
            "reason": "file created but empty"
        }
    )

    brain.observe_side_effect(
        "UserDataPipeline",
        "write_to_database",
        False,  # No actual write
        {
            "connection_active": True,
            "rows_affected": 0,
            "reason": "connection exists but no data transferred"
        }
    )

    brain.observe_side_effect(
        "UserDataPipeline",
        "log_completion",
        True,  # Logging works
        {"logged": True, "message": "Pipeline completed successfully"}
    )

    # Check component health
    report = brain.check_component_health("UserDataPipeline")

    print(f"\n[Pipeline Health]")
    print(f"Status: {report.status}")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"Outputs manifested: {report.outputs_manifested}/{report.outputs_expected}")
    print(f"Side effects manifested: {report.effects_manifested}/{report.effects_expected}")

    if report.ghost_issues:
        print(f"\n[Ghost Issues Detected]")
        for issue in report.ghost_issues:
            print(f"  {issue}")

    print(f"\n[Recommendation]")
    print(f"  {report.recommendation}")

    # Validation
    success = (
        report.confidence < 0.5 and  # Low confidence (ghost)
        report.status == "FULL_GHOST" and
        len(report.ghost_issues) >= 2  # Multiple ghost issues
    )

    print(f"\n[Validation] {'✓ PASS' if success else '✗ FAIL'}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Semantic Loop Detection
# ─────────────────────────────────────────────────────────────────────────────

def scenario_semantic_loop():
    """
    Scenario: AI searches different places for same thing.

    Problem: AI searches kitchen, bedroom, bathroom for "food" without
    recognizing the pattern (different locations, same action).

    Expected: Loop detector recognizes semantic similarity.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Semantic Loop Detection")
    print("=" * 70)

    brain = get_enhanced_brain()

    print("\n[Setup] AI searching for food in different rooms")

    search_actions = [
        ("search", "kitchen", {"target": "food", "urgency": "high"}),
        ("search", "bedroom", {"target": "food", "urgency": "high"}),
        ("search", "bathroom", {"target": "food", "urgency": "high"}),
        ("search", "garage", {"target": "food", "urgency": "high"}),
    ]

    results = []
    for i, (action, room, context) in enumerate(search_actions):
        print(f"\n[Step {i+1}] Searching {room} for food")

        thought = ThoughtProcess(
            process_id=f"search_{i}",
            context=context,
            action_type=action,
            target=room,
        )

        result = brain.execute_thought_safely(thought)

        if result.loop_status and result.loop_status.is_loop:
            print(f"  🔄 SEMANTIC LOOP DETECTED!")
            print(f"  Pattern: Same action type across different targets")
            print(f"  Recommendation: {result.loop_status.recommendation}")

        results.append(result)

    # Validation: Should detect semantic pattern
    loop_detected = any(
        r.loop_status and r.loop_status.is_loop
        for r in results
    )

    print(f"\n[Validation] {'✓ PASS' if loop_detected else '✗ FAIL'}")
    print(f"Semantic pattern detected: {loop_detected}")

    return loop_detected


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Combined Multi-System Detection
# ─────────────────────────────────────────────────────────────────────────────

def scenario_combined_detection():
    """
    Scenario: Complex failure with multiple detection systems.

    Problem: AI repeatedly tries to render a component that never
    shows up, creating both a loop and ghost code situation.

    Expected: Both loop detector and ghost detector trigger.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: Combined Multi-System Detection")
    print("=" * 70)

    brain = get_enhanced_brain()

    # Register component
    brain.ghost_detector.register_component(
        "NotificationToast",
        expected_outputs=["toast_element"],
        expected_side_effects=["show_animation"],
        dependencies=["ui_framework"],
    )

    print("\n[Setup] AI repeatedly attempting to show notification")

    results = []
    for attempt in range(5):
        print(f"\n[Attempt {attempt + 1}] Trying to render notification...")

        thought = ThoughtProcess(
            process_id=f"show_notification_{attempt}",
            context={"message": "Update available", "attempt": attempt + 1},
            action_type="render",
            target="notification",
            produces_output=True,
            output_id="toast_element",
            component="NotificationToast",
        )

        # Ghost renderer - always returns invisible
        def ghost_renderer(t):
            return {
                "rendered": True,
                "visible": False,
                "z_index": -1,  # Behind everything
            }

        result = brain.execute_thought_safely(thought, ghost_renderer)

        # Record ghost observation
        brain.observe_output(
            "NotificationToast",
            "toast_element",
            False,
            {"z_index": -1, "reason": "rendered behind other elements"}
        )

        # Check for warnings
        if result.loop_status and result.loop_status.is_loop:
            print("  🔄 Loop detected!")
        if result.ghost_report and result.ghost_report.confidence < 0.5:
            print("  👻 Ghost code detected!")

        results.append(result)

    # Validation: Both systems should trigger
    loop_detected = any(r.loop_status and r.loop_status.is_loop for r in results)
    ghost_detected = any(
        r.ghost_report and r.ghost_report.confidence < 0.5
        for r in results
    )

    print(f"\n[Results]")
    print(f"Loop detection triggered: {loop_detected}")
    print(f"Ghost detection triggered: {ghost_detected}")

    success = loop_detected and ghost_detected

    print(f"\n[Validation] {'✓ PASS' if success else '✗ FAIL'}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios():
    """Run all validation scenarios and generate report."""
    print("\n" + "=" * 70)
    print("JANUS DETECTION SYSTEMS - REAL-WORLD VALIDATION")
    print("=" * 70)

    scenarios = [
        ("Infinite Loop Prevention", scenario_infinite_loop_prevention),
        ("Repeated Action Recognition", scenario_repeated_action_recognition),
        ("Off-Screen UI Element", scenario_offscreen_ui_element),
        ("Silent Data Pipeline", scenario_silent_data_pipeline),
        ("Semantic Loop Detection", scenario_semantic_loop),
        ("Combined Multi-System", scenario_combined_detection),
    ]

    results = {}
    for name, scenario_func in scenarios:
        try:
            results[name] = scenario_func()
        except Exception as e:
            print(f"\n❌ SCENARIO FAILED WITH EXCEPTION: {e}")
            results[name] = False

    # Summary Report
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    print(f"\nResults: {passed}/{total} scenarios passed\n")

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {name}")

    overall_success = passed == total

    print("\n" + "=" * 70)
    if overall_success:
        print("✓ ALL VALIDATION SCENARIOS PASSED")
    else:
        print(f"⚠ {total - passed} SCENARIO(S) FAILED")
    print("=" * 70)

    return overall_success


if __name__ == "__main__":
    success = run_all_scenarios()
    sys.exit(0 if success else 1)

