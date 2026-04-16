"""
Janus Ghost Code Detector: Finds "working" code that produces no visible output

Problem: Code runs perfectly, no errors, but nothing shows up.
- Button renders off-screen
- Component has zero height
- Pipeline completes but output never connects to UI
- No errors in console, just silence

This is more dangerous than bugs because the system thinks everything is fine.
"""

import ast
import inspect
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class ComponentManifest:
    """Tracks what a component SHOULD be doing"""
    component_name: str
    expected_outputs: List[str]  # What should be visible/callable
    expected_side_effects: List[str]  # What should happen
    dependencies: List[str]  # What it needs
    registered_at: float = field(default_factory=time.time)


@dataclass
class ObservabilityReport:
    """Report on what's actually manifesting"""
    component_name: str
    outputs_manifested: Dict[str, bool]  # output_name -> is_visible
    side_effects_observed: Dict[str, bool]  # effect_name -> did_happen
    ghost_issues: List[str]  # Things that should exist but don't
    confidence: float
    recommendation: str


class JanusGhostCodeDetector:
    """
    Observability layer that checks "am I visibly doing what I'm supposed to?"

    Not just "did the function return success" but "is the result actually manifesting?"
    """

    def __init__(self):
        # Registry of what components should be doing
        self.manifests: Dict[str, ComponentManifest] = {}

        # Observations of what's actually happening
        self.observations: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Detection history
        self.ghost_history: List[ObservabilityReport] = []

    def register_component(
        self,
        component_name: str,
        expected_outputs: List[str],
        expected_side_effects: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ):
        """
        Register what a component SHOULD be doing.

        Example:
        register_component(
            "UserProfileCard",
            expected_outputs=["profile_image", "username_text", "bio_section"],
            expected_side_effects=["fetch_user_data", "update_avatar_cache"],
            dependencies=["user_api", "image_loader"]
        )
        """
        self.manifests[component_name] = ComponentManifest(
            component_name=component_name,
            expected_outputs=expected_outputs,
            expected_side_effects=expected_side_effects or [],
            dependencies=dependencies or []
        )

    def observe_output(self, component_name: str, output_name: str, is_visible: bool, metadata: Optional[Dict] = None):
        """
        Record an observation about whether an output is actually visible.

        Example:
        observe_output("UserProfileCard", "profile_image", False, {
            "element_exists": True,
            "width": 0,
            "height": 0,
            "reason": "zero dimensions"
        })
        """
        if component_name not in self.observations:
            self.observations[component_name] = {}

        self.observations[component_name][output_name] = {
            "is_visible": is_visible,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

    def observe_side_effect(self, component_name: str, effect_name: str, did_occur: bool, metadata: Optional[Dict] = None):
        """
        Record whether a side effect actually happened.

        Example:
        observe_side_effect("DataPipeline", "output_written_to_db", False, {
            "function_called": True,
            "return_value": "success",
            "rows_affected": 0,
            "reason": "connection exists but no data transferred"
        })
        """
        if component_name not in self.observations:
            self.observations[component_name] = {}

        effect_key = f"effect:{effect_name}"
        self.observations[component_name][effect_key] = {
            "did_occur": did_occur,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

    def check_component_health(self, component_name: str) -> ObservabilityReport:
        """
        Check if a component is actually manifesting its expected behavior.

        Returns a report with ghost issues (things that should work but don't).
        """
        if component_name not in self.manifests:
            return ObservabilityReport(
                component_name=component_name,
                outputs_manifested={},
                side_effects_observed={},
                ghost_issues=[f"Component not registered in manifest"],
                confidence=0.0,
                recommendation="Register component with register_component() first"
            )

        manifest = self.manifests[component_name]
        observations = self.observations.get(component_name, {})

        # Check outputs
        outputs_manifested = {}
        output_ghost_issues = []

        for expected_output in manifest.expected_outputs:
            if expected_output in observations:
                is_visible = observations[expected_output].get("is_visible", False)
                outputs_manifested[expected_output] = is_visible

                if not is_visible:
                    metadata = observations[expected_output].get("metadata", {})
                    reason = metadata.get("reason", "unknown")
                    output_ghost_issues.append(
                        f"👻 Output '{expected_output}' exists but not visible: {reason}"
                    )
            else:
                outputs_manifested[expected_output] = False
                output_ghost_issues.append(
                    f"👻 Output '{expected_output}' never observed (no data)"
                )

        # Check side effects
        side_effects_observed = {}
        effect_ghost_issues = []

        for expected_effect in manifest.expected_side_effects:
            effect_key = f"effect:{expected_effect}"
            if effect_key in observations:
                did_occur = observations[effect_key].get("did_occur", False)
                side_effects_observed[expected_effect] = did_occur

                if not did_occur:
                    metadata = observations[effect_key].get("metadata", {})
                    reason = metadata.get("reason", "unknown")
                    effect_ghost_issues.append(
                        f"👻 Side effect '{expected_effect}' was attempted but didn't manifest: {reason}"
                    )
            else:
                side_effects_observed[expected_effect] = False
                effect_ghost_issues.append(
                    f"👻 Side effect '{expected_effect}' never observed"
                )

        # Combine issues
        all_ghost_issues = output_ghost_issues + effect_ghost_issues

        # Calculate confidence (1.0 = everything working, 0.0 = total ghost)
        total_expected = len(manifest.expected_outputs) + len(manifest.expected_side_effects)
        if total_expected == 0:
            confidence = 1.0
        else:
            successful = sum(outputs_manifested.values()) + sum(side_effects_observed.values())
            confidence = successful / total_expected

        # Recommendation
        if confidence >= 0.9:
            recommendation = "✅ Component healthy - all outputs manifesting"
        elif confidence >= 0.5:
            recommendation = f"⚠️ Partial ghost - {len(all_ghost_issues)} issues found. Investigate missing outputs."
        else:
            recommendation = f"🔴 GHOST CODE - Component running but not manifesting! {len(all_ghost_issues)} critical issues."

        report = ObservabilityReport(
            component_name=component_name,
            outputs_manifested=outputs_manifested,
            side_effects_observed=side_effects_observed,
            ghost_issues=all_ghost_issues,
            confidence=confidence,
            recommendation=recommendation
        )

        self.ghost_history.append(report)
        return report

    def scan_all_components(self) -> Dict[str, ObservabilityReport]:
        """Scan all registered components and return health reports"""
        reports = {}
        for component_name in self.manifests:
            reports[component_name] = self.check_component_health(component_name)
        return reports

    def get_ghost_summary(self) -> Dict[str, Any]:
        """Get system-wide ghost code statistics"""
        total_components = len(self.manifests)
        if total_components == 0:
            return {
                "total_components": 0,
                "healthy": 0,
                "partial_ghost": 0,
                "full_ghost": 0,
                "system_health": 0.0
            }

        reports = self.scan_all_components()

        healthy = sum(1 for r in reports.values() if r.confidence >= 0.9)
        partial = sum(1 for r in reports.values() if 0.5 <= r.confidence < 0.9)
        full_ghost = sum(1 for r in reports.values() if r.confidence < 0.5)

        avg_confidence = sum(r.confidence for r in reports.values()) / len(reports)

        return {
            "total_components": total_components,
            "healthy": healthy,
            "partial_ghost": partial,
            "full_ghost": full_ghost,
            "system_health": avg_confidence,
            "ghost_issues_count": sum(len(r.ghost_issues) for r in reports.values())
        }


def demo_ghost_detector():
    """Demonstrate ghost code detection"""
    print("="*70)
    print("JANUS GHOST CODE DETECTOR: FINDING SILENT FAILURES")
    print("="*70)

    detector = JanusGhostCodeDetector()

    # Scenario 1: UI Component that renders but is invisible
    print("\n[SCENARIO 1] UI Component - Button renders off-screen")
    print("-" * 70)

    detector.register_component(
        "SubmitButton",
        expected_outputs=["button_element", "click_handler"],
        expected_side_effects=["render_to_dom"],
        dependencies=["react", "event_system"]
    )

    # Observations: Button exists but is off-screen
    detector.observe_output("SubmitButton", "button_element", False, {
        "element_exists": True,
        "position": {"x": -1000, "y": 50},
        "width": 100,
        "height": 40,
        "reason": "positioned off-screen (x=-1000)"
    })

    detector.observe_output("SubmitButton", "click_handler", True, {
        "handler_registered": True
    })

    detector.observe_side_effect("SubmitButton", "render_to_dom", True, {
        "dom_node_created": True
    })

    report = detector.check_component_health("SubmitButton")
    print(f"\nComponent: {report.component_name}")
    print(f"Confidence: {report.confidence:.2%}")
    print(f"Ghost Issues: {len(report.ghost_issues)}")
    for issue in report.ghost_issues:
        print(f"  {issue}")
    print(f"\n{report.recommendation}")

    # Scenario 2: Data Pipeline that completes successfully but outputs nothing
    print("\n\n[SCENARIO 2] Data Pipeline - Runs successfully but no output")
    print("-" * 70)

    detector.register_component(
        "UserDataPipeline",
        expected_outputs=["processed_data", "export_file"],
        expected_side_effects=["write_to_database", "trigger_notification"],
        dependencies=["database_connection", "file_system"]
    )

    # Pipeline runs but produces no actual results
    detector.observe_output("UserDataPipeline", "processed_data", False, {
        "function_returned": "success",
        "data_object_exists": True,
        "data_length": 0,
        "reason": "empty result set despite success status"
    })

    detector.observe_output("UserDataPipeline", "export_file", False, {
        "file_created": True,
        "file_size": 0,
        "reason": "file exists but is empty"
    })

    detector.observe_side_effect("UserDataPipeline", "write_to_database", False, {
        "function_called": True,
        "connection_active": True,
        "rows_affected": 0,
        "reason": "connection exists but no data transferred"
    })

    detector.observe_side_effect("UserDataPipeline", "trigger_notification", False, {
        "function_called": True,
        "notification_queued": False,
        "reason": "notification system responded but nothing sent"
    })

    report = detector.check_component_health("UserDataPipeline")
    print(f"\nComponent: {report.component_name}")
    print(f"Confidence: {report.confidence:.2%}")
    print(f"Ghost Issues: {len(report.ghost_issues)}")
    for issue in report.ghost_issues:
        print(f"  {issue}")
    print(f"\n{report.recommendation}")

    # Scenario 3: Healthy component for comparison
    print("\n\n[SCENARIO 3] Healthy Component - Everything manifesting correctly")
    print("-" * 70)

    detector.register_component(
        "UserAvatar",
        expected_outputs=["avatar_image", "fallback_icon"],
        expected_side_effects=["load_image", "cache_result"],
        dependencies=["image_loader"]
    )

    detector.observe_output("UserAvatar", "avatar_image", True, {
        "visible": True,
        "dimensions": {"width": 64, "height": 64}
    })

    detector.observe_output("UserAvatar", "fallback_icon", True, {
        "visible": False,  # Fallback not needed, but it's available
        "reason": "primary image loaded successfully"
    })

    detector.observe_side_effect("UserAvatar", "load_image", True)
    detector.observe_side_effect("UserAvatar", "cache_result", True)

    report = detector.check_component_health("UserAvatar")
    print(f"\nComponent: {report.component_name}")
    print(f"Confidence: {report.confidence:.2%}")
    print(f"Ghost Issues: {len(report.ghost_issues)}")
    print(f"\n{report.recommendation}")

    # System-wide summary
    print("\n" + "="*70)
    print("SYSTEM-WIDE GHOST CODE SUMMARY:")
    print("="*70)

    summary = detector.get_ghost_summary()
    print(f"Total Components: {summary['total_components']}")
    print(f"✅ Healthy: {summary['healthy']}")
    print(f"⚠️  Partial Ghost: {summary['partial_ghost']}")
    print(f"🔴 Full Ghost: {summary['full_ghost']}")
    print(f"System Health: {summary['system_health']:.2%}")
    print(f"Total Ghost Issues: {summary['ghost_issues_count']}")

    print("\n" + "="*70)
    print("GHOST DETECTION SUMMARY:")
    print("="*70)
    print("✓ Detects components that run but produce no visible output")
    print("✓ Distinguishes between 'function succeeded' and 'result manifested'")
    print("✓ Identifies silent failures (zero dimensions, empty data, etc.)")
    print("✓ Tracks side effects that should happen but don't")
    print("✓ Provides system-wide observability metrics")
    print("\nThis catches what traditional debugging misses!")
    print("="*70)


if __name__ == "__main__":
    demo_ghost_detector()
