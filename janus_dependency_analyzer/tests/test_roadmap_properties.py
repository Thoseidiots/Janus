"""
Property-based tests for the Roadmap Generator.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the RoadmapGenerator component.

# Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..core.models import Capability, CapabilityCategory, InterfaceType
from ..roadmap.generator import RoadmapGenerator, ImplementationRoadmap


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def cap_strategy():
    return st.builds(
        Capability,
        name=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz "),
        category=st.sampled_from(list(CapabilityCategory)),
        interface_type=st.sampled_from(list(InterfaceType)),
        description=st.text(min_size=0, max_size=100),
        confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        parameters=st.just([]),
        supported_formats=st.just([]),
        detection_method=st.just("test"),
        examples=st.just([]),
    )


# ---------------------------------------------------------------------------
# Property 8: Complete Roadmap Generation
# Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
# ---------------------------------------------------------------------------


class TestCompleteRoadmapGeneration:
    """
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**

    Property 8: Complete Roadmap Generation
    For any capability selected for internalization, the Roadmap_Generator SHALL
    create implementation plans with technical components, effort estimates,
    testing requirements, success criteria, risk mitigation strategies, and
    milestone checkpoints.
    """

    # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_has_technical_components(self, capability: Capability) -> None:
        """
        For any capability, the generated roadmap has a non-empty list of
        technical components.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert isinstance(roadmap.technical_components, list), (
            "technical_components must be a list"
        )
        assert len(roadmap.technical_components) > 0, (
            f"technical_components must be non-empty for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_has_positive_effort_estimate(self, capability: Capability) -> None:
        """
        For any capability, the estimated_effort_hours is strictly positive.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert roadmap.estimated_effort_hours > 0, (
            f"estimated_effort_hours must be > 0, got {roadmap.estimated_effort_hours} "
            f"for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_effort_equals_sum_of_components(self, capability: Capability) -> None:
        """
        The roadmap's estimated_effort_hours equals the sum of all component
        effort_hours.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        component_total = sum(c.effort_hours for c in roadmap.technical_components)
        assert roadmap.estimated_effort_hours == component_total, (
            f"estimated_effort_hours ({roadmap.estimated_effort_hours}) must equal "
            f"sum of component effort_hours ({component_total}) "
            f"for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_has_milestones(self, capability: Capability) -> None:
        """
        For any capability, the generated roadmap has at least one milestone.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert isinstance(roadmap.milestones, list), (
            "milestones must be a list"
        )
        assert len(roadmap.milestones) >= 1, (
            f"milestones must have at least 1 entry for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_milestones_are_ordered(self, capability: Capability) -> None:
        """
        Milestone estimated_hours_from_start values are non-decreasing
        (milestones are ordered chronologically).

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        hours = [m.estimated_hours_from_start for m in roadmap.milestones]
        for i in range(len(hours) - 1):
            assert hours[i] <= hours[i + 1], (
                f"Milestones are not ordered: milestone {i} has "
                f"{hours[i]}h but milestone {i + 1} has {hours[i + 1]}h "
                f"for capability '{capability.name}'"
            )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_has_risks(self, capability: Capability) -> None:
        """
        For any capability, the generated roadmap has at least one risk.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert isinstance(roadmap.risks, list), (
            "risks must be a list"
        )
        assert len(roadmap.risks) >= 1, (
            f"risks must have at least 1 entry for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_all_risks_have_mitigation(self, capability: Capability) -> None:
        """
        Every risk in the roadmap has a non-empty mitigation string.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        for risk in roadmap.risks:
            assert risk.mitigation, (
                f"Risk '{risk.name}' has empty mitigation "
                f"for capability '{capability.name}'"
            )
            assert len(risk.mitigation.strip()) > 0, (
                f"Risk '{risk.name}' has blank/whitespace mitigation "
                f"for capability '{capability.name}'"
            )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_has_success_criteria(self, capability: Capability) -> None:
        """
        For any capability, the generated roadmap has at least one success criterion.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert isinstance(roadmap.success_criteria, list), (
            "success_criteria must be a list"
        )
        assert len(roadmap.success_criteria) >= 1, (
            f"success_criteria must have at least 1 entry for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_testing_requirements_are_complete(self, capability: Capability) -> None:
        """
        The testing_requirements has non-empty unit_tests, integration_tests,
        and performance_tests lists, and test_coverage_target is in [0.0, 1.0].

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        tr = roadmap.testing_requirements
        assert len(tr.unit_tests) > 0, (
            f"unit_tests must be non-empty for capability '{capability.name}'"
        )
        assert len(tr.integration_tests) > 0, (
            f"integration_tests must be non-empty for capability '{capability.name}'"
        )
        assert len(tr.performance_tests) > 0, (
            f"performance_tests must be non-empty for capability '{capability.name}'"
        )
        assert 0.0 <= tr.test_coverage_target <= 1.0, (
            f"test_coverage_target must be in [0.0, 1.0], "
            f"got {tr.test_coverage_target} for capability '{capability.name}'"
        )

    @given(capability=cap_strategy())
    @settings(max_examples=25, deadline=None)
    def test_roadmap_capability_id_matches(self, capability: Capability) -> None:
        """
        The roadmap's capability_id matches the input capability's id.

        # Feature: janus-dependency-analyzer, Property 8: Complete Roadmap Generation
        """
        generator = RoadmapGenerator()
        roadmap = generator.generate(capability)

        assert roadmap.capability_id == capability.id, (
            f"roadmap.capability_id '{roadmap.capability_id}' does not match "
            f"capability.id '{capability.id}'"
        )
