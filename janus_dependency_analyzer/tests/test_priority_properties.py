"""
Property-based tests for the Priority Engine.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the PriorityEngine component.

Feature: janus-dependency-analyzer
"""

import pytest
from typing import Dict, List

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from ..core.models import (
    Capability,
    CapabilityCategory,
    InterfaceType,
    Parameter,
)
from ..priority.engine import (
    AnalysisContext,
    PriorityEngine,
    PriorityScore,
    PriorityWeights,
    RankedCapability,
)
from ..dependency.mapper import DependencyMapping, ExternalInvocation


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

capability_category_strategy = st.sampled_from(list(CapabilityCategory))
interface_type_strategy = st.sampled_from(list(InterfaceType))

parameter_strategy = st.builds(
    Parameter,
    name=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20),
    type=st.sampled_from(["str", "int", "bool", "float", "list"]),
    description=st.text(min_size=0, max_size=50),
    required=st.booleans(),
    default_value=st.one_of(st.none(), st.text(min_size=0, max_size=10)),
)

supported_format_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789.",
    min_size=1,
    max_size=10,
)


def capability_strategy() -> st.SearchStrategy:
    """Build a Hypothesis strategy for generating Capability objects."""
    return st.builds(
        Capability,
        name=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-",
            min_size=1,
            max_size=40,
        ),
        category=capability_category_strategy,
        description=st.text(min_size=0, max_size=100),
        interface_type=interface_type_strategy,
        parameters=st.lists(parameter_strategy, min_size=0, max_size=8),
        supported_formats=st.lists(supported_format_strategy, min_size=0, max_size=6),
        confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )


def analysis_context_strategy(
    min_freq: int = 0,
    max_freq_upper: int = 100,
) -> st.SearchStrategy:
    """Build a Hypothesis strategy for generating AnalysisContext objects."""
    return st.integers(min_value=1, max_value=max_freq_upper).flatmap(
        lambda max_freq: st.integers(min_value=0, max_value=max_freq).map(
            lambda usage_freq: AnalysisContext(
                usage_frequency=usage_freq,
                max_frequency=max_freq,
            )
        )
    )


# ---------------------------------------------------------------------------
# Property 7: Multi-Factor Priority Calculation
# Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
# ---------------------------------------------------------------------------


class TestMultiFactorPriorityCalculation:
    """
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**

    Property 7: Multi-Factor Priority Calculation
    For any capability with usage data, the Priority_Engine SHALL calculate
    priority scores considering usage frequency, implementation complexity,
    maintenance burden, security benefits, and performance improvements, then
    rank capabilities in descending order with justification.
    """

    # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation

    @given(
        capability=capability_strategy(),
        context=analysis_context_strategy(),
    )
    @settings(max_examples=25)
    def test_total_score_is_between_0_and_1(
        self, capability: Capability, context: AnalysisContext
    ) -> None:
        """
        For any capability and context, total_score is in [0.0, 1.0].

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        score = engine.calculate_priority(capability, context)

        assert 0.0 <= score.total_score <= 1.0, (
            f"total_score {score.total_score} is outside [0.0, 1.0] "
            f"for capability '{capability.name}'"
        )

    @given(
        capability=capability_strategy(),
        context=analysis_context_strategy(),
    )
    @settings(max_examples=25)
    def test_all_factor_scores_are_between_0_and_1(
        self, capability: Capability, context: AnalysisContext
    ) -> None:
        """
        All individual factor scores (usage, complexity, security, performance,
        maintenance) are in [0.0, 1.0].

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        score = engine.calculate_priority(capability, context)

        assert 0.0 <= score.usage_frequency_score <= 1.0, (
            f"usage_frequency_score {score.usage_frequency_score} out of range"
        )
        assert 0.0 <= score.implementation_complexity_score <= 1.0, (
            f"implementation_complexity_score {score.implementation_complexity_score} out of range"
        )
        assert 0.0 <= score.security_benefit_score <= 1.0, (
            f"security_benefit_score {score.security_benefit_score} out of range"
        )
        assert 0.0 <= score.performance_impact_score <= 1.0, (
            f"performance_impact_score {score.performance_impact_score} out of range"
        )
        assert 0.0 <= score.maintenance_burden_score <= 1.0, (
            f"maintenance_burden_score {score.maintenance_burden_score} out of range"
        )

    @given(
        capability=capability_strategy(),
        low_freq=st.integers(min_value=1, max_value=100),
        high_freq=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=25)
    def test_higher_usage_gives_higher_score(
        self,
        capability: Capability,
        low_freq: int,
        high_freq: int,
    ) -> None:
        """
        When two identical capabilities differ only in usage_frequency
        (one higher, one lower), the higher-usage one gets a higher total_score.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        assume(high_freq > low_freq)

        max_freq = high_freq  # same max for both so the ratio differs

        ctx_low = AnalysisContext(
            usage_frequency=low_freq,
            max_frequency=max_freq,
        )
        ctx_high = AnalysisContext(
            usage_frequency=high_freq,
            max_frequency=max_freq,
        )

        engine = PriorityEngine()
        score_low = engine.calculate_priority(capability, ctx_low)
        score_high = engine.calculate_priority(capability, ctx_high)

        assert score_high.total_score >= score_low.total_score, (
            f"Higher usage_frequency ({high_freq}) should yield a higher or equal "
            f"total_score than lower usage_frequency ({low_freq}), but got "
            f"{score_high.total_score} vs {score_low.total_score}"
        )

    @given(
        capabilities=st.lists(capability_strategy(), min_size=2, max_size=10),
    )
    @settings(max_examples=25)
    def test_ranking_is_descending(
        self, capabilities: List[Capability]
    ) -> None:
        """
        rank_capabilities() returns capabilities in descending order of
        total_score (rank 1 = highest score).

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        max_freq = max(len(capabilities), 1)
        contexts: Dict[str, AnalysisContext] = {
            cap.id: AnalysisContext(
                usage_frequency=i + 1,
                max_frequency=max_freq,
            )
            for i, cap in enumerate(capabilities)
        }

        ranked = engine.rank_capabilities(capabilities, contexts)

        for i in range(len(ranked) - 1):
            assert ranked[i].priority_score.total_score >= ranked[i + 1].priority_score.total_score, (
                f"Ranking is not descending at position {i}: "
                f"score {ranked[i].priority_score.total_score} < "
                f"score {ranked[i + 1].priority_score.total_score}"
            )

    @given(
        capabilities=st.lists(capability_strategy(), min_size=1, max_size=10),
    )
    @settings(max_examples=25)
    def test_rank_numbers_are_sequential(
        self, capabilities: List[Capability]
    ) -> None:
        """
        Ranks are 1, 2, 3, ... N with no gaps.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        contexts: Dict[str, AnalysisContext] = {
            cap.id: AnalysisContext(
                usage_frequency=1,
                max_frequency=max(len(capabilities), 1),
            )
            for cap in capabilities
        }

        ranked = engine.rank_capabilities(capabilities, contexts)

        ranks = [rc.rank for rc in ranked]
        expected = list(range(1, len(capabilities) + 1))
        assert ranks == expected, (
            f"Expected sequential ranks {expected}, got {ranks}"
        )

    @given(
        capability=capability_strategy(),
        context=analysis_context_strategy(),
    )
    @settings(max_examples=25)
    def test_justification_is_non_empty(
        self, capability: Capability, context: AnalysisContext
    ) -> None:
        """
        Every PriorityScore has a non-empty justification string.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        score = engine.calculate_priority(capability, context)

        assert score.justification, (
            f"justification is empty for capability '{capability.name}'"
        )
        assert len(score.justification.strip()) > 0, (
            f"justification is blank/whitespace for capability '{capability.name}'"
        )

    @given(
        capability=capability_strategy(),
        context=analysis_context_strategy(),
    )
    @settings(max_examples=25)
    def test_justification_contains_score(
        self, capability: Capability, context: AnalysisContext
    ) -> None:
        """
        The justification string contains the total score value.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        score = engine.calculate_priority(capability, context)

        # The justification should contain the formatted total score
        formatted_score = f"{score.total_score:.2f}"
        assert formatted_score in score.justification, (
            f"Justification '{score.justification}' does not contain "
            f"the total score '{formatted_score}'"
        )

    @given(
        capability=st.builds(
            Capability,
            name=st.text(min_size=1, max_size=30),
            category=st.just(CapabilityCategory.SECURITY),
            description=st.text(min_size=0, max_size=50),
            interface_type=interface_type_strategy,
            parameters=st.lists(parameter_strategy, min_size=0, max_size=4),
            supported_formats=st.lists(supported_format_strategy, min_size=0, max_size=4),
            confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        context=analysis_context_strategy(),
    )
    @settings(max_examples=25)
    def test_security_category_gets_high_security_score(
        self, capability: Capability, context: AnalysisContext
    ) -> None:
        """
        A capability with category=CapabilityCategory.SECURITY gets
        security_benefit_score >= 0.8.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        engine = PriorityEngine()
        score = engine.calculate_priority(capability, context)

        assert score.security_benefit_score >= 0.8, (
            f"SECURITY category capability '{capability.name}' should have "
            f"security_benefit_score >= 0.8, got {score.security_benefit_score}"
        )

    def test_weights_sum_to_one(self) -> None:
        """
        Default PriorityWeights validates (weights sum to ~1.0).

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        weights = PriorityWeights()
        assert weights.validate(), (
            f"Default PriorityWeights should sum to ~1.0, but validate() returned False. "
            f"Sum = {weights.usage + weights.complexity + weights.security + weights.performance + weights.maintenance}"
        )

    @given(
        capability=capability_strategy(),
        usage_freq=st.integers(min_value=0, max_value=100),
        max_freq=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=25)
    def test_custom_weights_respected(
        self,
        capability: Capability,
        usage_freq: int,
        max_freq: int,
    ) -> None:
        """
        When usage weight is 1.0 and all others are 0.0, the total_score
        equals the usage_frequency_score.

        # Feature: janus-dependency-analyzer, Property 7: Multi-Factor Priority Calculation
        """
        assume(usage_freq <= max_freq)

        weights = PriorityWeights(
            usage=1.0,
            complexity=0.0,
            security=0.0,
            performance=0.0,
            maintenance=0.0,
        )
        context = AnalysisContext(
            usage_frequency=usage_freq,
            max_frequency=max_freq,
            priority_weights=weights,
        )

        engine = PriorityEngine(weights=weights)
        score = engine.calculate_priority(capability, context)

        expected_usage_score = min(usage_freq / max(max_freq, 1), 1.0)
        assert abs(score.total_score - expected_usage_score) < 1e-9, (
            f"With usage weight=1.0 and all others=0.0, total_score should equal "
            f"usage_frequency_score ({expected_usage_score:.6f}), "
            f"but got total_score={score.total_score:.6f}"
        )
