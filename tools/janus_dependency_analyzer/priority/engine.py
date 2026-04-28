"""
Priority Engine for the Janus Dependency Analyzer.

This module implements multi-factor priority scoring for capabilities,
ranking them by implementation priority based on usage frequency,
implementation complexity, security benefits, performance improvements,
and maintenance burden.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.models import Capability, CapabilityCategory, InterfaceType
from ..dependency.mapper import DependencyMapping


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Threshold for "High / Moderate / Low" labels in justification
HIGH_THRESHOLD = 0.7
MODERATE_THRESHOLD = 0.4

# Security-related keywords to look for in capability name/description
SECURITY_KEYWORDS = {
    "auth", "encrypt", "sign", "verify", "certificate", "token", "credential",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PriorityWeights:
    """Configurable weights for the priority scoring factors."""

    usage: float = 0.35          # Usage frequency weight
    complexity: float = 0.25     # Implementation complexity weight (inverted)
    security: float = 0.20       # Security benefit weight
    performance: float = 0.15    # Performance improvement weight
    maintenance: float = 0.05    # Maintenance burden weight (inverted)

    def validate(self) -> bool:
        """Return True if weights sum to approximately 1.0."""
        total = (
            self.usage
            + self.complexity
            + self.security
            + self.performance
            + self.maintenance
        )
        return abs(total - 1.0) < 0.01


@dataclass
class AnalysisContext:
    """Context for priority calculation."""

    usage_frequency: int                          # How many times this capability is used
    max_frequency: int                            # Maximum frequency across all capabilities
    dependency_mapping: Optional[DependencyMapping] = None
    priority_weights: PriorityWeights = field(default_factory=PriorityWeights)


@dataclass
class PriorityScore:
    """Complete priority score with all factor breakdowns."""

    capability_id: str
    capability_name: str
    usage_frequency_score: float           # 0.0 to 1.0
    implementation_complexity_score: float  # 0.0 to 1.0 (higher = simpler)
    security_benefit_score: float          # 0.0 to 1.0
    performance_impact_score: float        # 0.0 to 1.0
    maintenance_burden_score: float        # 0.0 to 1.0 (higher = lower burden)
    total_score: float                     # weighted combination
    justification: str                     # human-readable explanation
    rank: int = 0                          # set after ranking


@dataclass
class RankedCapability:
    """A capability with its priority score and rank."""

    capability: Capability
    priority_score: PriorityScore
    rank: int


# ---------------------------------------------------------------------------
# PriorityEngine
# ---------------------------------------------------------------------------


class PriorityEngine:
    """
    Calculates and ranks implementation priorities for capabilities.

    Uses a multi-factor scoring model that considers:
    - Usage frequency (how often the capability is invoked)
    - Implementation complexity (how hard it is to implement natively)
    - Security benefit (how much security improves by internalizing)
    - Performance impact (how much performance improves)
    - Maintenance burden (ongoing cost of maintaining the integration)
    """

    def __init__(self, weights: Optional[PriorityWeights] = None) -> None:
        self.weights = weights or PriorityWeights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_priority(
        self,
        capability: Capability,
        context: AnalysisContext,
    ) -> PriorityScore:
        """Calculate implementation priority for a capability."""
        usage_score = self._calculate_usage_score(context)
        complexity_score = self._estimate_complexity_score(capability)
        security_score = self._calculate_security_benefit(capability)
        performance_score = self._estimate_performance_gain(capability, context)
        maintenance_score = self._calculate_maintenance_burden(capability, context)

        weights = context.priority_weights
        total_score = (
            usage_score * weights.usage
            + complexity_score * weights.complexity
            + security_score * weights.security
            + performance_score * weights.performance
            + maintenance_score * weights.maintenance
        )
        # Clamp to [0.0, 1.0] to guard against floating-point drift
        total_score = max(0.0, min(1.0, total_score))

        scores = {
            "usage": usage_score,
            "complexity": complexity_score,
            "security": security_score,
            "performance": performance_score,
            "maintenance": maintenance_score,
        }
        justification = self._generate_justification(capability, context, scores)

        return PriorityScore(
            capability_id=capability.id,
            capability_name=capability.name,
            usage_frequency_score=usage_score,
            implementation_complexity_score=complexity_score,
            security_benefit_score=security_score,
            performance_impact_score=performance_score,
            maintenance_burden_score=maintenance_score,
            total_score=total_score,
            justification=justification,
        )

    def rank_capabilities(
        self,
        capabilities: List[Capability],
        contexts: Dict[str, AnalysisContext],
    ) -> List[RankedCapability]:
        """
        Rank capabilities by implementation priority.

        Returns list sorted descending by total_score (rank 1 = highest score).
        ``contexts`` maps capability.id -> AnalysisContext.
        Capabilities without a context entry receive a default context with
        zero usage frequency.
        """
        scored: List[tuple] = []
        for cap in capabilities:
            ctx = contexts.get(cap.id)
            if ctx is None:
                # Default context: zero usage, max_frequency=1
                ctx = AnalysisContext(
                    usage_frequency=0,
                    max_frequency=1,
                    priority_weights=self.weights,
                )
            score = self.calculate_priority(cap, ctx)
            scored.append((cap, score))

        # Sort descending by total_score, then by capability name for stability
        scored.sort(key=lambda t: (-t[1].total_score, t[0].name))

        ranked: List[RankedCapability] = []
        for rank_idx, (cap, score) in enumerate(scored, start=1):
            score.rank = rank_idx
            ranked.append(
                RankedCapability(
                    capability=cap,
                    priority_score=score,
                    rank=rank_idx,
                )
            )

        return ranked

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _calculate_usage_score(self, context: AnalysisContext) -> float:
        """Calculate normalized usage frequency score (0.0 to 1.0)."""
        return min(
            context.usage_frequency / max(context.max_frequency, 1),
            1.0,
        )

    def _estimate_complexity_score(self, capability: Capability) -> float:
        """
        Estimate implementation complexity score (0.0 to 1.0, higher = simpler).

        Base complexity on:
        - Number of parameters (more = more complex)
        - Number of supported formats (more = more complex)
        - Category (SECURITY and NETWORK_OPERATIONS are more complex)
        - Interface type (GUI is more complex than COMMAND_LINE)
        """
        score = 0.8  # base

        # Parameters penalty
        score -= 0.05 * len(capability.parameters)

        # Supported formats penalty
        score -= 0.02 * len(capability.supported_formats)

        # Category penalties
        if capability.category == CapabilityCategory.SECURITY:
            score -= 0.2
        elif capability.category == CapabilityCategory.NETWORK_OPERATIONS:
            score -= 0.15

        # Interface type penalties
        if capability.interface_type == InterfaceType.GUI:
            score -= 0.15
        elif capability.interface_type == InterfaceType.REST_API:
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _calculate_security_benefit(self, capability: Capability) -> float:
        """
        Calculate security benefit score (0.0 to 1.0).

        Higher score for:
        - SECURITY category capabilities
        - Capabilities that contain security-related keywords in name/description
        - NETWORK_OPERATIONS (reduces external network calls)
        - FILE_PROCESSING (reduces external file access)
        """
        if capability.category == CapabilityCategory.SECURITY:
            return 0.9

        # Check name and description for security keywords
        combined_text = (capability.name + " " + capability.description).lower()
        if any(kw in combined_text for kw in SECURITY_KEYWORDS):
            return 0.7

        if capability.category == CapabilityCategory.NETWORK_OPERATIONS:
            return 0.5

        if capability.category == CapabilityCategory.FILE_PROCESSING:
            return 0.4

        return 0.2

    def _estimate_performance_gain(
        self,
        capability: Capability,
        context: AnalysisContext,
    ) -> float:
        """
        Estimate performance improvement score (0.0 to 1.0).

        Higher score for:
        - High usage frequency (more subprocess calls eliminated)
        - FILE_PROCESSING and DATA_TRANSFORMATION categories
        - Capabilities with many parameters (rich interface = more overhead)
        """
        base = (
            min(context.usage_frequency / max(context.max_frequency, 1), 1.0) * 0.6
        )

        if capability.category in (
            CapabilityCategory.FILE_PROCESSING,
            CapabilityCategory.DATA_TRANSFORMATION,
        ):
            base += 0.2

        if len(capability.parameters) > 3:
            base += 0.1

        return max(0.0, min(1.0, base))

    def _calculate_maintenance_burden(
        self,
        capability: Capability,
        context: AnalysisContext,
    ) -> float:
        """
        Calculate maintenance burden score (0.0 to 1.0, higher = lower burden).

        Lower burden (higher score) for:
        - Capabilities used from fewer languages
        - Simpler interface types
        - Lower invocation count (less critical path)
        """
        score = 0.8  # base

        # Language diversity penalty
        if context.dependency_mapping is not None:
            extra_languages = max(
                0, len(context.dependency_mapping.languages_used_from) - 1
            )
            score -= 0.1 * extra_languages

        # Interface type penalties
        if capability.interface_type == InterfaceType.GUI:
            score -= 0.2
        elif capability.interface_type == InterfaceType.REST_API:
            score -= 0.1

        return max(0.1, min(1.0, score))

    # ------------------------------------------------------------------
    # Justification generation
    # ------------------------------------------------------------------

    @staticmethod
    def _score_label(score: float) -> str:
        """Return 'High', 'Moderate', or 'Low' based on score thresholds."""
        if score >= HIGH_THRESHOLD:
            return "High"
        if score >= MODERATE_THRESHOLD:
            return "Moderate"
        return "Low"

    def _generate_justification(
        self,
        capability: Capability,  # noqa: ARG002  (kept for API consistency)
        context: AnalysisContext,  # noqa: ARG002
        scores: dict,
    ) -> str:
        """
        Generate a human-readable justification for the priority ranking.

        Format:
        "Priority score X.XX: [factor1 reason], [factor2 reason], ..."
        """
        weights = context.priority_weights
        total = (
            scores["usage"] * weights.usage
            + scores["complexity"] * weights.complexity
            + scores["security"] * weights.security
            + scores["performance"] * weights.performance
            + scores["maintenance"] * weights.maintenance
        )
        total = max(0.0, min(1.0, total))

        factor_descriptions = [
            (
                f"{self._score_label(scores['usage'])} usage frequency "
                f"(score: {scores['usage']:.2f})"
            ),
            (
                f"{self._score_label(scores['complexity'])} implementation complexity "
                f"(score: {scores['complexity']:.2f})"
            ),
            (
                f"{self._score_label(scores['security'])} security benefit "
                f"(score: {scores['security']:.2f})"
            ),
            (
                f"{self._score_label(scores['performance'])} performance improvement "
                f"(score: {scores['performance']:.2f})"
            ),
            (
                f"{self._score_label(scores['maintenance'])} maintenance burden "
                f"(score: {scores['maintenance']:.2f})"
            ),
        ]

        factors_str = ", ".join(factor_descriptions)
        return f"Priority score {total:.2f}: {factors_str}"
