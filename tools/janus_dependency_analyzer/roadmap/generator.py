"""
Roadmap Generator for the Janus Dependency Analyzer.

This module implements the RoadmapGenerator, which produces detailed
implementation plans for capabilities selected for internalization.
Plans include technical components, effort estimates, milestones,
risk assessments, success criteria, and testing requirements.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum

from ..core.models import Capability, CapabilityCategory, InterfaceType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplexityLevel(Enum):
    """Complexity levels for technical components."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TechnicalComponent:
    """A discrete technical component required for implementation."""
    name: str
    description: str
    effort_hours: int
    dependencies: List[str]
    complexity: ComplexityLevel


@dataclass
class Milestone:
    """A progress checkpoint in the implementation plan."""
    name: str
    description: str
    deliverables: List[str]
    estimated_hours_from_start: int  # cumulative hours from project start


@dataclass
class Risk:
    """A potential risk with mitigation strategy."""
    name: str
    description: str
    level: RiskLevel
    mitigation: str
    probability: float  # 0.0 to 1.0


@dataclass
class TestingRequirements:
    """Testing requirements for the implementation."""
    unit_tests: List[str]
    integration_tests: List[str]
    performance_tests: List[str]
    test_coverage_target: float  # 0.0 to 1.0


@dataclass
class ImplementationRoadmap:
    """Complete implementation plan for a capability."""
    capability_id: str
    capability_name: str
    estimated_effort_hours: int
    technical_components: List[TechnicalComponent]
    milestones: List[Milestone]
    risks: List[Risk]
    success_criteria: List[str]
    testing_requirements: TestingRequirements


# ---------------------------------------------------------------------------
# Complexity mapping by interface type
# ---------------------------------------------------------------------------

_INTERFACE_COMPLEXITY: dict = {
    InterfaceType.GUI: ComplexityLevel.VERY_HIGH,
    InterfaceType.REST_API: ComplexityLevel.HIGH,
    InterfaceType.GRAPHQL_API: ComplexityLevel.HIGH,
    InterfaceType.COMMAND_LINE: ComplexityLevel.MEDIUM,
    InterfaceType.LIBRARY: ComplexityLevel.LOW,
}

_COMPLEXITY_HOURS: dict = {
    ComplexityLevel.LOW: 8,
    ComplexityLevel.MEDIUM: 16,
    ComplexityLevel.HIGH: 32,
    ComplexityLevel.VERY_HIGH: 64,
}


# ---------------------------------------------------------------------------
# RoadmapGenerator
# ---------------------------------------------------------------------------


class RoadmapGenerator:
    """
    Generates detailed implementation roadmaps for capabilities.

    For each capability selected for internalization, the generator produces:
    - Technical components with effort estimates (Requirements 5.1, 5.2, 5.3)
    - Testing requirements and success criteria (Requirements 5.4, 5.5, 5.7)
    - Risk assessment and mitigation strategies (Requirement 5.6)
    - Milestone checkpoints for progress tracking (Requirement 5.7)
    """

    def generate(self, capability: Capability) -> ImplementationRoadmap:
        """Generate a complete implementation roadmap for a capability."""
        components = self._identify_technical_components(capability)
        total_hours = sum(c.effort_hours for c in components)
        milestones = self._create_milestones(capability, components, total_hours)
        risks = self._assess_risks(capability)
        success_criteria = self._define_success_criteria(capability)
        testing_reqs = self._define_testing_requirements(capability)

        return ImplementationRoadmap(
            capability_id=capability.id,
            capability_name=capability.name,
            estimated_effort_hours=total_hours,
            technical_components=components,
            milestones=milestones,
            risks=risks,
            success_criteria=success_criteria,
            testing_requirements=testing_reqs,
        )

    # ------------------------------------------------------------------
    # Technical component identification (Requirements 5.1, 5.2, 5.3)
    # ------------------------------------------------------------------

    def _identify_technical_components(
        self, capability: Capability
    ) -> List[TechnicalComponent]:
        """
        Identify all technical components required to implement the capability.

        Always includes base components (core implementation, unit tests,
        integration tests, documentation) plus category-specific extras.
        """
        complexity = _INTERFACE_COMPLEXITY.get(
            capability.interface_type, ComplexityLevel.MEDIUM
        )
        core_hours = _COMPLEXITY_HOURS[complexity]

        components: List[TechnicalComponent] = [
            TechnicalComponent(
                name="Core Implementation",
                description=f"Core implementation of {capability.name}",
                effort_hours=core_hours,
                dependencies=[],
                complexity=complexity,
            ),
            TechnicalComponent(
                name="Unit Tests",
                description="Unit test suite for core functionality",
                effort_hours=4,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.LOW,
            ),
            TechnicalComponent(
                name="Integration Tests",
                description="Integration test suite verifying API compatibility",
                effort_hours=8,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.MEDIUM,
            ),
            TechnicalComponent(
                name="Documentation",
                description="API documentation and usage examples",
                effort_hours=4,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.LOW,
            ),
        ]

        # Category-specific extra components
        category = capability.category

        if category == CapabilityCategory.NETWORK_OPERATIONS:
            components.append(TechnicalComponent(
                name="HTTP Client",
                description="HTTP client implementation for network operations",
                effort_hours=16,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.HIGH,
            ))
            components.append(TechnicalComponent(
                name="Error Handling & Retry",
                description="Robust error handling and retry logic for network failures",
                effort_hours=8,
                dependencies=["HTTP Client"],
                complexity=ComplexityLevel.MEDIUM,
            ))

        elif category == CapabilityCategory.FILE_PROCESSING:
            components.append(TechnicalComponent(
                name="File Format Parser",
                description="Parser for file formats handled by this capability",
                effort_hours=12,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.MEDIUM,
            ))

        elif category == CapabilityCategory.SECURITY:
            components.append(TechnicalComponent(
                name="Security Audit",
                description="Security audit of the implementation",
                effort_hours=16,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.HIGH,
            ))
            components.append(TechnicalComponent(
                name="Cryptography Implementation",
                description="Cryptographic primitives and algorithms",
                effort_hours=24,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.VERY_HIGH,
            ))

        elif category == CapabilityCategory.DATA_TRANSFORMATION:
            components.append(TechnicalComponent(
                name="Data Schema Validation",
                description="Schema validation for input/output data",
                effort_hours=8,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.MEDIUM,
            ))

        elif category == CapabilityCategory.DATABASE:
            components.append(TechnicalComponent(
                name="Database Schema",
                description="Database schema design and migration scripts",
                effort_hours=8,
                dependencies=["Core Implementation"],
                complexity=ComplexityLevel.MEDIUM,
            ))
            components.append(TechnicalComponent(
                name="Query Optimization",
                description="Query optimization and indexing strategy",
                effort_hours=12,
                dependencies=["Database Schema"],
                complexity=ComplexityLevel.HIGH,
            ))

        return components

    # ------------------------------------------------------------------
    # Milestone creation (Requirement 5.7)
    # ------------------------------------------------------------------

    def _create_milestones(
        self,
        capability: Capability,
        components: List[TechnicalComponent],
        total_hours: int,
    ) -> List[Milestone]:
        """
        Create three milestone checkpoints for progress tracking.

        Milestones are placed at 33%, 66%, and 100% of total effort.
        """
        m1_hours = max(1, round(total_hours * 0.33))
        m2_hours = max(2, round(total_hours * 0.66))
        m3_hours = total_hours

        return [
            Milestone(
                name="Foundation",
                description=f"Core {capability.name} implementation complete with basic tests",
                deliverables=["Core implementation", "Basic unit tests"],
                estimated_hours_from_start=m1_hours,
            ),
            Milestone(
                name="Integration",
                description="Integration verified and API compatibility confirmed",
                deliverables=["Integration tests", "API compatibility verified"],
                estimated_hours_from_start=m2_hours,
            ),
            Milestone(
                name="Production Ready",
                description="Full test coverage, documentation, and performance benchmarks met",
                deliverables=[
                    "Full test coverage",
                    "Documentation complete",
                    "Performance benchmarks met",
                ],
                estimated_hours_from_start=m3_hours,
            ),
        ]

    # ------------------------------------------------------------------
    # Risk assessment (Requirement 5.6)
    # ------------------------------------------------------------------

    def _assess_risks(self, capability: Capability) -> List[Risk]:
        """
        Identify implementation risks and mitigation strategies.

        Always includes integration complexity and performance regression risks,
        plus category-specific risks.
        """
        risks: List[Risk] = [
            Risk(
                name="Integration Complexity",
                description=(
                    "Integrating with existing Janus codebase may require refactoring"
                ),
                level=RiskLevel.MEDIUM,
                mitigation="Conduct thorough code review before implementation",
                probability=0.4,
            ),
            Risk(
                name="Performance Regression",
                description=(
                    "Native implementation may not match external tool performance"
                ),
                level=RiskLevel.MEDIUM,
                mitigation=(
                    "Implement performance benchmarks and optimize critical paths"
                ),
                probability=0.3,
            ),
        ]

        category = capability.category

        if category == CapabilityCategory.SECURITY:
            risks.append(Risk(
                name="Security Vulnerability",
                description="Implementation may introduce security vulnerabilities",
                level=RiskLevel.HIGH,
                mitigation="Conduct security audit and penetration testing",
                probability=0.3,
            ))

        elif category == CapabilityCategory.NETWORK_OPERATIONS:
            risks.append(Risk(
                name="Network Reliability",
                description="Network operations may fail in various environments",
                level=RiskLevel.MEDIUM,
                mitigation=(
                    "Implement robust retry logic and fallback mechanisms"
                ),
                probability=0.5,
            ))

        elif category == CapabilityCategory.FILE_PROCESSING:
            risks.append(Risk(
                name="File Format Compatibility",
                description="May not support all file format variants",
                level=RiskLevel.LOW,
                mitigation=(
                    "Test with diverse file samples and implement format detection"
                ),
                probability=0.4,
            ))

        return risks

    # ------------------------------------------------------------------
    # Success criteria (Requirement 5.5)
    # ------------------------------------------------------------------

    def _define_success_criteria(self, capability: Capability) -> List[str]:
        """
        Define measurable success criteria for replacing the external dependency.

        Always includes core criteria plus category-specific ones.
        """
        criteria: List[str] = [
            f"All {capability.name} functionality replaces external dependency",
            "Unit test coverage >= 80%",
            "Integration tests pass in CI/CD pipeline",
            "Performance within 10% of external tool baseline",
        ]

        category = capability.category

        if category == CapabilityCategory.SECURITY:
            criteria.append(
                "Security audit completed with no critical findings"
            )
        elif category == CapabilityCategory.NETWORK_OPERATIONS:
            criteria.append(
                "Handles network failures gracefully with retry logic"
            )
        elif category == CapabilityCategory.FILE_PROCESSING:
            criteria.append(
                "Supports all file formats previously handled by external tool"
            )

        return criteria

    # ------------------------------------------------------------------
    # Testing requirements (Requirement 5.4)
    # ------------------------------------------------------------------

    def _define_testing_requirements(
        self, capability: Capability
    ) -> TestingRequirements:
        """
        Define comprehensive testing requirements for the implementation.

        Specifies unit, integration, and performance test requirements
        along with a coverage target.
        """
        return TestingRequirements(
            unit_tests=[
                "Test core functionality",
                "Test error handling",
                "Test edge cases",
            ],
            integration_tests=[
                "Test with real Janus codebase",
                "Test API compatibility",
            ],
            performance_tests=[
                "Benchmark against external tool",
                "Memory usage profiling",
            ],
            test_coverage_target=0.8,
        )
