"""
Property-based tests for the Capability Analyzer.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the CapabilityAnalyzerImpl.

Feature: janus-dependency-analyzer
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..analyzers.capability_analyzer import CapabilityAnalyzerImpl
from ..core.interfaces import AnalysisStrategy
from ..core.models import (
    Application,
    ApplicationMetadata,
    Capability,
    CapabilityCategory,
    InterfaceType,
    Parameter,
    Platform,
)


# ---------------------------------------------------------------------------
# MockStrategy helper
# ---------------------------------------------------------------------------

class MockStrategy(AnalysisStrategy):
    """
    A thread-safe mock analysis strategy that returns a pre-built list of
    capabilities.  can_analyze returns True iff the list is non-empty.
    """

    def __init__(self, capabilities: List[Capability], name: str = "mock_strategy",
                 confidence_factor: float = 1.0):
        self._capabilities = list(capabilities)  # copy for thread safety
        self._name = name
        self._confidence_factor = confidence_factor

    def can_analyze(self, app: Application) -> bool:
        return bool(self._capabilities)

    def extract_capabilities(self, app: Application) -> List[Capability]:
        return list(self._capabilities)  # return a copy

    def get_strategy_name(self) -> str:
        return self._name

    def get_confidence_factor(self) -> float:
        return self._confidence_factor


class ErrorStrategy(AnalysisStrategy):
    """A strategy whose extract_capabilities always raises RuntimeError."""

    def can_analyze(self, app: Application) -> bool:
        return True

    def extract_capabilities(self, app: Application) -> List[Capability]:
        raise RuntimeError("Simulated strategy failure")

    def get_strategy_name(self) -> str:
        return "error_strategy"

    def get_confidence_factor(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
    min_size=1,
    max_size=64,
)

path_text = st.text(
    alphabet=st.characters(
        blacklist_characters="\x00",
        whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Po"),
    ),
    min_size=1,
    max_size=128,
)


def parameter_strategy() -> st.SearchStrategy:
    """Generate a valid Parameter object."""
    return st.builds(
        Parameter,
        name=non_empty_text,
        type=st.sampled_from(["string", "int", "bool", "float", "path"]),
        description=st.text(min_size=0, max_size=128),
        required=st.booleans(),
        default_value=st.one_of(st.none(), non_empty_text),
    )


def capability_strategy(
    application_id: str = "test-app-id",
    confidence_score: st.SearchStrategy = st.floats(min_value=0.0, max_value=1.0,
                                                     allow_nan=False, allow_infinity=False),
) -> st.SearchStrategy:
    """Generate a valid Capability object."""
    return st.builds(
        Capability,
        id=st.builds(lambda: str(uuid.uuid4())),
        application_id=st.just(application_id),
        name=non_empty_text,
        category=st.sampled_from(list(CapabilityCategory)),
        description=st.text(min_size=0, max_size=256),
        interface_type=st.sampled_from(list(InterfaceType)),
        parameters=st.lists(parameter_strategy(), min_size=0, max_size=5),
        confidence_score=confidence_score,
        detection_method=st.text(min_size=0, max_size=64),
        examples=st.lists(st.text(min_size=1, max_size=64), min_size=0, max_size=5),
        supported_formats=st.lists(
            st.sampled_from(["json", "xml", "csv", "txt", "pdf", "png", "mp4"]),
            min_size=0,
            max_size=5,
        ),
    )


def capabilities_list_strategy(min_size: int = 0, max_size: int = 20) -> st.SearchStrategy:
    """Generate a list of Capability objects."""
    return st.lists(capability_strategy(), min_size=min_size, max_size=max_size)


def application_metadata_strategy() -> st.SearchStrategy:
    """Generate a valid ApplicationMetadata object."""
    return st.builds(
        ApplicationMetadata,
        vendor=st.one_of(st.none(), non_empty_text),
        description=st.one_of(st.none(), non_empty_text),
        file_size=st.integers(min_value=0, max_value=10_000_000),
        install_date=st.one_of(st.none(), st.just(datetime.now())),
        digital_signature=st.one_of(st.none(), non_empty_text),
    )


def application_strategy(is_accessible: bool = True) -> st.SearchStrategy:
    """Generate a valid Application object."""
    return st.builds(
        Application,
        id=st.builds(lambda: str(uuid.uuid4())),
        name=non_empty_text,
        version=st.text(alphabet="0123456789.", min_size=1, max_size=20),
        installation_path=path_text.map(lambda p: Path(f"/opt/{p}")),
        executable_path=path_text.map(lambda p: Path(f"/usr/bin/{p}")),
        platform=st.sampled_from(list(Platform)),
        metadata=application_metadata_strategy(),
        discovered_at=st.just(datetime.now()),
        is_accessible=st.just(is_accessible),
        access_error=st.just(None) if is_accessible else st.just("Permission denied"),
    )


def _make_analyzer_with_strategies(strategies: List[AnalysisStrategy]) -> CapabilityAnalyzerImpl:
    """Return a CapabilityAnalyzerImpl pre-wired with the given strategies only."""
    analyzer = CapabilityAnalyzerImpl.__new__(CapabilityAnalyzerImpl)
    analyzer.logger = MagicMock()
    analyzer._strategies = list(strategies)
    return analyzer


# ---------------------------------------------------------------------------
# Property 3: Comprehensive Capability Detection
# Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------

class TestComprehensiveCapabilityDetection:
    """
    **Validates: Requirements 2.1, 2.2**

    Property 3: Comprehensive Capability Detection
    For any discovered application, the Capability_Analyzer SHALL identify at
    least one capability OR report inability to analyze, and all detected
    capabilities SHALL be assigned to valid categories from the defined
    capability taxonomy.
    """

    @given(
        capabilities=capabilities_list_strategy(min_size=1, max_size=20),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_all_capabilities_have_valid_category(
        self, capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        Every capability returned by the analyzer has a valid CapabilityCategory enum member.
        """
        strategy = MockStrategy(capabilities)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        valid_categories = set(CapabilityCategory)
        for cap in result:
            assert cap.category in valid_categories, (
                f"Capability '{cap.name}' has invalid category: {cap.category!r}"
            )

    @given(
        capabilities=capabilities_list_strategy(min_size=1, max_size=20),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_all_capabilities_have_valid_interface_type(
        self, capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        Every capability returned by the analyzer has a valid InterfaceType enum member.
        """
        strategy = MockStrategy(capabilities)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        valid_interface_types = set(InterfaceType)
        for cap in result:
            assert cap.interface_type in valid_interface_types, (
                f"Capability '{cap.name}' has invalid interface_type: {cap.interface_type!r}"
            )

    @given(
        capabilities=capabilities_list_strategy(min_size=1, max_size=20),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_all_capabilities_have_non_empty_name(
        self, capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        Every capability returned by the analyzer has a non-empty name.
        """
        strategy = MockStrategy(capabilities)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        for cap in result:
            assert cap.name, (
                f"Capability with id={cap.id} has an empty name"
            )

    @given(
        capabilities=capabilities_list_strategy(min_size=1, max_size=20),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_all_capabilities_have_valid_confidence_score(
        self, capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        Every capability's confidence_score is between 0.0 and 1.0 inclusive.
        """
        strategy = MockStrategy(capabilities)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        for cap in result:
            assert 0.0 <= cap.confidence_score <= 1.0, (
                f"Capability '{cap.name}' has out-of-range confidence_score: "
                f"{cap.confidence_score}"
            )

    @given(
        capabilities=capabilities_list_strategy(min_size=0, max_size=20),
        app=application_strategy(is_accessible=False),
    )
    @settings(max_examples=25)
    def test_inaccessible_app_returns_empty(
        self, capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        When app.is_accessible is False, analyze_application always returns an empty list
        regardless of what strategies would return.
        """
        strategy = MockStrategy(capabilities)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        assert result == [], (
            f"Expected empty list for inaccessible app, got {len(result)} capabilities"
        )

    @given(
        good_capabilities=capabilities_list_strategy(min_size=1, max_size=10),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_analyzer_never_raises(
        self, good_capabilities: List[Capability], app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 3: Comprehensive Capability Detection
        analyze_application never raises an unhandled exception even when a strategy
        raises RuntimeError internally.  The analyzer catches the error and returns
        whatever other strategies produced.
        """
        good_strategy = MockStrategy(good_capabilities, name="good_strategy")
        bad_strategy = ErrorStrategy()
        analyzer = _make_analyzer_with_strategies([good_strategy, bad_strategy])

        try:
            result = analyzer.analyze_application(app)
        except Exception as exc:
            pytest.fail(
                f"analyze_application raised an unhandled exception when a strategy "
                f"threw RuntimeError: {exc!r}"
            )

        # The good strategy's capabilities should still be present (after merging)
        assert isinstance(result, list), "analyze_application must return a list"


# ---------------------------------------------------------------------------
# Property 4: Multi-Strategy Capability Analysis
# Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
# Validates: Requirements 2.3, 2.4, 2.5, 2.6
# ---------------------------------------------------------------------------

class TestMultiStrategyCapabilityAnalysis:
    """
    **Validates: Requirements 2.3, 2.4, 2.5, 2.6**

    Property 4: Multi-Strategy Capability Analysis
    For any application with available documentation, help text, CLI interfaces,
    or API endpoints, the Capability_Analyzer SHALL extract capabilities using
    appropriate analysis strategies and assign confidence scores between 0.0 and 1.0.
    """

    @given(
        base_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                             allow_infinity=False),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25, deadline=None)
    def test_confidence_scores_are_scaled_by_strategy_factor(
        self, base_score: float, app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When a strategy with confidence_factor=0.5 returns a capability with
        confidence_score=1.0, the final capability has confidence_score <= 0.5.
        """
        cap = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name="test-capability",
            category=CapabilityCategory.FILE_PROCESSING,
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=1.0,
        )
        strategy = MockStrategy([cap], name="half_confidence_strategy",
                                confidence_factor=0.5)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        assert len(result) >= 1
        for final_cap in result:
            if final_cap.name == "test-capability":
                assert final_cap.confidence_score <= 0.5 + 1e-9, (
                    f"Expected confidence_score <= 0.5 after 0.5 factor scaling, "
                    f"got {final_cap.confidence_score}"
                )

    @given(
        name=non_empty_text,
        category=st.sampled_from(list(CapabilityCategory)),
        interface_type=st.sampled_from(list(InterfaceType)),
        score_a=st.floats(min_value=0.1, max_value=1.0, allow_nan=False,
                          allow_infinity=False),
        score_b=st.floats(min_value=0.1, max_value=1.0, allow_nan=False,
                          allow_infinity=False),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_merge_deduplicates_same_name_category_interface(
        self,
        name: str,
        category: CapabilityCategory,
        interface_type: InterfaceType,
        score_a: float,
        score_b: float,
        app: Application,
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When two strategies return capabilities with identical (name, category,
        interface_type), merge_capabilities produces exactly one capability for
        that group.
        """
        cap_a = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=score_a,
            detection_method="strategy_a",
        )
        cap_b = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=score_b,
            detection_method="strategy_b",
        )
        analyzer = CapabilityAnalyzerImpl.__new__(CapabilityAnalyzerImpl)
        analyzer.logger = MagicMock()
        analyzer._strategies = []

        merged = analyzer.merge_capabilities([cap_a, cap_b])

        # There should be exactly one capability for this (name, category, interface_type)
        matching = [
            c for c in merged
            if c.name.lower().strip() == name.lower().strip()
            and c.category == category
            and c.interface_type == interface_type
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 merged capability for ({name!r}, {category}, "
            f"{interface_type}), got {len(matching)}"
        )

    @given(
        name=non_empty_text,
        category=st.sampled_from(list(CapabilityCategory)),
        interface_type=st.sampled_from(list(InterfaceType)),
        score_a=st.floats(min_value=0.1, max_value=0.9, allow_nan=False,
                          allow_infinity=False),
        score_b=st.floats(min_value=0.1, max_value=0.9, allow_nan=False,
                          allow_infinity=False),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_merge_boosts_confidence_for_multiple_detections(
        self,
        name: str,
        category: CapabilityCategory,
        interface_type: InterfaceType,
        score_a: float,
        score_b: float,
        app: Application,
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When two strategies detect the same capability, the merged confidence is
        higher than the average of the two individual scores (up to 1.0 cap).
        """
        avg = (score_a + score_b) / 2.0

        cap_a = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=score_a,
        )
        cap_b = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=score_b,
        )
        analyzer = CapabilityAnalyzerImpl.__new__(CapabilityAnalyzerImpl)
        analyzer.logger = MagicMock()
        analyzer._strategies = []

        merged = analyzer.merge_capabilities([cap_a, cap_b])

        matching = [
            c for c in merged
            if c.name.lower().strip() == name.lower().strip()
            and c.category == category
            and c.interface_type == interface_type
        ]
        assert len(matching) == 1
        merged_score = matching[0].confidence_score

        # Merged score must be >= average (boosted) and <= 1.0
        assert merged_score >= avg - 1e-9, (
            f"Merged confidence {merged_score} is less than average {avg}"
        )
        assert merged_score <= 1.0 + 1e-9, (
            f"Merged confidence {merged_score} exceeds 1.0"
        )

    @given(
        name=non_empty_text,
        category=st.sampled_from(list(CapabilityCategory)),
        interface_type=st.sampled_from(list(InterfaceType)),
        formats_a=st.lists(
            st.sampled_from(["json", "xml", "csv", "txt", "pdf"]),
            min_size=1, max_size=3, unique=True,
        ),
        formats_b=st.lists(
            st.sampled_from(["png", "mp4", "wav", "zip", "html"]),
            min_size=1, max_size=3, unique=True,
        ),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_merge_combines_supported_formats(
        self,
        name: str,
        category: CapabilityCategory,
        interface_type: InterfaceType,
        formats_a: List[str],
        formats_b: List[str],
        app: Application,
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When two capabilities for the same group have different supported_formats,
        the merged capability contains all formats from both.
        """
        cap_a = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=0.5,
            supported_formats=list(formats_a),
        )
        cap_b = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=0.5,
            supported_formats=list(formats_b),
        )
        analyzer = CapabilityAnalyzerImpl.__new__(CapabilityAnalyzerImpl)
        analyzer.logger = MagicMock()
        analyzer._strategies = []

        merged = analyzer.merge_capabilities([cap_a, cap_b])

        matching = [
            c for c in merged
            if c.name.lower().strip() == name.lower().strip()
            and c.category == category
            and c.interface_type == interface_type
        ]
        assert len(matching) == 1
        merged_formats = set(matching[0].supported_formats)

        for fmt in formats_a:
            assert fmt in merged_formats, (
                f"Format '{fmt}' from strategy A missing in merged capability"
            )
        for fmt in formats_b:
            assert fmt in merged_formats, (
                f"Format '{fmt}' from strategy B missing in merged capability"
            )

    @given(
        name=non_empty_text,
        category=st.sampled_from(list(CapabilityCategory)),
        interface_type=st.sampled_from(list(InterfaceType)),
        params_a=st.lists(
            st.builds(
                Parameter,
                name=st.sampled_from(["input", "output", "format", "verbose"]),
                type=st.just("string"),
                description=st.just("param a"),
                required=st.just(True),
            ),
            min_size=1, max_size=3,
        ),
        params_b=st.lists(
            st.builds(
                Parameter,
                name=st.sampled_from(["timeout", "retries", "encoding", "mode"]),
                type=st.just("string"),
                description=st.just("param b"),
                required=st.just(False),
            ),
            min_size=1, max_size=3,
        ),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_merge_combines_parameters(
        self,
        name: str,
        category: CapabilityCategory,
        interface_type: InterfaceType,
        params_a: List[Parameter],
        params_b: List[Parameter],
        app: Application,
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When two capabilities for the same group have different parameters, the
        merged capability contains all unique parameters.
        """
        cap_a = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=0.5,
            parameters=list(params_a),
        )
        cap_b = Capability(
            id=str(uuid.uuid4()),
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            confidence_score=0.5,
            parameters=list(params_b),
        )
        analyzer = CapabilityAnalyzerImpl.__new__(CapabilityAnalyzerImpl)
        analyzer.logger = MagicMock()
        analyzer._strategies = []

        merged = analyzer.merge_capabilities([cap_a, cap_b])

        matching = [
            c for c in merged
            if c.name.lower().strip() == name.lower().strip()
            and c.category == category
            and c.interface_type == interface_type
        ]
        assert len(matching) == 1
        merged_param_keys = {(p.name, p.type) for p in matching[0].parameters}

        for param in params_a:
            assert (param.name, param.type) in merged_param_keys, (
                f"Parameter ({param.name}, {param.type}) from strategy A missing "
                f"in merged capability"
            )
        for param in params_b:
            assert (param.name, param.type) in merged_param_keys, (
                f"Parameter ({param.name}, {param.type}) from strategy B missing "
                f"in merged capability"
            )

    @given(
        capabilities=capabilities_list_strategy(min_size=1, max_size=5),
        strategy_name=non_empty_text,
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_detection_method_tagged_correctly(
        self,
        capabilities: List[Capability],
        strategy_name: str,
        app: Application,
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        After analyze_application, each capability's detection_method contains the
        strategy name that produced it (before merging combines them).
        We use a single strategy so no merging occurs for unique capabilities.
        """
        # Give each capability a unique name so they won't be merged together
        unique_caps = []
        for i, cap in enumerate(capabilities):
            unique_cap = Capability(
                id=str(uuid.uuid4()),
                application_id=app.id,
                name=f"unique_cap_{i}_{cap.name}",
                category=cap.category,
                interface_type=cap.interface_type,
                confidence_score=cap.confidence_score,
                detection_method="",  # will be set by analyzer
            )
            unique_caps.append(unique_cap)

        strategy = MockStrategy(unique_caps, name=strategy_name)
        analyzer = _make_analyzer_with_strategies([strategy])

        result = analyzer.analyze_application(app)

        for cap in result:
            assert strategy_name in cap.detection_method, (
                f"Expected strategy name '{strategy_name}' in detection_method "
                f"'{cap.detection_method}' for capability '{cap.name}'"
            )

    @given(
        n_strategies=st.integers(min_value=1, max_value=5),
        app=application_strategy(is_accessible=True),
    )
    @settings(max_examples=25)
    def test_multiple_strategies_all_contribute(
        self, n_strategies: int, app: Application
    ) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When N strategies each return 1 unique capability (different names), the
        analyzer returns N capabilities total.
        """
        strategies = []
        for i in range(n_strategies):
            cap = Capability(
                id=str(uuid.uuid4()),
                application_id=app.id,
                name=f"unique_capability_from_strategy_{i}",
                category=CapabilityCategory.FILE_PROCESSING,
                interface_type=InterfaceType.COMMAND_LINE,
                confidence_score=0.8,
            )
            strategies.append(MockStrategy([cap], name=f"strategy_{i}"))

        analyzer = _make_analyzer_with_strategies(strategies)
        result = analyzer.analyze_application(app)

        assert len(result) == n_strategies, (
            f"Expected {n_strategies} capabilities from {n_strategies} strategies "
            f"with unique names, got {len(result)}"
        )

    @given(app=application_strategy(is_accessible=True))
    @settings(max_examples=25)
    def test_empty_strategies_returns_empty(self, app: Application) -> None:
        """
        # Feature: janus-dependency-analyzer, Property 4: Multi-Strategy Capability Analysis
        When no strategies can analyze the app, analyze_application returns an
        empty list.
        """
        analyzer = _make_analyzer_with_strategies([])
        result = analyzer.analyze_application(app)

        assert result == [], (
            f"Expected empty list when no strategies are registered, "
            f"got {len(result)} capabilities"
        )
