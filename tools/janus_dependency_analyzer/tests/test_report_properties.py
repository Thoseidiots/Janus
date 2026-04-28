"""
Property-based tests for Comprehensive Report Generation.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the ReportGenerator component.

# Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..core.models import (
    Capability,
    CapabilityCategory,
    DependencyMapping,
    InterfaceType,
    Platform,
    ScanResult,
    UsagePattern,
)
from ..reports.generator import ReportGenerator


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

scan_result_strategy = st.builds(
    ScanResult,
    platform=st.sampled_from(list(Platform)),
    total_applications=st.integers(min_value=0, max_value=1000),
    accessible_applications=st.integers(min_value=0, max_value=1000),
    scan_type=st.sampled_from(["full", "incremental"]),
    errors=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    warnings=st.lists(st.text(min_size=1, max_size=50), max_size=5),
)

capability_strategy = st.builds(
    Capability,
    name=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" _-",
        ),
    ),
    category=st.sampled_from(list(CapabilityCategory)),
    interface_type=st.sampled_from(list(InterfaceType)),
    confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    description=st.text(max_size=100),
    detection_method=st.just("test"),
    parameters=st.just([]),
    examples=st.just([]),
    supported_formats=st.just([]),
)

usage_pattern_strategy = st.builds(
    UsagePattern,
    invocation_method=st.sampled_from(["subprocess", "api_call", "library_import"]),
    frequency_per_day=st.integers(min_value=0, max_value=100),
    parameters=st.just([]),
    input_types=st.just([]),
    output_types=st.just([]),
)

dependency_strategy = st.builds(
    DependencyMapping,
    janus_component=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_",
        ),
    ),
    external_application=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
    ),
    usage_pattern=usage_pattern_strategy,
    frequency=st.integers(min_value=0, max_value=10000),
    last_used=st.just(datetime.now()),
    context=st.text(max_size=100),
    criticality=st.sampled_from(["low", "medium", "high", "critical"]),
    alternatives=st.just([]),
)


# ---------------------------------------------------------------------------
# Property 11: Comprehensive Report Generation
# Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
# Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
# ---------------------------------------------------------------------------


class TestComprehensiveReportGeneration:
    """
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**

    Property 11: Comprehensive Report Generation
    For any analysis results, the Report_Generator SHALL create accurate summary
    reports, detailed capability inventories, dependency usage reports,
    priority-ranked lists, and actionable project plans in multiple export
    formats with visual charts.
    """

    # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation

    @given(scan_results=st.lists(scan_result_strategy, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_summary_total_applications_is_sum(
        self, scan_results: list
    ) -> None:
        """
        total_applications in summary equals sum of all scan_result.total_applications.

        For any list of ScanResult objects, generate_summary_report SHALL return
        a dict whose total_applications equals the arithmetic sum of each
        scan_result's total_applications field.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_summary_report(scan_results)

        expected = sum(r.total_applications for r in scan_results)
        assert report["total_applications"] == expected, (
            f"total_applications {report['total_applications']} != "
            f"expected sum {expected}"
        )

    @given(capabilities=st.lists(capability_strategy, max_size=20))
    @settings(max_examples=25, deadline=None)
    def test_capability_inventory_total_matches_list(
        self, capabilities: list
    ) -> None:
        """
        total_capabilities equals len(capabilities) and len(report['capabilities']).

        For any list of Capability objects, generate_capability_inventory SHALL
        return a dict whose total_capabilities equals both the length of the
        input list and the length of the returned 'capabilities' list.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_capability_inventory(capabilities)

        assert report["total_capabilities"] == len(capabilities), (
            f"total_capabilities {report['total_capabilities']} != "
            f"len(capabilities) {len(capabilities)}"
        )
        assert report["total_capabilities"] == len(report["capabilities"]), (
            f"total_capabilities {report['total_capabilities']} != "
            f"len(report['capabilities']) {len(report['capabilities'])}"
        )

    @given(capabilities=st.lists(capability_strategy, min_size=1, max_size=20))
    @settings(max_examples=25, deadline=None)
    def test_capability_by_category_sums_to_total(
        self, capabilities: list
    ) -> None:
        """
        sum of by_category values equals total_capabilities.

        For any non-empty list of Capability objects, the sum of all values in
        the by_category dict SHALL equal total_capabilities.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_capability_inventory(capabilities)

        category_sum = sum(report["by_category"].values())
        assert category_sum == report["total_capabilities"], (
            f"sum of by_category values {category_sum} != "
            f"total_capabilities {report['total_capabilities']}"
        )

    @given(dependencies=st.lists(dependency_strategy, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_dependency_total_frequency_is_sum(
        self, dependencies: list
    ) -> None:
        """
        total_frequency equals sum of all dependency.frequency values.

        For any list of DependencyMapping objects, generate_dependency_report
        SHALL return a dict whose total_frequency equals the arithmetic sum of
        each dependency's frequency field.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_dependency_report(dependencies)

        expected = sum(d.frequency for d in dependencies)
        assert report["total_frequency"] == expected, (
            f"total_frequency {report['total_frequency']} != "
            f"expected sum {expected}"
        )

    @given(scan_results=st.lists(scan_result_strategy, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_summary_generated_at_is_valid_iso(
        self, scan_results: list
    ) -> None:
        """
        generated_at field can be parsed as ISO datetime.

        For any list of ScanResult objects, generate_summary_report SHALL return
        a dict whose generated_at value is a string parseable by
        datetime.fromisoformat().

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_summary_report(scan_results)

        assert "generated_at" in report, "generated_at key must be present in summary report"
        generated_at = report["generated_at"]
        assert isinstance(generated_at, str), (
            f"generated_at must be a string, got {type(generated_at)}"
        )
        # Must parse without raising
        parsed = datetime.fromisoformat(generated_at)
        assert isinstance(parsed, datetime), (
            f"generated_at '{generated_at}' did not parse to a datetime"
        )

    @given(capabilities=st.lists(capability_strategy, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_export_json_produces_valid_json(
        self, capabilities: list
    ) -> None:
        """
        export_report to JSON produces a file parseable by json.loads.

        For any list of Capability objects, exporting the capability inventory
        report to JSON SHALL produce a file whose contents are valid JSON.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_capability_inventory(capabilities)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            gen.export_report(report, "json", output_path)

            assert output_path.exists(), "JSON export file must exist"
            content = output_path.read_text(encoding="utf-8")
            # Must parse without raising
            parsed = json.loads(content)
            assert isinstance(parsed, dict), (
                f"Parsed JSON must be a dict, got {type(parsed)}"
            )
            assert parsed["total_capabilities"] == len(capabilities), (
                f"Parsed JSON total_capabilities {parsed['total_capabilities']} != "
                f"len(capabilities) {len(capabilities)}"
            )

    @given(capabilities=st.lists(capability_strategy, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_export_html_contains_html_tag(
        self, capabilities: list
    ) -> None:
        """
        export_report to HTML produces a file containing <html>.

        For any list of Capability objects, exporting the capability inventory
        report to HTML SHALL produce a file whose contents contain the string
        '<html>'.

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_capability_inventory(capabilities)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            gen.export_report(report, "html", output_path)

            assert output_path.exists(), "HTML export file must exist"
            content = output_path.read_text(encoding="utf-8")
            assert "<html>" in content, (
                "HTML export must contain '<html>' tag"
            )

    @given(capabilities=st.lists(capability_strategy, min_size=1, max_size=20))
    @settings(max_examples=25, deadline=None)
    def test_average_confidence_in_valid_range(
        self, capabilities: list
    ) -> None:
        """
        average_confidence is always in [0.0, 1.0].

        For any non-empty list of Capability objects whose confidence_score
        values are in [0.0, 1.0], generate_capability_inventory SHALL return
        a dict whose average_confidence is also in [0.0, 1.0].

        # Feature: janus-dependency-analyzer, Property 11: Comprehensive Report Generation
        """
        gen = ReportGenerator()
        report = gen.generate_capability_inventory(capabilities)

        avg = report["average_confidence"]
        assert isinstance(avg, float), (
            f"average_confidence must be a float, got {type(avg)}"
        )
        assert 0.0 <= avg <= 1.0, (
            f"average_confidence {avg} is not in [0.0, 1.0]"
        )
