"""
Unit tests for the ReportGenerator.

Tests cover summary reports, capability inventory, dependency reports,
export functionality (JSON, CSV, HTML), priority reports, and full reports.

Requirements: 7.1, 7.2, 7.3
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ..core.models import (
    Capability,
    CapabilityCategory,
    DependencyMapping,
    InterfaceType,
    Platform,
    ScanResult,
    UsagePattern,
)
from ..priority.engine import (
    AnalysisContext,
    PriorityEngine,
    PriorityScore,
    RankedCapability,
)
from ..reports.generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_scan_result(
    platform: Platform = Platform.LINUX,
    total_apps: int = 5,
    accessible_apps: int = 4,
    errors: list = None,
    warnings: list = None,
    scan_type: str = "full",
) -> ScanResult:
    result = ScanResult(
        platform=platform,
        total_applications=total_apps,
        accessible_applications=accessible_apps,
        scan_type=scan_type,
    )
    for e in (errors or []):
        result.add_error(e)
    for w in (warnings or []):
        result.add_warning(w)
    return result


def make_capability(
    name: str = "test_cap",
    category: CapabilityCategory = CapabilityCategory.FILE_PROCESSING,
    interface_type: InterfaceType = InterfaceType.COMMAND_LINE,
    confidence: float = 0.8,
    application_id: str = "app-1",
) -> Capability:
    return Capability(
        name=name,
        category=category,
        interface_type=interface_type,
        confidence_score=confidence,
        application_id=application_id,
        description="A test capability",
        detection_method="test",
    )


def make_dependency(
    janus_component: str = "core",
    external_application: str = "ffmpeg",
    frequency: int = 10,
    criticality: str = "high",
    invocation_method: str = "subprocess",
) -> DependencyMapping:
    return DependencyMapping(
        janus_component=janus_component,
        external_application=external_application,
        usage_pattern=UsagePattern(invocation_method=invocation_method),
        frequency=frequency,
        last_used=datetime.now(),
        context="video processing",
        criticality=criticality,
    )


def make_ranked_capability(rank: int = 1, score: float = 0.75) -> RankedCapability:
    cap = make_capability(name=f"cap_rank_{rank}")
    priority_score = PriorityScore(
        capability_id=cap.id,
        capability_name=cap.name,
        usage_frequency_score=score,
        implementation_complexity_score=score,
        security_benefit_score=score,
        performance_impact_score=score,
        maintenance_burden_score=score,
        total_score=score,
        justification=f"Priority score {score:.2f}: test justification",
        rank=rank,
    )
    return RankedCapability(capability=cap, priority_score=priority_score, rank=rank)


# ---------------------------------------------------------------------------
# generate_summary_report
# ---------------------------------------------------------------------------


class TestGenerateSummaryReport:
    def test_empty_list(self):
        gen = ReportGenerator()
        report = gen.generate_summary_report([])

        assert report["total_scans"] == 0
        assert report["total_applications"] == 0
        assert report["accessible_applications"] == 0
        assert report["total_errors"] == 0
        assert report["total_warnings"] == 0
        assert report["platforms"] == []
        assert report["scan_types"] == []
        assert "generated_at" in report

    def test_single_scan(self):
        gen = ReportGenerator()
        scan = make_scan_result(
            platform=Platform.LINUX,
            total_apps=10,
            accessible_apps=8,
            errors=["err1"],
            warnings=["warn1", "warn2"],
            scan_type="full",
        )
        report = gen.generate_summary_report([scan])

        assert report["total_scans"] == 1
        assert report["total_applications"] == 10
        assert report["accessible_applications"] == 8
        assert report["total_errors"] == 1
        assert report["total_warnings"] == 2
        assert "linux" in report["platforms"]
        assert "full" in report["scan_types"]

    def test_multiple_scans(self):
        gen = ReportGenerator()
        scans = [
            make_scan_result(Platform.LINUX, 10, 8, ["e1"], ["w1"], "full"),
            make_scan_result(Platform.WINDOWS, 5, 5, [], ["w2"], "incremental"),
            make_scan_result(Platform.LINUX, 3, 2, ["e2", "e3"], [], "full"),
        ]
        report = gen.generate_summary_report(scans)

        assert report["total_scans"] == 3
        assert report["total_applications"] == 18
        assert report["accessible_applications"] == 15
        assert report["total_errors"] == 3
        assert report["total_warnings"] == 2
        assert set(report["platforms"]) == {"linux", "windows"}
        assert set(report["scan_types"]) == {"full", "incremental"}

    def test_generated_at_is_iso_format(self):
        gen = ReportGenerator()
        report = gen.generate_summary_report([])
        # Should parse without error
        datetime.fromisoformat(report["generated_at"])


# ---------------------------------------------------------------------------
# generate_capability_inventory
# ---------------------------------------------------------------------------


class TestGenerateCapabilityInventory:
    def test_empty_list(self):
        gen = ReportGenerator()
        report = gen.generate_capability_inventory([])

        assert report["total_capabilities"] == 0
        assert report["by_category"] == {}
        assert report["by_interface_type"] == {}
        assert report["average_confidence"] == 0.0
        assert report["capabilities"] == []
        assert "generated_at" in report

    def test_mixed_categories(self):
        gen = ReportGenerator()
        caps = [
            make_capability("cap1", CapabilityCategory.FILE_PROCESSING, InterfaceType.COMMAND_LINE, 0.9),
            make_capability("cap2", CapabilityCategory.SECURITY, InterfaceType.REST_API, 0.7),
            make_capability("cap3", CapabilityCategory.FILE_PROCESSING, InterfaceType.LIBRARY, 0.5),
        ]
        report = gen.generate_capability_inventory(caps)

        assert report["total_capabilities"] == 3
        assert report["by_category"]["file_processing"] == 2
        assert report["by_category"]["security"] == 1
        assert report["by_interface_type"]["command_line"] == 1
        assert report["by_interface_type"]["rest_api"] == 1
        assert report["by_interface_type"]["library"] == 1
        assert abs(report["average_confidence"] - (0.9 + 0.7 + 0.5) / 3) < 1e-9

    def test_capability_dict_fields(self):
        gen = ReportGenerator()
        cap = make_capability("my_cap", application_id="app-42")
        report = gen.generate_capability_inventory([cap])

        cap_dict = report["capabilities"][0]
        assert cap_dict["id"] == cap.id
        assert cap_dict["name"] == "my_cap"
        assert cap_dict["category"] == cap.category.value
        assert cap_dict["interface_type"] == cap.interface_type.value
        assert cap_dict["application_id"] == "app-42"
        assert "confidence_score" in cap_dict
        assert "detection_method" in cap_dict
        assert "description" in cap_dict


# ---------------------------------------------------------------------------
# generate_dependency_report
# ---------------------------------------------------------------------------


class TestGenerateDependencyReport:
    def test_empty_list(self):
        gen = ReportGenerator()
        report = gen.generate_dependency_report([])

        assert report["total_dependencies"] == 0
        assert report["by_component"] == {}
        assert report["by_application"] == {}
        assert report["by_criticality"] == {}
        assert report["total_frequency"] == 0
        assert report["dependencies"] == []
        assert "generated_at" in report

    def test_multiple_dependencies(self):
        gen = ReportGenerator()
        deps = [
            make_dependency("core", "ffmpeg", 10, "high", "subprocess"),
            make_dependency("core", "git", 5, "medium", "subprocess"),
            make_dependency("scanner", "ffmpeg", 3, "low", "api_call"),
        ]
        report = gen.generate_dependency_report(deps)

        assert report["total_dependencies"] == 3
        assert report["by_component"]["core"] == 2
        assert report["by_component"]["scanner"] == 1
        assert report["by_application"]["ffmpeg"] == 2
        assert report["by_application"]["git"] == 1
        assert report["by_criticality"]["high"] == 1
        assert report["by_criticality"]["medium"] == 1
        assert report["by_criticality"]["low"] == 1
        assert report["total_frequency"] == 18

    def test_dependency_dict_fields(self):
        gen = ReportGenerator()
        dep = make_dependency("comp", "tool", 7, "critical", "subprocess")
        report = gen.generate_dependency_report([dep])

        dep_dict = report["dependencies"][0]
        assert dep_dict["janus_component"] == "comp"
        assert dep_dict["external_application"] == "tool"
        assert dep_dict["frequency"] == 7
        assert dep_dict["criticality"] == "critical"
        assert dep_dict["invocation_method"] == "subprocess"
        assert "context" in dep_dict


# ---------------------------------------------------------------------------
# export_report — JSON
# ---------------------------------------------------------------------------


class TestExportReportJson:
    def test_json_creates_valid_file(self, tmp_path):
        gen = ReportGenerator()
        data = {"key": "value", "count": 42, "items": [1, 2, 3]}
        output = tmp_path / "report.json"
        gen.export_report(data, "json", output)

        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["key"] == "value"
        assert loaded["count"] == 42
        assert loaded["items"] == [1, 2, 3]

    def test_json_is_indented(self, tmp_path):
        gen = ReportGenerator()
        data = {"a": 1}
        output = tmp_path / "report.json"
        gen.export_report(data, "json", output)

        content = output.read_text()
        # Indented JSON has newlines
        assert "\n" in content


# ---------------------------------------------------------------------------
# export_report — CSV
# ---------------------------------------------------------------------------


class TestExportReportCsv:
    def test_csv_creates_file(self, tmp_path):
        gen = ReportGenerator()
        data = {
            "total": 2,
            "capabilities": [
                {"name": "cap1", "category": "security"},
                {"name": "cap2", "category": "database"},
            ],
        }
        output = tmp_path / "report.csv"
        gen.export_report(data, "csv", output)

        assert output.exists()

    def test_csv_with_capabilities_list(self, tmp_path):
        gen = ReportGenerator()
        data = {
            "capabilities": [
                {"name": "cap1", "score": 0.9},
                {"name": "cap2", "score": 0.5},
            ]
        }
        output = tmp_path / "caps.csv"
        gen.export_report(data, "csv", output)

        content = output.read_text()
        assert "name" in content
        assert "cap1" in content
        assert "cap2" in content

    def test_csv_scalar_fallback(self, tmp_path):
        gen = ReportGenerator()
        data = {"total_scans": 3, "total_errors": 1}
        output = tmp_path / "scalar.csv"
        gen.export_report(data, "csv", output)

        assert output.exists()
        content = output.read_text()
        assert "total_scans" in content


# ---------------------------------------------------------------------------
# export_report — HTML
# ---------------------------------------------------------------------------


class TestExportReportHtml:
    def test_html_creates_file_with_html_tag(self, tmp_path):
        gen = ReportGenerator()
        data = {"total": 5, "items": [{"name": "a"}, {"name": "b"}]}
        output = tmp_path / "report.html"
        gen.export_report(data, "html", output)

        assert output.exists()
        content = output.read_text()
        assert "<html>" in content

    def test_html_contains_table_for_list(self, tmp_path):
        gen = ReportGenerator()
        data = {"capabilities": [{"name": "cap1", "score": 0.9}]}
        output = tmp_path / "report.html"
        gen.export_report(data, "html", output)

        content = output.read_text()
        assert "<table" in content
        assert "cap1" in content

    def test_html_contains_p_for_scalar(self, tmp_path):
        gen = ReportGenerator()
        data = {"total_scans": 3}
        output = tmp_path / "report.html"
        gen.export_report(data, "html", output)

        content = output.read_text()
        assert "<p>" in content or "<p " in content


# ---------------------------------------------------------------------------
# export_report — unsupported format
# ---------------------------------------------------------------------------


class TestExportReportUnsupportedFormat:
    def test_raises_value_error(self, tmp_path):
        gen = ReportGenerator()
        data = {"key": "value"}
        output = tmp_path / "report.xml"

        with pytest.raises(ValueError, match="Unsupported report format"):
            gen.export_report(data, "xml", output)

    def test_raises_for_yaml(self, tmp_path):
        gen = ReportGenerator()
        with pytest.raises(ValueError):
            gen.export_report({}, "yaml", tmp_path / "out.yaml")


# ---------------------------------------------------------------------------
# generate_priority_report
# ---------------------------------------------------------------------------


class TestGeneratePriorityReport:
    def test_empty_list(self):
        gen = ReportGenerator()
        report = gen.generate_priority_report([])

        assert report["total_ranked"] == 0
        assert report["top_10"] == []
        assert report["average_score"] == 0.0
        assert "generated_at" in report

    def test_with_ranked_capabilities(self):
        gen = ReportGenerator()
        ranked = [make_ranked_capability(rank=i + 1, score=0.9 - i * 0.1) for i in range(5)]
        report = gen.generate_priority_report(ranked)

        assert report["total_ranked"] == 5
        assert len(report["top_10"]) == 5
        assert abs(report["average_score"] - sum(0.9 - i * 0.1 for i in range(5)) / 5) < 1e-9

    def test_top_10_capped_at_10(self):
        gen = ReportGenerator()
        ranked = [make_ranked_capability(rank=i + 1, score=0.5) for i in range(15)]
        report = gen.generate_priority_report(ranked)

        assert report["total_ranked"] == 15
        assert len(report["top_10"]) == 10

    def test_top_10_dict_fields(self):
        gen = ReportGenerator()
        ranked = [make_ranked_capability(rank=1, score=0.8)]
        report = gen.generate_priority_report(ranked)

        entry = report["top_10"][0]
        assert "rank" in entry
        assert "capability_name" in entry
        assert "category" in entry
        assert "total_score" in entry
        assert "justification" in entry

    def test_top_10_ordered_by_rank(self):
        gen = ReportGenerator()
        # Create in reverse order to test sorting
        ranked = [make_ranked_capability(rank=i + 1, score=0.9 - i * 0.05) for i in range(5)]
        import random
        random.shuffle(ranked)
        report = gen.generate_priority_report(ranked)

        ranks = [entry["rank"] for entry in report["top_10"]]
        assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# generate_full_report
# ---------------------------------------------------------------------------


class TestGenerateFullReport:
    def test_has_all_expected_keys(self):
        gen = ReportGenerator()
        scan = make_scan_result()
        cap = make_capability()
        dep = make_dependency()
        ranked = [make_ranked_capability()]

        report = gen.generate_full_report([scan], [cap], [dep], ranked)

        assert "summary" in report
        assert "capability_inventory" in report
        assert "dependency_report" in report
        assert "priority_report" in report
        assert "generated_at" in report

    def test_sub_reports_are_dicts(self):
        gen = ReportGenerator()
        report = gen.generate_full_report([], [], [], [])

        assert isinstance(report["summary"], dict)
        assert isinstance(report["capability_inventory"], dict)
        assert isinstance(report["dependency_report"], dict)
        assert isinstance(report["priority_report"], dict)

    def test_full_report_with_data(self):
        gen = ReportGenerator()
        scans = [make_scan_result(Platform.MACOS, 20, 18)]
        caps = [make_capability("cap_a"), make_capability("cap_b")]
        deps = [make_dependency("comp_x", "tool_y", 15, "critical")]
        ranked = [make_ranked_capability(1, 0.95), make_ranked_capability(2, 0.80)]

        report = gen.generate_full_report(scans, caps, deps, ranked)

        assert report["summary"]["total_scans"] == 1
        assert report["capability_inventory"]["total_capabilities"] == 2
        assert report["dependency_report"]["total_dependencies"] == 1
        assert report["priority_report"]["total_ranked"] == 2


# ---------------------------------------------------------------------------
# generate_roadmap_report
# ---------------------------------------------------------------------------


from ..roadmap.generator import RoadmapGenerator, ImplementationRoadmap, RiskLevel


def make_roadmap(cap_name: str = "test_cap") -> ImplementationRoadmap:
    """Helper to create a test ImplementationRoadmap via RoadmapGenerator."""
    cap = make_capability(
        name=cap_name,
        category=CapabilityCategory.FILE_PROCESSING,
        interface_type=InterfaceType.COMMAND_LINE,
    )
    return RoadmapGenerator().generate(cap)


class TestGenerateRoadmapReport:
    def test_empty_list_returns_zeros(self):
        gen = ReportGenerator()
        report = gen.generate_roadmap_report([])

        assert report["total_roadmaps"] == 0
        assert report["total_estimated_hours"] == 0
        assert report["total_estimated_weeks"] == 0.0
        assert report["by_capability"] == []
        assert report["all_milestones"] == []
        assert report["risk_summary"] == {"low": 0, "medium": 0, "high": 0}
        assert "generated_at" in report

    def test_single_roadmap_correct_fields(self):
        gen = ReportGenerator()
        roadmap = make_roadmap("my_cap")
        report = gen.generate_roadmap_report([roadmap])

        assert report["total_roadmaps"] == 1
        assert report["total_estimated_hours"] == roadmap.estimated_effort_hours
        assert abs(report["total_estimated_weeks"] - roadmap.estimated_effort_hours / 40.0) < 1e-9

        assert len(report["by_capability"]) == 1
        entry = report["by_capability"][0]
        assert entry["capability_id"] == roadmap.capability_id
        assert entry["capability_name"] == roadmap.capability_name
        assert entry["estimated_effort_hours"] == roadmap.estimated_effort_hours
        assert entry["component_count"] == len(roadmap.technical_components)
        assert entry["milestone_count"] == len(roadmap.milestones)
        assert entry["risk_count"] == len(roadmap.risks)
        assert entry["success_criteria_count"] == len(roadmap.success_criteria)
        assert isinstance(entry["high_risks"], list)

    def test_multiple_roadmaps_totals_summed(self):
        gen = ReportGenerator()
        roadmap1 = make_roadmap("cap_a")
        roadmap2 = make_roadmap("cap_b")
        report = gen.generate_roadmap_report([roadmap1, roadmap2])

        expected_hours = roadmap1.estimated_effort_hours + roadmap2.estimated_effort_hours
        assert report["total_roadmaps"] == 2
        assert report["total_estimated_hours"] == expected_hours
        assert abs(report["total_estimated_weeks"] - expected_hours / 40.0) < 1e-9
        assert len(report["by_capability"]) == 2

    def test_risk_summary_counts_by_level(self):
        gen = ReportGenerator()
        # SECURITY category adds a HIGH risk; FILE_PROCESSING adds a LOW risk
        cap_security = make_capability(
            name="sec_cap",
            category=CapabilityCategory.SECURITY,
            interface_type=InterfaceType.COMMAND_LINE,
        )
        cap_file = make_capability(
            name="file_cap",
            category=CapabilityCategory.FILE_PROCESSING,
            interface_type=InterfaceType.COMMAND_LINE,
        )
        roadmap_sec = RoadmapGenerator().generate(cap_security)
        roadmap_file = RoadmapGenerator().generate(cap_file)

        report = gen.generate_roadmap_report([roadmap_sec, roadmap_file])
        risk_summary = report["risk_summary"]

        # Count expected risks manually
        all_risks = roadmap_sec.risks + roadmap_file.risks
        expected_high = sum(1 for r in all_risks if r.level == RiskLevel.HIGH)
        expected_medium = sum(1 for r in all_risks if r.level == RiskLevel.MEDIUM)
        expected_low = sum(1 for r in all_risks if r.level == RiskLevel.LOW)

        assert risk_summary["high"] == expected_high
        assert risk_summary["medium"] == expected_medium
        assert risk_summary["low"] == expected_low

    def test_all_milestones_contains_entries_from_all_roadmaps(self):
        gen = ReportGenerator()
        roadmap1 = make_roadmap("cap_x")
        roadmap2 = make_roadmap("cap_y")
        report = gen.generate_roadmap_report([roadmap1, roadmap2])

        total_milestones = len(roadmap1.milestones) + len(roadmap2.milestones)
        assert len(report["all_milestones"]) == total_milestones

        # Each milestone entry has the required fields
        for entry in report["all_milestones"]:
            assert "capability_name" in entry
            assert "milestone_name" in entry
            assert "description" in entry
            assert "estimated_hours_from_start" in entry
            assert "deliverable_count" in entry

        # Capability names are present
        cap_names = {e["capability_name"] for e in report["all_milestones"]}
        assert "cap_x" in cap_names
        assert "cap_y" in cap_names

    def test_high_risks_list_contains_only_high_level_risk_names(self):
        gen = ReportGenerator()
        cap = make_capability(
            name="sec_cap",
            category=CapabilityCategory.SECURITY,
            interface_type=InterfaceType.COMMAND_LINE,
        )
        roadmap = RoadmapGenerator().generate(cap)
        report = gen.generate_roadmap_report([roadmap])

        entry = report["by_capability"][0]
        high_risk_names = entry["high_risks"]

        # Verify only HIGH-level risks are included
        expected_high_names = [r.name for r in roadmap.risks if r.level == RiskLevel.HIGH]
        assert sorted(high_risk_names) == sorted(expected_high_names)

    def test_generated_at_is_iso_format(self):
        gen = ReportGenerator()
        report = gen.generate_roadmap_report([])
        datetime.fromisoformat(report["generated_at"])


# ---------------------------------------------------------------------------
# generate_full_report — roadmap integration
# ---------------------------------------------------------------------------


class TestGenerateFullReportWithRoadmap:
    def test_full_report_with_roadmaps_includes_roadmap_report_key(self):
        gen = ReportGenerator()
        roadmap = make_roadmap("cap_full")
        report = gen.generate_full_report([], [], [], [], roadmaps=[roadmap])

        assert "roadmap_report" in report
        assert isinstance(report["roadmap_report"], dict)
        assert report["roadmap_report"]["total_roadmaps"] == 1

    def test_full_report_without_roadmaps_default_still_works(self):
        gen = ReportGenerator()
        report = gen.generate_full_report([], [], [], [])

        assert "roadmap_report" in report
        assert report["roadmap_report"]["total_roadmaps"] == 0
        assert report["roadmap_report"]["total_estimated_hours"] == 0

    def test_full_report_has_all_expected_keys_with_roadmap(self):
        gen = ReportGenerator()
        scan = make_scan_result()
        cap = make_capability()
        dep = make_dependency()
        ranked = [make_ranked_capability()]
        roadmap = make_roadmap()

        report = gen.generate_full_report([scan], [cap], [dep], ranked, roadmaps=[roadmap])

        assert "summary" in report
        assert "capability_inventory" in report
        assert "dependency_report" in report
        assert "priority_report" in report
        assert "roadmap_report" in report
        assert "generated_at" in report


# ---------------------------------------------------------------------------
# TestMultiFormatExport — task 11.3
# ---------------------------------------------------------------------------


class TestMultiFormatExport:
    """Tests for export_all_formats and enhanced HTML output."""

    def test_export_all_formats_creates_all_files(self, tmp_path):
        """All three format files should be created on disk."""
        gen = ReportGenerator()
        data = {"total": 3, "items": [{"name": "a"}, {"name": "b"}]}
        gen.export_all_formats(data, tmp_path)

        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.csv").exists()
        assert (tmp_path / "report.html").exists()

    def test_export_all_formats_returns_correct_paths(self, tmp_path):
        """Returned dict must have json/csv/html keys pointing to the right paths."""
        gen = ReportGenerator()
        data = {"total": 1}
        result = gen.export_all_formats(data, tmp_path)

        assert set(result.keys()) == {"json", "csv", "html"}
        assert result["json"] == tmp_path / "report.json"
        assert result["csv"] == tmp_path / "report.csv"
        assert result["html"] == tmp_path / "report.html"

    def test_export_all_formats_custom_filename(self, tmp_path):
        """Custom base_filename should be reflected in all output paths."""
        gen = ReportGenerator()
        data = {"x": 1}
        result = gen.export_all_formats(data, tmp_path, base_filename="my_report")

        assert result["json"] == tmp_path / "my_report.json"
        assert result["csv"] == tmp_path / "my_report.csv"
        assert result["html"] == tmp_path / "my_report.html"
        for path in result.values():
            assert path.exists()

    def test_html_contains_svg_for_dict_values(self, tmp_path):
        """HTML output should contain an <svg element when a dict-of-ints is present."""
        gen = ReportGenerator()
        data = {
            "by_category": {"security": 3, "file_processing": 5, "database": 1},
        }
        output = tmp_path / "report.html"
        gen.export_report(data, "html", output)

        content = output.read_text()
        assert "<svg" in content

    def test_html_executive_summary_rendered(self, tmp_path):
        """HTML should contain the executive summary div when 'summary' key is present."""
        gen = ReportGenerator()
        data = {
            "summary": {"total_scans": 2, "total_applications": 10},
            "other": "value",
        }
        output = tmp_path / "report.html"
        gen.export_report(data, "html", output)

        content = output.read_text()
        assert "Executive Summary" in content
        assert "background:#f0f4f8" in content
        assert "total_scans" in content
