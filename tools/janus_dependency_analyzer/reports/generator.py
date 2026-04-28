"""
Report Generator for the Janus Dependency Analyzer.

This module implements the ReportGenerator, which produces comprehensive
analysis reports in multiple formats (JSON, CSV, HTML) covering scan
summaries, capability inventories, dependency mappings, and priority rankings.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.interfaces import ReportGenerator as AbstractReportGenerator
from ..core.models import (
    Capability,
    DependencyMapping,
    ScanResult,
)
from ..priority.engine import RankedCapability
from ..roadmap.generator import ImplementationRoadmap, RiskLevel


class ReportGenerator(AbstractReportGenerator):
    """
    Concrete implementation of the ReportGenerator interface.

    Generates comprehensive reports from scan results, capability inventories,
    dependency mappings, and priority rankings. Supports export to JSON, CSV,
    and HTML formats.
    """

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def generate_summary_report(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """
        Generate a summary report of scan results.

        Args:
            scan_results: Results from system scans

        Returns:
            Dict with total_scans, total_applications, accessible_applications,
            total_errors, total_warnings, platforms, scan_types, generated_at.
        """
        total_applications = sum(r.total_applications for r in scan_results)
        accessible_applications = sum(r.accessible_applications for r in scan_results)
        total_errors = sum(len(r.errors) for r in scan_results)
        total_warnings = sum(len(r.warnings) for r in scan_results)

        platforms = list({r.platform.value for r in scan_results})
        scan_types = list({r.scan_type for r in scan_results})

        return {
            "total_scans": len(scan_results),
            "total_applications": total_applications,
            "accessible_applications": accessible_applications,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "platforms": sorted(platforms),
            "scan_types": sorted(scan_types),
            "generated_at": datetime.now().isoformat(),
        }

    def generate_capability_inventory(self, capabilities: List[Capability]) -> Dict[str, Any]:
        """
        Generate a detailed capability inventory report.

        Args:
            capabilities: Discovered capabilities

        Returns:
            Dict with total_capabilities, by_category, by_interface_type,
            average_confidence, capabilities list, generated_at.
        """
        by_category: Dict[str, int] = {}
        by_interface_type: Dict[str, int] = {}

        for cap in capabilities:
            cat_key = cap.category.value
            by_category[cat_key] = by_category.get(cat_key, 0) + 1

            iface_key = cap.interface_type.value
            by_interface_type[iface_key] = by_interface_type.get(iface_key, 0) + 1

        average_confidence = (
            sum(c.confidence_score for c in capabilities) / len(capabilities)
            if capabilities
            else 0.0
        )

        cap_list = [
            {
                "id": cap.id,
                "name": cap.name,
                "category": cap.category.value,
                "interface_type": cap.interface_type.value,
                "description": cap.description,
                "confidence_score": cap.confidence_score,
                "application_id": cap.application_id,
                "detection_method": cap.detection_method,
            }
            for cap in capabilities
        ]

        return {
            "total_capabilities": len(capabilities),
            "by_category": by_category,
            "by_interface_type": by_interface_type,
            "average_confidence": average_confidence,
            "capabilities": cap_list,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_dependency_report(self, dependencies: List[DependencyMapping]) -> Dict[str, Any]:
        """
        Generate a dependency usage report.

        Args:
            dependencies: Current dependency mappings

        Returns:
            Dict with total_dependencies, by_component, by_application,
            by_criticality, total_frequency, dependencies list, generated_at.
        """
        by_component: Dict[str, int] = {}
        by_application: Dict[str, int] = {}
        by_criticality: Dict[str, int] = {}
        total_frequency = 0

        for dep in dependencies:
            by_component[dep.janus_component] = by_component.get(dep.janus_component, 0) + 1
            by_application[dep.external_application] = (
                by_application.get(dep.external_application, 0) + 1
            )
            by_criticality[dep.criticality] = by_criticality.get(dep.criticality, 0) + 1
            total_frequency += dep.frequency

        dep_list = [
            {
                "janus_component": dep.janus_component,
                "external_application": dep.external_application,
                "frequency": dep.frequency,
                "criticality": dep.criticality,
                "context": dep.context,
                "invocation_method": dep.usage_pattern.invocation_method,
            }
            for dep in dependencies
        ]

        return {
            "total_dependencies": len(dependencies),
            "by_component": by_component,
            "by_application": by_application,
            "by_criticality": by_criticality,
            "total_frequency": total_frequency,
            "dependencies": dep_list,
            "generated_at": datetime.now().isoformat(),
        }

    def export_report(
        self,
        report_data: Dict[str, Any],
        format: str,
        output_path: Path,
    ) -> None:
        """
        Export report data to the specified format and path.

        Args:
            report_data: Report data to export
            format: Export format — "json", "csv", or "html"
            output_path: Path to write the exported report

        Raises:
            ValueError: If the format is not supported
        """
        fmt = format.lower()
        if fmt == "json":
            self._export_json(report_data, output_path)
        elif fmt == "csv":
            self._export_csv(report_data, output_path)
        elif fmt == "html":
            self._export_html(report_data, output_path)
        else:
            raise ValueError(
                f"Unsupported report format: '{format}'. "
                "Supported formats are: json, csv, html"
            )

    # ------------------------------------------------------------------
    # Convenience methods (not in abstract base)
    # ------------------------------------------------------------------

    def generate_priority_report(
        self, ranked_capabilities: List[RankedCapability]
    ) -> Dict[str, Any]:
        """
        Generate a priority ranking report.

        Args:
            ranked_capabilities: Capabilities with priority scores and ranks

        Returns:
            Dict with total_ranked, top_10, average_score, generated_at.
        """
        total_ranked = len(ranked_capabilities)

        average_score = (
            sum(rc.priority_score.total_score for rc in ranked_capabilities) / total_ranked
            if ranked_capabilities
            else 0.0
        )

        # Sort by rank ascending and take top 10
        sorted_ranked = sorted(ranked_capabilities, key=lambda rc: rc.rank)
        top_10 = [
            {
                "rank": rc.rank,
                "capability_name": rc.capability.name,
                "category": rc.capability.category.value,
                "total_score": rc.priority_score.total_score,
                "justification": rc.priority_score.justification,
            }
            for rc in sorted_ranked[:10]
        ]

        return {
            "total_ranked": total_ranked,
            "top_10": top_10,
            "average_score": average_score,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_roadmap_report(
        self, roadmaps: List[ImplementationRoadmap]
    ) -> Dict[str, Any]:
        """
        Compile implementation roadmaps into an actionable project plan.

        Args:
            roadmaps: List of ImplementationRoadmap objects

        Returns:
            Dict with total_roadmaps, total_estimated_hours, total_estimated_weeks,
            by_capability, all_milestones, risk_summary, generated_at.
        """
        total_estimated_hours = sum(r.estimated_effort_hours for r in roadmaps)
        total_estimated_weeks = total_estimated_hours / 40.0

        by_capability = []
        for roadmap in roadmaps:
            high_risks = [
                risk.name
                for risk in roadmap.risks
                if risk.level == RiskLevel.HIGH
            ]
            by_capability.append({
                "capability_id": roadmap.capability_id,
                "capability_name": roadmap.capability_name,
                "estimated_effort_hours": roadmap.estimated_effort_hours,
                "component_count": len(roadmap.technical_components),
                "milestone_count": len(roadmap.milestones),
                "risk_count": len(roadmap.risks),
                "success_criteria_count": len(roadmap.success_criteria),
                "high_risks": high_risks,
            })

        all_milestones = []
        for roadmap in roadmaps:
            for milestone in roadmap.milestones:
                all_milestones.append({
                    "capability_name": roadmap.capability_name,
                    "milestone_name": milestone.name,
                    "description": milestone.description,
                    "estimated_hours_from_start": milestone.estimated_hours_from_start,
                    "deliverable_count": len(milestone.deliverables),
                })

        risk_summary: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for roadmap in roadmaps:
            for risk in roadmap.risks:
                level_key = risk.level.value
                if level_key in risk_summary:
                    risk_summary[level_key] += 1

        return {
            "total_roadmaps": len(roadmaps),
            "total_estimated_hours": total_estimated_hours,
            "total_estimated_weeks": total_estimated_weeks,
            "by_capability": by_capability,
            "all_milestones": all_milestones,
            "risk_summary": risk_summary,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_full_report(
        self,
        scan_results: List[ScanResult],
        capabilities: List[Capability],
        dependencies: List[DependencyMapping],
        ranked_capabilities: List[RankedCapability],
        roadmaps: List[ImplementationRoadmap] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report combining all sub-reports.

        Args:
            scan_results: Results from system scans
            capabilities: Discovered capabilities
            dependencies: Current dependency mappings
            ranked_capabilities: Capabilities with priority scores and ranks
            roadmaps: Optional list of ImplementationRoadmap objects

        Returns:
            Dict with keys: summary, capability_inventory, dependency_report,
            priority_report, roadmap_report, generated_at.
        """
        return {
            "summary": self.generate_summary_report(scan_results),
            "capability_inventory": self.generate_capability_inventory(capabilities),
            "dependency_report": self.generate_dependency_report(dependencies),
            "priority_report": self.generate_priority_report(ranked_capabilities),
            "roadmap_report": self.generate_roadmap_report(roadmaps or []),
            "generated_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Private export helpers
    # ------------------------------------------------------------------

    def _export_json(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Write report data as indented JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report_data, fh, indent=2, default=str)

    def _export_csv(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """
        Write report data as CSV.

        If the report contains a list value (keyed "capabilities", "dependencies",
        or "applications"), that list is used as rows. Otherwise, top-level scalar
        values are flattened into a single-row CSV.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find the first list value to use as rows
        list_keys = ["capabilities", "dependencies", "applications"]
        rows: List[Dict[str, Any]] = []
        for key in list_keys:
            if key in report_data and isinstance(report_data[key], list):
                rows = report_data[key]
                break

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            if rows:
                # Use the keys from the first row as fieldnames
                fieldnames = list(rows[0].keys()) if rows else []
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            else:
                # Flatten scalar top-level values into a single row
                scalar_data = {
                    k: v
                    for k, v in report_data.items()
                    if not isinstance(v, (list, dict))
                }
                if scalar_data:
                    writer = csv.DictWriter(fh, fieldnames=list(scalar_data.keys()))
                    writer.writeheader()
                    writer.writerow(scalar_data)

    def export_all_formats(
        self,
        report_data: Dict[str, Any],
        output_dir: Path,
        base_filename: str = "report",
    ) -> Dict[str, Path]:
        """
        Export report to all supported formats (json, csv, html).

        Creates {output_dir}/{base_filename}.json, .csv, and .html files.

        Args:
            report_data: Report data to export
            output_dir: Directory to write the exported files
            base_filename: Base name for the output files (without extension)

        Returns:
            Dict mapping format name to output path: {"json": path, "csv": path, "html": path}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}
        for fmt in ("json", "csv", "html"):
            out_path = output_dir / f"{base_filename}.{fmt}"
            self.export_report(report_data, fmt, out_path)
            paths[fmt] = out_path

        return paths

    def _export_html(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """
        Write report data as an HTML page with charts and executive summary.

        Renders an executive summary box when a "summary" key is present,
        inline SVG bar charts for dict-of-ints values, a <table> for each
        list value, and <p> tags for scalar values.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "<!DOCTYPE html>",
            "<html>",
            "<head><meta charset='utf-8'><title>Janus Dependency Analyzer Report</title></head>",
            "<body>",
            "<h1>Janus Dependency Analyzer Report</h1>",
        ]

        # Executive summary section — rendered before other sections
        if "summary" in report_data and isinstance(report_data["summary"], dict):
            summary = report_data["summary"]
            lines.append(
                '<div style="background:#f0f4f8;padding:16px;border-radius:8px;margin-bottom:24px;">'
            )
            lines.append('<h2 style="margin-top:0;">Executive Summary</h2>')
            for k, v in summary.items():
                lines.append(
                    f"<p><strong>{self._html_escape(str(k))}:</strong> "
                    f"{self._html_escape(str(v))}</p>"
                )
            lines.append("</div>")

        for key, value in report_data.items():
            if isinstance(value, list):
                lines.append(f"<h2>{key}</h2>")
                if value:
                    lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
                    # Header row from first item's keys
                    if isinstance(value[0], dict):
                        headers = list(value[0].keys())
                        lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
                        for row in value:
                            cells = "".join(
                                f"<td>{self._html_escape(str(row.get(h, '')))}</td>"
                                for h in headers
                            )
                            lines.append(f"<tr>{cells}</tr>")
                    else:
                        lines.append("<tr><th>value</th></tr>")
                        for item in value:
                            lines.append(f"<tr><td>{self._html_escape(str(item))}</td></tr>")
                    lines.append("</table>")
                else:
                    lines.append("<p><em>(empty)</em></p>")
            elif isinstance(value, dict):
                lines.append(f"<h2>{key}</h2>")
                lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
                lines.append("<tr><th>key</th><th>value</th></tr>")
                for k, v in value.items():
                    lines.append(
                        f"<tr><td>{self._html_escape(str(k))}</td>"
                        f"<td>{self._html_escape(str(v))}</td></tr>"
                    )
                lines.append("</table>")
                # Render SVG bar chart if all values are integers (count dict)
                if value and all(isinstance(v, int) for v in value.values()):
                    lines.append(self._render_bar_chart_svg(value, title=key))
            else:
                lines.append(
                    f"<p><strong>{self._html_escape(str(key))}:</strong> "
                    f"{self._html_escape(str(value))}</p>"
                )

        lines.extend(["</body>", "</html>"])

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _render_bar_chart_svg(
        self,
        data: Dict[str, int],
        title: str,
        width: int = 400,
        height: int = 200,
    ) -> str:
        """
        Generate an inline SVG bar chart for a dict of string→int counts.

        Args:
            data: Mapping of label to integer count
            title: Chart title displayed above the bars
            width: Total SVG width in pixels
            height: Total SVG height in pixels (excluding title area)

        Returns:
            An SVG string like ``<svg width="..." height="...">...</svg>``
        """
        if not data:
            return ""

        title_height = 24
        label_height = 20
        chart_height = height - label_height
        total_svg_height = height + title_height

        items = list(data.items())
        n = len(items)
        max_val = max(v for _, v in items) if items else 1
        if max_val == 0:
            max_val = 1

        bar_area_width = width - 20  # 10px padding each side
        bar_width = max(4, bar_area_width // n - 4)
        gap = (bar_area_width - bar_width * n) // max(n, 1)

        svg_lines: List[str] = [
            f'<svg width="{width}" height="{total_svg_height}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;margin:8px 0;">',
        ]

        # Title
        svg_lines.append(
            f'  <text x="{width // 2}" y="16" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="#333">'
            f'{self._html_escape(title)}</text>'
        )

        for i, (label, val) in enumerate(items):
            bar_h = int((val / max_val) * (chart_height - 20))
            x = 10 + i * (bar_width + gap)
            y = title_height + chart_height - 20 - bar_h

            # Bar rectangle
            svg_lines.append(
                f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                f'fill="#4a90d9" rx="2"/>'
            )

            # Value label above bar
            svg_lines.append(
                f'  <text x="{x + bar_width // 2}" y="{max(y - 2, title_height + 10)}" '
                f'text-anchor="middle" font-size="10" fill="#333">{val}</text>'
            )

            # Label below bar (truncated to 12 chars)
            truncated = label[:12]
            label_y = title_height + chart_height - 20 + 14
            svg_lines.append(
                f'  <text x="{x + bar_width // 2}" y="{label_y}" '
                f'text-anchor="middle" font-size="9" fill="#555">'
                f'{self._html_escape(truncated)}</text>'
            )

        svg_lines.append("</svg>")
        return "\n".join(svg_lines)

    # ------------------------------------------------------------------
    # Path sanitization (Req 9.5)
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_path(path: str, home_dir: str = None) -> str:
        """
        Sanitize a file system path to prevent information disclosure.

        Replaces the home directory prefix with ``~`` and truncates long
        paths to show only the last 3 components.

        Requirement 9.5: WHEN generating reports, THE Report_Generator SHALL
        sanitize file paths and system information to prevent information
        disclosure.

        Args:
            path: The raw file system path string.
            home_dir: The home directory prefix to replace with ``~``.
                      Defaults to the current user's home directory.

        Returns:
            A sanitized path string.
        """
        import os

        if not path:
            return path

        # Resolve the home directory to use for replacement
        if home_dir is None:
            home_dir = str(Path.home())

        # Normalise separators for comparison (handle both / and \)
        normalised = path.replace("\\", "/")
        normalised_home = home_dir.replace("\\", "/").rstrip("/")

        # Replace home directory prefix with ~
        if normalised.startswith(normalised_home + "/") or normalised == normalised_home:
            path = "~" + path[len(home_dir):]
            normalised = "~" + normalised[len(normalised_home):]

        # Expand ~ at the start for component counting purposes
        # Split on either separator
        parts = [p for p in normalised.replace("\\", "/").split("/") if p]

        # If the path has more than 3 meaningful components, truncate
        # (keep the last 3 components, prepend "..." to indicate truncation)
        if len(parts) > 3:
            last_three = parts[-3:]
            # Preserve leading ~ or / or drive letter
            if normalised.startswith("~"):
                path = "~/.../{}".format("/".join(last_three))
            elif normalised.startswith("/"):
                path = "/.../{}".format("/".join(last_three))
            else:
                # Windows-style or relative — just show last 3 parts
                path = ".../{}".format("/".join(last_three))

        return path

    def sanitize_report_paths(
        self, report_data: Dict[str, Any], home_dir: str = None
    ) -> Dict[str, Any]:
        """
        Recursively sanitize all path-like string values in a report dict.

        Walks the report dict and replaces any string value that looks like
        an absolute path (starts with ``/``, ``C:\\``, or ``~``) with a
        sanitized version produced by :meth:`sanitize_path`.

        Does not mutate the original dict — returns a new dict.

        Requirement 9.5: WHEN generating reports, THE Report_Generator SHALL
        sanitize file paths and system information to prevent information
        disclosure.

        Args:
            report_data: The report dict to sanitize.
            home_dir: Optional home directory override passed to
                      :meth:`sanitize_path`.

        Returns:
            A new dict with sanitized path values.
        """
        return self._sanitize_value(report_data, home_dir)

    def _sanitize_value(self, value: Any, home_dir: str) -> Any:
        """Recursively sanitize a value (dict, list, or scalar)."""
        if isinstance(value, dict):
            return {k: self._sanitize_value(v, home_dir) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item, home_dir) for item in value]
        if isinstance(value, str) and self._looks_like_path(value):
            return self.sanitize_path(value, home_dir)
        return value

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        """Return True if *value* looks like an absolute or home-relative path."""
        if not value:
            return False
        # Unix absolute path
        if value.startswith("/"):
            return True
        # Windows absolute path (e.g. C:\...)
        if len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\"):
            return True
        # Home-relative path
        if value.startswith("~"):
            return True
        return False

    @staticmethod
    def _html_escape(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
