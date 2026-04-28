"""
Integration tests for the Janus Dependency Analyzer CLI.

Tests cover:
- Full and incremental scan commands
- Configuration file loading and the configure subcommand
- Report generation commands (summary, capabilities, dependencies, priority, full)
- Output format selection (json, table, csv, html)
- --output flag for saving results to files
- Error handling for invalid inputs
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from janus_dependency_analyzer.cli import cli
from janus_dependency_analyzer.core.models import (
    Application,
    Capability,
    CapabilityCategory,
    Configuration,
    InterfaceType,
    Platform,
    ScanResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def minimal_scan_result():
    """A minimal ScanResult with two applications."""
    result = ScanResult(
        scan_type="full",
        platform=Platform.LINUX,
        scan_start_time=datetime.now(),
    )
    app1 = Application(
        id="app-001",
        name="git",
        version="2.40.0",
        installation_path=Path("/usr/bin"),
        executable_path=Path("/usr/bin/git"),
        platform=Platform.LINUX,
        is_accessible=True,
    )
    app2 = Application(
        id="app-002",
        name="curl",
        version="7.88.0",
        installation_path=Path("/usr/bin"),
        executable_path=Path("/usr/bin/curl"),
        platform=Platform.LINUX,
        is_accessible=True,
    )
    result.add_application(app1)
    result.add_application(app2)
    result.finalize()
    return result


@pytest.fixture
def minimal_capabilities():
    """A list of two minimal Capability objects."""
    return [
        Capability(
            id="cap-001",
            application_id="app-001",
            name="Version Control",
            category=CapabilityCategory.SYSTEM_INTEGRATION,
            description="Git version control operations",
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=0.9,
            detection_method="help_text",
        ),
        Capability(
            id="cap-002",
            application_id="app-002",
            name="HTTP Transfer",
            category=CapabilityCategory.NETWORK_OPERATIONS,
            description="HTTP/HTTPS data transfer",
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=0.85,
            detection_method="help_text",
        ),
    ]


@pytest.fixture
def default_config_file(tmp_path):
    """Write a default JSON config file and return its path."""
    from janus_dependency_analyzer.config.configuration import ConfigurationManager
    manager = ConfigurationManager()
    config = Configuration()
    config_text = manager.format(config, format="json")
    config_path = tmp_path / "config.json"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_scanner(scan_result):
    """Return a mock SystemScannerImpl that returns scan_result."""
    mock = MagicMock()
    mock.scan_full.return_value = scan_result
    mock.scan_incremental.return_value = scan_result
    mock.detect_platform.return_value = Platform.LINUX
    mock.get_platform_scanner.return_value = MagicMock(__class__=MagicMock(__name__="MockScanner"))
    return mock


def _make_mock_analyzer(capabilities):
    """Return a mock CapabilityAnalyzerImpl that returns capabilities."""
    mock = MagicMock()
    mock.analyze_application.return_value = capabilities
    return mock


def _extract_json(output: str) -> dict:
    """Extract the first JSON object from a CLI output string."""
    json_start = output.find('{')
    assert json_start != -1, f"No JSON found in output:\n{output}"
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(output, json_start)
    return data


# ---------------------------------------------------------------------------
# CLI group tests
# ---------------------------------------------------------------------------

class TestCLIGroup:
    """Tests for the top-level CLI group."""

    def test_help_shows_all_commands(self, runner):
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'scan' in result.output
        assert 'analyze' in result.output
        assert 'report' in result.output
        assert 'configure' in result.output
        assert 'info' in result.output

    def test_config_flag_loads_file(self, runner, default_config_file):
        result = runner.invoke(cli, ['--config', str(default_config_file), 'configure', '--show'])
        assert result.exit_code == 0
        assert 'Configuration loaded' in result.output

    def test_config_flag_nonexistent_file_exits(self, runner, tmp_path):
        missing = tmp_path / "nonexistent.json"
        result = runner.invoke(cli, ['--config', str(missing), 'configure', '--show'])
        # Click validates exists=True before our code runs
        assert result.exit_code != 0

    def test_verbose_flag_accepted(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['--verbose', 'scan', '--format', 'json'])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# scan command tests
# ---------------------------------------------------------------------------

class TestScanCommand:
    """Tests for the scan command."""

    def test_full_scan_table_format(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['scan'])
        assert result.exit_code == 0
        assert 'git' in result.output
        assert 'curl' in result.output

    def test_full_scan_json_format(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['scan', '--format', 'json'])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert 'scan_info' in data
        assert data['scan_info']['total_applications'] == 2

    def test_full_scan_json_to_file(self, runner, minimal_scan_result, tmp_path):
        out_file = tmp_path / "scan.json"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['scan', '--format', 'json', '--output', str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data['scan_info']['total_applications'] == 2

    def test_incremental_scan(self, runner, minimal_scan_result):
        mock_scanner = _make_mock_scanner(minimal_scan_result)
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=mock_scanner,
        ):
            result = runner.invoke(cli, ['scan', '--incremental'])
        assert result.exit_code == 0
        assert 'incremental' in result.output.lower()
        mock_scanner.scan_incremental.assert_called_once()

    def test_incremental_scan_with_since(self, runner, minimal_scan_result):
        mock_scanner = _make_mock_scanner(minimal_scan_result)
        since_str = "2024-01-01T00:00:00"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=mock_scanner,
        ):
            result = runner.invoke(cli, ['scan', '--incremental', '--since', since_str])
        assert result.exit_code == 0
        mock_scanner.scan_incremental.assert_called_once()
        call_args = mock_scanner.scan_incremental.call_args[0]
        assert call_args[0] == datetime.fromisoformat(since_str)

    def test_incremental_scan_invalid_since(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['scan', '--incremental', '--since', 'not-a-date'])
        assert result.exit_code != 0
        assert 'Invalid' in result.output

    def test_scan_shows_summary_counts(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['scan'])
        assert result.exit_code == 0
        assert '2' in result.output  # 2 applications found


# ---------------------------------------------------------------------------
# analyze command tests
# ---------------------------------------------------------------------------

class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_found_app_table(self, runner, minimal_scan_result, minimal_capabilities):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            'janus_dependency_analyzer.cli.CapabilityAnalyzerImpl',
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            result = runner.invoke(cli, ['analyze', 'git'])
        assert result.exit_code == 0
        assert 'git' in result.output.lower()

    def test_analyze_found_app_json(self, runner, minimal_scan_result, minimal_capabilities):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            'janus_dependency_analyzer.cli.CapabilityAnalyzerImpl',
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            result = runner.invoke(cli, ['analyze', 'git', '--format', 'json'])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert 'capabilities' in data
        assert len(data['capabilities']) == 2

    def test_analyze_json_to_file(self, runner, minimal_scan_result, minimal_capabilities, tmp_path):
        out_file = tmp_path / "caps.json"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            'janus_dependency_analyzer.cli.CapabilityAnalyzerImpl',
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            result = runner.invoke(
                cli, ['analyze', 'git', '--format', 'json', '--output', str(out_file)]
            )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert 'capabilities' in data

    def test_analyze_app_not_found(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['analyze', 'nonexistent_app_xyz'])
        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_analyze_case_insensitive_match(self, runner, minimal_scan_result, minimal_capabilities):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            'janus_dependency_analyzer.cli.CapabilityAnalyzerImpl',
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            result = runner.invoke(cli, ['analyze', 'GIT'])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# report command tests
# ---------------------------------------------------------------------------

class TestReportCommand:
    """Tests for the report command."""

    def test_summary_report_table(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['report', '--type', 'summary'])
        assert result.exit_code == 0
        assert 'summary' in result.output.lower()

    def test_summary_report_json(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['report', '--type', 'summary', '--format', 'json'])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert 'total_applications' in data
        assert data['total_applications'] == 2

    def test_summary_report_json_to_file(self, runner, minimal_scan_result, tmp_path):
        out_file = tmp_path / "summary.json"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli,
                ['report', '--type', 'summary', '--format', 'json', '--output', str(out_file)],
            )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert 'total_applications' in data

    def test_summary_report_html_to_file(self, runner, minimal_scan_result, tmp_path):
        out_file = tmp_path / "summary.html"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli,
                ['report', '--type', 'summary', '--format', 'html', '--output', str(out_file)],
            )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert '<html>' in content

    def test_capabilities_report(self, runner, minimal_scan_result, minimal_capabilities):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            'janus_dependency_analyzer.cli.CapabilityAnalyzerImpl',
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            result = runner.invoke(
                cli, ['report', '--type', 'capabilities', '--format', 'json']
            )
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert 'total_capabilities' in data
        assert data['total_capabilities'] == 4  # 2 apps × 2 caps each

    def test_dependencies_report(self, runner, minimal_scan_result, tmp_path):
        # Create a minimal Python file with a subprocess call for the mapper to find
        code_file = tmp_path / "test_code.py"
        code_file.write_text('import subprocess\nsubprocess.run(["git", "status"])\n')

        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli,
                [
                    'report', '--type', 'dependencies',
                    '--format', 'json',
                    '--codebase', str(tmp_path),
                ],
            )
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert 'total_dependencies' in data

    def test_all_formats_flag(self, runner, minimal_scan_result, tmp_path):
        out_dir = tmp_path / "reports"
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli,
                [
                    'report', '--type', 'summary',
                    '--all-formats',
                    '--output', str(out_dir),
                ],
            )
        assert result.exit_code == 0
        assert (out_dir / "summary_report.json").exists()
        assert (out_dir / "summary_report.csv").exists()
        assert (out_dir / "summary_report.html").exists()

    def test_report_invalid_type_rejected(self, runner):
        result = runner.invoke(cli, ['report', '--type', 'invalid_type'])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# configure command tests
# ---------------------------------------------------------------------------

class TestConfigureCommand:
    """Tests for the configure command."""

    def test_show_displays_configuration(self, runner):
        result = runner.invoke(cli, ['configure', '--show'])
        assert result.exit_code == 0
        assert 'Scan Timeout' in result.output
        assert 'Priority Weights' in result.output

    def test_generate_json_config(self, runner, tmp_path):
        out_file = tmp_path / "config.json"
        result = runner.invoke(cli, ['configure', '--generate', str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert 'scan_timeout_seconds' in data

    def test_generate_yaml_config(self, runner, tmp_path):
        out_file = tmp_path / "config.yaml"
        result = runner.invoke(
            cli, ['configure', '--generate', str(out_file), '--format', 'yaml']
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert 'scan_timeout_seconds' in content

    def test_validate_valid_config(self, runner, default_config_file):
        result = runner.invoke(cli, ['configure', '--validate', str(default_config_file)])
        assert result.exit_code == 0
        assert 'valid' in result.output.lower()

    def test_validate_invalid_config(self, runner, tmp_path):
        bad_config = tmp_path / "bad.json"
        bad_config.write_text(
            '{"scan_timeout_seconds": -1, "max_applications_per_scan": 10000, '
            '"analysis_timeout_seconds": 60, "min_confidence_threshold": 0.5, '
            '"max_concurrent_analyses": 4, "cache_expiry_hours": 24}',
            encoding='utf-8',
        )
        result = runner.invoke(cli, ['configure', '--validate', str(bad_config)])
        # Should report invalid (negative timeout)
        assert result.exit_code != 0 or 'invalid' in result.output.lower()

    def test_configure_no_args_shows_config(self, runner):
        """configure with no args should show the configuration."""
        result = runner.invoke(cli, ['configure'])
        assert result.exit_code == 0
        assert 'Scan Timeout' in result.output

    def test_config_flag_with_configure_show(self, runner, default_config_file):
        result = runner.invoke(
            cli, ['--config', str(default_config_file), 'configure', '--show']
        )
        assert result.exit_code == 0
        assert 'Configuration loaded' in result.output
        assert 'Scan Timeout' in result.output


# ---------------------------------------------------------------------------
# info command tests
# ---------------------------------------------------------------------------

class TestInfoCommand:
    """Tests for the info command."""

    def test_info_shows_platform(self, runner, minimal_scan_result):
        mock_scanner = _make_mock_scanner(minimal_scan_result)
        mock_scanner.detect_platform.return_value = Platform.LINUX
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=mock_scanner,
        ):
            result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'Platform' in result.output

    def test_info_shows_configuration(self, runner, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'Scan Timeout' in result.output


# ---------------------------------------------------------------------------
# Configuration file loading integration
# ---------------------------------------------------------------------------

class TestConfigFileLoading:
    """Tests for --config flag integration with all commands."""

    def test_config_file_applied_to_scan(self, runner, default_config_file, minimal_scan_result):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli, ['--config', str(default_config_file), 'scan', '--format', 'json']
            )
        assert result.exit_code == 0

    def test_config_file_applied_to_report(
        self, runner, default_config_file, minimal_scan_result
    ):
        with patch(
            'janus_dependency_analyzer.cli.SystemScannerImpl',
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            result = runner.invoke(
                cli,
                [
                    '--config', str(default_config_file),
                    'report', '--type', 'summary', '--format', 'json',
                ],
            )
        assert result.exit_code == 0

    def test_invalid_config_file_exits_early(self, runner, tmp_path):
        bad_config = tmp_path / "bad.json"
        bad_config.write_text('{"scan_timeout_seconds": -999}', encoding='utf-8')
        result = runner.invoke(cli, ['--config', str(bad_config), 'scan'])
        assert result.exit_code != 0
