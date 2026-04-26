"""Integration tests for CLI and API interfaces."""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from janus_dependency_analyzer.api.app import create_app
from janus_dependency_analyzer.api.rate_limit import RateLimiter
from janus_dependency_analyzer.cli import cli
from janus_dependency_analyzer.core.models import Application, Platform, ScanResult

VALID_API_KEY = "test-key-123"

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def api_client():
    app = create_app(api_keys=[VALID_API_KEY], rate_limiter=RateLimiter(1000, 60))
    return TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def auth_headers():
    return {"X-API-Key": VALID_API_KEY}

@pytest.fixture
def scan_result():
    result = ScanResult(scan_type="full", platform=Platform.LINUX, scan_start_time=datetime.now())
    result.add_application(Application(
        id="app-1", name="git", version="2.40.0",
        installation_path=Path("/usr/bin"), executable_path=Path("/usr/bin/git"),
        platform=Platform.LINUX, is_accessible=True
    ))
    result.finalize()
    return result

def _mock_scanner(scan_result):
    m = MagicMock()
    m.scan_full.return_value = scan_result
    m.detect_platform.return_value = Platform.LINUX
    return m

class TestCLIBasics:
    """Test basic CLI commands work end-to-end."""
    
    def test_scan_command_works(self, runner, scan_result):
        with patch("janus_dependency_analyzer.cli.SystemScannerImpl", return_value=_mock_scanner(scan_result)):
            result = runner.invoke(cli, ["scan", "--format", "json"])
        assert result.exit_code == 0
        assert "git" in result.output

    def test_report_command_works(self, runner, scan_result):
        with patch("janus_dependency_analyzer.cli.SystemScannerImpl", return_value=_mock_scanner(scan_result)):
            result = runner.invoke(cli, ["report", "--type", "summary", "--format", "json"])
        assert result.exit_code == 0

class TestAPIBasics:
    """Test basic API endpoints work end-to-end."""
    
    def test_scan_endpoint_works(self, api_client, auth_headers, scan_result):
        with patch("janus_dependency_analyzer.api.routes.scan.SystemScannerImpl", return_value=_mock_scanner(scan_result)):
            resp = api_client.post("/scan/full", json={}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["summary"]["total_applications"] == 1

    def test_auth_required(self, api_client):
        resp = api_client.post("/scan/full", json={})
        assert resp.status_code == 401

class TestConsistency:
    """Test CLI and API produce consistent results."""
    
    def test_cli_and_api_agree_on_app_count(self, runner, api_client, auth_headers, scan_result):
        mock = _mock_scanner(scan_result)
        with patch("janus_dependency_analyzer.cli.SystemScannerImpl", return_value=mock):
            cli_result = runner.invoke(cli, ["scan", "--format", "json"])
        start = cli_result.output.find("{")
        end = cli_result.output.rfind("}") + 1
        cli_data = json.loads(cli_result.output[start:end])
        with patch("janus_dependency_analyzer.api.routes.scan.SystemScannerImpl", return_value=mock):
            api_resp = api_client.post("/scan/full", json={}, headers=auth_headers)
        assert cli_data["scan_info"]["total_applications"] == api_resp.json()["summary"]["total_applications"]