"""
Integration tests for the Janus Dependency Analyzer REST API.

Tests cover:
- Authentication (missing key, invalid key, valid key)
- Rate limiting middleware
- Scan endpoints (full, incremental, async)
- Analyze endpoints (sync, async)
- Report endpoints (sync, async)
- Configure endpoints (GET, PUT, POST reset, GET validate)
- Health and root endpoints
- Error handling (404, 422, 500)
- Async job lifecycle (submit → poll → result)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from janus_dependency_analyzer.api.app import create_app
from janus_dependency_analyzer.api.auth import APIKeyStore
from janus_dependency_analyzer.api.rate_limit import RateLimiter
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
# Constants
# ---------------------------------------------------------------------------

VALID_KEY = "test-api-key-12345"
INVALID_KEY = "wrong-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a test application with a known API key and relaxed rate limits."""
    return create_app(
        api_keys=[VALID_KEY],
        rate_limiter=RateLimiter(max_requests=1000, window_seconds=60),
    )


@pytest.fixture
def client(app):
    """TestClient for the test application."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    """Headers with a valid API key."""
    return {"X-API-Key": VALID_KEY}


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
            application_id="app-001",
            name="Repository Management",
            category=CapabilityCategory.DEVELOPMENT_TOOLS,
            description="Manage git repositories",
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=0.85,
            detection_method="documentation",
        ),
    ]


def _make_mock_scanner(scan_result):
    mock = MagicMock()
    mock.scan_full.return_value = scan_result
    mock.scan_incremental.return_value = scan_result
    mock.detect_platform.return_value = Platform.LINUX
    mock.get_platform_scanner.return_value = MagicMock(
        __class__=MagicMock(__name__="MockScanner")
    )
    return mock


def _make_mock_analyzer(capabilities):
    mock = MagicMock()
    mock.analyze_application.return_value = capabilities
    return mock


# ---------------------------------------------------------------------------
# Health / root endpoints (no auth required)
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "openapi" in schema
        assert "paths" in schema


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_missing_key_returns_401(self, client):
        resp = client.post("/scan/full", json={"incremental": False})
        assert resp.status_code == 401

    def test_invalid_key_returns_403(self, client):
        resp = client.post(
            "/scan/full",
            json={"incremental": False},
            headers={"X-API-Key": INVALID_KEY},
        )
        assert resp.status_code == 403

    def test_valid_key_passes_auth(self, client, auth_headers, minimal_scan_result):
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post(
                "/scan/full",
                json={"incremental": False},
                headers=auth_headers,
            )
        assert resp.status_code == 200

    def test_health_does_not_require_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_does_not_require_auth(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_headers_present(self, client, auth_headers, minimal_scan_result):
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post(
                "/scan/full",
                json={"incremental": False},
                headers=auth_headers,
            )
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_exceeded_returns_429(self, app, minimal_scan_result):
        """Use a very tight limiter (1 req/60s) to trigger 429."""
        tight_app = create_app(
            api_keys=[VALID_KEY],
            rate_limiter=RateLimiter(max_requests=1, window_seconds=60),
        )
        tight_client = TestClient(tight_app, raise_server_exceptions=False)
        headers = {"X-API-Key": VALID_KEY}

        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            # First request should succeed
            resp1 = tight_client.post("/scan/full", json={}, headers=headers)
            assert resp1.status_code == 200

            # Second request should be rate-limited
            resp2 = tight_client.post("/scan/full", json={}, headers=headers)
            assert resp2.status_code == 429
            assert "Retry-After" in resp2.headers

    def test_health_exempt_from_rate_limit(self, app):
        """Health endpoint should never be rate-limited."""
        tight_app = create_app(
            api_keys=[VALID_KEY],
            rate_limiter=RateLimiter(max_requests=1, window_seconds=60),
        )
        tight_client = TestClient(tight_app, raise_server_exceptions=False)
        for _ in range(5):
            resp = tight_client.get("/health")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Scan endpoints
# ---------------------------------------------------------------------------

class TestScanEndpoints:
    def test_full_scan_returns_applications(self, client, auth_headers, minimal_scan_result):
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post("/scan/full", json={}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert "summary" in data
        assert "applications" in data
        assert data["summary"]["total_applications"] == 2
        assert len(data["applications"]) == 2

    def test_full_scan_application_fields(self, client, auth_headers, minimal_scan_result):
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post("/scan/full", json={}, headers=auth_headers)
        app_data = resp.json()["applications"][0]
        assert "id" in app_data
        assert "name" in app_data
        assert "platform" in app_data
        assert "is_accessible" in app_data

    def test_incremental_scan(self, client, auth_headers, minimal_scan_result):
        mock_scanner = _make_mock_scanner(minimal_scan_result)
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=mock_scanner,
        ):
            resp = client.post(
                "/scan/incremental",
                json={"incremental": True},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        mock_scanner.scan_incremental.assert_called_once()

    def test_incremental_scan_with_since(self, client, auth_headers, minimal_scan_result):
        mock_scanner = _make_mock_scanner(minimal_scan_result)
        since = "2024-01-01T00:00:00"
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=mock_scanner,
        ):
            resp = client.post(
                "/scan/incremental",
                json={"incremental": True, "since": since},
                headers=auth_headers,
            )
        assert resp.status_code == 200

    def test_async_scan_returns_202(self, client, auth_headers):
        resp = client.post("/scan/async", json={}, headers=auth_headers)
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "status_url" in data

    def test_async_scan_job_status_pending(self, client, auth_headers):
        resp = client.post("/scan/async", json={}, headers=auth_headers)
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/scan/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("pending", "running", "completed", "failed")

    def test_async_scan_job_not_found(self, client, auth_headers):
        resp = client.get(f"/scan/jobs/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404

    def test_scan_scanner_error_returns_500(self, client, auth_headers):
        mock_scanner = MagicMock()
        mock_scanner.scan_full.side_effect = RuntimeError("Scanner exploded")
        with patch(
            "janus_dependency_analyzer.api.routes.scan.SystemScannerImpl",
            return_value=mock_scanner,
        ):
            resp = client.post("/scan/full", json={}, headers=auth_headers)
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Analyze endpoints
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoints:
    def test_analyze_found_app(
        self, client, auth_headers, minimal_scan_result, minimal_capabilities
    ):
        with patch(
            "janus_dependency_analyzer.api.routes.analyze.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            "janus_dependency_analyzer.api.routes.analyze.CapabilityAnalyzerImpl",
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            resp = client.post(
                "/analyze/",
                json={"app_name": "git"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "application" in data
        assert "capabilities" in data
        assert data["total_capabilities"] == 2
        assert data["application"]["name"] == "git"

    def test_analyze_case_insensitive(
        self, client, auth_headers, minimal_scan_result, minimal_capabilities
    ):
        with patch(
            "janus_dependency_analyzer.api.routes.analyze.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            "janus_dependency_analyzer.api.routes.analyze.CapabilityAnalyzerImpl",
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            resp = client.post(
                "/analyze/",
                json={"app_name": "GIT"},
                headers=auth_headers,
            )
        assert resp.status_code == 200

    def test_analyze_app_not_found_returns_404(
        self, client, auth_headers, minimal_scan_result
    ):
        with patch(
            "janus_dependency_analyzer.api.routes.analyze.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post(
                "/analyze/",
                json={"app_name": "nonexistent_xyz_app"},
                headers=auth_headers,
            )
        assert resp.status_code == 404

    def test_analyze_missing_app_name_returns_422(self, client, auth_headers):
        resp = client.post("/analyze/", json={}, headers=auth_headers)
        assert resp.status_code == 422

    def test_async_analyze_returns_202(self, client, auth_headers):
        resp = client.post(
            "/analyze/async",
            json={"app_name": "git"},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_async_analyze_job_status(self, client, auth_headers):
        resp = client.post(
            "/analyze/async",
            json={"app_name": "git"},
            headers=auth_headers,
        )
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/analyze/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_async_analyze_job_not_found(self, client, auth_headers):
        resp = client.get(f"/analyze/jobs/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

class TestReportEndpoints:
    def test_summary_report(self, client, auth_headers, minimal_scan_result):
        with patch(
            "janus_dependency_analyzer.api.routes.report.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ):
            resp = client.post(
                "/report/",
                json={"report_type": "summary", "output_format": "json"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert data["report_type"] == "summary"
        assert "total_applications" in data["data"]

    def test_capabilities_report(
        self, client, auth_headers, minimal_scan_result, minimal_capabilities
    ):
        with patch(
            "janus_dependency_analyzer.api.routes.report.SystemScannerImpl",
            return_value=_make_mock_scanner(minimal_scan_result),
        ), patch(
            "janus_dependency_analyzer.api.routes.report.CapabilityAnalyzerImpl",
            return_value=_make_mock_analyzer(minimal_capabilities),
        ):
            resp = client.post(
                "/report/",
                json={"report_type": "capabilities", "output_format": "json"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_capabilities" in data["data"]

    def test_invalid_report_type_returns_422(self, client, auth_headers):
        resp = client.post(
            "/report/",
            json={"report_type": "invalid_type", "output_format": "json"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_async_report_returns_202(self, client, auth_headers):
        resp = client.post(
            "/report/async",
            json={"report_type": "summary", "output_format": "json"},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_async_report_job_status(self, client, auth_headers):
        resp = client.post(
            "/report/async",
            json={"report_type": "summary", "output_format": "json"},
            headers=auth_headers,
        )
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/report/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_async_report_job_not_found(self, client, auth_headers):
        resp = client.get(f"/report/jobs/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Configure endpoints
# ---------------------------------------------------------------------------

class TestConfigureEndpoints:
    def test_get_configuration(self, client, auth_headers):
        resp = client.get("/configure/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "scan_timeout_seconds" in data
        assert "priority_weights" in data
        assert "min_confidence_threshold" in data

    def test_update_configuration_partial(self, client, auth_headers):
        resp = client.put(
            "/configure/",
            json={"scan_timeout_seconds": 600},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["scan_timeout_seconds"] == 600

    def test_update_configuration_priority_weights(self, client, auth_headers):
        resp = client.put(
            "/configure/",
            json={
                "priority_weights": {
                    "usage": 0.4,
                    "complexity": 0.3,
                    "security": 0.2,
                    "performance": 0.1,
                }
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        weights = resp.json()["priority_weights"]
        assert weights["usage"] == pytest.approx(0.4)

    def test_update_configuration_invalid_returns_422(self, client, auth_headers):
        resp = client.put(
            "/configure/",
            json={"scan_timeout_seconds": -1},
            headers=auth_headers,
        )
        # Pydantic rejects negative value at the model level (gt=0)
        assert resp.status_code == 422

    def test_reset_configuration(self, client, auth_headers):
        # First change something
        client.put(
            "/configure/",
            json={"scan_timeout_seconds": 999},
            headers=auth_headers,
        )
        # Then reset
        resp = client.post("/configure/reset", headers=auth_headers)
        assert resp.status_code == 200
        # Default is 300
        assert resp.json()["scan_timeout_seconds"] == 300

    def test_validate_configuration(self, client, auth_headers):
        resp = client.get("/configure/validate", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_valid" in data
        assert data["is_valid"] is True

    def test_configure_requires_auth(self, client):
        resp = client.get("/configure/")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# OpenAPI schema completeness
# ---------------------------------------------------------------------------

class TestOpenAPISchema:
    def test_all_route_prefixes_in_schema(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        prefixes = {"/scan/", "/analyze/", "/report/", "/configure/"}
        for prefix in prefixes:
            assert any(p.startswith(prefix) for p in paths), (
                f"No paths found with prefix {prefix}"
            )

    def test_security_scheme_defined(self, client):
        schema = client.get("/openapi.json").json()
        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        assert len(security_schemes) > 0, "No security schemes defined in OpenAPI schema"
