"""
Analyze routes for the Janus Dependency Analyzer REST API.

Provides endpoints for analysing the capabilities of discovered applications,
both synchronously and as background jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from ...analyzers.capability_analyzer import CapabilityAnalyzerImpl
from ...scanners.system_scanner import SystemScannerImpl
from ..auth import require_api_key
from ..models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ApplicationResponse,
    AsyncJobResponse,
    CapabilityResponse,
    CapabilityCategoryEnum,
    InterfaceTypeEnum,
    JobStatusEnum,
    JobStatusResponse,
    PlatformEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analyze"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app_response(app) -> ApplicationResponse:
    return ApplicationResponse(
        id=app.id,
        name=app.name,
        version=app.version or None,
        installation_path=str(app.installation_path),
        executable_path=str(app.executable_path),
        platform=PlatformEnum(app.platform.value),
        is_accessible=app.is_accessible,
        access_error=app.access_error,
        discovered_at=app.discovered_at,
    )


def _build_cap_response(cap) -> CapabilityResponse:
    return CapabilityResponse(
        id=cap.id,
        application_id=cap.application_id,
        name=cap.name,
        category=CapabilityCategoryEnum(cap.category.value),
        description=cap.description,
        interface_type=InterfaceTypeEnum(cap.interface_type.value),
        confidence_score=cap.confidence_score,
        detection_method=cap.detection_method,
        examples=cap.examples,
        documentation_url=cap.documentation_url,
        supported_formats=cap.supported_formats,
    )


def _find_application(scan_result, app_name: str):
    """Return the first application whose name contains app_name (case-insensitive)."""
    needle = app_name.lower()
    for app in scan_result.applications:
        if needle in app.name.lower():
            return app
    return None


def _run_analysis(
    analyze_request: AnalyzeRequest,
    job_store: Dict[str, Any],
    job_id: str,
) -> None:
    """Execute capability analysis and store the result in the job store."""
    try:
        job_store[job_id]["status"] = JobStatusEnum.RUNNING
        job_store[job_id]["started_at"] = datetime.utcnow()
        job_store[job_id]["progress_message"] = "Scanning for application…"

        scanner = SystemScannerImpl()
        scan_result = scanner.scan_full()

        target_app = _find_application(scan_result, analyze_request.app_name)
        if target_app is None:
            job_store[job_id].update(
                {
                    "status": JobStatusEnum.FAILED,
                    "completed_at": datetime.utcnow(),
                    "error": f"Application '{analyze_request.app_name}' not found.",
                }
            )
            return

        job_store[job_id]["progress_message"] = "Analysing capabilities…"
        analyzer = CapabilityAnalyzerImpl()
        capabilities = analyzer.analyze_application(target_app)

        result_data = {
            "application": _build_app_response(target_app).model_dump(mode="json"),
            "capabilities": [_build_cap_response(c).model_dump(mode="json") for c in capabilities],
            "total_capabilities": len(capabilities),
        }

        job_store[job_id].update(
            {
                "status": JobStatusEnum.COMPLETED,
                "completed_at": datetime.utcnow(),
                "progress_message": "Analysis completed successfully.",
                "result": result_data,
            }
        )
        logger.info(
            "Analysis job %s completed: %d capabilities for '%s'",
            job_id,
            len(capabilities),
            target_app.name,
        )

    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        job_store[job_id].update(
            {
                "status": JobStatusEnum.FAILED,
                "completed_at": datetime.utcnow(),
                "error": str(exc),
            }
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=AnalyzeResponse,
    summary="Analyse application capabilities (synchronous)",
    description=(
        "Scans the system, finds the named application, and analyses its capabilities. "
        "The ``app_name`` is matched case-insensitively as a substring. "
        "For long-running analyses use POST /analyze/async instead."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_application(
    request: AnalyzeRequest,
    _api_key: str = Depends(require_api_key),
) -> AnalyzeResponse:
    """Synchronous capability analysis for a named application."""
    try:
        scanner = SystemScannerImpl()
        scan_result = scanner.scan_full()
    except Exception as exc:
        logger.exception("Scan phase of analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {exc}",
        )

    target_app = _find_application(scan_result, request.app_name)
    if target_app is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application '{request.app_name}' not found on this system.",
        )

    try:
        analyzer = CapabilityAnalyzerImpl()
        capabilities = analyzer.analyze_application(target_app)
    except Exception as exc:
        logger.exception("Capability analysis failed for '%s'", request.app_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        )

    job_id = str(uuid.uuid4())
    return AnalyzeResponse(
        job_id=job_id,
        application=_build_app_response(target_app),
        capabilities=[_build_cap_response(c) for c in capabilities],
        total_capabilities=len(capabilities),
    )


@router.post(
    "/async",
    response_model=AsyncJobResponse,
    summary="Start an async capability analysis job",
    description=(
        "Submits a capability analysis job for background processing and returns "
        "immediately with a job ID. Poll GET /analyze/jobs/{job_id} for results."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_async_analysis(
    http_request: Request,
    analyze_request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
) -> AsyncJobResponse:
    """Submit a capability analysis as a background job."""
    job_id = str(uuid.uuid4())
    job_store: Dict[str, Any] = http_request.app.state.job_store

    job_store[job_id] = {
        "job_id": job_id,
        "status": JobStatusEnum.PENDING,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "progress_message": "Job queued.",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_analysis, analyze_request, job_store, job_id)

    base_url = str(http_request.base_url).rstrip("/")
    return AsyncJobResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        message="Analysis job submitted. Poll the status URL for updates.",
        created_at=job_store[job_id]["created_at"],
        status_url=f"{base_url}/analyze/jobs/{job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get async analysis job status",
    description="Returns the current status and result (when complete) of an async analysis job.",
)
async def get_analysis_job_status(
    job_id: str,
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> JobStatusResponse:
    """Retrieve the status of an async analysis job."""
    job_store: Dict[str, Any] = request.app.state.job_store
    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress_message=job.get("progress_message"),
        result=job.get("result"),
        error=job.get("error"),
    )
