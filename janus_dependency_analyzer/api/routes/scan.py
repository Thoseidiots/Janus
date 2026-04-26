"""
Scan routes for the Janus Dependency Analyzer REST API.

Provides endpoints for initiating full and incremental system scans,
both synchronously and as background jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from ...scanners.system_scanner import SystemScannerImpl
from ..auth import require_api_key
from ..models import (
    ApplicationResponse,
    AsyncJobResponse,
    JobStatusEnum,
    JobStatusResponse,
    PlatformEnum,
    ScanRequest,
    ScanResponse,
    ScanSummaryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan", tags=["Scan"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app_response(app) -> ApplicationResponse:
    """Convert a core Application dataclass to an API response model."""
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


def _build_summary(scan_result) -> ScanSummaryResponse:
    return ScanSummaryResponse(
        platform=PlatformEnum(scan_result.platform.value),
        scan_type=scan_result.scan_type,
        scan_start_time=scan_result.scan_start_time,
        scan_end_time=scan_result.scan_end_time,
        total_applications=scan_result.total_applications,
        accessible_applications=scan_result.accessible_applications,
        errors=scan_result.errors,
        warnings=scan_result.warnings,
    )


def _run_scan(request: ScanRequest, job_store: Dict[str, Any], job_id: str) -> None:
    """Execute a scan and store the result in the job store."""
    try:
        job_store[job_id]["status"] = JobStatusEnum.RUNNING
        job_store[job_id]["started_at"] = datetime.utcnow()
        job_store[job_id]["progress_message"] = "Scan in progress…"

        scanner = SystemScannerImpl()

        if request.incremental:
            since = request.since or (datetime.now() - timedelta(hours=24))
            scan_result = scanner.scan_incremental(since)
        else:
            scan_result = scanner.scan_full()

        result_data = {
            "summary": _build_summary(scan_result).model_dump(mode="json"),
            "applications": [
                _build_app_response(a).model_dump(mode="json")
                for a in scan_result.applications
            ],
        }

        job_store[job_id].update(
            {
                "status": JobStatusEnum.COMPLETED,
                "completed_at": datetime.utcnow(),
                "progress_message": "Scan completed successfully.",
                "result": result_data,
            }
        )
        logger.info("Scan job %s completed: %d apps found", job_id, scan_result.total_applications)

    except Exception as exc:
        logger.exception("Scan job %s failed", job_id)
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
    "/full",
    response_model=ScanResponse,
    summary="Run a full system scan (synchronous)",
    description=(
        "Performs a complete scan of all installed applications on the system. "
        "This endpoint blocks until the scan completes. "
        "For long-running scans use POST /scan/async instead."
    ),
    status_code=status.HTTP_200_OK,
)
async def full_scan(
    request: ScanRequest,
    _api_key: str = Depends(require_api_key),
) -> ScanResponse:
    """Synchronous full system scan."""
    try:
        scanner = SystemScannerImpl()
        scan_result = scanner.scan_full()
    except Exception as exc:
        logger.exception("Full scan failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {exc}",
        )

    job_id = str(uuid.uuid4())
    return ScanResponse(
        job_id=job_id,
        summary=_build_summary(scan_result),
        applications=[_build_app_response(a) for a in scan_result.applications],
    )


@router.post(
    "/incremental",
    response_model=ScanResponse,
    summary="Run an incremental scan (synchronous)",
    description=(
        "Scans for application changes since the specified baseline time. "
        "Defaults to 24 hours ago when ``since`` is not provided."
    ),
    status_code=status.HTTP_200_OK,
)
async def incremental_scan(
    request: ScanRequest,
    _api_key: str = Depends(require_api_key),
) -> ScanResponse:
    """Synchronous incremental scan."""
    since = request.since or (datetime.now() - timedelta(hours=24))
    try:
        scanner = SystemScannerImpl()
        scan_result = scanner.scan_incremental(since)
    except Exception as exc:
        logger.exception("Incremental scan failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Incremental scan failed: {exc}",
        )

    job_id = str(uuid.uuid4())
    return ScanResponse(
        job_id=job_id,
        summary=_build_summary(scan_result),
        applications=[_build_app_response(a) for a in scan_result.applications],
    )


@router.post(
    "/async",
    response_model=AsyncJobResponse,
    summary="Start an async scan job",
    description=(
        "Submits a scan job for background processing and returns immediately "
        "with a job ID. Poll GET /scan/jobs/{job_id} to check progress and "
        "retrieve results."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_async_scan(
    request: Request,
    scan_request: ScanRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
) -> AsyncJobResponse:
    """Submit a scan as a background job."""
    job_id = str(uuid.uuid4())
    job_store: Dict[str, Any] = request.app.state.job_store

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

    background_tasks.add_task(_run_scan, scan_request, job_store, job_id)

    base_url = str(request.base_url).rstrip("/")
    return AsyncJobResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        message="Scan job submitted. Poll the status URL for updates.",
        created_at=job_store[job_id]["created_at"],
        status_url=f"{base_url}/scan/jobs/{job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get async scan job status",
    description="Returns the current status and result (when complete) of an async scan job.",
)
async def get_scan_job_status(
    job_id: str,
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> JobStatusResponse:
    """Retrieve the status of an async scan job."""
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
