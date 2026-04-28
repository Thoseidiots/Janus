"""
Report routes for the Janus Dependency Analyzer REST API.

Provides endpoints for generating analysis reports in multiple formats,
both synchronously and as background jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from ...analyzers.capability_analyzer import CapabilityAnalyzerImpl
from ...core.models import (
    DependencyMapping as CoreDependencyMapping,
    UsagePattern,
)
from ...dependency.mapper import DependencyMapper
from ...priority.engine import AnalysisContext, PriorityEngine
from ...reports.generator import ReportGenerator
from ...scanners.system_scanner import SystemScannerImpl
from ..auth import require_api_key
from ..models import (
    AsyncJobResponse,
    JobStatusEnum,
    JobStatusResponse,
    ReportFormatEnum,
    ReportRequest,
    ReportResponse,
    ReportTypeEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report", tags=["Report"])


# ---------------------------------------------------------------------------
# Core report-building logic (shared by sync and async paths)
# ---------------------------------------------------------------------------

def _build_report(request: ReportRequest) -> Dict[str, Any]:
    """
    Execute the full report pipeline and return the report data dict.

    Raises exceptions on failure; callers are responsible for error handling.
    """
    scanner = SystemScannerImpl()
    scan_result = scanner.scan_full()

    capabilities = []
    if request.report_type in (
        ReportTypeEnum.CAPABILITIES,
        ReportTypeEnum.PRIORITY,
        ReportTypeEnum.FULL,
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        analyzer = CapabilityAnalyzerImpl()
        accessible_apps = [a for a in scan_result.applications if a.is_accessible]
        
        # Use parallel processing with worker count based on CPU cores
        max_workers = min(multiprocessing.cpu_count() * 2, len(accessible_apps))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_app = {
                executor.submit(analyzer.analyze_application, app): app
                for app in accessible_apps
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_app):
                app = future_to_app[future]
                try:
                    caps = future.result(timeout=60)
                    capabilities.extend(caps)
                except Exception as exc:
                    logger.error(f"Failed to analyze {app.name}: {exc}")

    dependency_mappings: list = []
    if request.report_type in (ReportTypeEnum.DEPENDENCIES, ReportTypeEnum.FULL):
        codebase = Path(request.codebase_path) if request.codebase_path else Path(".")
        mapper = DependencyMapper(codebase)
        dep_map = mapper.scan_codebase()
        for dm in dep_map.values():
            dependency_mappings.append(
                CoreDependencyMapping(
                    janus_component=dm.first_seen,
                    external_application=dm.application_name,
                    usage_pattern=UsagePattern(
                        invocation_method=(
                            dm.invocations[0].invocation_method
                            if dm.invocations
                            else "unknown"
                        ),
                    ),
                    frequency=dm.invocation_count,
                    last_used=datetime.now(),
                    context=dm.first_seen,
                )
            )

    ranked_capabilities = []
    if request.report_type in (ReportTypeEnum.PRIORITY, ReportTypeEnum.FULL) and capabilities:
        engine = PriorityEngine()
        contexts = {
            cap.id: AnalysisContext(usage_frequency=0, max_frequency=1)
            for cap in capabilities
        }
        ranked_capabilities = engine.rank_capabilities(capabilities, contexts)

    generator = ReportGenerator()

    if request.report_type == ReportTypeEnum.SUMMARY:
        report_data = generator.generate_summary_report([scan_result])
    elif request.report_type == ReportTypeEnum.CAPABILITIES:
        report_data = generator.generate_capability_inventory(capabilities)
    elif request.report_type == ReportTypeEnum.DEPENDENCIES:
        report_data = generator.generate_dependency_report(dependency_mappings)
    elif request.report_type == ReportTypeEnum.PRIORITY:
        report_data = generator.generate_priority_report(ranked_capabilities)
    elif request.report_type == ReportTypeEnum.FULL:
        report_data = generator.generate_full_report(
            scan_results=[scan_result],
            capabilities=capabilities,
            dependencies=dependency_mappings,
            ranked_capabilities=ranked_capabilities,
        )
    else:
        raise ValueError(f"Unknown report type: {request.report_type}")

    return report_data


def _run_report_job(
    report_request: ReportRequest,
    job_store: Dict[str, Any],
    job_id: str,
) -> None:
    """Execute report generation and store the result in the job store."""
    try:
        job_store[job_id]["status"] = JobStatusEnum.RUNNING
        job_store[job_id]["started_at"] = datetime.utcnow()
        job_store[job_id]["progress_message"] = "Generating report…"

        report_data = _build_report(report_request)

        job_store[job_id].update(
            {
                "status": JobStatusEnum.COMPLETED,
                "completed_at": datetime.utcnow(),
                "progress_message": "Report generated successfully.",
                "result": {
                    "report_type": report_request.report_type.value,
                    "output_format": report_request.output_format.value,
                    "data": report_data,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            }
        )
        logger.info("Report job %s completed (type=%s)", job_id, report_request.report_type)

    except Exception as exc:
        logger.exception("Report job %s failed", job_id)
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
    response_model=ReportResponse,
    summary="Generate a report (synchronous)",
    description=(
        "Runs a full scan and generates the requested report type. "
        "For long-running reports use POST /report/async instead."
    ),
    status_code=status.HTTP_200_OK,
)
async def generate_report(
    request: ReportRequest,
    _api_key: str = Depends(require_api_key),
) -> ReportResponse:
    """Synchronous report generation."""
    try:
        report_data = _build_report(request)
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {exc}",
        )

    job_id = str(uuid.uuid4())
    return ReportResponse(
        job_id=job_id,
        report_type=request.report_type,
        output_format=request.output_format,
        data=report_data,
        generated_at=datetime.utcnow(),
    )


@router.post(
    "/async",
    response_model=AsyncJobResponse,
    summary="Start an async report generation job",
    description=(
        "Submits a report generation job for background processing and returns "
        "immediately with a job ID. Poll GET /report/jobs/{job_id} for results."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_async_report(
    http_request: Request,
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
) -> AsyncJobResponse:
    """Submit a report generation as a background job."""
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

    background_tasks.add_task(_run_report_job, report_request, job_store, job_id)

    base_url = str(http_request.base_url).rstrip("/")
    return AsyncJobResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        message="Report job submitted. Poll the status URL for updates.",
        created_at=job_store[job_id]["created_at"],
        status_url=f"{base_url}/report/jobs/{job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get async report job status",
    description="Returns the current status and result (when complete) of an async report job.",
)
async def get_report_job_status(
    job_id: str,
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> JobStatusResponse:
    """Retrieve the status of an async report job."""
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
