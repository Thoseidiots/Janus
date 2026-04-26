"""
Pydantic request/response models for the Janus Dependency Analyzer REST API.

These models define the shape of all API inputs and outputs, providing
automatic validation and OpenAPI documentation generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums mirroring core models (kept as strings for JSON serialisation)
# ---------------------------------------------------------------------------

class PlatformEnum(str, Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


class CapabilityCategoryEnum(str, Enum):
    FILE_PROCESSING = "file_processing"
    NETWORK_OPERATIONS = "network_operations"
    DATA_TRANSFORMATION = "data_transformation"
    USER_INTERFACE = "user_interface"
    SYSTEM_INTEGRATION = "system_integration"
    DEVELOPMENT_TOOLS = "development_tools"
    MULTIMEDIA = "multimedia"
    SECURITY = "security"
    DATABASE = "database"
    COMMUNICATION = "communication"


class InterfaceTypeEnum(str, Enum):
    COMMAND_LINE = "command_line"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    RPC = "rpc"
    LIBRARY = "library"
    GUI = "gui"
    WEB_INTERFACE = "web_interface"
    PLUGIN = "plugin"


class ReportTypeEnum(str, Enum):
    SUMMARY = "summary"
    CAPABILITIES = "capabilities"
    DEPENDENCIES = "dependencies"
    PRIORITY = "priority"
    FULL = "full"


class ReportFormatEnum(str, Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class ApplicationResponse(BaseModel):
    """Serialised representation of a discovered application."""

    id: str
    name: str
    version: Optional[str] = None
    installation_path: str
    executable_path: str
    platform: PlatformEnum
    is_accessible: bool
    access_error: Optional[str] = None
    discovered_at: datetime

    model_config = {"from_attributes": True}


class CapabilityResponse(BaseModel):
    """Serialised representation of an application capability."""

    id: str
    application_id: str
    name: str
    category: CapabilityCategoryEnum
    description: str
    interface_type: InterfaceTypeEnum
    confidence_score: float = Field(ge=0.0, le=1.0)
    detection_method: str
    examples: List[str] = []
    documentation_url: Optional[str] = None
    supported_formats: List[str] = []

    model_config = {"from_attributes": True}


class ScanSummaryResponse(BaseModel):
    """Summary information from a scan operation."""

    platform: PlatformEnum
    scan_type: str
    scan_start_time: datetime
    scan_end_time: Optional[datetime] = None
    total_applications: int
    accessible_applications: int
    errors: List[str] = []
    warnings: List[str] = []


# ---------------------------------------------------------------------------
# Scan endpoints
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    """Request body for initiating a scan."""

    incremental: bool = Field(
        default=False,
        description="Perform an incremental scan instead of a full scan.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description=(
            "Baseline datetime for incremental scans. "
            "Defaults to 24 hours ago when not provided."
        ),
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration overrides for this scan.",
    )


class ScanResponse(BaseModel):
    """Response from a synchronous scan operation."""

    job_id: str
    summary: ScanSummaryResponse
    applications: List[ApplicationResponse]


class AsyncJobResponse(BaseModel):
    """Response returned immediately when an async job is submitted."""

    job_id: str
    status: JobStatusEnum
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status_url: str = Field(description="URL to poll for job status.")


class JobStatusResponse(BaseModel):
    """Current status of an async background job."""

    job_id: str
    status: JobStatusEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Analyse endpoints
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for capability analysis."""

    app_name: str = Field(
        description="Application name to analyse (case-insensitive substring match).",
        min_length=1,
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration overrides for this analysis.",
    )


class AnalyzeResponse(BaseModel):
    """Response from a capability analysis operation."""

    job_id: str
    application: ApplicationResponse
    capabilities: List[CapabilityResponse]
    total_capabilities: int


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    """Request body for report generation."""

    report_type: ReportTypeEnum = Field(
        default=ReportTypeEnum.SUMMARY,
        description="Type of report to generate.",
    )
    output_format: ReportFormatEnum = Field(
        default=ReportFormatEnum.JSON,
        description="Output format for the report.",
    )
    codebase_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the Janus codebase for dependency analysis. "
            "Required for 'dependencies' and 'full' report types."
        ),
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration overrides for this report.",
    )


class ReportResponse(BaseModel):
    """Response from a report generation operation."""

    job_id: str
    report_type: ReportTypeEnum
    output_format: ReportFormatEnum
    data: Dict[str, Any]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Configuration endpoints
# ---------------------------------------------------------------------------

class PriorityWeightsRequest(BaseModel):
    """Priority weight configuration."""

    usage: float = Field(default=0.3, ge=0.0, le=1.0)
    complexity: float = Field(default=0.2, ge=0.0, le=1.0)
    security: float = Field(default=0.25, ge=0.0, le=1.0)
    performance: float = Field(default=0.25, ge=0.0, le=1.0)


class ConfigurationRequest(BaseModel):
    """Request body for updating configuration."""

    scan_exclusion_patterns: Optional[List[str]] = None
    scan_timeout_seconds: Optional[int] = Field(default=None, gt=0)
    max_applications_per_scan: Optional[int] = Field(default=None, gt=0)
    capability_detection_rules: Optional[Dict[str, Any]] = None
    analysis_timeout_seconds: Optional[int] = Field(default=None, gt=0)
    min_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    priority_weights: Optional[PriorityWeightsRequest] = None
    respect_access_controls: Optional[bool] = None
    encrypt_stored_data: Optional[bool] = None
    audit_logging_enabled: Optional[bool] = None
    max_concurrent_analyses: Optional[int] = Field(default=None, gt=0)
    cache_analysis_results: Optional[bool] = None
    cache_expiry_hours: Optional[int] = Field(default=None, gt=0)
    report_formats: Optional[List[str]] = None
    include_charts: Optional[bool] = None
    sanitize_paths: Optional[bool] = None


class ConfigurationResponse(BaseModel):
    """Current configuration state."""

    scan_exclusion_patterns: List[str]
    scan_timeout_seconds: int
    max_applications_per_scan: int
    capability_detection_rules: Dict[str, Any]
    analysis_timeout_seconds: int
    min_confidence_threshold: float
    priority_weights: PriorityWeightsRequest
    respect_access_controls: bool
    encrypt_stored_data: bool
    audit_logging_enabled: bool
    max_concurrent_analyses: int
    cache_analysis_results: bool
    cache_expiry_hours: int
    report_formats: List[str]
    include_charts: bool
    sanitize_paths: bool


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response body."""

    error: str = "Validation error"
    detail: List[Dict[str, Any]]
