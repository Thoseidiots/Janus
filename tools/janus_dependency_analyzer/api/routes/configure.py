"""
Configuration routes for the Janus Dependency Analyzer REST API.

Provides endpoints for reading and updating the analyzer configuration
at runtime without restarting the server.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ...core.models import Configuration, PriorityWeights
from ..auth import require_api_key
from ..models import (
    ConfigurationRequest,
    ConfigurationResponse,
    PriorityWeightsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/configure", tags=["Configure"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_response(config: Configuration) -> ConfigurationResponse:
    """Convert a core Configuration dataclass to an API response model."""
    return ConfigurationResponse(
        scan_exclusion_patterns=config.scan_exclusion_patterns,
        scan_timeout_seconds=config.scan_timeout_seconds,
        max_applications_per_scan=config.max_applications_per_scan,
        capability_detection_rules=config.capability_detection_rules,
        analysis_timeout_seconds=config.analysis_timeout_seconds,
        min_confidence_threshold=config.min_confidence_threshold,
        priority_weights=PriorityWeightsRequest(
            usage=config.priority_weights.usage,
            complexity=config.priority_weights.complexity,
            security=config.priority_weights.security,
            performance=config.priority_weights.performance,
        ),
        respect_access_controls=config.respect_access_controls,
        encrypt_stored_data=config.encrypt_stored_data,
        audit_logging_enabled=config.audit_logging_enabled,
        max_concurrent_analyses=config.max_concurrent_analyses,
        cache_analysis_results=config.cache_analysis_results,
        cache_expiry_hours=config.cache_expiry_hours,
        report_formats=config.report_formats,
        include_charts=config.include_charts,
        sanitize_paths=config.sanitize_paths,
    )


def _apply_overrides(config: Configuration, req: ConfigurationRequest) -> Configuration:
    """Return a new Configuration with the requested fields overridden."""
    # Build a dict of current values and apply non-None overrides
    updates: dict = {}

    if req.scan_exclusion_patterns is not None:
        updates["scan_exclusion_patterns"] = req.scan_exclusion_patterns
    if req.scan_timeout_seconds is not None:
        updates["scan_timeout_seconds"] = req.scan_timeout_seconds
    if req.max_applications_per_scan is not None:
        updates["max_applications_per_scan"] = req.max_applications_per_scan
    if req.capability_detection_rules is not None:
        updates["capability_detection_rules"] = req.capability_detection_rules
    if req.analysis_timeout_seconds is not None:
        updates["analysis_timeout_seconds"] = req.analysis_timeout_seconds
    if req.min_confidence_threshold is not None:
        updates["min_confidence_threshold"] = req.min_confidence_threshold
    if req.priority_weights is not None:
        updates["priority_weights"] = PriorityWeights(
            usage=req.priority_weights.usage,
            complexity=req.priority_weights.complexity,
            security=req.priority_weights.security,
            performance=req.priority_weights.performance,
        )
    if req.respect_access_controls is not None:
        updates["respect_access_controls"] = req.respect_access_controls
    if req.encrypt_stored_data is not None:
        updates["encrypt_stored_data"] = req.encrypt_stored_data
    if req.audit_logging_enabled is not None:
        updates["audit_logging_enabled"] = req.audit_logging_enabled
    if req.max_concurrent_analyses is not None:
        updates["max_concurrent_analyses"] = req.max_concurrent_analyses
    if req.cache_analysis_results is not None:
        updates["cache_analysis_results"] = req.cache_analysis_results
    if req.cache_expiry_hours is not None:
        updates["cache_expiry_hours"] = req.cache_expiry_hours
    if req.report_formats is not None:
        updates["report_formats"] = req.report_formats
    if req.include_charts is not None:
        updates["include_charts"] = req.include_charts
    if req.sanitize_paths is not None:
        updates["sanitize_paths"] = req.sanitize_paths

    # Create a new Configuration with the merged values
    import dataclasses
    return dataclasses.replace(config, **updates)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=ConfigurationResponse,
    summary="Get current configuration",
    description="Returns the active analyzer configuration.",
)
async def get_configuration(
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> ConfigurationResponse:
    """Return the current configuration."""
    config: Configuration = request.app.state.config
    return _config_to_response(config)


@router.put(
    "/",
    response_model=ConfigurationResponse,
    summary="Update configuration",
    description=(
        "Applies partial configuration updates. Only the fields provided in the "
        "request body are changed; all other settings retain their current values. "
        "Changes take effect immediately for subsequent requests."
    ),
    status_code=status.HTTP_200_OK,
)
async def update_configuration(
    config_request: ConfigurationRequest,
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> ConfigurationResponse:
    """Apply partial configuration updates."""
    current_config: Configuration = request.app.state.config
    updated_config = _apply_overrides(current_config, config_request)

    # Validate the updated configuration
    validation = updated_config.validate()
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Invalid configuration",
                "validation_errors": validation.errors,
                "warnings": validation.warnings,
            },
        )

    request.app.state.config = updated_config
    logger.info("Configuration updated via API")

    return _config_to_response(updated_config)


@router.post(
    "/reset",
    response_model=ConfigurationResponse,
    summary="Reset configuration to defaults",
    description="Resets all configuration settings to their default values.",
    status_code=status.HTTP_200_OK,
)
async def reset_configuration(
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> ConfigurationResponse:
    """Reset configuration to defaults."""
    default_config = Configuration()
    request.app.state.config = default_config
    logger.info("Configuration reset to defaults via API")
    return _config_to_response(default_config)


@router.get(
    "/validate",
    summary="Validate current configuration",
    description="Validates the current configuration and returns any errors or warnings.",
)
async def validate_configuration(
    request: Request,
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Validate the current configuration."""
    config: Configuration = request.app.state.config
    result = config.validate()
    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }
