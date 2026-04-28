"""
FastAPI application factory for the Janus Dependency Analyzer REST API.

Usage
-----
Start the server with uvicorn::

    uvicorn janus_dependency_analyzer.api.app:app --reload

Or programmatically::

    from janus_dependency_analyzer.api import create_app
    app = create_app()

Environment variables
---------------------
JANUS_API_KEYS
    Comma-separated list of valid API keys.  When not set a single
    development key is auto-generated and logged at WARNING level.

JANUS_RATE_LIMIT_REQUESTS
    Maximum number of requests per rate-limit window (default: 60).

JANUS_RATE_LIMIT_WINDOW
    Rate-limit window size in seconds (default: 60).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.models import Configuration
from .auth import APIKeyStore, initialise_keys_from_env
from .rate_limit import RateLimitMiddleware, RateLimiter
from .routes import analyze, configure, report, scan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    *,
    initial_config: Optional[Configuration] = None,
    api_keys: Optional[list] = None,
    rate_limiter: Optional[RateLimiter] = None,
    title: str = "Janus Dependency Analyzer API",
    version: str = "1.0.0",
) -> FastAPI:  # noqa: C901
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    initial_config:
        Starting ``Configuration`` object.  Defaults to ``Configuration()``.
    api_keys:
        Explicit list of plaintext API keys to register.  When ``None`` the
        keys are loaded from the ``JANUS_API_KEYS`` environment variable (or
        an auto-generated development key is used as a fallback).
    rate_limiter:
        Custom ``RateLimiter`` instance.  Defaults to the module-level limiter
        configured via environment variables.
    title:
        OpenAPI title shown in the docs UI.
    version:
        API version string shown in the docs UI.
    """
    # ------------------------------------------------------------------
    # Startup / shutdown events (using modern lifespan handler)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(application: FastAPI):  # type: ignore[override]
        logger.info("Janus Dependency Analyzer API starting up (version %s)", version)
        yield
        logger.info("Janus Dependency Analyzer API shutting down")

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
        description=(
            "REST API for the Janus Dependency Analyzer. "
            "Provides programmatic access to system scanning, capability analysis, "
            "dependency mapping, report generation, and configuration management. "
            "\n\n"
            "**Authentication**: All endpoints (except `/health`, `/docs`, `/redoc`, "
            "and `/openapi.json`) require an API key passed via the `X-API-Key` header."
            "\n\n"
            "**Rate limiting**: Requests are rate-limited per API key "
            "(or IP address when no key is present). "
            "The current limits are returned in `X-RateLimit-*` response headers."
            "\n\n"
            "**Async processing**: Long-running operations (scans, analysis, reports) "
            "can be submitted as background jobs via the `/*/async` endpoints. "
            "Poll the returned `status_url` to check progress and retrieve results."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Janus Development Team",
            "email": "dev@janus.ai",
        },
        license_info={
            "name": "MIT",
        },
    )

    # ------------------------------------------------------------------
    # Application state
    # ------------------------------------------------------------------

    app.state.config = initial_config or Configuration()
    app.state.job_store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    key_store = APIKeyStore()
    if api_keys:
        for key in api_keys:
            key_store.add_key(key)
    else:
        initialise_keys_from_env(key_store)

    # Override the module-level default store so the auth dependency picks
    # up the keys we just registered.
    import janus_dependency_analyzer.api.auth as _auth_module
    _auth_module._default_store = key_store

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    # CORS – allow all origins by default (tighten in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    limiter = rate_limiter or RateLimiter()
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    app.include_router(scan.router)
    app.include_router(analyze.router)
    app.include_router(report.router)
    app.include_router(configure.router)

    # ------------------------------------------------------------------
    # Root and health endpoints (no auth required)
    # ------------------------------------------------------------------

    @app.get(
        "/",
        tags=["Health"],
        summary="API root",
        description="Returns basic API information.",
        include_in_schema=True,
    )
    async def root() -> dict:
        return {
            "name": title,
            "version": version,
            "status": "ok",
            "docs": "/docs",
        }

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check",
        description="Returns the health status of the API server.",
        include_in_schema=True,
    )
    async def health() -> dict:
        return {"status": "healthy"}

    return app


# ---------------------------------------------------------------------------
# Default application instance (for uvicorn / gunicorn)
# ---------------------------------------------------------------------------

app = create_app()
