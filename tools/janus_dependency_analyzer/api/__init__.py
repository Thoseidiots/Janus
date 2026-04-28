"""
REST API for the Janus Dependency Analyzer.

Provides programmatic access to all analyzer functionality via a FastAPI-based
HTTP interface with API key authentication, rate limiting, and async processing
for long-running operations.
"""

from .app import create_app

__all__ = ["create_app"]
