"""
API key authentication for the Janus Dependency Analyzer REST API.

Provides a simple but secure API key scheme:
- Keys are passed via the ``X-API-Key`` request header.
- The set of valid keys is managed through the application state so it can be
  configured at startup without restarting the process.
- A default development key is provided when no keys are configured, but a
  warning is logged to encourage proper key management in production.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from typing import Optional, Set

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header scheme
# ---------------------------------------------------------------------------

API_KEY_HEADER_NAME = "X-API-Key"

_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,  # We raise our own 401 so the message is consistent
    description="API key for authenticating requests. Pass via the X-API-Key header.",
)


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------

class APIKeyStore:
    """
    In-memory store for valid API keys.

    Keys are stored as SHA-256 hashes so that the plaintext values are never
    held in memory after initial registration.
    """

    def __init__(self) -> None:
        self._hashed_keys: Set[str] = set()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_key(self, plaintext_key: str) -> None:
        """Register a new API key (stores its hash)."""
        self._hashed_keys.add(self._hash(plaintext_key))

    def remove_key(self, plaintext_key: str) -> bool:
        """Remove an API key. Returns True if the key existed."""
        h = self._hash(plaintext_key)
        if h in self._hashed_keys:
            self._hashed_keys.discard(h)
            return True
        return False

    def is_valid(self, plaintext_key: str) -> bool:
        """Return True if the given key is registered."""
        return self._hash(plaintext_key) in self._hashed_keys

    def is_empty(self) -> bool:
        """Return True when no keys have been registered."""
        return len(self._hashed_keys) == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Module-level default store (shared across the application)
# ---------------------------------------------------------------------------

_default_store = APIKeyStore()


def get_key_store() -> APIKeyStore:
    """Dependency that returns the module-level key store."""
    return _default_store


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def initialise_keys_from_env(store: Optional[APIKeyStore] = None) -> None:
    """
    Populate the key store from the ``JANUS_API_KEYS`` environment variable.

    The variable should contain one or more comma-separated API keys.  If the
    variable is not set and the store is empty, a single development key is
    generated and logged at WARNING level.
    """
    target = store or _default_store
    raw = os.environ.get("JANUS_API_KEYS", "")
    if raw:
        for key in (k.strip() for k in raw.split(",") if k.strip()):
            target.add_key(key)
        logger.info("Loaded %d API key(s) from JANUS_API_KEYS", len(raw.split(",")))
    elif target.is_empty():
        dev_key = secrets.token_urlsafe(32)
        target.add_key(dev_key)
        logger.warning(
            "No API keys configured. Using auto-generated development key: %s  "
            "Set JANUS_API_KEYS in the environment for production use.",
            dev_key,
        )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def require_api_key(
    request: Request,
    raw_key: Optional[str] = Security(_api_key_header),
    store: APIKeyStore = Depends(get_key_store),
) -> str:
    """
    FastAPI dependency that enforces API key authentication.

    Raises HTTP 401 when no key is provided and HTTP 403 when the key is
    invalid.  Returns the validated key on success.
    """
    if raw_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not store.is_valid(raw_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return raw_key
