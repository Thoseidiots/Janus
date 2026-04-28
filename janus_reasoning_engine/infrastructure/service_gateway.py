"""
ServiceGateway — task submission, status tracking, and credits balance.

All external integrations are optional; degrades gracefully when modules
are unavailable.

Requirements: REQ-12.1, REQ-12.4
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_service_gateway import JanusServiceGateway  # type: ignore
    _GATEWAY_AVAILABLE = True
except ImportError:
    _GATEWAY_AVAILABLE = False
    logger.debug("janus_service_gateway not available — using in-memory stub")

try:
    from janus_credits import JanusCredits  # type: ignore
    _CREDITS_AVAILABLE = True
except ImportError:
    _CREDITS_AVAILABLE = False
    logger.debug("janus_credits not available — balance returns 0.0")


# ---------------------------------------------------------------------------
# ServiceGateway
# ---------------------------------------------------------------------------

class ServiceGateway:
    """
    Unified interface for task submission, status queries, and credits.
    """

    _VALID_STATUSES = {"pending", "running", "done", "unknown"}

    def __init__(self) -> None:
        self._tasks: Dict[str, str] = {}  # task_id → status
        self._gateway: Optional[object] = None
        self._credits: Optional[object] = None

        if _GATEWAY_AVAILABLE:
            try:
                self._gateway = JanusServiceGateway()
                logger.info("JanusServiceGateway backend initialised")
            except Exception as exc:
                logger.warning("JanusServiceGateway init failed: %s", exc)

        if _CREDITS_AVAILABLE:
            try:
                self._credits = JanusCredits()
                logger.info("JanusCredits backend initialised")
            except Exception as exc:
                logger.warning("JanusCredits init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_task(self, task_description: str, priority: int = 5) -> str:
        """
        Submit a task for execution.

        Returns a unique task_id string.  Delegates to janus_service_gateway
        when available; otherwise stores the task locally with status "pending".
        """
        task_id = str(uuid.uuid4())

        if self._gateway is not None:
            try:
                result = self._gateway.submit(  # type: ignore[union-attr]
                    description=task_description,
                    priority=priority,
                )
                task_id = str(getattr(result, "task_id", task_id) or task_id)
                logger.info("Task submitted via gateway: task_id=%s", task_id)
            except Exception as exc:
                logger.warning("gateway.submit failed: %s", exc)

        self._tasks[task_id] = "pending"
        return task_id

    def get_task_status(self, task_id: str) -> str:
        """
        Return the status of a task: "pending" | "running" | "done" | "unknown".
        """
        if self._gateway is not None:
            try:
                raw = self._gateway.get_status(task_id=task_id)  # type: ignore[union-attr]
                status = str(raw).lower() if raw else "unknown"
                if status in self._VALID_STATUSES:
                    self._tasks[task_id] = status
                    return status
            except Exception as exc:
                logger.debug("gateway.get_status failed: %s", exc)

        return self._tasks.get(task_id, "unknown")

    def get_credits_balance(self) -> float:
        """Return the current credits balance (0.0 if unavailable)."""
        if self._credits is not None:
            try:
                balance = self._credits.get_balance()  # type: ignore[union-attr]
                return float(balance)
            except Exception as exc:
                logger.debug("credits.get_balance failed: %s", exc)

        return 0.0
