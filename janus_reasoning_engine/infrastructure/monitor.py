"""
SystemMonitor — performance snapshots, metric recording, history, and threshold alerts.

All external integrations are optional; degrades gracefully when modules
are unavailable.

Requirements: REQ-12.2, REQ-9.3
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_monitoring_dashboard import MonitoringDashboard  # type: ignore
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    logger.debug("janus_monitoring_dashboard not available — using built-in monitor")

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.debug("psutil not available — CPU/memory will report 0.0")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    timestamp: float
    cpu_pct: float
    memory_pct: float
    active_tasks: int
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------------

class SystemMonitor:
    """
    Collects and exposes system performance metrics.

    Wraps janus_monitoring_dashboard when available; otherwise uses
    psutil (if installed) and in-memory metric storage.
    """

    def __init__(self, history_limit: int = 1000) -> None:
        self._history_limit = history_limit
        self._metric_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._history_limit)
        )
        self._active_tasks: int = 0
        self._dashboard: Optional[object] = None

        if _DASHBOARD_AVAILABLE:
            try:
                self._dashboard = MonitoringDashboard()
                logger.info("MonitoringDashboard backend initialised")
            except Exception as exc:
                logger.warning("MonitoringDashboard init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Return a current performance snapshot."""
        if self._dashboard is not None:
            try:
                raw = self._dashboard.get_snapshot()  # type: ignore[union-attr]
                if raw is not None:
                    return PerformanceSnapshot(
                        timestamp=float(getattr(raw, "timestamp", time.time())),
                        cpu_pct=float(getattr(raw, "cpu_pct", 0.0)),
                        memory_pct=float(getattr(raw, "memory_pct", 0.0)),
                        active_tasks=int(getattr(raw, "active_tasks", self._active_tasks)),
                        metrics=dict(getattr(raw, "metrics", {})),
                    )
            except Exception as exc:
                logger.debug("dashboard.get_snapshot failed: %s", exc)

        cpu_pct = 0.0
        memory_pct = 0.0
        if _PSUTIL_AVAILABLE:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
                memory_pct = float(psutil.virtual_memory().percent)
            except Exception as exc:
                logger.debug("psutil read failed: %s", exc)

        # Snapshot of latest recorded metrics
        latest_metrics: Dict[str, float] = {
            name: hist[-1] for name, hist in self._metric_history.items() if hist
        }

        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_pct=cpu_pct,
            memory_pct=memory_pct,
            active_tasks=self._active_tasks,
            metrics=latest_metrics,
        )

    def record_metric(self, name: str, value: float) -> None:
        """Record a named metric value."""
        self._metric_history[name].append(value)
        logger.debug("Metric recorded: %s = %s", name, value)

        if self._dashboard is not None:
            try:
                self._dashboard.record(name=name, value=value)  # type: ignore[union-attr]
            except Exception as exc:
                logger.debug("dashboard.record failed: %s", exc)

    def get_metric_history(self, name: str, limit: int = 100) -> List[float]:
        """Return the last *limit* recorded values for a named metric."""
        hist = self._metric_history.get(name)
        if not hist:
            return []
        values = list(hist)
        return values[-limit:]

    def alert_if_threshold(
        self,
        name: str,
        threshold: float,
        direction: str = "above",
    ) -> bool:
        """
        Return True (and log a warning) if the latest value for *name*
        breaches *threshold* in the given *direction* ("above" or "below").
        """
        hist = self._metric_history.get(name)
        if not hist:
            return False

        latest = hist[-1]
        breached = (
            (direction == "above" and latest > threshold)
            or (direction == "below" and latest < threshold)
        )

        if breached:
            logger.warning(
                "Threshold alert: %s = %.4f is %s %.4f",
                name,
                latest,
                direction,
                threshold,
            )

        return breached
