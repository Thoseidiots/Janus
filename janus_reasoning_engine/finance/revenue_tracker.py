"""
RevenueTracker — SQLite-backed revenue tracking and optimization.

Covers tasks 11.2 (core tracking) and 11.4 (profit-per-hour, revenue mix
optimization, top sources).

Requirements: REQ-10.2, REQ-10.4, REQ-10.5, REQ-5.3
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_EARNINGS_TABLE = """
CREATE TABLE IF NOT EXISTS earnings (
    id          TEXT PRIMARY KEY,
    amount      REAL NOT NULL,
    source      TEXT NOT NULL,
    job_id      TEXT,
    recorded_at TEXT NOT NULL
);
"""

_CREATE_JOB_HOURS_TABLE = """
CREATE TABLE IF NOT EXISTS job_hours (
    job_id      TEXT PRIMARY KEY,
    hours       REAL NOT NULL
);
"""


class RevenueTracker:
    """
    Tracks earnings, calculates profit, forecasts revenue, and optimises
    the revenue mix.

    All data is persisted in a local SQLite database so it survives
    process restarts.
    """

    def __init__(self, db_path: str = "janus_revenue.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_EARNINGS_TABLE)
            self._conn.execute(_CREATE_JOB_HOURS_TABLE)

    # ------------------------------------------------------------------
    # Task 11.2 — Core tracking
    # ------------------------------------------------------------------

    def record_earning(
        self,
        amount: float,
        source: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Record a new earning entry.

        Parameters
        ----------
        amount:  Positive monetary amount (USD).
        source:  Revenue source label (e.g. "upwork", "fiverr", "paypal").
        job_id:  Optional job identifier for per-job analytics.

        Returns
        -------
        The unique earning ID (UUID string).
        """
        if amount < 0:
            raise ValueError("amount must be non-negative")
        earning_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO earnings (id, amount, source, job_id, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (earning_id, amount, source, job_id, now),
            )
        logger.debug("Recorded earning %.2f from %s (id=%s)", amount, source, earning_id)
        return earning_id

    def get_total_earnings(self, period_days: int = 30) -> float:
        """
        Return total earnings over the last *period_days* days.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=period_days)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM earnings WHERE recorded_at >= ?",
            (cutoff,),
        ).fetchone()
        return float(row[0])

    def get_earnings_by_source(self) -> Dict[str, float]:
        """
        Return a mapping of source → total earnings (all time).
        """
        rows = self._conn.execute(
            "SELECT source, COALESCE(SUM(amount), 0) FROM earnings GROUP BY source"
        ).fetchall()
        return {row[0]: float(row[1]) for row in rows}

    def calculate_profit(self, revenue: float, expenses: float) -> float:
        """
        Calculate profit as revenue minus expenses.
        """
        return revenue - expenses

    def forecast_monthly(self, history_days: int = 30) -> float:
        """
        Forecast monthly revenue based on the average daily rate over
        the last *history_days* days.
        """
        total = self.get_total_earnings(period_days=history_days)
        if history_days <= 0:
            return 0.0
        daily_avg = total / history_days
        return daily_avg * 30.0

    # ------------------------------------------------------------------
    # Task 11.4 — Money-making optimisation
    # ------------------------------------------------------------------

    def record_job_hours(self, job_id: str, hours: float) -> None:
        """
        Record (or update) the number of hours spent on a job.
        Used by get_profit_per_hour.
        """
        if hours <= 0:
            raise ValueError("hours must be positive")
        with self._conn:
            self._conn.execute(
                "INSERT INTO job_hours (job_id, hours) VALUES (?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET hours = excluded.hours",
                (job_id, hours),
            )

    def get_profit_per_hour(self, job_id: str) -> float:
        """
        Return the profit-per-hour for a specific job.

        Profit = total earnings for that job_id.
        Hours  = recorded via record_job_hours().

        Returns 0.0 if the job has no earnings or no hours recorded.
        """
        earnings_row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM earnings WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        hours_row = self._conn.execute(
            "SELECT hours FROM job_hours WHERE job_id = ?",
            (job_id,),
        ).fetchone()

        total_earnings = float(earnings_row[0])
        hours = float(hours_row[0]) if hours_row else 0.0

        if hours == 0.0:
            return 0.0
        return total_earnings / hours

    def optimize_revenue_mix(
        self, opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank a list of opportunities by estimated profit-per-hour (descending).

        Each opportunity dict should contain at least:
            - "profit_per_hour" (float)  — used directly if present, OR
            - "estimated_revenue" (float) and "estimated_hours" (float)

        Returns the same list sorted best-first.
        """
        def _score(opp: Dict[str, Any]) -> float:
            if "profit_per_hour" in opp:
                return float(opp["profit_per_hour"])
            revenue = float(opp.get("estimated_revenue", 0))
            hours = float(opp.get("estimated_hours", 1)) or 1.0
            return revenue / hours

        return sorted(opportunities, key=_score, reverse=True)

    def get_top_revenue_sources(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Return the top *n* revenue sources by total earnings.

        Each entry is a dict with keys: source, total_earnings, percentage.
        """
        rows = self._conn.execute(
            "SELECT source, COALESCE(SUM(amount), 0) AS total "
            "FROM earnings GROUP BY source ORDER BY total DESC LIMIT ?",
            (n,),
        ).fetchall()

        grand_total = self.get_total_earnings(period_days=36500)  # all-time
        result: List[Dict[str, Any]] = []
        for row in rows:
            pct = (float(row["total"]) / grand_total * 100) if grand_total > 0 else 0.0
            result.append(
                {
                    "source": row["source"],
                    "total_earnings": float(row["total"]),
                    "percentage": round(pct, 2),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
