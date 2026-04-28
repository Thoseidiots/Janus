"""
Transparency and logging for the Janus Autonomous Reasoning Engine.

Persists all decisions and activities to SQLite, provides query APIs,
and optionally sends alerts via janus_notify.

Requirements: REQ-9.3, REQ-8.5
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.safety.transparency")

# Optional janus_notify integration
try:
    import janus_notify as _janus_notify  # type: ignore
    _NOTIFY_AVAILABLE = True
except ImportError:
    _janus_notify = None
    _NOTIFY_AVAILABLE = False


_CREATE_DECISIONS_SQL = """
CREATE TABLE IF NOT EXISTS transparency_decisions (
    log_id      TEXT PRIMARY KEY,
    action      TEXT NOT NULL,
    reasoning   TEXT NOT NULL,
    context     TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""

_CREATE_ACTIVITIES_SQL = """
CREATE TABLE IF NOT EXISTS transparency_activities (
    log_id          TEXT PRIMARY KEY,
    activity_type   TEXT NOT NULL,
    details         TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""


class TransparencyLogger:
    """
    Logs all Janus decisions and activities to SQLite for full transparency.

    Optionally sends alerts via janus_notify when important events occur.

    Usage::

        tl = TransparencyLogger()
        log_id = tl.log_decision("apply for job", "best match for skills", {})
        recent = tl.get_recent_decisions(limit=5)
        summary = tl.get_activity_summary(hours=24)
    """

    def __init__(self, db_path: str = "janus_transparency.db") -> None:
        """
        Args:
            db_path: Path to the SQLite database file.
                     Use ":memory:" for an in-process database (useful for tests).
        """
        self.db_path = db_path
        # For in-memory databases we must keep a single persistent connection
        # because each new connect(":memory:") creates a fresh empty database.
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_DECISIONS_SQL)
        conn.execute(_CREATE_ACTIVITIES_SQL)
        conn.commit()
        if self._persistent_conn is None:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _exec(self, sql: str, params: tuple = ()) -> list:
        """Execute a SQL statement, committing writes. Returns fetchall() for SELECTs."""
        conn = self._connect()
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def log_decision(
        self,
        action: str,
        reasoning: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Log a decision with its reasoning and context.

        Args:
            action: Description of the action decided upon.
            reasoning: Explanation of why this action was chosen.
            context: Additional context dict (serialised to JSON).

        Returns:
            Unique log_id string.
        """
        log_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        context_json = json.dumps(context, default=str)

        self._exec(
            "INSERT INTO transparency_decisions "
            "(log_id, action, reasoning, context, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (log_id, action, reasoning, context_json, created_at),
        )

        logger.debug("Decision logged: %s — %s", log_id, action)
        return log_id

    def log_activity(self, activity_type: str, details: Dict[str, Any]) -> str:
        """
        Log a general activity event.

        Args:
            activity_type: Category of activity (e.g. "opportunity_scan").
            details: Activity details dict (serialised to JSON).

        Returns:
            Unique log_id string.
        """
        log_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        details_json = json.dumps(details, default=str)

        self._exec(
            "INSERT INTO transparency_activities "
            "(log_id, activity_type, details, created_at) "
            "VALUES (?, ?, ?, ?)",
            (log_id, activity_type, details_json, created_at),
        )

        logger.debug("Activity logged: %s — %s", log_id, activity_type)
        return log_id

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent decisions.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of decision dicts ordered newest-first.
        """
        rows = self._exec(
            "SELECT log_id, action, reasoning, context, created_at "
            "FROM transparency_decisions "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

        results = []
        for row in rows:
            log_id, action, reasoning, context_json, created_at = row
            results.append(
                {
                    "log_id": log_id,
                    "action": action,
                    "reasoning": reasoning,
                    "context": json.loads(context_json),
                    "created_at": created_at,
                }
            )
        return results

    def get_activity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Summarise activity over the last *hours* hours.

        Args:
            hours: Look-back window in hours.

        Returns:
            Dict with total_activities, by_type counts, and time_window.
        """
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        rows = self._exec(
            "SELECT activity_type, COUNT(*) "
            "FROM transparency_activities "
            "WHERE created_at >= ? "
            "GROUP BY activity_type",
            (since,),
        )

        total_rows = self._exec(
            "SELECT COUNT(*) FROM transparency_activities WHERE created_at >= ?",
            (since,),
        )
        total = total_rows[0][0] if total_rows else 0

        by_type: Dict[str, int] = {row[0]: row[1] for row in rows}

        return {
            "time_window_hours": hours,
            "total_activities": total,
            "by_type": by_type,
        }

    def send_alert(self, event_type: str, message: str) -> None:
        """
        Send an alert to the owner about an important event.

        Uses janus_notify if available; otherwise logs at WARNING level.

        Args:
            event_type: Category of the alert (e.g. "budget_exceeded").
            message: Human-readable alert message.
        """
        # Always log locally
        logger.warning("ALERT [%s]: %s", event_type, message)

        # Also persist as an activity
        self.log_activity("alert", {"event_type": event_type, "message": message})

        # Optionally notify via janus_notify
        if _NOTIFY_AVAILABLE and _janus_notify is not None:
            try:
                notify_fn = getattr(_janus_notify, "notify", None) or getattr(
                    _janus_notify, "send_notification", None
                )
                if notify_fn is not None:
                    notify_fn(f"[{event_type}] {message}")
                    logger.debug("Alert sent via janus_notify")
            except Exception as exc:
                logger.debug("janus_notify send failed (non-critical): %s", exc)
