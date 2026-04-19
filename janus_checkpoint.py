"""
janus_checkpoint.py
====================
Job state checkpointing — saves and restores mid-job progress to SQLite
so a crash never loses work in progress.

Usage:
    from janus_checkpoint import Checkpointer

    cp = Checkpointer()

    # Save progress
    cp.save(job_id, stage="generating", data={"prompt": ..., "partial_work": ...})

    # Restore after crash
    state = cp.load(job_id)
    if state:
        resume from state["stage"] and state["data"]

    # Mark complete (removes checkpoint)
    cp.complete(job_id)
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("janus_checkpoints.db")

# Stages a job passes through
STAGES = [
    "claimed",
    "generating",
    "generated",
    "validating",
    "validated",
    "submitting",
    "submitted",
    "completed",
    "failed",
]


class Checkpointer:
    """
    Persists job execution state to SQLite so Janus can resume after a crash.

    Each checkpoint stores:
      - job_id       : unique job identifier
      - stage        : current execution stage (see STAGES)
      - data_json    : arbitrary JSON payload (partial work, prompts, etc.)
      - attempt      : how many times this job has been attempted
      - updated_at   : ISO timestamp of last update
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id      TEXT PRIMARY KEY,
                    stage       TEXT NOT NULL,
                    data_json   TEXT NOT NULL DEFAULT '{}',
                    attempt     INTEGER NOT NULL DEFAULT 1,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage ON checkpoints(stage)"
            )
            conn.commit()
        except Exception as e:
            logger.error("[Checkpointer] DB init failed: %s", e)
        finally:
            conn.close()

    # ── Core operations ───────────────────────────────────────────────────────

    def save(
        self,
        job_id: str,
        stage: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save or update a checkpoint for job_id.

        Args:
            job_id: Unique job identifier.
            stage:  Current execution stage.
            data:   Arbitrary dict payload (partial work, prompts, etc.).
        """
        now = datetime.now(timezone.utc).isoformat()
        data_json = json.dumps(data or {})
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO checkpoints (job_id, stage, data_json, attempt, updated_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    stage      = excluded.stage,
                    data_json  = excluded.data_json,
                    attempt    = attempt + 1,
                    updated_at = excluded.updated_at
                """,
                (job_id, stage, data_json, now),
            )
            conn.commit()
            logger.debug("[Checkpointer] saved %s → stage=%s", job_id, stage)
        except Exception as e:
            logger.error("[Checkpointer] save failed for %s: %s", job_id, e)
        finally:
            conn.close()

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the checkpoint for job_id.

        Returns a dict with keys: job_id, stage, data, attempt, updated_at
        or None if no checkpoint exists.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT job_id, stage, data_json, attempt, updated_at "
                "FROM checkpoints WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "job_id":     row[0],
                "stage":      row[1],
                "data":       json.loads(row[2]),
                "attempt":    row[3],
                "updated_at": row[4],
            }
        except Exception as e:
            logger.error("[Checkpointer] load failed for %s: %s", job_id, e)
            return None
        finally:
            conn.close()

    def complete(self, job_id: str) -> None:
        """Remove the checkpoint for a successfully completed job."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM checkpoints WHERE job_id = ?", (job_id,))
            conn.commit()
            logger.debug("[Checkpointer] cleared checkpoint for %s", job_id)
        except Exception as e:
            logger.error("[Checkpointer] complete failed for %s: %s", job_id, e)
        finally:
            conn.close()

    def fail(self, job_id: str, reason: str) -> None:
        """Mark a job as failed in the checkpoint (keeps record for audit)."""
        self.save(job_id, stage="failed", data={"reason": reason})

    def get_incomplete(self) -> List[Dict[str, Any]]:
        """
        Return all checkpoints that are not in 'completed' or 'failed' stage.
        Used on startup to resume interrupted jobs.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT job_id, stage, data_json, attempt, updated_at "
                "FROM checkpoints WHERE stage NOT IN ('completed', 'failed') "
                "ORDER BY updated_at ASC"
            ).fetchall()
            return [
                {
                    "job_id":     r[0],
                    "stage":      r[1],
                    "data":       json.loads(r[2]),
                    "attempt":    r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[Checkpointer] get_incomplete failed: %s", e)
            return []
        finally:
            conn.close()

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all checkpoints (for debugging/monitoring)."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT job_id, stage, data_json, attempt, updated_at "
                "FROM checkpoints ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {
                    "job_id":     r[0],
                    "stage":      r[1],
                    "data":       json.loads(r[2]),
                    "attempt":    r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[Checkpointer] get_all failed: %s", e)
            return []
        finally:
            conn.close()

    def summary(self) -> Dict[str, int]:
        """Return a count of checkpoints per stage."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT stage, COUNT(*) FROM checkpoints GROUP BY stage"
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception as e:
            logger.error("[Checkpointer] summary failed: %s", e)
            return {}
        finally:
            conn.close()


# ── Module-level singleton ────────────────────────────────────────────────────
_checkpointer: Optional[Checkpointer] = None


def get_checkpointer() -> Checkpointer:
    """Return the module-level Checkpointer singleton."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = Checkpointer()
    return _checkpointer
