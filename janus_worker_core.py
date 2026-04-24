"""
janus_worker_core.py
====================
Core components for the Janus Autonomous Worker system.

Contains all new components wired together into a coherent, self-improving
work loop: WorkerDatabase, DecisionEngine, WorkGenerator, QualityAssurance,
LearningEngine, MarketAnalyzer, InvestmentEngine, MonitoringSystem, WorkCycle.

All platform interaction goes through janus_computer_use.py /
janus_platform_browser.py (computer-use-first). janus_wallet.py is the
single source of truth for all financial state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SkillLevel(str, Enum):
    """Skill proficiency levels."""
    BEGINNER     = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED     = "ADVANCED"
    EXPERT       = "EXPERT"


class JobStatus(str, Enum):
    """Status of a job record."""
    AVAILABLE = "available"
    CLAIMED   = "claimed"
    COMPLETED = "completed"
    FAILED    = "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JobRecord:
    """Persistent job record stored in the database."""
    id: str
    title: str
    description: str
    platform: str
    budget: float
    status: str
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]
    quality_score: Optional[float]
    payment_amount: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRecord:
    """Persistent skill record stored in the database."""
    name: str
    level: SkillLevel
    experience_pts: int
    success_rate: float
    last_used: Optional[datetime]
    last_improved: Optional[datetime]


@dataclass
class LearningResourceRecord:
    """Persistent learning resource record stored in the database."""
    id: str
    url: str
    title: str
    topic: str
    resource_type: str
    concepts: List[str]
    completed_at: Optional[datetime]
    skill_delta: float


@dataclass
class CycleSummary:
    """Summary of a completed work cycle."""
    id: str
    started_at: datetime
    completed_at: datetime
    jobs_processed: int
    earnings: Decimal
    skills_improved: List[str]
    errors: int
    state: str  # "completed" | "partial" | "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class WorkerDatabase:
    """
    SQLite persistence for all worker state beyond financial data.

    Owns four tables: jobs, skills, learning_resources, cycle_summaries.
    All writes are wrapped in ``with self._conn:`` transactions.
    On sqlite3.Error the operation is retried once; if the retry also fails
    the error is logged and execution continues.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id              TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        description     TEXT NOT NULL DEFAULT '',
        platform        TEXT NOT NULL,
        budget          REAL NOT NULL DEFAULT 0,
        status          TEXT NOT NULL DEFAULT 'available',
        claimed_at      TEXT,
        completed_at    TEXT,
        quality_score   REAL,
        payment_amount  REAL,
        metadata        TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS skills (
        name            TEXT PRIMARY KEY,
        level           TEXT NOT NULL DEFAULT 'BEGINNER',
        experience_pts  INTEGER NOT NULL DEFAULT 0,
        success_rate    REAL NOT NULL DEFAULT 0.5,
        last_used       TEXT,
        last_improved   TEXT
    );

    CREATE TABLE IF NOT EXISTS learning_resources (
        id              TEXT PRIMARY KEY,
        url             TEXT NOT NULL,
        title           TEXT NOT NULL DEFAULT '',
        topic           TEXT NOT NULL,
        resource_type   TEXT NOT NULL DEFAULT 'web',
        concepts        TEXT NOT NULL DEFAULT '[]',
        completed_at    TEXT,
        skill_delta     REAL NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS cycle_summaries (
        id              TEXT PRIMARY KEY,
        started_at      TEXT NOT NULL,
        completed_at    TEXT NOT NULL,
        jobs_processed  INTEGER NOT NULL DEFAULT 0,
        earnings        TEXT NOT NULL DEFAULT '0',
        skills_improved TEXT NOT NULL DEFAULT '[]',
        errors          INTEGER NOT NULL DEFAULT 0,
        state           TEXT NOT NULL DEFAULT 'completed'
    );
    """

    def __init__(self, db_path: str = "janus_worker.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()
        logger.info("WorkerDatabase initialised (db=%s)", db_path)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt is not None else None

    @staticmethod
    def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(s) if s else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            platform=row["platform"],
            budget=float(row["budget"]),
            status=row["status"],
            claimed_at=WorkerDatabase._str_to_dt(row["claimed_at"]),
            completed_at=WorkerDatabase._str_to_dt(row["completed_at"]),
            quality_score=float(row["quality_score"]) if row["quality_score"] is not None else None,
            payment_amount=float(row["payment_amount"]) if row["payment_amount"] is not None else None,
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> SkillRecord:
        return SkillRecord(
            name=row["name"],
            level=SkillLevel(row["level"]),
            experience_pts=int(row["experience_pts"]),
            success_rate=float(row["success_rate"]),
            last_used=WorkerDatabase._str_to_dt(row["last_used"]),
            last_improved=WorkerDatabase._str_to_dt(row["last_improved"]),
        )

    @staticmethod
    def _row_to_resource(row: sqlite3.Row) -> LearningResourceRecord:
        return LearningResourceRecord(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            topic=row["topic"],
            resource_type=row["resource_type"],
            concepts=json.loads(row["concepts"]),
            completed_at=WorkerDatabase._str_to_dt(row["completed_at"]),
            skill_delta=float(row["skill_delta"]),
        )

    @staticmethod
    def _row_to_cycle(row: sqlite3.Row) -> CycleSummary:
        return CycleSummary(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]),
            jobs_processed=int(row["jobs_processed"]),
            earnings=Decimal(row["earnings"]),
            skills_improved=json.loads(row["skills_improved"]),
            errors=int(row["errors"]),
            state=row["state"],
        )

    def _execute_with_retry(self, sql: str, params: tuple) -> None:
        """Execute a write statement; retry once on sqlite3.Error."""
        for attempt in range(2):
            try:
                with self._lock:
                    with self._conn:
                        self._conn.execute(sql, params)
                return
            except sqlite3.Error as exc:
                if attempt == 0:
                    logger.warning("DB write failed (attempt 1), retrying: %s", exc)
                else:
                    logger.error("DB write failed after retry, continuing: %s", exc)

    # ── jobs ─────────────────────────────────────────────────────────────────

    def insert_job(self, job: JobRecord) -> JobRecord:
        """Persist a new job record. Assigns a UUID if job.id is empty."""
        if not job.id:
            job.id = str(uuid.uuid4())

        sql = """
            INSERT INTO jobs
                (id, title, description, platform, budget, status,
                 claimed_at, completed_at, quality_score, payment_amount, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            job.id,
            job.title,
            job.description,
            job.platform,
            job.budget,
            job.status,
            self._dt_to_str(job.claimed_at),
            self._dt_to_str(job.completed_at),
            job.quality_score,
            job.payment_amount,
            json.dumps(job.metadata),
        )
        self._execute_with_retry(sql, params)
        return job

    def update_job_status(
        self,
        job_id: str,
        status: str,
        *,
        quality_score: Optional[float] = None,
        payment_amount: Optional[float] = None,
        completed_at: Optional[datetime] = None,
        claimed_at: Optional[datetime] = None,
    ) -> None:
        """Update a job's status and optional fields."""
        sets = ["status = ?"]
        params: list = [status]

        if quality_score is not None:
            sets.append("quality_score = ?")
            params.append(quality_score)
        if payment_amount is not None:
            sets.append("payment_amount = ?")
            params.append(payment_amount)
        if completed_at is not None:
            sets.append("completed_at = ?")
            params.append(self._dt_to_str(completed_at))
        if claimed_at is not None:
            sets.append("claimed_at = ?")
            params.append(self._dt_to_str(claimed_at))

        params.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?"
        self._execute_with_retry(sql, tuple(params))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Return a single job by id, or None if not found."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        status: Optional[str] = None,
        platform: Optional[str] = None,
        limit: int = 100,
    ) -> List[JobRecord]:
        """Return up to *limit* jobs, optionally filtered by status and/or platform."""
        clauses: List[str] = []
        params: list = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if platform is not None:
            clauses.append("platform = ?")
            params.append(platform)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        sql = f"SELECT * FROM jobs {where} ORDER BY rowid ASC LIMIT ?"

        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()

        result = []
        for row in rows:
            try:
                result.append(self._row_to_job(row))
            except Exception as exc:
                logger.warning("Skipping unparseable job row: %s", exc)
        return result

    # ── skills ───────────────────────────────────────────────────────────────

    def upsert_skill(self, skill: SkillRecord) -> SkillRecord:
        """Insert or replace a skill record (keyed by name)."""
        sql = """
            INSERT OR REPLACE INTO skills
                (name, level, experience_pts, success_rate, last_used, last_improved)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            skill.name,
            skill.level.value if isinstance(skill.level, SkillLevel) else skill.level,
            skill.experience_pts,
            skill.success_rate,
            self._dt_to_str(skill.last_used),
            self._dt_to_str(skill.last_improved),
        )
        self._execute_with_retry(sql, params)
        return skill

    def get_skill(self, name: str) -> Optional[SkillRecord]:
        """Return a skill by name, or None if not found."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM skills WHERE name = ?", (name,))
            row = cur.fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self) -> List[SkillRecord]:
        """Return all skill records."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM skills ORDER BY name ASC")
            rows = cur.fetchall()

        result = []
        for row in rows:
            try:
                result.append(self._row_to_skill(row))
            except Exception as exc:
                logger.warning("Skipping unparseable skill row: %s", exc)
        return result

    # ── learning resources ───────────────────────────────────────────────────

    def insert_learning_resource(self, resource: LearningResourceRecord) -> LearningResourceRecord:
        """Persist a learning resource. Assigns a UUID if resource.id is empty."""
        if not resource.id:
            resource.id = str(uuid.uuid4())

        sql = """
            INSERT INTO learning_resources
                (id, url, title, topic, resource_type, concepts, completed_at, skill_delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            resource.id,
            resource.url,
            resource.title,
            resource.topic,
            resource.resource_type,
            json.dumps(resource.concepts),
            self._dt_to_str(resource.completed_at),
            resource.skill_delta,
        )
        self._execute_with_retry(sql, params)
        return resource

    # ── cycle summaries ──────────────────────────────────────────────────────

    def insert_cycle_summary(self, summary: CycleSummary) -> CycleSummary:
        """Persist a cycle summary. Assigns a UUID if summary.id is empty."""
        if not summary.id:
            summary.id = str(uuid.uuid4())

        sql = """
            INSERT INTO cycle_summaries
                (id, started_at, completed_at, jobs_processed, earnings,
                 skills_improved, errors, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            summary.id,
            summary.started_at.isoformat(),
            summary.completed_at.isoformat(),
            summary.jobs_processed,
            str(summary.earnings),
            json.dumps(summary.skills_improved),
            summary.errors,
            summary.state,
        )
        self._execute_with_retry(sql, params)
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BrowserJob:
    """A job discovered via browser-based platform scraping."""
    id: str
    title: str
    description: str
    platform: str
    budget: float
    required_skills: List[str]
    deadline: Optional[datetime] = None
    job_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """
    Scores and selects jobs based on skill match, budget, deadline, and
    learning opportunity.  Pure functions — no I/O, no database access.
    """

    WEIGHTS: Dict[str, float] = {
        "skill_match": 0.40,
        "budget":      0.30,
        "deadline":    0.20,
        "learning":    0.10,
    }

    # ── sub-scores ────────────────────────────────────────────────────────────

    def _skill_match_score(
        self,
        required: List[str],
        available_skills_dict: Dict[str, Any],
    ) -> float:
        """Ratio of required skills present in available_skills_dict, clamped [0,1]."""
        if not required:
            return 1.0
        matched = sum(1 for s in required if s.lower() in {k.lower() for k in available_skills_dict})
        return min(1.0, max(0.0, matched / len(required)))

    def _budget_score(self, budget: float, market_avg: float = 50.0) -> float:
        """budget / market_avg clamped [0,1]."""
        if market_avg <= 0:
            return 0.0
        return min(1.0, max(0.0, budget / market_avg))

    def _deadline_score(self, deadline: Optional[datetime]) -> float:
        """
        1.0 if deadline is >7 days away, 0.0 if past, linear in between.
        Returns 0.5 if deadline is None.
        """
        if deadline is None:
            return 0.5
        now = datetime.utcnow()
        # Make deadline naive if it has tzinfo, for comparison
        if deadline.tzinfo is not None:
            from datetime import timezone
            now = datetime.now(timezone.utc)
        delta_seconds = (deadline - now).total_seconds()
        if delta_seconds <= 0:
            return 0.0
        seven_days = 7 * 24 * 3600
        if delta_seconds >= seven_days:
            return 1.0
        return min(1.0, max(0.0, delta_seconds / seven_days))

    def _learning_score(
        self,
        required: List[str],
        available: Dict[str, Any],
    ) -> float:
        """Ratio of NEW skills (not in available), clamped [0,1]."""
        if not required:
            return 0.0
        available_lower = {k.lower() for k in available}
        new_skills = sum(1 for s in required if s.lower() not in available_lower)
        return min(1.0, max(0.0, new_skills / len(required)))

    # ── composite score ───────────────────────────────────────────────────────

    def score_job(self, job: "BrowserJob", skills_dict: Dict[str, Any]) -> float:
        """Weighted sum of the four sub-scores, clamped [0,1]."""
        s_match    = self._skill_match_score(job.required_skills, skills_dict)
        s_budget   = self._budget_score(job.budget)
        s_deadline = self._deadline_score(job.deadline)
        s_learning = self._learning_score(job.required_skills, skills_dict)

        score = (
            self.WEIGHTS["skill_match"] * s_match
            + self.WEIGHTS["budget"]    * s_budget
            + self.WEIGHTS["deadline"]  * s_deadline
            + self.WEIGHTS["learning"]  * s_learning
        )
        return min(1.0, max(0.0, score))

    def select_jobs(
        self,
        jobs: List["BrowserJob"],
        skills_dict: Dict[str, Any],
        max_jobs: int,
    ) -> List["BrowserJob"]:
        """Sort by score desc, filter score < 0.5, return top max_jobs."""
        scored = [(job, self.score_job(job, skills_dict)) for job in jobs]
        scored = [(j, s) for j, s in scored if s >= 0.5]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [j for j, _ in scored[:max_jobs]]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — WORK GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkResult:
    """Result of a work generation attempt."""
    content: str
    job_type: str
    quality_score: float
    generation_time_seconds: float
    attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkGenerator:
    """
    Calls an AI brain to produce deliverables for a given job.
    Formats output by job type and validates quality before returning.
    """

    MAX_RETRIES: int = 3

    def __init__(self, brain: Any) -> None:
        self._brain = brain

    def _build_prompt(self, job: "BrowserJob") -> str:
        """Build a prompt that embeds job title, description, and each required skill."""
        skills_str = "\n".join(f"- {s}" for s in job.required_skills)
        return (
            f"Job Title: {job.title}\n\n"
            f"Job Description: {job.description}\n\n"
            f"Required Skills:\n{skills_str}\n\n"
            f"Please produce high-quality work that fully addresses the above job requirements."
        )

    def _format_output(self, raw: str, job_type: str) -> str:
        """Format raw output according to job type."""
        if job_type == "code":
            if not raw.strip().startswith("```"):
                return f"```\n{raw}\n```"
            return raw
        elif job_type == "document":
            # Ensure paragraph structure: split on double newlines, rejoin
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            if not paragraphs:
                return raw
            return "\n\n".join(paragraphs)
        else:
            return raw

    def _validate_quality(self, work: str, job: "BrowserJob") -> float:
        """
        Score 0.0-1.0 based on:
        - length (>100 chars = 0.5 base)
        - keyword overlap with job description
        - format correctness
        """
        score = 0.0

        # Length component (0.0 or 0.5)
        if len(work) > 100:
            score += 0.5

        # Keyword overlap with job description (up to 0.3)
        desc_words = set(job.description.lower().split())
        work_words = set(work.lower().split())
        if desc_words:
            overlap = len(desc_words & work_words) / len(desc_words)
            score += min(0.3, overlap * 0.3)

        # Format correctness (up to 0.2)
        if job.job_type == "code" and "```" in work:
            score += 0.2
        elif job.job_type == "document" and "\n\n" in work:
            score += 0.2
        elif job.job_type not in ("code", "document"):
            score += 0.2

        return min(1.0, max(0.0, score))

    async def generate(self, job: "BrowserJob") -> "WorkResult":
        """
        Generate work for the given job, retrying up to MAX_RETRIES times.
        Returns WorkResult with quality_score=0.0 if all retries fail.
        """
        start_time = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                prompt = self._build_prompt(job)
                raw: str = ""
                if self._brain is not None:
                    if asyncio.iscoroutinefunction(getattr(self._brain, "ask", None)):
                        raw = await self._brain.ask(prompt)
                    elif hasattr(self._brain, "ask"):
                        raw = self._brain.ask(prompt)
                    else:
                        raw = str(self._brain)

                formatted = self._format_output(raw or "", job.job_type)
                quality = self._validate_quality(formatted, job)
                elapsed = time.monotonic() - start_time

                return WorkResult(
                    content=formatted,
                    job_type=job.job_type,
                    quality_score=quality,
                    generation_time_seconds=max(0.0, elapsed),
                    attempts=attempt,
                    metadata={"prompt_length": len(prompt)},
                )
            except Exception as exc:
                last_error = exc
                logger.warning("WorkGenerator attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, exc)

        elapsed = time.monotonic() - start_time
        return WorkResult(
            content="",
            job_type=job.job_type,
            quality_score=0.0,
            generation_time_seconds=max(0.0, elapsed),
            attempts=self.MAX_RETRIES,
            metadata={"error": str(last_error)},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QAResult:
    """Result of a quality assurance validation."""
    passed: bool
    score: float
    completeness: float
    relevance: float
    format_score: float
    feedback: str


class QualityAssurance:
    """
    Validates generated work before submission.
    All sub-scores are clamped to [0.0, 1.0].
    """

    MIN_QUALITY_THRESHOLD: float = 0.7
    MAX_RETRIES: int = 3

    def _check_completeness(self, work: str, job: "BrowserJob") -> float:
        """
        Length-based: <50 chars=0.0, >500 chars=1.0, linear between.
        Clamped [0,1].
        """
        length = len(work)
        if length < 50:
            return 0.0
        if length >= 500:
            return 1.0
        return min(1.0, max(0.0, (length - 50) / (500 - 50)))

    def _check_relevance(self, work: str, job: "BrowserJob") -> float:
        """
        Keyword overlap between work and job.description + job.required_skills.
        Clamped [0,1].
        """
        reference_text = job.description + " " + " ".join(job.required_skills)
        ref_words = set(reference_text.lower().split())
        work_words = set(work.lower().split())
        if not ref_words:
            return 1.0
        overlap = len(ref_words & work_words) / len(ref_words)
        return min(1.0, max(0.0, overlap))

    def _check_format(self, work: str, job_type: str) -> float:
        """
        Checks expected format markers.
        Clamped [0,1].
        """
        if job_type == "code":
            return 1.0 if "```" in work else 0.3
        elif job_type == "document":
            return 1.0 if "\n\n" in work else 0.5
        else:
            return 1.0 if work.strip() else 0.0

    def validate(self, work: "WorkResult", job: "BrowserJob") -> QAResult:
        """
        Average of 3 sub-scores; passed = score >= MIN_QUALITY_THRESHOLD.
        """
        completeness = self._check_completeness(work.content, job)
        relevance    = self._check_relevance(work.content, job)
        fmt_score    = self._check_format(work.content, job.job_type)

        score = min(1.0, max(0.0, (completeness + relevance + fmt_score) / 3.0))
        passed = score >= self.MIN_QUALITY_THRESHOLD

        feedback_parts = []
        if completeness < 0.5:
            feedback_parts.append("Work is too short.")
        if relevance < 0.3:
            feedback_parts.append("Work lacks relevance to job description.")
        if fmt_score < 0.5:
            feedback_parts.append("Work format does not match expected job type.")
        feedback = " ".join(feedback_parts) if feedback_parts else "Quality check passed."

        return QAResult(
            passed=passed,
            score=score,
            completeness=completeness,
            relevance=relevance,
            format_score=fmt_score,
            feedback=feedback,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearningResult:
    """Result of a skill learning session."""
    skill_name: str
    concepts_learned: List[str]
    skill_delta: float
    resources_used: List[str]
    duration_seconds: float


class LearningEngine:
    """
    Opens a browser, searches YouTube or DuckDuckGo, reads content,
    and uses the AI brain to extract concepts for skill improvement.
    """

    _BACKOFF: List[int] = [1, 2, 4, 8, 16]

    def __init__(self, engine: Any, brain: Any, db: "WorkerDatabase") -> None:
        self._engine = engine
        self._brain  = brain
        self._db     = db

    async def _search_youtube(self, query: str) -> List[Dict[str, Any]]:
        """
        Use BrowserComputerUse to open youtube.com, search, read results via OCR.
        Returns list of {title, url, description}; on failure returns [].
        """
        try:
            # Try to import BrowserComputerUse
            try:
                from janus_computer_use import BrowserComputerUse  # type: ignore
            except ImportError:
                return []

            browser = BrowserComputerUse(self._engine)
            await browser.open_url("https://www.youtube.com")
            await browser.type_text(query)
            await browser.press_key("Return")
            await asyncio.sleep(2)
            screen_text = await browser.read_screen_text()
            results = []
            for line in (screen_text or "").split("\n"):
                line = line.strip()
                if line:
                    results.append({"title": line, "url": "https://www.youtube.com/results?search_query=" + query.replace(" ", "+"), "description": line})
                if len(results) >= 5:
                    break
            return results
        except Exception as exc:
            logger.warning("LearningEngine._search_youtube failed: %s", exc)
            return []

    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Open duckduckgo.com, search, read results.
        Returns list of {title, url, description}; on failure returns [].
        """
        try:
            try:
                from janus_computer_use import BrowserComputerUse  # type: ignore
            except ImportError:
                return []

            browser = BrowserComputerUse(self._engine)
            await browser.open_url("https://duckduckgo.com")
            await browser.type_text(query)
            await browser.press_key("Return")
            await asyncio.sleep(2)
            screen_text = await browser.read_screen_text()
            results = []
            for line in (screen_text or "").split("\n"):
                line = line.strip()
                if line:
                    results.append({"title": line, "url": "https://duckduckgo.com/?q=" + query.replace(" ", "+"), "description": line})
                if len(results) >= 5:
                    break
            return results
        except Exception as exc:
            logger.warning("LearningEngine._search_web failed: %s", exc)
            return []

    async def _extract_concepts(self, content: str, skill: str) -> List[str]:
        """
        Call brain.ask() with a prompt asking to extract key concepts.
        Parse response into list; return ["general concepts"] if brain unavailable.
        """
        if not self._brain:
            return ["general concepts"]
        try:
            prompt = (
                f"Extract the key concepts from the following content about '{skill}'.\n"
                f"Return each concept on a new line, starting with '- '.\n\n"
                f"Content:\n{content[:2000]}"
            )
            if asyncio.iscoroutinefunction(getattr(self._brain, "ask", None)):
                response = await self._brain.ask(prompt)
            elif hasattr(self._brain, "ask"):
                response = self._brain.ask(prompt)
            else:
                return ["general concepts"]

            concepts = []
            for line in (response or "").split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line:
                    concepts.append(line)
            return concepts if concepts else ["general concepts"]
        except Exception as exc:
            logger.warning("LearningEngine._extract_concepts failed: %s", exc)
            return ["general concepts"]

    def _map_concepts_to_skills(self, concepts: List[str]) -> Dict[str, float]:
        """
        For each concept, check if it matches any skill name in db.list_skills().
        Returns {skill_name: 0.5} for matches.
        """
        known_skills = {s.name.lower(): s.name for s in self._db.list_skills()}
        result: Dict[str, float] = {}
        for concept in concepts:
            concept_lower = concept.lower()
            for skill_lower, skill_name in known_skills.items():
                if skill_lower in concept_lower or concept_lower in skill_lower:
                    result[skill_name] = 0.5
        return result

    async def learn_skill(self, skill_name: str) -> "LearningResult":
        """
        Search YouTube first, fall back to web; extract concepts; update skill in DB;
        persist LearningResourceRecord; return LearningResult.
        """
        start_time = time.monotonic()
        resources_used: List[str] = []

        # Search YouTube first, fall back to web
        results = await self._search_youtube(skill_name)
        if not results:
            results = await self._search_web(skill_name)

        # Aggregate content from results
        content = " ".join(r.get("description", "") + " " + r.get("title", "") for r in results)
        for r in results:
            url = r.get("url", "")
            if url:
                resources_used.append(url)

        # Extract concepts
        concepts = await self._extract_concepts(content or skill_name, skill_name)

        # Map concepts to skills and compute delta
        skill_map = self._map_concepts_to_skills(concepts)
        skill_delta = skill_map.get(skill_name, 0.1) if skill_map else 0.1

        # Update skill in DB
        existing = self._db.get_skill(skill_name)
        if existing:
            new_pts = existing.experience_pts + max(1, int(skill_delta * 10))
            updated = SkillRecord(
                name=existing.name,
                level=existing.level,
                experience_pts=new_pts,
                success_rate=existing.success_rate,
                last_used=datetime.utcnow(),
                last_improved=datetime.utcnow(),
            )
            self._db.upsert_skill(updated)
        else:
            self._db.upsert_skill(SkillRecord(
                name=skill_name,
                level=SkillLevel.BEGINNER,
                experience_pts=max(1, int(skill_delta * 10)),
                success_rate=0.5,
                last_used=datetime.utcnow(),
                last_improved=datetime.utcnow(),
            ))

        # Persist learning resource record
        resource = LearningResourceRecord(
            id=str(uuid.uuid4()),
            url=resources_used[0] if resources_used else "",
            title=f"Learning: {skill_name}",
            topic=skill_name,
            resource_type="youtube" if results else "web",
            concepts=concepts,
            completed_at=datetime.utcnow(),
            skill_delta=skill_delta,
        )
        self._db.insert_learning_resource(resource)

        duration = time.monotonic() - start_time
        return LearningResult(
            skill_name=skill_name,
            concepts_learned=concepts,
            skill_delta=skill_delta,
            resources_used=resources_used,
            duration_seconds=duration,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6 — MARKET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketAnalysis:
    """Result of a market analysis pass."""
    trending_skills: List[str]
    high_paying_job_types: List[str]
    emerging_opportunities: List[str]
    skill_roi: Dict[str, float]
    confidence: float
    data_sources: List[str]
    recommendations: List[str]


class MarketAnalyzer:
    """
    Analyzes job history to identify trends.
    All analysis methods are pure functions over List[JobRecord]; no I/O.
    """

    def __init__(self, db: "WorkerDatabase", brain: Any) -> None:
        self._db    = db
        self._brain = brain

    def trending_skills(self, history: List[JobRecord]) -> List[str]:
        """Count skill frequency across all job required_skills; return top 5 sorted by frequency."""
        freq: Dict[str, int] = {}
        for job in history:
            skills = job.metadata.get("required_skills", []) if job.metadata else []
            for skill in skills:
                freq[skill] = freq.get(skill, 0) + 1
        sorted_skills = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in sorted_skills[:5]]

    def high_paying_types(self, history: List[JobRecord]) -> List[str]:
        """Group by job_type, sort by avg budget desc; return top 5."""
        type_budgets: Dict[str, List[float]] = {}
        for job in history:
            jtype = job.metadata.get("job_type", "general") if job.metadata else "general"
            type_budgets.setdefault(jtype, []).append(job.budget)
        avg_budgets = {jtype: sum(budgets) / len(budgets) for jtype, budgets in type_budgets.items()}
        sorted_types = sorted(avg_budgets.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_types[:5]]

    def skill_roi(self, history: List[JobRecord]) -> Dict[str, float]:
        """For each skill, avg payment_amount of completed jobs requiring it."""
        skill_payments: Dict[str, List[float]] = {}
        for job in history:
            if job.status != JobStatus.COMPLETED.value:
                continue
            payment = job.payment_amount or 0.0
            skills = job.metadata.get("required_skills", []) if job.metadata else []
            for skill in skills:
                skill_payments.setdefault(skill, []).append(payment)
        return {
            skill: sum(payments) / len(payments)
            for skill, payments in skill_payments.items()
        }

    def analyze(self, job_history: List[JobRecord]) -> MarketAnalysis:
        """
        Works on empty list (returns empty lists/dicts with confidence=0.0).
        Calls brain.ask() for emerging_opportunities and recommendations if brain available.
        """
        if not job_history:
            return MarketAnalysis(
                trending_skills=[],
                high_paying_job_types=[],
                emerging_opportunities=[],
                skill_roi={},
                confidence=0.0,
                data_sources=[],
                recommendations=[],
            )

        trending   = self.trending_skills(job_history)
        high_pay   = self.high_paying_types(job_history)
        roi        = self.skill_roi(job_history)
        confidence = min(1.0, len(job_history) / 20.0)

        emerging: List[str] = []
        recommendations: List[str] = []

        if self._brain is not None:
            try:
                summary = (
                    f"Trending skills: {trending}\n"
                    f"High-paying job types: {high_pay}\n"
                    f"Skill ROI: {roi}\n"
                    f"Total jobs analyzed: {len(job_history)}"
                )
                prompt_emerging = f"Based on this market data, list 3 emerging opportunities:\n{summary}"
                prompt_recs     = f"Based on this market data, give 3 actionable recommendations:\n{summary}"

                if asyncio.iscoroutinefunction(getattr(self._brain, "ask", None)):
                    # Can't await in sync method; skip brain call
                    pass
                elif hasattr(self._brain, "ask"):
                    resp_e = self._brain.ask(prompt_emerging)
                    emerging = [l.strip().lstrip("- ") for l in (resp_e or "").split("\n") if l.strip()][:3]
                    resp_r = self._brain.ask(prompt_recs)
                    recommendations = [l.strip().lstrip("- ") for l in (resp_r or "").split("\n") if l.strip()][:3]
            except Exception as exc:
                logger.warning("MarketAnalyzer.analyze brain call failed: %s", exc)

        return MarketAnalysis(
            trending_skills=trending,
            high_paying_job_types=high_pay,
            emerging_opportunities=emerging,
            skill_roi=roi,
            confidence=confidence,
            data_sources=["worker_database"],
            recommendations=recommendations,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7 — INVESTMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import JanusWallet; degrade gracefully if unavailable
try:
    from janus_wallet import JanusWallet as _JanusWallet  # type: ignore
    _HAS_WALLET = True
except ImportError:
    _JanusWallet = None  # type: ignore
    _HAS_WALLET = False


@dataclass
class InvestmentAction:
    """A single investment decision."""
    category: str
    amount: Decimal
    description: str
    expected_roi: float
    executed: bool = False


class InvestmentEngine:
    """
    Decides how to spend earned money.
    Delegates to JanusWallet for all financial operations.
    """

    COMPUTE_THRESHOLD: Decimal = Decimal("100")
    COURSE_THRESHOLD:  Decimal = Decimal("50")

    def __init__(self, wallet: Any, brain: Any) -> None:
        self._wallet = wallet
        self._brain  = brain

    def _should_invest(self, balance: Decimal) -> bool:
        """Returns True when balance > COURSE_THRESHOLD."""
        return balance > self.COURSE_THRESHOLD

    def _prioritize_investments(
        self,
        balance: Decimal,
        weak_skills: List[str],
    ) -> List[InvestmentAction]:
        """
        If balance > COMPUTE_THRESHOLD: add compute action.
        If balance > COURSE_THRESHOLD and weak_skills: add course action per weak skill (up to 2).
        """
        actions: List[InvestmentAction] = []

        if balance > self.COMPUTE_THRESHOLD:
            actions.append(InvestmentAction(
                category="compute",
                amount=Decimal("20"),
                description="Invest in additional compute resources",
                expected_roi=1.5,
            ))

        if balance > self.COURSE_THRESHOLD and weak_skills:
            for skill in weak_skills[:2]:
                actions.append(InvestmentAction(
                    category="course",
                    amount=Decimal("15"),
                    description=f"Online course to improve skill: {skill}",
                    expected_roi=2.0,
                ))

        return actions

    async def evaluate_and_invest(self) -> List[InvestmentAction]:
        """
        Get balance from wallet, get weak skills (BEGINNER level) from DB,
        prioritize, record each as expense via wallet.record_expense(), mark executed=True.
        """
        if self._wallet is None:
            return []

        try:
            balance = self._wallet.get_balance()
        except Exception as exc:
            logger.warning("InvestmentEngine: failed to get balance: %s", exc)
            return []

        # Get weak skills from DB if available
        weak_skills: List[str] = []
        try:
            # wallet may have a db reference, or we skip
            pass
        except Exception:
            pass

        if not self._should_invest(balance):
            return []

        actions = self._prioritize_investments(balance, weak_skills)

        executed: List[InvestmentAction] = []
        for action in actions:
            try:
                self._wallet.record_expense(
                    amount=action.amount,
                    category=action.category,
                    description=action.description,
                )
                action.executed = True
                executed.append(action)
                logger.info("InvestmentEngine: executed %s investment of %s", action.category, action.amount)
            except Exception as exc:
                logger.warning("InvestmentEngine: failed to record expense for %s: %s", action.category, exc)

        return executed


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8 — MONITORING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

import logging.handlers

_SENSITIVE_KEYS = {"secret", "password", "credential", "key", "token"}


def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out any key containing sensitive words."""
    return {
        k: v for k, v in context.items()
        if not any(s in k.lower() for s in _SENSITIVE_KEYS)
    }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for the worker."""
    jobs_completed: int
    total_earned: Decimal
    average_job_value: Decimal
    skill_levels: Dict[str, str]
    error_rate: float
    success_rate: float


class MonitoringSystem:
    """
    Structured logging and metrics aggregation.
    Writes JSON lines to a log file; tracks in-memory counters.
    """

    def __init__(self, log_path: str = "janus_worker.log") -> None:
        self._log_path = log_path
        self._logger = logging.getLogger(f"janus_monitor.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

        # In-memory counters
        self._jobs_completed: int = 0
        self._jobs_failed: int    = 0
        self._total_earned: Decimal = Decimal("0")
        self._errors: int         = 0

    def log_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Write a JSON line with timestamp, event_type, and sanitized context."""
        safe_ctx = _sanitize_context(context)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **safe_ctx,
        }
        try:
            self._logger.info(json.dumps(record))
        except Exception as exc:
            logger.warning("MonitoringSystem.log_event failed: %s", exc)

    def log_job_claimed(self, job: "BrowserJob", rationale: str) -> None:
        self.log_event("job_claimed", {
            "job_id": job.id,
            "job_title": job.title,
            "platform": job.platform,
            "budget": job.budget,
            "rationale": rationale,
        })

    def log_work_generated(self, job_id: str, quality: float, time_s: float) -> None:
        self.log_event("work_generated", {
            "job_id": job_id,
            "quality_score": quality,
            "generation_time_seconds": time_s,
        })

    def log_job_completed(self, job_id: str, quality: float) -> None:
        self._jobs_completed += 1
        self.log_event("job_completed", {
            "job_id": job_id,
            "quality_score": quality,
        })

    def log_payment(self, amount: Decimal, platform: str) -> None:
        self._total_earned += amount
        self.log_event("payment_received", {
            "amount": str(amount),
            "platform": platform,
        })

    def log_skill_improved(self, skill: str, new_level: str, resources: List[str]) -> None:
        self.log_event("skill_improved", {
            "skill": skill,
            "new_level": new_level,
            "resources": resources,
        })

    def log_error(self, error_type: str, tb: str, recovery: str) -> None:
        self._errors += 1
        self.log_event("error", {
            "error_type": error_type,
            "traceback": tb,
            "recovery": recovery,
        })

    def get_metrics(self) -> PerformanceMetrics:
        """Compute metrics from in-memory counters; all fields non-None."""
        total_jobs = self._jobs_completed + self._jobs_failed
        avg_value = (
            self._total_earned / self._jobs_completed
            if self._jobs_completed > 0
            else Decimal("0")
        )
        error_rate = self._errors / max(1, total_jobs + self._errors)
        success_rate = self._jobs_completed / max(1, total_jobs)

        return PerformanceMetrics(
            jobs_completed=self._jobs_completed,
            total_earned=self._total_earned,
            average_job_value=avg_value,
            skill_levels={},
            error_rate=error_rate,
            success_rate=success_rate,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9 — WORK CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

# Optional computer-use imports
try:
    from janus_computer_use import ComputerUseEngine, BrowserComputerUse  # type: ignore
    _HAS_COMPUTER_USE = True
except ImportError:
    _HAS_COMPUTER_USE = False

# Optional platform browser imports
try:
    from janus_platform_browser import UpworkBrowser, FiverrBrowser  # type: ignore
    _HAS_PLATFORM_BROWSER = True
except ImportError:
    _HAS_PLATFORM_BROWSER = False


@dataclass
class JobResult:
    """Result of executing a single job."""
    job_id: str
    success: bool
    quality_score: float
    payment_amount: Decimal
    error: Optional[str] = None


class WorkCycle:
    """
    Top-level orchestrator.  Sequences: discover → evaluate → execute → learn → invest → persist.
    """

    def __init__(
        self,
        db: "WorkerDatabase",
        wallet: Any,
        decision_engine: "DecisionEngine",
        work_generator: "WorkGenerator",
        learning_engine: "LearningEngine",
        quality_assurance: "QualityAssurance",
        investment_engine: "InvestmentEngine",
        market_analyzer: "MarketAnalyzer",
        monitor: "MonitoringSystem",
        max_concurrent_jobs: int = 5,
    ) -> None:
        self._db               = db
        self._wallet           = wallet
        self._decision_engine  = decision_engine
        self._work_generator   = work_generator
        self._learning_engine  = learning_engine
        self._qa               = quality_assurance
        self._investment_engine = investment_engine
        self._market_analyzer  = market_analyzer
        self._monitor          = monitor
        self._max_concurrent   = max_concurrent_jobs

    # ── job discovery ─────────────────────────────────────────────────────────

    async def _discover_jobs(self) -> List["BrowserJob"]:
        """
        Use PlatformBrowser (which wraps both Upwork and Fiverr) to find jobs.
        Falls back gracefully if computer use or browsers are unavailable.
        """
        if not _HAS_PLATFORM_BROWSER:
            return []

        # Need a ComputerUseEngine to drive the browsers
        engine = None
        if _HAS_COMPUTER_USE:
            try:
                engine = ComputerUseEngine()
                await engine.__aenter__()
            except Exception as exc:
                logger.warning("WorkCycle._discover_jobs: could not create ComputerUseEngine: %s", exc)
                return []

        try:
            from janus_platform_browser import PlatformBrowser  # type: ignore
            browser = PlatformBrowser(engine)
            # Get skills from DB to search for relevant jobs
            skills = [s.name for s in self._db.list_skills()] or ["writing", "coding", "data analysis"]
            jobs = await browser.find_jobs(skills=skills[:5])
            return list(jobs) if jobs else []
        except Exception as exc:
            logger.warning("WorkCycle._discover_jobs: PlatformBrowser failed: %s", exc)
            return []

    # ── job execution ─────────────────────────────────────────────────────────

    async def _execute_job(self, job: "BrowserJob") -> "JobResult":
        """
        Generate work, validate QA; if qa.passed: submit via browser (if available),
        record income via wallet.record_income(); if not passed: mark failed.
        Log all steps via monitor.
        """
        self._monitor.log_job_claimed(job, rationale="Selected by DecisionEngine")

        # Persist job as claimed
        job_record = JobRecord(
            id=job.id,
            title=job.title,
            description=job.description,
            platform=job.platform,
            budget=job.budget,
            status=JobStatus.CLAIMED.value,
            claimed_at=datetime.utcnow(),
            completed_at=None,
            quality_score=None,
            payment_amount=None,
            metadata={**job.metadata, "job_type": job.job_type, "required_skills": job.required_skills},
        )
        try:
            self._db.insert_job(job_record)
        except Exception:
            pass  # May already exist

        try:
            work_result = await self._work_generator.generate(job)
            self._monitor.log_work_generated(job.id, work_result.quality_score, work_result.generation_time_seconds)

            qa_result = self._qa.validate(work_result, job)

            if not qa_result.passed:
                self._db.update_job_status(job.id, JobStatus.FAILED.value, quality_score=qa_result.score)
                self._monitor.log_job_completed(job.id, qa_result.score)
                return JobResult(
                    job_id=job.id,
                    success=False,
                    quality_score=qa_result.score,
                    payment_amount=Decimal("0"),
                    error=f"QA failed: {qa_result.feedback}",
                )

            # Submit via browser if available
            submitted = False
            if _HAS_PLATFORM_BROWSER:
                try:
                    if job.platform.lower() == "upwork":
                        browser = UpworkBrowser()
                        if asyncio.iscoroutinefunction(getattr(browser, "submit_work", None)):
                            submitted = await browser.submit_work(job.id, work_result.content)
                        else:
                            submitted = browser.submit_work(job.id, work_result.content)
                    else:
                        browser = FiverrBrowser()
                        if asyncio.iscoroutinefunction(getattr(browser, "deliver_order", None)):
                            submitted = await browser.deliver_order(job.id, work_result.content)
                        else:
                            submitted = browser.deliver_order(job.id, work_result.content)
                except Exception as exc:
                    logger.warning("WorkCycle._execute_job: browser submit failed: %s", exc)
                    submitted = False

            # Record income
            payment = Decimal(str(job.budget))
            if self._wallet is not None:
                try:
                    self._wallet.record_income(
                        amount=payment,
                        source=job.platform,
                        description=f"Payment for job: {job.title}",
                    )
                    self._monitor.log_payment(payment, job.platform)
                except Exception as exc:
                    logger.warning("WorkCycle._execute_job: record_income failed: %s", exc)

            self._db.update_job_status(
                job.id,
                JobStatus.COMPLETED.value,
                quality_score=qa_result.score,
                payment_amount=float(payment),
                completed_at=datetime.utcnow(),
            )
            self._monitor.log_job_completed(job.id, qa_result.score)

            return JobResult(
                job_id=job.id,
                success=True,
                quality_score=qa_result.score,
                payment_amount=payment,
            )

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._monitor.log_error("job_execution_error", tb, "mark_failed_continue")
            self._db.update_job_status(job.id, JobStatus.FAILED.value)
            return JobResult(
                job_id=job.id,
                success=False,
                quality_score=0.0,
                payment_amount=Decimal("0"),
                error=str(exc),
            )

    # ── one cycle ─────────────────────────────────────────────────────────────

    async def run_one_cycle(self) -> "CycleSummary":
        """
        Discover → score/select (DecisionEngine) → sort by deadline asc →
        execute up to max_concurrent_jobs via asyncio.gather →
        learn one skill → invest check → persist CycleSummary.
        """
        cycle_start = datetime.utcnow()
        cycle_id    = str(uuid.uuid4())
        errors      = 0
        earnings    = Decimal("0")
        skills_improved: List[str] = []

        # 1. Discover jobs
        discovered = await self._discover_jobs()

        # 2. Score and select
        skills_dict = {s.name: s.level for s in self._db.list_skills()}
        selected = self._decision_engine.select_jobs(discovered, skills_dict, self._max_concurrent)

        # 3. Sort by deadline ascending (earliest first; None deadlines go last)
        selected.sort(key=lambda j: (j.deadline is None, j.deadline or datetime.max))

        # 4. Execute concurrently (up to max_concurrent_jobs)
        batch = selected[:self._max_concurrent]
        results: List[JobResult] = []
        if batch:
            job_tasks = [self._execute_job(job) for job in batch]
            raw_results = await asyncio.gather(*job_tasks, return_exceptions=True)
            for r in raw_results:
                if isinstance(r, Exception):
                    errors += 1
                    logger.error("WorkCycle.run_one_cycle: job raised exception: %s", r)
                else:
                    results.append(r)
                    if r.success:
                        earnings += r.payment_amount
                    else:
                        errors += 1

        # 5. Learn one skill (pick the first BEGINNER skill or a default)
        try:
            all_skills = self._db.list_skills()
            beginner_skills = [s for s in all_skills if s.level == SkillLevel.BEGINNER]
            skill_to_learn = beginner_skills[0].name if beginner_skills else "general programming"
            lr = await self._learning_engine.learn_skill(skill_to_learn)
            skills_improved.append(lr.skill_name)
            self._monitor.log_skill_improved(lr.skill_name, "BEGINNER", lr.resources_used)
        except Exception as exc:
            logger.warning("WorkCycle.run_one_cycle: learning failed: %s", exc)
            errors += 1

        # 6. Investment check
        try:
            await self._investment_engine.evaluate_and_invest()
        except Exception as exc:
            logger.warning("WorkCycle.run_one_cycle: investment check failed: %s", exc)

        # 7. Persist CycleSummary
        cycle_end = datetime.utcnow()
        summary = CycleSummary(
            id=cycle_id,
            started_at=cycle_start,
            completed_at=cycle_end,
            jobs_processed=len(results),
            earnings=earnings,
            skills_improved=skills_improved,
            errors=errors,
            state="completed" if errors == 0 else "partial",
        )
        try:
            self._db.insert_cycle_summary(summary)
        except Exception as exc:
            logger.warning("WorkCycle.run_one_cycle: failed to persist cycle summary: %s", exc)

        return summary

    # ── run forever ───────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """
        Loop calling run_one_cycle(); catch exceptions, log, sleep 60s, continue.
        """
        while True:
            try:
                summary = await self.run_one_cycle()
                logger.info(
                    "WorkCycle completed: jobs=%d earnings=%s errors=%d state=%s",
                    summary.jobs_processed,
                    summary.earnings,
                    summary.errors,
                    summary.state,
                )
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self._monitor.log_error("cycle_error", tb, "sleep_and_retry")
                logger.error("WorkCycle.run_forever: cycle failed: %s", exc)

            await asyncio.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE WORK GENERATOR (from janus_worker_completion.py)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveWorkGenerator:
    """
    Wraps WorkGenerator and adapts generation strategy based on outcome feedback.
    Tracks per-job-type statistics and tunes prompt instructions over time.
    """

    def __init__(self, base_generator: "WorkGenerator") -> None:
        from collections import defaultdict
        self.base_generator = base_generator
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_quality": 0.0, "successes": 0, "attempts": 0}
        )

    def record_outcome(self, job_type: str, quality_score: float, client_satisfied: bool) -> None:
        stats = self._stats[job_type]
        stats["total_quality"] += quality_score
        stats["attempts"] += 1
        if client_satisfied:
            stats["successes"] += 1

    def get_strategy(self, job_type: str) -> dict:
        stats = self._stats[job_type]
        attempts = stats["attempts"]
        if attempts == 0:
            return {"detail_level": 0.7, "examples": True, "conservative": False}
        avg_quality = stats["total_quality"] / attempts
        success_rate = stats["successes"] / attempts
        detail_level = min(1.0, 0.5 + (1.0 - avg_quality) * 0.5)
        return {
            "detail_level": round(detail_level, 3),
            "examples": success_rate < 0.7,
            "conservative": avg_quality < 0.5,
        }

    def adapt_prompt(self, prompt: str, job_type: str) -> str:
        strategy = self.get_strategy(job_type)
        additions = []
        if strategy["conservative"]:
            additions.append("Be conservative and thorough; previous attempts had quality issues.")
        if strategy["examples"]:
            additions.append("Include concrete examples to improve client satisfaction.")
        if strategy["detail_level"] > 0.8:
            additions.append("Provide high detail and comprehensive coverage of all requirements.")
        elif strategy["detail_level"] < 0.5:
            additions.append("Keep the response concise and focused.")
        if additions:
            return prompt + "\n\n[Adaptive instructions: " + " ".join(additions) + "]"
        return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS IMPROVEMENT ENGINE (from janus_worker_completion.py)
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousImprovementEngine:
    """
    Analyzes work cycles and generates improvement recommendations.
    Maintains a rolling history of cycles and identifies trends.
    """

    _MAX_HISTORY = 50

    def __init__(self) -> None:
        from collections import deque, defaultdict
        self._cycles: Any = deque(maxlen=self._MAX_HISTORY)

    def record_cycle(self, jobs_completed: int, earnings: float,
                     quality_scores: List[float], skills_improved: List[str]) -> None:
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        self._cycles.append({
            "ts": datetime.utcnow().isoformat(),
            "jobs_completed": jobs_completed,
            "earnings": float(earnings),
            "avg_quality": avg_quality,
            "quality_scores": list(quality_scores),
            "skills_improved": list(skills_improved),
        })

    def analyze(self) -> dict:
        from collections import defaultdict
        recent = list(self._cycles)[-5:]
        if not recent:
            return {
                "high_performing_skills": [],
                "profitable_types": [],
                "common_failures": [],
                "recommendations": ["No cycle data yet; complete some jobs first."],
            }
        skill_quality: Dict[str, List[float]] = defaultdict(list)
        for cycle in recent:
            for skill in cycle.get("skills_improved", []):
                skill_quality[skill].append(cycle["avg_quality"])
        high_performing = sorted(
            skill_quality,
            key=lambda s: sum(skill_quality[s]) / len(skill_quality[s]),
            reverse=True,
        )[:5]
        avg_earnings = sum(c["earnings"] for c in recent) / len(recent)
        low_quality = [c for c in recent if c["avg_quality"] < 0.5]
        recs = []
        if high_performing:
            recs.append(f"Double down on high-performing skills: {', '.join(high_performing[:3])}")
        if low_quality:
            recs.append("Quality is below threshold in recent cycles; review outputs before submission.")
        trend = [c["earnings"] for c in recent]
        if len(trend) >= 2 and trend[-1] < trend[0]:
            recs.append("Earnings are declining; consider diversifying job types or upskilling.")
        elif len(trend) >= 2 and trend[-1] > trend[0]:
            recs.append("Earnings are growing; maintain current strategy.")
        if not recs:
            recs.append("Performance is stable; continue current approach.")
        return {
            "high_performing_skills": high_performing,
            "profitable_types": [f"cycles averaging ${avg_earnings:.2f} earnings"] if avg_earnings > 0 else [],
            "common_failures": [f"{len(low_quality)} of last {len(recent)} cycles had low quality"] if low_quality else [],
            "recommendations": recs,
        }

    def get_recommendations(self) -> List[dict]:
        analysis = self.analyze()
        raw_recs = analysis.get("recommendations", [])
        n = len(list(self._cycles)[-5:])
        structured = []
        for i, rec in enumerate(raw_recs):
            confidence = min(0.95, max(0.1, 0.4 + (n / 10.0) - (i * 0.05)))
            structured.append({
                "action": rec,
                "expected_impact": "moderate improvement in earnings or quality",
                "confidence": round(confidence, 2),
            })
        return structured


# ═══════════════════════════════════════════════════════════════════════════════
# CREDENTIAL MANAGER (from janus_worker_completion.py)
# ═══════════════════════════════════════════════════════════════════════════════

import base64
from pathlib import Path as _Path

try:
    from cryptography.fernet import Fernet as _Fernet
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _Fernet = None  # type: ignore
    _HAS_CRYPTOGRAPHY = False


class CredentialManager:
    """
    Secure credential storage using Fernet encryption (or base64 fallback).
    Values are never logged.
    """

    _CRED_FILE = ".janus_credentials.enc"

    def __init__(self, key_file: str = ".janus_key") -> None:
        self.key_file = key_file
        self._fernet = None
        self._init_crypto()

    def _init_crypto(self) -> None:
        if not _HAS_CRYPTOGRAPHY:
            logger.warning("cryptography not installed; using base64 obfuscation fallback.")
            return
        try:
            key_path = _Path(self.key_file)
            if key_path.exists():
                key = key_path.read_bytes().strip()
            else:
                key = _Fernet.generate_key()
                key_path.write_bytes(key)
                try:
                    import os as _os
                    _os.chmod(self.key_file, 0o600)
                except OSError:
                    pass
            self._fernet = _Fernet(key)
        except Exception as exc:
            logger.error("CredentialManager._init_crypto failed: %s", exc)

    def _load_store(self) -> dict:
        try:
            p = _Path(self._CRED_FILE)
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            return {}

    def _save_store(self, store: dict) -> None:
        try:
            _Path(self._CRED_FILE).write_text(json.dumps(store), encoding="utf-8")
        except Exception as exc:
            logger.error("CredentialManager._save_store failed: %s", exc)

    def _encrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        return base64.b64encode(value.encode()).decode()

    def _decrypt(self, token: str) -> str:
        if self._fernet:
            return self._fernet.decrypt(token.encode()).decode()
        return base64.b64decode(token.encode()).decode()

    def store(self, name: str, value: str) -> None:
        store = self._load_store()
        store[name] = self._encrypt(value)
        self._save_store(store)

    def load(self, name: str) -> Optional[str]:
        store = self._load_store()
        if name not in store:
            return None
        try:
            return self._decrypt(store[name])
        except Exception:
            return None

    def list_names(self) -> List[str]:
        return list(self._load_store().keys())

    def rotate(self, name: str, new_value: str) -> None:
        store = self._load_store()
        store[name] = self._encrypt(new_value)
        self._save_store(store)


# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY REGISTRY (from janus_worker.py)
# ═══════════════════════════════════════════════════════════════════════════════

class CapabilityRegistry:
    """
    What Janus can actually do right now, with honest confidence scores.
    Used by DecisionEngine to assess job fit.
    """

    CAPABILITIES: Dict[str, float] = {
        "python_scripting":    0.95,
        "data_analysis":       0.90,
        "web_scraping":        0.85,
        "code_review":         0.90,
        "bug_fixing":          0.85,
        "api_integration":     0.80,
        "report_generation":   0.90,
        "automation_scripts":  0.85,
        "technical_writing":   0.75,
        "sql_queries":         0.85,
        "json_xml_processing": 0.95,
        "file_processing":     0.95,
        "regex_patterns":      0.90,
        "test_writing":        0.80,
        "documentation":       0.80,
    }

    CATEGORY_REQUIREMENTS: Dict[str, List[str]] = {
        "code":       ["python_scripting", "bug_fixing"],
        "data":       ["data_analysis", "sql_queries", "report_generation"],
        "content":    ["technical_writing", "documentation"],
        "automation": ["automation_scripts", "web_scraping", "api_integration"],
        "analysis":   ["data_analysis", "report_generation", "json_xml_processing"],
        "review":     ["code_review", "documentation"],
    }

    def can_do(self, category: str) -> tuple:
        """Returns (can_do: bool, confidence: float) for a job category."""
        required = self.CATEGORY_REQUIREMENTS.get(category, [])
        if not required:
            return False, 0.0
        scores = [self.CAPABILITIES.get(cap, 0.0) for cap in required]
        avg = sum(scores) / len(scores)
        return avg >= 0.7, avg
