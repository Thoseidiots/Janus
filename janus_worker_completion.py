import base64
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class MarketAnalyzer:
    """Analyzes job market trends to identify opportunities and optimize strategy.

    Tracks skill demand, average budgets, and job type frequency to surface
    actionable recommendations for the autonomous worker.
    """

    def __init__(self):
        self._skill_demand: Dict[str, int] = defaultdict(int)
        self._skill_budgets: Dict[str, List[float]] = defaultdict(list)
        self._job_type_freq: Dict[str, int] = defaultdict(int)

    def analyze_market(self, jobs: List[dict]) -> dict:
        """Analyze a list of job postings and return market insights.

        Args:
            jobs: List of job dicts with keys like 'skills', 'budget', 'type'.

        Returns:
            dict with trending_skills, high_paying_types, avg_budget, recommendations.
        """
        try:
            skill_demand: Dict[str, int] = defaultdict(int)
            type_budgets: Dict[str, List[float]] = defaultdict(list)
            all_budgets: List[float] = []

            for job in jobs:
                job_type = job.get("type", "unknown")
                budget = job.get("budget", 0.0)
                skills = job.get("skills", [])

                self._job_type_freq[job_type] += 1

                if isinstance(budget, (int, float)) and budget > 0:
                    all_budgets.append(float(budget))
                    type_budgets[job_type].append(float(budget))

                for skill in skills:
                    skill_demand[skill] += 1
                    self._skill_demand[skill] += 1
                    if isinstance(budget, (int, float)) and budget > 0:
                        self._skill_budgets[skill].append(float(budget))

            trending_skills = sorted(skill_demand, key=lambda s: skill_demand[s], reverse=True)[:10]

            high_paying_types = sorted(
                type_budgets,
                key=lambda t: sum(type_budgets[t]) / len(type_budgets[t]) if type_budgets[t] else 0,
                reverse=True,
            )[:5]

            avg_budget = sum(all_budgets) / len(all_budgets) if all_budgets else 0.0

            recommendations = self._build_recommendations(trending_skills, high_paying_types, avg_budget)

            return {
                "trending_skills": trending_skills,
                "high_paying_types": high_paying_types,
                "avg_budget": round(avg_budget, 2),
                "recommendations": recommendations,
            }
        except Exception as exc:
            logger.error("MarketAnalyzer.analyze_market failed: %s", exc)
            return {"trending_skills": [], "high_paying_types": [], "avg_budget": 0.0, "recommendations": []}

    def _build_recommendations(
        self, trending_skills: List[str], high_paying_types: List[str], avg_budget: float
    ) -> List[str]:
        recs = []
        if trending_skills:
            recs.append(f"Focus on high-demand skills: {', '.join(trending_skills[:3])}")
        if high_paying_types:
            recs.append(f"Prioritize job types with best pay: {', '.join(high_paying_types[:2])}")
        if avg_budget > 0:
            recs.append(f"Market average budget is ${avg_budget:.2f}; price competitively above this")
        return recs

    def detect_opportunities(self, skills: List[str], jobs: List[dict]) -> List[str]:
        """Identify jobs that match the given skill set and represent good opportunities.

        Args:
            skills: Skills the worker currently has.
            jobs: Available job postings.

        Returns:
            List of human-readable opportunity descriptions.
        """
        try:
            skill_set = set(s.lower() for s in skills)
            opportunities = []

            for job in jobs:
                job_skills = set(s.lower() for s in job.get("skills", []))
                if not job_skills:
                    continue
                overlap = skill_set & job_skills
                match_ratio = len(overlap) / len(job_skills) if job_skills else 0.0
                if match_ratio >= 0.5:
                    budget = job.get("budget", 0)
                    title = job.get("title", job.get("type", "Unnamed job"))
                    matched = ", ".join(sorted(overlap))
                    opportunities.append(
                        f"'{title}' matches {int(match_ratio * 100)}% of required skills "
                        f"({matched}); budget ${budget}"
                    )

            return opportunities
        except Exception as exc:
            logger.error("MarketAnalyzer.detect_opportunities failed: %s", exc)
            return []


class ConcurrentJobManager:
    """Manages concurrent job execution up to a configurable limit.

    Maintains active, queued, completed, and failed job sets and provides
    queue promotion so work flows smoothly without exceeding the concurrency cap.
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._active: set = set()
        self._queue: deque = deque()
        self._completed: set = set()
        self._failed: set = set()

    def can_accept(self) -> bool:
        """Return True if a new job can be started immediately."""
        return len(self._active) < self.max_concurrent

    def add_job(self, job_id: str) -> None:
        """Add a job to the active set.

        Args:
            job_id: Unique identifier for the job.
        """
        try:
            self._active.add(job_id)
        except Exception as exc:
            logger.error("ConcurrentJobManager.add_job failed for %s: %s", job_id, exc)

    def complete_job(self, job_id: str) -> None:
        """Move a job from active to completed.

        Args:
            job_id: Unique identifier for the job.
        """
        try:
            self._active.discard(job_id)
            self._completed.add(job_id)
        except Exception as exc:
            logger.error("ConcurrentJobManager.complete_job failed for %s: %s", job_id, exc)

    def fail_job(self, job_id: str) -> None:
        """Move a job from active to failed.

        Args:
            job_id: Unique identifier for the job.
        """
        try:
            self._active.discard(job_id)
            self._failed.add(job_id)
        except Exception as exc:
            logger.error("ConcurrentJobManager.fail_job failed for %s: %s", job_id, exc)

    def queue_job(self, job_id: str) -> None:
        """Add a job to the waiting queue.

        Args:
            job_id: Unique identifier for the job.
        """
        try:
            self._queue.append(job_id)
        except Exception as exc:
            logger.error("ConcurrentJobManager.queue_job failed for %s: %s", job_id, exc)

    def promote_from_queue(self) -> Optional[str]:
        """Move the oldest queued job to active if capacity allows.

        Returns:
            The promoted job_id, or None if queue is empty or at capacity.
        """
        try:
            if self._queue and self.can_accept():
                job_id = self._queue.popleft()
                self._active.add(job_id)
                return job_id
            return None
        except Exception as exc:
            logger.error("ConcurrentJobManager.promote_from_queue failed: %s", exc)
            return None

    def status(self) -> dict:
        """Return counts of jobs in each state."""
        return {
            "active": len(self._active),
            "queued": len(self._queue),
            "completed": len(self._completed),
            "failed": len(self._failed),
        }


class AdaptiveWorkGenerator:
    """Wraps a base WorkGenerator and adapts generation strategy based on outcome feedback.

    Tracks per-job-type statistics (average quality, success rate, attempt count) and
    uses them to tune prompt instructions and generation parameters over time.
    """

    def __init__(self, base_generator):
        self.base_generator = base_generator
        # Per job_type: {"total_quality": float, "successes": int, "attempts": int}
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_quality": 0.0, "successes": 0, "attempts": 0}
        )

    def record_outcome(self, job_type: str, quality_score: float, client_satisfied: bool) -> None:
        """Record the outcome of a completed job to inform future strategy.

        Args:
            job_type: Category/type of the job.
            quality_score: Numeric quality rating (0.0 – 1.0).
            client_satisfied: Whether the client accepted the work.
        """
        try:
            stats = self._stats[job_type]
            stats["total_quality"] += quality_score
            stats["attempts"] += 1
            if client_satisfied:
                stats["successes"] += 1
        except Exception as exc:
            logger.error("AdaptiveWorkGenerator.record_outcome failed: %s", exc)

    def get_strategy(self, job_type: str) -> dict:
        """Return generation strategy parameters for the given job type.

        Args:
            job_type: Category/type of the job.

        Returns:
            dict with detail_level (float), examples (bool), conservative (bool).
        """
        try:
            stats = self._stats[job_type]
            attempts = stats["attempts"]
            if attempts == 0:
                return {"detail_level": 0.7, "examples": True, "conservative": False}

            avg_quality = stats["total_quality"] / attempts
            success_rate = stats["successes"] / attempts

            # Increase detail when quality is low; add examples when success rate is low
            detail_level = min(1.0, 0.5 + (1.0 - avg_quality) * 0.5)
            examples = success_rate < 0.7
            conservative = avg_quality < 0.5

            return {
                "detail_level": round(detail_level, 3),
                "examples": examples,
                "conservative": conservative,
            }
        except Exception as exc:
            logger.error("AdaptiveWorkGenerator.get_strategy failed: %s", exc)
            return {"detail_level": 0.7, "examples": True, "conservative": False}

    def adapt_prompt(self, prompt: str, job_type: str) -> str:
        """Append adaptive instructions to a prompt based on historical performance.

        Args:
            prompt: The original generation prompt.
            job_type: Category/type of the job.

        Returns:
            The prompt with adaptive instructions appended.
        """
        try:
            strategy = self.get_strategy(job_type)
            additions = []

            if strategy["conservative"]:
                additions.append(
                    "Be conservative and thorough; previous attempts had quality issues."
                )
            if strategy["examples"]:
                additions.append(
                    "Include concrete examples to improve client satisfaction."
                )
            if strategy["detail_level"] > 0.8:
                additions.append(
                    "Provide high detail and comprehensive coverage of all requirements."
                )
            elif strategy["detail_level"] < 0.5:
                additions.append("Keep the response concise and focused.")

            if additions:
                return prompt + "\n\n[Adaptive instructions: " + " ".join(additions) + "]"
            return prompt
        except Exception as exc:
            logger.error("AdaptiveWorkGenerator.adapt_prompt failed: %s", exc)
            return prompt


class FinancialReporter:
    """Tracks income and expenses in a local SQLite database and produces financial reports.

    Provides summaries, per-skill earnings breakdowns, daily trend data, and
    low-balance alerts to keep the autonomous worker financially aware.
    """

    _CREATE_FINANCIALS = """
        CREATE TABLE IF NOT EXISTS financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            job_id TEXT,
            platform TEXT,
            category TEXT,
            description TEXT
        )
    """

    def __init__(self, db_path: str = "janus_worker.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(self._CREATE_FINANCIALS)
            conn.commit()
        except Exception as exc:
            logger.error("FinancialReporter._init_db failed: %s", exc)
        finally:
            if conn:
                conn.close()

    def record_income(self, amount: float, job_id: str, platform: str) -> None:
        """Record an income transaction.

        Args:
            amount: Dollar amount earned.
            job_id: Identifier of the completed job.
            platform: Platform the job came from.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO financials (ts, type, amount, job_id, platform) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), "income", float(amount), job_id, platform),
            )
            conn.commit()
        except Exception as exc:
            logger.error("FinancialReporter.record_income failed: %s", exc)
        finally:
            if conn:
                conn.close()

    def record_expense(self, amount: float, category: str, description: str) -> None:
        """Record an expense transaction.

        Args:
            amount: Dollar amount spent.
            category: Expense category (e.g. 'api_cost', 'tooling').
            description: Human-readable description of the expense.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO financials (ts, type, amount, category, description) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), "expense", float(amount), category, description),
            )
            conn.commit()
        except Exception as exc:
            logger.error("FinancialReporter.record_expense failed: %s", exc)
        finally:
            if conn:
                conn.close()

    def get_summary(self) -> dict:
        """Return a financial summary.

        Returns:
            dict with total_earned, total_spent, current_balance, avg_job_value, jobs_paid.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            cur.execute("SELECT COALESCE(SUM(amount), 0) FROM financials WHERE type='income'")
            total_earned = cur.fetchone()[0]

            cur.execute("SELECT COALESCE(SUM(amount), 0) FROM financials WHERE type='expense'")
            total_spent = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM financials WHERE type='income'")
            jobs_paid = cur.fetchone()[0]

            avg_job_value = total_earned / jobs_paid if jobs_paid > 0 else 0.0

            return {
                "total_earned": round(total_earned, 2),
                "total_spent": round(total_spent, 2),
                "current_balance": round(total_earned - total_spent, 2),
                "avg_job_value": round(avg_job_value, 2),
                "jobs_paid": jobs_paid,
            }
        except Exception as exc:
            logger.error("FinancialReporter.get_summary failed: %s", exc)
            return {"total_earned": 0.0, "total_spent": 0.0, "current_balance": 0.0, "avg_job_value": 0.0, "jobs_paid": 0}
        finally:
            if conn:
                conn.close()

    def get_earnings_by_skill(self, jobs_completed: List[dict]) -> dict:
        """Map each skill to total earnings from jobs that required it.

        Args:
            jobs_completed: List of job dicts with 'skills' and 'earnings' keys.

        Returns:
            dict mapping skill name -> total earned.
        """
        try:
            skill_earnings: Dict[str, float] = defaultdict(float)
            for job in jobs_completed:
                earnings = float(job.get("earnings", 0.0))
                for skill in job.get("skills", []):
                    skill_earnings[skill] += earnings
            return dict(skill_earnings)
        except Exception as exc:
            logger.error("FinancialReporter.get_earnings_by_skill failed: %s", exc)
            return {}

    def get_trend(self, days: int = 7) -> dict:
        """Return daily earnings totals for the last N days.

        Args:
            days: Number of days to look back.

        Returns:
            dict mapping date string (YYYY-MM-DD) -> earnings float.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()
            cur.execute(
                "SELECT DATE(ts), SUM(amount) FROM financials "
                "WHERE type='income' AND ts >= ? GROUP BY DATE(ts)",
                (since,),
            )
            rows = cur.fetchall()
            trend = {row[0]: round(row[1], 2) for row in rows}
            # Fill missing days with 0
            for i in range(days):
                day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                trend.setdefault(day, 0.0)
            return trend
        except Exception as exc:
            logger.error("FinancialReporter.get_trend failed: %s", exc)
            return {}
        finally:
            if conn:
                conn.close()

    def alert_low_balance(self, threshold: float = 20.0) -> Optional[str]:
        """Return an alert string if current balance is below the threshold.

        Args:
            threshold: Minimum acceptable balance in dollars.

        Returns:
            Alert string if balance is low, otherwise None.
        """
        try:
            summary = self.get_summary()
            balance = summary.get("current_balance", 0.0)
            if balance < threshold:
                return (
                    f"LOW BALANCE ALERT: Current balance ${balance:.2f} is below "
                    f"threshold ${threshold:.2f}. Consider accepting more jobs."
                )
            return None
        except Exception as exc:
            logger.error("FinancialReporter.alert_low_balance failed: %s", exc)
            return None


class ContinuousImprovementEngine:
    """Analyzes work cycles and generates improvement recommendations.

    Maintains a rolling history of cycles and identifies trends in quality,
    earnings, and skill performance to guide the worker's strategy.
    """

    _MAX_HISTORY = 50

    def __init__(self):
        self._cycles: deque = deque(maxlen=self._MAX_HISTORY)

    def record_cycle(
        self,
        jobs_completed: int,
        earnings: float,
        quality_scores: List[float],
        skills_improved: List[str],
    ) -> None:
        """Record the results of one work cycle.

        Args:
            jobs_completed: Number of jobs finished in this cycle.
            earnings: Total earnings for the cycle.
            quality_scores: List of quality scores (0.0 – 1.0) for each job.
            skills_improved: Skills that were exercised or improved.
        """
        try:
            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            )
            self._cycles.append(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "jobs_completed": jobs_completed,
                    "earnings": float(earnings),
                    "avg_quality": avg_quality,
                    "quality_scores": list(quality_scores),
                    "skills_improved": list(skills_improved),
                }
            )
        except Exception as exc:
            logger.error("ContinuousImprovementEngine.record_cycle failed: %s", exc)

    def analyze(self) -> dict:
        """Analyze recent cycles and return performance insights.

        Returns:
            dict with high_performing_skills, profitable_types, common_failures,
            and recommendations list.
        """
        try:
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

            high_performing_skills = sorted(
                skill_quality,
                key=lambda s: sum(skill_quality[s]) / len(skill_quality[s]),
                reverse=True,
            )[:5]

            avg_earnings = sum(c["earnings"] for c in recent) / len(recent)
            profitable_types: List[str] = []
            if avg_earnings > 0:
                profitable_types = [
                    f"cycles averaging ${avg_earnings:.2f} earnings"
                ]

            low_quality_cycles = [c for c in recent if c["avg_quality"] < 0.5]
            common_failures = (
                [f"{len(low_quality_cycles)} of last {len(recent)} cycles had low quality"]
                if low_quality_cycles
                else []
            )

            recommendations = self._build_recommendations(
                high_performing_skills, avg_earnings, low_quality_cycles, recent
            )

            return {
                "high_performing_skills": high_performing_skills,
                "profitable_types": profitable_types,
                "common_failures": common_failures,
                "recommendations": recommendations,
            }
        except Exception as exc:
            logger.error("ContinuousImprovementEngine.analyze failed: %s", exc)
            return {"high_performing_skills": [], "profitable_types": [], "common_failures": [], "recommendations": []}

    def _build_recommendations(
        self,
        high_performing_skills: List[str],
        avg_earnings: float,
        low_quality_cycles: list,
        recent: list,
    ) -> List[str]:
        recs = []
        if high_performing_skills:
            recs.append(f"Double down on high-performing skills: {', '.join(high_performing_skills[:3])}")
        if low_quality_cycles:
            recs.append("Quality is below threshold in recent cycles; slow down and review outputs before submission.")
        trend = [c["earnings"] for c in recent]
        if len(trend) >= 2 and trend[-1] < trend[0]:
            recs.append("Earnings are declining; consider diversifying job types or upskilling.")
        elif len(trend) >= 2 and trend[-1] > trend[0]:
            recs.append("Earnings are growing; maintain current strategy.")
        if not recs:
            recs.append("Performance is stable; continue current approach.")
        return recs

    def get_recommendations(self) -> List[dict]:
        """Return structured recommendations with expected impact and confidence.

        Returns:
            List of dicts, each with action (str), expected_impact (str), confidence (float).
        """
        try:
            analysis = self.analyze()
            raw_recs = analysis.get("recommendations", [])
            recent = list(self._cycles)[-5:]
            n = len(recent)

            structured = []
            for i, rec in enumerate(raw_recs):
                # Confidence grows with more data; cap at 0.95
                confidence = min(0.95, 0.4 + (n / 10.0) - (i * 0.05))
                confidence = max(0.1, confidence)
                structured.append(
                    {
                        "action": rec,
                        "expected_impact": "moderate improvement in earnings or quality",
                        "confidence": round(confidence, 2),
                    }
                )
            return structured
        except Exception as exc:
            logger.error("ContinuousImprovementEngine.get_recommendations failed: %s", exc)
            return []


class CredentialManager:
    """Secure credential storage using Fernet encryption (or base64 fallback).

    Credentials are stored in an encrypted JSON file. The encryption key is
    persisted in a separate key file. Values are never logged.
    """

    _CRED_FILE = ".janus_credentials.enc"

    def __init__(self, key_file: str = ".janus_key"):
        self.key_file = key_file
        self._fernet = None
        self._init_crypto()

    def _init_crypto(self) -> None:
        if not HAS_CRYPTOGRAPHY:
            logger.warning("cryptography package not installed; using base64 obfuscation fallback.")
            return
        try:
            key_path = Path(self.key_file)
            if key_path.exists():
                key = key_path.read_bytes().strip()
            else:
                key = Fernet.generate_key()
                key_path.write_bytes(key)
                # Restrict permissions on Unix-like systems
                try:
                    os.chmod(self.key_file, 0o600)
                except OSError:
                    pass
            self._fernet = Fernet(key)
        except Exception as exc:
            logger.error("CredentialManager._init_crypto failed: %s", exc)
            self._fernet = None

    def _load_store(self) -> dict:
        try:
            cred_path = Path(self._CRED_FILE)
            if not cred_path.exists():
                return {}
            raw = cred_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except Exception as exc:
            logger.error("CredentialManager._load_store failed: %s", exc)
            return {}

    def _save_store(self, store: dict) -> None:
        try:
            Path(self._CRED_FILE).write_text(json.dumps(store), encoding="utf-8")
        except Exception as exc:
            logger.error("CredentialManager._save_store failed: %s", exc)

    def _encrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        # Fallback: base64 obfuscation (not secure, just obfuscated)
        return base64.b64encode(value.encode()).decode()

    def _decrypt(self, token: str) -> str:
        if self._fernet:
            return self._fernet.decrypt(token.encode()).decode()
        return base64.b64decode(token.encode()).decode()

    def store(self, name: str, value: str) -> None:
        """Encrypt and persist a credential.

        Args:
            name: Credential name/key.
            value: Plaintext credential value (never logged).
        """
        try:
            store = self._load_store()
            store[name] = self._encrypt(value)
            self._save_store(store)
        except Exception as exc:
            logger.error("CredentialManager.store failed for '%s': %s", name, exc)

    def load(self, name: str) -> Optional[str]:
        """Decrypt and return a stored credential value.

        Args:
            name: Credential name/key.

        Returns:
            Plaintext value, or None if not found.
        """
        try:
            store = self._load_store()
            if name not in store:
                return None
            return self._decrypt(store[name])
        except Exception as exc:
            logger.error("CredentialManager.load failed for '%s': %s", name, exc)
            return None

    def list_names(self) -> List[str]:
        """Return the names of all stored credentials (not their values).

        Returns:
            List of credential name strings.
        """
        try:
            return list(self._load_store().keys())
        except Exception as exc:
            logger.error("CredentialManager.list_names failed: %s", exc)
            return []

    def rotate(self, name: str, new_value: str) -> None:
        """Replace an existing credential with a new value.

        Args:
            name: Credential name/key to rotate.
            new_value: New plaintext credential value (never logged).
        """
        try:
            store = self._load_store()
            if name not in store:
                logger.warning("CredentialManager.rotate: '%s' not found; storing as new.", name)
            store[name] = self._encrypt(new_value)
            self._save_store(store)
        except Exception as exc:
            logger.error("CredentialManager.rotate failed for '%s': %s", name, exc)


class WorkerCompletionMixin:
    """Mixin that adds all completion subsystems to JanusAutonomousWorker.

    Call init_completion_systems() during worker initialisation to attach
    market_analyzer, job_manager, adaptive_generator, financial_reporter,
    improvement_engine, and credentials as instance attributes.
    """

    def init_completion_systems(self) -> None:
        """Instantiate and attach all six completion subsystems."""
        try:
            self.market_analyzer = MarketAnalyzer()
            self.job_manager = ConcurrentJobManager()
            # Use self as the base_generator if the worker itself generates work,
            # otherwise callers can replace adaptive_generator.base_generator later.
            self.adaptive_generator = AdaptiveWorkGenerator(base_generator=self)
            self.financial_reporter = FinancialReporter()
            self.improvement_engine = ContinuousImprovementEngine()
            self.credentials = CredentialManager()
            logger.info("WorkerCompletionMixin: all completion systems initialised.")
        except Exception as exc:
            logger.error("WorkerCompletionMixin.init_completion_systems failed: %s", exc)

    def get_full_status(self) -> dict:
        """Return a combined status snapshot from all subsystems.

        Returns:
            dict with keys: job_manager, financial_summary, improvement_analysis,
            credentials_stored, market_info.
        """
        try:
            status: dict = {}

            if hasattr(self, "job_manager"):
                status["job_manager"] = self.job_manager.status()

            if hasattr(self, "financial_reporter"):
                status["financial_summary"] = self.financial_reporter.get_summary()
                alert = self.financial_reporter.alert_low_balance()
                if alert:
                    status["financial_alert"] = alert

            if hasattr(self, "improvement_engine"):
                status["improvement_analysis"] = self.improvement_engine.analyze()

            if hasattr(self, "credentials"):
                status["credentials_stored"] = self.credentials.list_names()

            if hasattr(self, "market_analyzer"):
                status["market_info"] = {
                    "skill_demand_tracked": len(self.market_analyzer._skill_demand),
                    "job_types_tracked": len(self.market_analyzer._job_type_freq),
                }

            return status
        except Exception as exc:
            logger.error("WorkerCompletionMixin.get_full_status failed: %s", exc)
            return {}


__all__ = [
    "MarketAnalyzer",
    "ConcurrentJobManager",
    "AdaptiveWorkGenerator",
    "FinancialReporter",
    "ContinuousImprovementEngine",
    "CredentialManager",
    "WorkerCompletionMixin",
]
