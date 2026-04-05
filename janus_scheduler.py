"""
janus_scheduler.py
===================
Persistent task scheduler for Janus.
Runs 24/7 without human intervention. Survives crashes and restarts.
No API keys. No external services. Pure Python stdlib + existing Janus modules.

Features:
  - Cron-style recurring tasks (hourly, daily, weekly, custom intervals)
  - One-shot tasks with a future run time
  - Persistent state — survives restarts, picks up where it left off
  - Crash recovery — failed tasks are retried with backoff
  - Human-in-the-loop escalation — tasks that fail too many times get flagged
  - Lock file — prevents duplicate scheduler instances
  - Graceful shutdown on SIGINT/SIGTERM

Usage:
    # Start the scheduler (blocking)
    python janus_scheduler.py

    # Add a task programmatically
    from janus_scheduler import JanusScheduler, Task, Schedule
    sched = JanusScheduler()
    sched.add_task(Task(
        name="daily_report",
        schedule=Schedule.daily(hour=9),
        action="ceo_cycle",
    ))
    sched.run()
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("janus.scheduler")

# ── State file paths ──────────────────────────────────────────────────────────

_STATE_FILE = Path("scheduler_state.json")
_LOCK_FILE  = Path("scheduler.lock")
_LOG_FILE   = Path("scheduler_run.log")


# ── Schedule definition ───────────────────────────────────────────────────────

class ScheduleType(str, Enum):
    ONCE      = "once"       # run once at a specific time
    INTERVAL  = "interval"   # run every N seconds
    DAILY     = "daily"      # run every day at HH:MM
    WEEKLY    = "weekly"     # run every week on a given weekday at HH:MM
    HOURLY    = "hourly"     # run every hour at :MM


@dataclass
class Schedule:
    type: ScheduleType
    interval_seconds: Optional[int]  = None  # for INTERVAL
    hour: Optional[int]              = None  # for DAILY / WEEKLY
    minute: int                      = 0
    weekday: Optional[int]           = None  # 0=Mon … 6=Sun for WEEKLY
    run_at: Optional[str]            = None  # ISO datetime for ONCE

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def once(cls, dt: datetime) -> "Schedule":
        return cls(type=ScheduleType.ONCE, run_at=dt.isoformat())

    @classmethod
    def every(cls, seconds: int) -> "Schedule":
        return cls(type=ScheduleType.INTERVAL, interval_seconds=seconds)

    @classmethod
    def hourly(cls, minute: int = 0) -> "Schedule":
        return cls(type=ScheduleType.HOURLY, minute=minute)

    @classmethod
    def daily(cls, hour: int = 9, minute: int = 0) -> "Schedule":
        return cls(type=ScheduleType.DAILY, hour=hour, minute=minute)

    @classmethod
    def weekly(cls, weekday: int = 0, hour: int = 9, minute: int = 0) -> "Schedule":
        return cls(type=ScheduleType.WEEKLY, weekday=weekday, hour=hour, minute=minute)

    def next_run(self, after: datetime) -> datetime:
        """Calculate the next run time after `after`."""
        now = after

        if self.type == ScheduleType.ONCE:
            dt = datetime.fromisoformat(self.run_at)
            return dt if dt > now else now  # already past = run now

        if self.type == ScheduleType.INTERVAL:
            return now + timedelta(seconds=self.interval_seconds)

        if self.type == ScheduleType.HOURLY:
            candidate = now.replace(minute=self.minute, second=0, microsecond=0)
            if candidate <= now:
                candidate += timedelta(hours=1)
            return candidate

        if self.type == ScheduleType.DAILY:
            candidate = now.replace(hour=self.hour, minute=self.minute,
                                    second=0, microsecond=0)
            if candidate <= now:
                candidate += timedelta(days=1)
            return candidate

        if self.type == ScheduleType.WEEKLY:
            days_ahead = (self.weekday - now.weekday()) % 7
            candidate = (now + timedelta(days=days_ahead)).replace(
                hour=self.hour, minute=self.minute, second=0, microsecond=0
            )
            if candidate <= now:
                candidate += timedelta(weeks=1)
            return candidate

        return now + timedelta(seconds=60)  # fallback


# ── Task definition ───────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    ESCALATED = "escalated"  # too many failures, needs human review
    DISABLED  = "disabled"


@dataclass
class Task:
    name:        str
    schedule:    Schedule
    action:      str                    # action key (see ActionRegistry)
    params:      Dict[str, Any]         = field(default_factory=dict)
    task_id:     str                    = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status:      TaskStatus             = TaskStatus.PENDING
    next_run:    Optional[str]          = None   # ISO datetime
    last_run:    Optional[str]          = None
    last_result: Optional[str]          = None
    run_count:   int                    = 0
    fail_count:  int                    = 0
    max_retries: int                    = 3
    enabled:     bool                   = True

    def __post_init__(self):
        if self.next_run is None:
            self.next_run = self.schedule.next_run(datetime.now()).isoformat()

    def is_due(self) -> bool:
        if not self.enabled or self.status == TaskStatus.DISABLED:
            return False
        if self.next_run is None:
            return False
        return datetime.fromisoformat(self.next_run) <= datetime.now()

    def mark_started(self):
        self.status   = TaskStatus.RUNNING
        self.last_run = datetime.now().isoformat()
        self.run_count += 1

    def mark_success(self, result: str = ""):
        self.status      = TaskStatus.COMPLETED
        self.fail_count  = 0
        self.last_result = result[:500] if result else "ok"
        # Schedule next run (ONCE tasks don't repeat)
        if self.schedule.type != ScheduleType.ONCE:
            self.next_run = self.schedule.next_run(datetime.now()).isoformat()
            self.status   = TaskStatus.PENDING
        else:
            self.enabled = False

    def mark_failed(self, error: str = ""):
        self.fail_count += 1
        self.last_result = f"ERROR: {error[:400]}"
        if self.fail_count >= self.max_retries:
            self.status = TaskStatus.ESCALATED
            logger.warning(f"Task '{self.name}' escalated after {self.fail_count} failures")
        else:
            # Exponential backoff: 1min, 2min, 4min …
            backoff = 60 * (2 ** (self.fail_count - 1))
            self.next_run = (datetime.now() + timedelta(seconds=backoff)).isoformat()
            self.status   = TaskStatus.PENDING
            logger.info(f"Task '{self.name}' will retry in {backoff}s")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schedule"] = asdict(self.schedule)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        sched_d = d.pop("schedule")
        sched   = Schedule(**sched_d)
        return cls(schedule=sched, **d)


# ── Action registry ───────────────────────────────────────────────────────────

class ActionRegistry:
    """
    Maps action keys to callables.
    All Janus capabilities are registered here.
    No API keys — everything runs locally.
    """

    def __init__(self):
        self._actions: Dict[str, Callable] = {}
        self._register_defaults()

    def register(self, key: str, fn: Callable):
        self._actions[key] = fn
        logger.debug(f"Action registered: {key}")

    def get(self, key: str) -> Optional[Callable]:
        return self._actions.get(key)

    def list_actions(self) -> List[str]:
        return list(self._actions.keys())

    def _register_defaults(self):
        """Register all built-in Janus actions."""

        # ── CEO cycle ─────────────────────────────────────────────────────────
        def ceo_cycle(params: dict) -> str:
            try:
                from ceo_agent import CEOAgent, GoalPriority
                ceo = CEOAgent()
                rec = ceo.recommend_action(params.get("situation", "routine cycle"))
                ceo.save_state("ceo_state.json")
                return f"CEO cycle complete. {len(rec['recommendations'])} recommendations."
            except Exception as e:
                return f"CEO cycle error: {e}"

        # ── Task selection ────────────────────────────────────────────────────
        def select_and_log_task(params: dict) -> str:
            try:
                from autonomous_task_selector import AutonomousTaskSelector
                sel  = AutonomousTaskSelector()
                opps = sel.scan_opportunities()
                task = sel.select_next_task()
                if task:
                    return f"Selected task: {task.title} (ROI: {task.roi:.2f})"
                return "No tasks available"
            except Exception as e:
                return f"Task selection error: {e}"

        # ── Financial snapshot ────────────────────────────────────────────────
        def financial_snapshot(params: dict) -> str:
            try:
                from janus_finance import get_finance
                fin = get_finance()
                # Scan inbox for any new payments first
                confirmed = fin.scan_for_payments()
                pos = fin.get_position()
                result = (
                    f"Income: ${pos['income']:.2f} | "
                    f"Expenses: ${pos['expenses']:.2f} | "
                    f"Profit: ${pos['profit']:.2f} | "
                    f"Pending: ${pos['pending_income']:.2f}"
                )
                if confirmed:
                    result += f" | Auto-confirmed {len(confirmed)} payment(s)"
                if pos["pending_approvals"]:
                    result += f" | {pos['pending_approvals']} transfer(s) need approval"
                return result
            except Exception as e:
                # Fallback to old payment pipeline
                try:
                    from payment_pipeline import PaymentPipeline
                    pp = PaymentPipeline()
                    s  = pp.revenue_summary()
                    return (f"Revenue: ${s.total_earned:.2f} | "
                            f"Pending: ${s.total_pending:.2f} | "
                            f"Requests: {s.request_count}")
                except Exception as e2:
                    return f"Financial snapshot error: {e2}"

        # ── Brain reflection ──────────────────────────────────────────────────
        def brain_reflect(params: dict) -> str:
            try:
                from avus_brain import get_brain
                brain = get_brain()
                completed = params.get("completed", [])
                pending   = params.get("pending", ["review goals", "check revenue"])
                return brain.reflect(completed, pending)
            except Exception as e:
                return f"Brain reflect error: {e}"

        # ── Goal review ───────────────────────────────────────────────────────
        def goal_review(params: dict) -> str:
            try:
                state_file = Path("ceo_state.json")
                if not state_file.exists():
                    return "No CEO state found"
                state = json.loads(state_file.read_text())
                goals = state.get("goals", {})
                active = [g for g in goals.values()
                          if g.get("status") not in ("completed", "failed")]
                return f"Active goals: {len(active)} | Total: {len(goals)}"
            except Exception as e:
                return f"Goal review error: {e}"

        # ── Log heartbeat ─────────────────────────────────────────────────────
        def heartbeat(params: dict) -> str:
            msg = f"Janus alive at {datetime.now().isoformat()}"
            logger.info(msg)
            return msg

        # ── Shell command (whitelisted) ───────────────────────────────────────
        def shell_cmd(params: dict) -> str:
            import subprocess, shlex
            ALLOWED_PREFIXES = ("echo ", "python ", "dir", "ls")
            cmd = params.get("cmd", "")
            if not any(cmd.startswith(p) for p in ALLOWED_PREFIXES):
                return f"Command not whitelisted: {cmd}"
            try:
                r = subprocess.run(
                    shlex.split(cmd), capture_output=True, text=True, timeout=30
                )
                return r.stdout.strip() or r.stderr.strip()
            except Exception as e:
                return str(e)

        # ── Daily digest ──────────────────────────────────────────────────────
        def daily_digest(params: dict) -> str:
            try:
                from janus_comms import get_comms
                ok = get_comms().send_daily_digest()
                return "Daily digest sent" if ok else "Digest built (email not configured)"
            except Exception as e:
                return f"Digest error: {e}"

        # ── Register all ──────────────────────────────────────────────────────
        self.register("ceo_cycle",          ceo_cycle)
        self.register("select_task",        select_and_log_task)
        self.register("financial_snapshot", financial_snapshot)
        self.register("brain_reflect",      brain_reflect)
        self.register("goal_review",        goal_review)
        self.register("heartbeat",          heartbeat)
        self.register("shell_cmd",          shell_cmd)
        self.register("daily_digest",       daily_digest)


# ── Persistent state ──────────────────────────────────────────────────────────

class SchedulerState:
    """Loads and saves scheduler state to disk."""

    def __init__(self, path: Path = _STATE_FILE):
        self.path = path

    def load(self) -> List[Task]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            tasks = []
            for d in data.get("tasks", []):
                try:
                    tasks.append(Task.from_dict(d))
                except Exception as e:
                    logger.warning(f"Could not load task: {e}")
            logger.info(f"Loaded {len(tasks)} tasks from {self.path}")
            return tasks
        except Exception as e:
            logger.error(f"State load failed: {e}")
            return []

    def save(self, tasks: List[Task]):
        try:
            data = {"tasks": [t.to_dict() for t in tasks],
                    "saved_at": datetime.now().isoformat()}
            self.path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"State save failed: {e}")


# ── Lock file ─────────────────────────────────────────────────────────────────

class LockFile:
    """Prevents two scheduler instances from running simultaneously."""

    def __init__(self, path: Path = _LOCK_FILE):
        self.path = path

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                # Check if that PID is still alive
                os.kill(pid, 0)
                logger.error(f"Scheduler already running (PID {pid})")
                return False
            except (ProcessLookupError, ValueError):
                # Stale lock
                self.path.unlink(missing_ok=True)
        self.path.write_text(str(os.getpid()))
        return True

    def release(self):
        self.path.unlink(missing_ok=True)


# ── Main scheduler ────────────────────────────────────────────────────────────

class JanusScheduler:
    """
    The persistent task scheduler.
    Runs in a loop, checks for due tasks, executes them, saves state.
    Survives crashes — state is persisted after every execution.
    """

    TICK_INTERVAL = 10  # seconds between due-task checks

    def __init__(self, state_file: Path = _STATE_FILE):
        self.tasks:    List[Task]    = []
        self.registry: ActionRegistry = ActionRegistry()
        self.state:    SchedulerState = SchedulerState(state_file)
        self.lock:     LockFile       = LockFile()
        self._running  = False
        self._lock     = threading.Lock()

        # File handler for persistent log
        fh = logging.FileHandler(_LOG_FILE)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    # ── Task management ───────────────────────────────────────────────────────

    def add_task(self, task: Task) -> Task:
        with self._lock:
            # Replace if same name exists
            self.tasks = [t for t in self.tasks if t.name != task.name]
            self.tasks.append(task)
        self._save()
        logger.info(f"Task added: '{task.name}' | next run: {task.next_run}")
        return task

    def remove_task(self, name: str) -> bool:
        with self._lock:
            before = len(self.tasks)
            self.tasks = [t for t in self.tasks if t.name != name]
        if len(self.tasks) < before:
            self._save()
            logger.info(f"Task removed: '{name}'")
            return True
        return False

    def disable_task(self, name: str) -> bool:
        task = self._get_task(name)
        if task:
            task.enabled = False
            task.status  = TaskStatus.DISABLED
            self._save()
            return True
        return False

    def enable_task(self, name: str) -> bool:
        task = self._get_task(name)
        if task:
            task.enabled  = True
            task.status   = TaskStatus.PENDING
            task.fail_count = 0
            task.next_run = task.schedule.next_run(datetime.now()).isoformat()
            self._save()
            return True
        return False

    def get_status(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "name":       t.name,
                    "status":     t.status,
                    "next_run":   t.next_run,
                    "last_run":   t.last_run,
                    "run_count":  t.run_count,
                    "fail_count": t.fail_count,
                    "last_result": t.last_result,
                }
                for t in self.tasks
            ]

    # ── Default task setup ────────────────────────────────────────────────────

    def setup_default_tasks(self):
        """Register the standard Janus autonomous task set."""

        self.add_task(Task(
            name     = "heartbeat",
            schedule = Schedule.every(300),          # every 5 min
            action   = "heartbeat",
        ))
        self.add_task(Task(
            name     = "ceo_morning_cycle",
            schedule = Schedule.daily(hour=9, minute=0),
            action   = "ceo_cycle",
            params   = {"situation": "morning review"},
        ))
        self.add_task(Task(
            name     = "ceo_evening_cycle",
            schedule = Schedule.daily(hour=18, minute=0),
            action   = "ceo_cycle",
            params   = {"situation": "evening review"},
        ))
        self.add_task(Task(
            name     = "financial_snapshot",
            schedule = Schedule.daily(hour=8, minute=0),
            action   = "financial_snapshot",
        ))
        self.add_task(Task(
            name     = "task_selection",
            schedule = Schedule.every(3600),         # every hour
            action   = "select_task",
        ))
        self.add_task(Task(
            name     = "goal_review",
            schedule = Schedule.weekly(weekday=0, hour=8, minute=30),  # Monday 8:30
            action   = "goal_review",
        ))
        self.add_task(Task(
            name     = "brain_reflect",
            schedule = Schedule.daily(hour=21, minute=0),
            action   = "brain_reflect",
            params   = {"pending": ["review goals", "check revenue", "plan tomorrow"]},
        ))
        self.add_task(Task(
            name     = "daily_digest",
            schedule = Schedule.daily(hour=20, minute=0),
            action   = "daily_digest",
        ))

        logger.info(f"Default tasks registered: {len(self.tasks)}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Start the scheduler. Blocks until stopped."""
        if not self.lock.acquire():
            sys.exit(1)

        # Load persisted state
        saved = self.state.load()
        if saved:
            with self._lock:
                # Merge: keep saved tasks, add any new defaults not in saved
                saved_names = {t.name for t in saved}
                self.tasks = saved + [t for t in self.tasks if t.name not in saved_names]
        elif not self.tasks:
            self.setup_default_tasks()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._running = True
        logger.info(f"Janus Scheduler started | PID={os.getpid()} | "
                    f"Tasks={len(self.tasks)}")

        # Load persistent identity
        try:
            from janus_identity import get_identity
            identity = get_identity()
            logger.info(f"Identity loaded: session #{identity.get_stat('sessions')}")
            print(identity.startup_context())
        except Exception as e:
            logger.warning(f"Identity load failed: {e}")

        # Start escalation inbox polling
        try:
            from janus_escalation import get_escalation_manager
            get_escalation_manager().start_inbox_polling(interval_seconds=180)
            logger.info("Escalation inbox polling started")
        except Exception as e:
            logger.warning(f"Escalation polling could not start: {e}")

        try:
            while self._running:
                self._tick()
                time.sleep(self.TICK_INTERVAL)
        finally:
            self._save()
            self.lock.release()
            # Save identity on clean shutdown
            try:
                from janus_identity import get_identity
                get_identity().shutdown()
            except Exception:
                pass
            logger.info("Janus Scheduler stopped.")

    def _tick(self):
        """Check for due tasks and execute them."""
        due = []
        with self._lock:
            for task in self.tasks:
                if task.is_due():
                    due.append(task)

        for task in due:
            self._execute_task(task)

    def _execute_task(self, task: Task):
        """Execute a single task and update its state."""
        logger.info(f"Running task: '{task.name}' (action={task.action})")
        task.mark_started()
        self._save()

        action_fn = self.registry.get(task.action)
        if action_fn is None:
            task.mark_failed(f"Unknown action: {task.action}")
            self._save()
            logger.error(f"Task '{task.name}' failed: unknown action '{task.action}'")
            return

        try:
            result = action_fn(task.params)
            task.mark_success(str(result) if result else "ok")
            logger.info(f"Task '{task.name}' succeeded: {str(result)[:100]}")
            # Notify via comms
            try:
                from janus_comms import get_comms
                get_comms().report_task_complete(task.name, str(result)[:100])
            except Exception:
                pass
            # Persist to identity
            try:
                from janus_identity import on_task_complete
                on_task_complete(task.name, str(result)[:100])
            except Exception:
                pass
        except Exception as e:
            err = traceback.format_exc()
            task.mark_failed(str(e))
            logger.error(f"Task '{task.name}' failed: {e}\n{err}")
            # Persist to identity
            try:
                from janus_identity import on_task_failed
                on_task_failed(task.name, str(e))
            except Exception:
                pass
            # Escalation notification
            if task.status == TaskStatus.ESCALATED:
                try:
                    from janus_comms import get_comms
                    get_comms().report_escalation(task.name, str(e))
                except Exception:
                    pass
                # Create formal escalation record
                try:
                    from janus_escalation import get_escalation_manager, EscalationTrigger
                    get_escalation_manager().escalate(
                        title       = f"Task failed: {task.name}",
                        description = (f"Task '{task.name}' has failed {task.fail_count} times.\n\n"
                                       f"Last error: {str(e)[:300]}"),
                        trigger     = EscalationTrigger.TASK_FAILURE,
                        context     = {"task": task.name, "action": task.action,
                                       "fail_count": task.fail_count, "error": str(e)},
                        options     = ["Retry task", "Disable task", "Investigate manually"],
                        default_option = "Disable task",
                        callback    = lambda resp, t=task: self._handle_escalation_response(t, resp),
                    )
                except Exception as esc_err:
                    logger.warning(f"Could not create escalation record: {esc_err}")
        finally:
            self._save()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _handle_escalation_response(self, task: Task, response: str):
        """Handle a human response to a task escalation."""
        response_lower = response.lower()
        if "retry" in response_lower:
            task.fail_count  = 0
            task.status      = TaskStatus.PENDING
            task.next_run    = task.schedule.next_run(datetime.now()).isoformat()
            self._save()
            logger.info(f"Task '{task.name}' re-enabled after escalation response")
        elif "disable" in response_lower:
            task.enabled = False
            task.status  = TaskStatus.DISABLED
            self._save()
            logger.info(f"Task '{task.name}' disabled after escalation response")
        else:
            logger.info(f"Escalation response for '{task.name}': {response}")

    def _get_task(self, name: str) -> Optional[Task]:
        with self._lock:
            for t in self.tasks:
                if t.name == name:
                    return t
        return None

    def _save(self):
        with self._lock:
            self.state.save(self.tasks)

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} received — shutting down...")
        self._running = False

    def stop(self):
        self._running = False


# ── Background thread helper ──────────────────────────────────────────────────

def start_scheduler_thread(scheduler: Optional[JanusScheduler] = None) -> JanusScheduler:
    """
    Start the scheduler in a background daemon thread.
    Use this when embedding the scheduler inside another process.

    Returns the scheduler instance so you can add tasks to it.
    """
    if scheduler is None:
        scheduler = JanusScheduler()
        scheduler.setup_default_tasks()

    t = threading.Thread(target=scheduler.run, daemon=True, name="janus-scheduler")
    t.start()
    logger.info("Scheduler started in background thread")
    return scheduler


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Persistent Task Scheduler")
    parser.add_argument("--status",   action="store_true", help="Show task status and exit")
    parser.add_argument("--reset",    action="store_true", help="Clear saved state and start fresh")
    parser.add_argument("--add-task", type=str, metavar="JSON",
                        help='Add a task: \'{"name":"x","action":"heartbeat","every":300}\'')
    args = parser.parse_args()

    if args.reset and _STATE_FILE.exists():
        _STATE_FILE.unlink()
        print("State cleared.")

    sched = JanusScheduler()

    if args.status:
        saved = sched.state.load()
        if not saved:
            print("No saved tasks.")
        else:
            print(f"{'Name':<25} {'Status':<12} {'Next Run':<22} {'Runs':>5} {'Fails':>5}")
            print("-" * 75)
            for t in saved:
                print(f"{t.name:<25} {t.status:<12} "
                      f"{(t.next_run or 'N/A')[:19]:<22} "
                      f"{t.run_count:>5} {t.fail_count:>5}")
        sys.exit(0)

    if args.add_task:
        d = json.loads(args.add_task)
        every = d.pop("every", None)
        daily_hour = d.pop("daily_hour", None)
        if every:
            sched_obj = Schedule.every(int(every))
        elif daily_hour is not None:
            sched_obj = Schedule.daily(hour=int(daily_hour))
        else:
            sched_obj = Schedule.every(3600)
        task = Task(schedule=sched_obj, **d)
        sched.add_task(task)
        print(f"Task '{task.name}' added.")
        sys.exit(0)

    # Normal run
    sched.setup_default_tasks()
    print(f"Starting Janus Scheduler (PID {os.getpid()})...")
    print(f"State file: {_STATE_FILE.absolute()}")
    print(f"Log file:   {_LOG_FILE.absolute()}")
    print("Press Ctrl+C to stop.\n")
    sched.run()
