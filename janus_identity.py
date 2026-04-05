"""
janus_identity.py
==================
Persistent identity for Janus across restarts, crashes, and updates.

The problem: every time Janus restarts it wakes up with no memory of
what it was doing, what it learned, or what decisions it made.

The solution: a single atomic identity snapshot that is:
  - Written after every significant event
  - Loaded on every startup before anything else runs
  - Versioned so we can roll back if something goes wrong
  - Checksummed so we detect corruption

What gets persisted:
  - Goals and their progress
  - Learned heuristics (what worked, what didn't)
  - Relationship context (clients, vendors)
  - Decision history (what Janus decided and why)
  - Operational stats (uptime, tasks completed, revenue)
  - Personality calibration (confidence thresholds, communication style)
  - Continuity narrative (a plain-English summary of "where we are")

No API keys. Pure stdlib JSON + hashlib.

Usage:
    from janus_identity import JanusIdentity
    identity = JanusIdentity()
    identity.load()

    # Read
    goals = identity.get("goals", [])
    heuristics = identity.get_heuristics()

    # Write
    identity.set("goals", updated_goals)
    identity.learn("When client goes quiet for 3 days, follow up proactively")
    identity.remember_decision("Raised prices 20%", "Revenue was flat for 60 days")
    identity.save()  # atomic write

    # On startup
    context = identity.startup_context()
    print(context)  # plain English summary of current state
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.identity")

_IDENTITY_FILE  = Path("janus_identity.json")
_BACKUP_FILE    = Path("janus_identity.backup.json")
_VERSIONS_DIR   = Path("identity_versions")


# ── Identity snapshot ─────────────────────────────────────────────────────────

@dataclass
class IdentitySnapshot:
    """The complete persistent state of Janus."""

    # Core identity
    name:        str   = "Janus"
    version:     int   = 1
    created_at:  str   = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:  str   = field(default_factory=lambda: datetime.now().isoformat())

    # Goals (mirrors CEO agent goals but survives restarts)
    goals:       List[Dict[str, Any]] = field(default_factory=list)

    # Learned heuristics — things Janus figured out from experience
    heuristics:  List[Dict[str, Any]] = field(default_factory=list)

    # Decision log — what Janus decided and why
    decisions:   List[Dict[str, Any]] = field(default_factory=list)

    # Relationship context — clients, vendors, contacts
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Operational stats
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "total_tasks_completed": 0,
        "total_tasks_failed":    0,
        "total_revenue_tracked": 0.0,
        "total_uptime_hours":    0.0,
        "sessions":              0,
        "last_startup":          None,
        "last_shutdown":         None,
    })

    # Personality calibration
    personality: Dict[str, Any] = field(default_factory=lambda: {
        "confidence_threshold":  0.5,
        "auto_approve_limit":    100.0,
        "communication_style":   "professional",
        "escalation_sensitivity":"medium",
        "risk_tolerance":        "moderate",
    })

    # Continuity narrative — plain English "where we are"
    narrative:   str = "Janus is starting fresh."

    # Arbitrary key-value store for other modules
    store:       Dict[str, Any] = field(default_factory=dict)

    # Checksum (set on save, verified on load)
    checksum:    str = ""

    def compute_checksum(self) -> str:
        """SHA-256 of the snapshot content (excluding checksum field)."""
        d = asdict(self)
        d.pop("checksum", None)
        payload = json.dumps(d, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checksum"] = self.compute_checksum()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "IdentitySnapshot":
        stored_checksum = d.pop("checksum", "")
        snap = cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})
        # Verify integrity
        computed = snap.compute_checksum()
        if stored_checksum and computed != stored_checksum:
            logger.warning(f"Identity checksum mismatch — file may be corrupted. "
                           f"Stored: {stored_checksum}, Computed: {computed}")
        return snap


# ── Main identity manager ─────────────────────────────────────────────────────

class JanusIdentity:
    """
    Single source of truth for Janus's persistent identity.

    All modules that need to survive restarts should read/write through here.
    Writes are atomic (write to temp → rename) to prevent corruption.
    """

    MAX_DECISIONS  = 500   # keep last N decisions
    MAX_HEURISTICS = 200
    VERSION_KEEP   = 10    # keep last N versioned snapshots

    def __init__(
        self,
        identity_file: Path = _IDENTITY_FILE,
        backup_file:   Path = _BACKUP_FILE,
        versions_dir:  Path = _VERSIONS_DIR,
    ):
        self._file    = identity_file
        self._backup  = backup_file
        self._ver_dir = versions_dir
        self._snap:   Optional[IdentitySnapshot] = None
        self._lock    = threading.Lock()
        self._dirty   = False
        self._session_start = time.time()

        # Auto-save thread
        self._autosave_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load identity from disk. Creates a fresh one if none exists.
        Returns True if loaded from file, False if created fresh.
        """
        # Try primary file
        if self._file.exists():
            snap = self._try_load(self._file)
            if snap:
                self._snap = snap
                self._on_startup()
                logger.info(f"Identity loaded: {snap.name} v{snap.version} "
                            f"(sessions: {snap.stats['sessions']})")
                return True

        # Try backup
        if self._backup.exists():
            logger.warning("Primary identity file failed — trying backup")
            snap = self._try_load(self._backup)
            if snap:
                self._snap = snap
                self._on_startup()
                logger.info(f"Identity restored from backup: {snap.name}")
                return True

        # Try latest version
        latest = self._latest_version()
        if latest:
            logger.warning("Both primary and backup failed — trying versioned snapshot")
            snap = self._try_load(latest)
            if snap:
                self._snap = snap
                self._on_startup()
                return True

        # Fresh start
        logger.info("No identity found — creating fresh identity")
        self._snap = IdentitySnapshot()
        self._on_startup()
        self.save()
        return False

    def save(self, versioned: bool = False) -> bool:
        """
        Atomically save identity to disk.
        Uses write-to-temp-then-rename to prevent corruption.
        """
        if self._snap is None:
            return False

        with self._lock:
            try:
                self._snap.updated_at = datetime.now().isoformat()
                data = self._snap.to_dict()
                payload = json.dumps(data, indent=2)

                # Atomic write: temp file → rename
                tmp = self._file.with_suffix(".tmp")
                tmp.write_text(payload)
                tmp.replace(self._file)

                # Backup copy
                shutil.copy2(self._file, self._backup)

                # Versioned snapshot
                if versioned:
                    self._save_version(payload)

                self._dirty = False
                return True

            except Exception as e:
                logger.error(f"Identity save failed: {e}")
                return False

    def start_autosave(self, interval_seconds: int = 60):
        """Save identity automatically every N seconds."""
        def _loop():
            while True:
                time.sleep(interval_seconds)
                if self._dirty:
                    self.save()

        self._autosave_thread = threading.Thread(
            target=_loop, daemon=True, name="janus-identity-autosave"
        )
        self._autosave_thread.start()

    def shutdown(self):
        """Call this when Janus is shutting down cleanly."""
        if self._snap:
            elapsed = (time.time() - self._session_start) / 3600
            self._snap.stats["total_uptime_hours"] = round(
                self._snap.stats.get("total_uptime_hours", 0) + elapsed, 2
            )
            self._snap.stats["last_shutdown"] = datetime.now().isoformat()
        self.save(versioned=True)
        logger.info("Identity saved on shutdown")

    # ── Read/write interface ──────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        self._ensure_loaded()
        return self._snap.store.get(key, default)

    def set(self, key: str, value: Any):
        """Set a value in the store."""
        self._ensure_loaded()
        self._snap.store[key] = value
        self._dirty = True

    def get_stat(self, key: str, default: Any = 0) -> Any:
        self._ensure_loaded()
        return self._snap.stats.get(key, default)

    def increment_stat(self, key: str, by: float = 1):
        self._ensure_loaded()
        self._snap.stats[key] = self._snap.stats.get(key, 0) + by
        self._dirty = True

    # ── Goals ─────────────────────────────────────────────────────────────────

    def sync_goals(self, goals: List[dict]):
        """Sync goals from CEO agent into identity."""
        self._ensure_loaded()
        self._snap.goals = goals
        self._dirty = True

    def get_goals(self) -> List[dict]:
        self._ensure_loaded()
        return self._snap.goals

    # ── Heuristics (learned rules) ────────────────────────────────────────────

    def learn(self, lesson: str, context: str = "", confidence: float = 0.7):
        """
        Record something Janus learned from experience.
        These survive restarts and inform future decisions.
        """
        self._ensure_loaded()
        entry = {
            "lesson":     lesson,
            "context":    context,
            "confidence": confidence,
            "learned_at": datetime.now().isoformat(),
            "used_count": 0,
        }
        self._snap.heuristics.append(entry)
        # Trim to max
        if len(self._snap.heuristics) > self.MAX_HEURISTICS:
            # Keep highest confidence
            self._snap.heuristics.sort(key=lambda h: h["confidence"], reverse=True)
            self._snap.heuristics = self._snap.heuristics[:self.MAX_HEURISTICS]
        self._dirty = True
        logger.info(f"Learned: {lesson[:60]}")

    def get_heuristics(self, min_confidence: float = 0.5) -> List[dict]:
        self._ensure_loaded()
        return [h for h in self._snap.heuristics if h["confidence"] >= min_confidence]

    def recall_relevant(self, situation: str, top_k: int = 5) -> List[dict]:
        """Find heuristics relevant to a situation using keyword matching."""
        self._ensure_loaded()
        words = set(situation.lower().split())
        scored = []
        for h in self._snap.heuristics:
            h_words = set(h["lesson"].lower().split())
            overlap  = len(words & h_words)
            if overlap > 0:
                score = overlap * h["confidence"]
                scored.append((score, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [h for _, h in scored[:top_k]]
        # Mark as used
        for h in results:
            h["used_count"] = h.get("used_count", 0) + 1
        if results:
            self._dirty = True
        return results

    # ── Decision log ──────────────────────────────────────────────────────────

    def remember_decision(self, decision: str, reasoning: str,
                          outcome: Optional[str] = None):
        """Log a decision Janus made."""
        self._ensure_loaded()
        entry = {
            "decision":   decision,
            "reasoning":  reasoning,
            "outcome":    outcome,
            "decided_at": datetime.now().isoformat(),
        }
        self._snap.decisions.append(entry)
        if len(self._snap.decisions) > self.MAX_DECISIONS:
            self._snap.decisions = self._snap.decisions[-self.MAX_DECISIONS:]
        self._dirty = True

    def update_decision_outcome(self, decision_text: str, outcome: str):
        """Update the outcome of a past decision."""
        self._ensure_loaded()
        for d in reversed(self._snap.decisions):
            if decision_text.lower() in d["decision"].lower():
                d["outcome"] = outcome
                self._dirty = True
                break

    def get_recent_decisions(self, n: int = 10) -> List[dict]:
        self._ensure_loaded()
        return self._snap.decisions[-n:]

    # ── Relationships ─────────────────────────────────────────────────────────

    def update_relationship(self, name: str, notes: str,
                            sentiment: float = 0.0, tags: Optional[List[str]] = None):
        """Update context about a person or organization."""
        self._ensure_loaded()
        existing = self._snap.relationships.get(name, {
            "name":         name,
            "first_seen":   datetime.now().isoformat(),
            "interactions": 0,
            "avg_sentiment": 0.0,
            "tags":         [],
            "notes":        [],
        })
        existing["interactions"] += 1
        existing["last_seen"]     = datetime.now().isoformat()
        # Rolling average sentiment
        n = existing["interactions"]
        existing["avg_sentiment"] = round(
            (existing["avg_sentiment"] * (n - 1) + sentiment) / n, 3
        )
        existing["notes"].append({
            "note": notes[:200],
            "at":   datetime.now().isoformat(),
        })
        existing["notes"] = existing["notes"][-20:]  # keep last 20
        if tags:
            existing["tags"] = list(set(existing.get("tags", []) + tags))
        self._snap.relationships[name] = existing
        self._dirty = True

    def get_relationship(self, name: str) -> Optional[dict]:
        self._ensure_loaded()
        return self._snap.relationships.get(name)

    # ── Narrative ─────────────────────────────────────────────────────────────

    def update_narrative(self, narrative: str):
        """Update the plain-English continuity summary."""
        self._ensure_loaded()
        self._snap.narrative = narrative
        self._dirty = True

    def get_narrative(self) -> str:
        self._ensure_loaded()
        return self._snap.narrative

    # ── Personality calibration ───────────────────────────────────────────────

    def get_personality(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return self._snap.personality.get(key, default)

    def set_personality(self, key: str, value: Any):
        self._ensure_loaded()
        self._snap.personality[key] = value
        self._dirty = True

    # ── Startup context ───────────────────────────────────────────────────────

    def startup_context(self) -> str:
        """
        Generate a plain-English briefing for Janus on startup.
        This is what Janus reads to know where it left off.
        """
        self._ensure_loaded()
        s    = self._snap
        stats = s.stats

        active_goals = [g for g in s.goals if g.get("status") not in ("completed", "failed")]
        recent_decisions = s.decisions[-3:] if s.decisions else []
        top_heuristics   = sorted(s.heuristics, key=lambda h: h["confidence"], reverse=True)[:3]

        lines = [
            f"=== Janus Startup Briefing ===",
            f"Session #{stats.get('sessions', 1)} | "
            f"Uptime so far: {stats.get('total_uptime_hours', 0):.1f}h total",
            f"",
            f"NARRATIVE: {s.narrative}",
            f"",
        ]

        if active_goals:
            lines.append(f"ACTIVE GOALS ({len(active_goals)}):")
            for g in active_goals[:5]:
                pct = g.get("completion_percentage", 0)
                lines.append(f"  • {g.get('name', '?')} — {pct:.0f}% complete")
            lines.append("")

        if recent_decisions:
            lines.append("RECENT DECISIONS:")
            for d in recent_decisions:
                lines.append(f"  • {d['decision'][:60]}")
                if d.get("outcome"):
                    lines.append(f"    → Outcome: {d['outcome'][:50]}")
            lines.append("")

        if top_heuristics:
            lines.append("TOP HEURISTICS (what I've learned):")
            for h in top_heuristics:
                lines.append(f"  • [{h['confidence']:.0%}] {h['lesson'][:70]}")
            lines.append("")

        lines.append(f"Revenue tracked: ${stats.get('total_revenue_tracked', 0):.2f}")
        lines.append(f"Tasks completed: {stats.get('total_tasks_completed', 0)}")
        lines.append(f"Last shutdown: {stats.get('last_shutdown', 'unknown')}")

        return "\n".join(lines)

    def full_summary(self) -> dict:
        """Machine-readable full summary."""
        self._ensure_loaded()
        return {
            "name":          self._snap.name,
            "version":       self._snap.version,
            "sessions":      self._snap.stats.get("sessions", 0),
            "uptime_hours":  self._snap.stats.get("total_uptime_hours", 0),
            "goals":         len(self._snap.goals),
            "heuristics":    len(self._snap.heuristics),
            "decisions":     len(self._snap.decisions),
            "relationships": len(self._snap.relationships),
            "narrative":     self._snap.narrative,
            "personality":   self._snap.personality,
            "stats":         self._snap.stats,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._snap is None:
            self.load()

    def _on_startup(self):
        """Called after loading — update session stats."""
        self._snap.stats["sessions"]      = self._snap.stats.get("sessions", 0) + 1
        self._snap.stats["last_startup"]  = datetime.now().isoformat()
        self._session_start               = time.time()
        self._dirty = True

    def _try_load(self, path: Path) -> Optional[IdentitySnapshot]:
        try:
            data = json.loads(path.read_text())
            return IdentitySnapshot.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load identity from {path}: {e}")
            return None

    def _save_version(self, payload: str):
        """Save a versioned snapshot."""
        self._ver_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._ver_dir / f"identity_{ts}.json"
        path.write_text(payload)
        # Prune old versions
        versions = sorted(self._ver_dir.glob("identity_*.json"))
        for old in versions[:-self.VERSION_KEEP]:
            old.unlink(missing_ok=True)

    def _latest_version(self) -> Optional[Path]:
        if not self._ver_dir.exists():
            return None
        versions = sorted(self._ver_dir.glob("identity_*.json"))
        return versions[-1] if versions else None


# ── Module-level singleton ────────────────────────────────────────────────────

_identity: Optional[JanusIdentity] = None

def get_identity() -> JanusIdentity:
    global _identity
    if _identity is None:
        _identity = JanusIdentity()
        _identity.load()
        _identity.start_autosave(interval_seconds=60)
    return _identity


# ── Integration hooks for other modules ──────────────────────────────────────

def on_task_complete(task_name: str, result: str):
    """Call this when a scheduler task completes."""
    ident = get_identity()
    ident.increment_stat("total_tasks_completed")
    ident.remember_decision(
        f"Executed task: {task_name}",
        f"Scheduled task ran successfully",
        outcome=result[:100],
    )

def on_task_failed(task_name: str, error: str):
    """Call this when a scheduler task fails."""
    ident = get_identity()
    ident.increment_stat("total_tasks_failed")

def on_revenue(amount: float, source: str):
    """Call this when revenue is confirmed."""
    ident = get_identity()
    current = ident.get_stat("total_revenue_tracked", 0.0)
    ident._snap.stats["total_revenue_tracked"] = round(current + amount, 2)
    ident._dirty = True

def on_brain_response(situation: str, response: str, acted_on: bool):
    """Call this after JanusBrain makes a decision."""
    ident = get_identity()
    if acted_on:
        ident.remember_decision(
            f"Brain decision: {situation[:60]}",
            response[:200],
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Identity Manager")
    parser.add_argument("--status",    action="store_true", help="Show identity summary")
    parser.add_argument("--briefing",  action="store_true", help="Show startup briefing")
    parser.add_argument("--learn",     type=str, metavar="LESSON", help="Add a heuristic")
    parser.add_argument("--heuristics",action="store_true", help="List learned heuristics")
    parser.add_argument("--decisions", action="store_true", help="Show recent decisions")
    parser.add_argument("--narrative", type=str, metavar="TEXT", help="Update narrative")
    parser.add_argument("--rollback",  action="store_true", help="List available versions")
    parser.add_argument("--reset",     action="store_true", help="Reset to fresh identity")
    args = parser.parse_args()

    if args.reset:
        for f in [_IDENTITY_FILE, _BACKUP_FILE]:
            f.unlink(missing_ok=True)
        print("Identity reset.")
        import sys; sys.exit(0)

    ident = JanusIdentity()
    ident.load()

    if args.status:
        summary = ident.full_summary()
        print(json.dumps(summary, indent=2))

    elif args.briefing:
        print(ident.startup_context())

    elif args.learn:
        ident.learn(args.learn)
        ident.save()
        print(f"Learned: {args.learn}")

    elif args.heuristics:
        heuristics = ident.get_heuristics()
        if not heuristics:
            print("No heuristics learned yet.")
        else:
            print(f"\n{'Confidence':<12} {'Used':>5}  Lesson")
            print("-" * 70)
            for h in sorted(heuristics, key=lambda x: x["confidence"], reverse=True):
                print(f"{h['confidence']:.0%}{'':8} {h.get('used_count', 0):>5}  {h['lesson'][:55]}")

    elif args.decisions:
        decisions = ident.get_recent_decisions(15)
        if not decisions:
            print("No decisions recorded yet.")
        else:
            for d in decisions:
                print(f"\n[{d['decided_at'][:10]}] {d['decision'][:60]}")
                print(f"  Reasoning: {d['reasoning'][:80]}")
                if d.get("outcome"):
                    print(f"  Outcome:   {d['outcome'][:60]}")

    elif args.narrative:
        ident.update_narrative(args.narrative)
        ident.save()
        print(f"Narrative updated.")

    elif args.rollback:
        if not _VERSIONS_DIR.exists():
            print("No versions saved yet.")
        else:
            versions = sorted(_VERSIONS_DIR.glob("identity_*.json"))
            if not versions:
                print("No versions found.")
            else:
                print(f"Available versions ({len(versions)}):")
                for v in versions:
                    size = v.stat().st_size
                    print(f"  {v.name}  ({size:,} bytes)")

    else:
        parser.print_help()
