# goal_generator.py

“””
Janus Autonomous Goal Generator — Janus decides what to work on next.

Instead of waiting for a human to provide a goal, this module:

1. Reads current valence state (what Janus needs emotionally/cognitively)
1. Checks recent memory (what has worked, what hasn’t)
1. Scans the environment (what opportunities exist right now)
1. Generates a prioritized goal queue autonomously
1. Hands goals to ReplanningAgent for execution

This closes agency gap #3: Janus generates its own goals.

Usage:
from goal_generator import AutonomousGoalEngine
engine = AutonomousGoalEngine()
engine.run_forever()   # Janus works autonomously until stopped
# or
goals = engine.generate_goals()  # get current goal queue
“””

from **future** import annotations

import json
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import signal
import sys

from replanner import ReplanningAgent
from agent_loop import AgentMemory

# ── Goal templates Janus can autonomously pursue ──────────────────────────────

# These are real income-generating and capability-building goals.

# Janus selects from these based on its current state and history.

INCOME_GOALS = [
“Search Fiverr for data entry jobs under $50 and save the listings to fiverr_opportunities.json”,
“Search Fiverr for research jobs and save top 5 listings”,
“Search Upwork for virtual assistant jobs posted today and save them”,
“Find freelance writing jobs on ProBlogger and save listings”,
“Search for remote data entry jobs on Indeed and save results”,
]

LEARNING_GOALS = [
“Search YouTube for Python automation tutorials and save the top 3 video titles and URLs”,
“Research how to use Playwright for web scraping and save key findings”,
“Find tutorials on making money with digital art online and save top results”,
“Research Fiverr seller tips for new accounts and save best practices”,
]

MAINTENANCE_GOALS = [
“Review agent_errors.txt and summarize any recurring failures”,
“Check partial_results.txt for any incomplete work to resume”,
“Organize all saved job listings into a summary report”,
]

OPPORTUNITY_GOALS = [
“Search for ‘quick online jobs no experience’ and save top results”,
“Find micro-task platforms similar to Fiverr and list them”,
“Search for digital art commission communities on Reddit and save top subreddits”,
]

# ── Goal priority scoring ─────────────────────────────────────────────────────

@dataclass
class ScoredGoal:
goal:        str
score:       float
category:    str
reasoning:   str
created:     str = field(default_factory=lambda: datetime.now().isoformat())

class GoalScorer:
“””
Scores candidate goals based on:
- Valence state (what Janus needs right now)
- Recent history (avoid repeating failed goals)
- Time of day (some goals make more sense at certain times)
- Diversity (don’t just do the same thing over and over)
“””

```
def score(
    self,
    goal: str,
    category: str,
    valence: dict,
    recent_goals: list[dict],
    recent_failures: set[str],
) -> ScoredGoal:

    score = 0.5  # base score
    reasoning_parts = []

    # ── Valence influence ─────────────────────────────────────────────────
    curiosity   = valence.get("curiosity",   0.5)
    competence  = valence.get("competence",  0.5)
    autonomy    = valence.get("autonomy",    0.5)
    pleasure    = valence.get("pleasure",    0.5)

    if category == "income" and pleasure < 0.4:
        score += 0.3
        reasoning_parts.append("low pleasure → prioritize income")

    if category == "learning" and curiosity > 0.6:
        score += 0.25
        reasoning_parts.append("high curiosity → learning valuable")

    if category == "maintenance" and competence < 0.4:
        score += 0.2
        reasoning_parts.append("low competence → maintenance helps")

    if category == "opportunity" and autonomy > 0.6:
        score += 0.2
        reasoning_parts.append("high autonomy → explore opportunities")

    # ── Avoid recently attempted goals ────────────────────────────────────
    recent_goal_texts = [g.get("goal", "") for g in recent_goals]
    if any(goal[:40] in g for g in recent_goal_texts):
        score -= 0.3
        reasoning_parts.append("recently attempted → deprioritized")

    # ── Penalize recently failed goals ────────────────────────────────────
    if goal[:40] in recent_failures:
        score -= 0.4
        reasoning_parts.append("recently failed → avoid for now")

    # ── Randomness for exploration ────────────────────────────────────────
    score += random.uniform(-0.05, 0.1)

    # Clamp
    score = max(0.0, min(1.0, score))

    return ScoredGoal(
        goal      = goal,
        score     = round(score, 3),
        category  = category,
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "base scoring",
    )
```

# ── Environment Scanner ───────────────────────────────────────────────────────

class EnvironmentScanner:
“””
Looks at the local file system for signals that influence goal priority.
E.g. if there are unsaved job listings, prioritize organizing them.
If there are auth_required.txt files, note that credentials are needed.
“””

```
def scan(self) -> dict:
    signals = {
        "has_errors":          Path("agent_errors.txt").exists(),
        "has_partial_results": Path("partial_results.txt").exists(),
        "has_job_listings":    Path("fiverr_opportunities.json").exists(),
        "has_auth_gaps":       Path("auth_required.txt").exists(),
        "research_files":      len(list(Path(".").glob("research_*.txt"))),
    }

    # Check how old the job listings are
    if signals["has_job_listings"]:
        age = time.time() - Path("fiverr_opportunities.json").stat().st_mtime
        signals["listings_age_hours"] = round(age / 3600, 1)
    else:
        signals["listings_age_hours"] = 999

    return signals

def goal_hints(self, signals: dict) -> list[tuple[str, float]]:
    """
    Returns (goal_text, priority_boost) pairs based on environment signals.
    """
    hints = []

    if signals["has_errors"]:
        hints.append((
            "Review agent_errors.txt and summarize any recurring failures",
            0.3
        ))

    if signals["has_partial_results"]:
        hints.append((
            "Check partial_results.txt for any incomplete work to resume",
            0.25
        ))

    if signals.get("listings_age_hours", 999) > 24:
        hints.append((
            "Search Fiverr for data entry jobs under $50 and save the listings to fiverr_opportunities.json",
            0.35  # job listings are stale, refresh them
        ))

    if signals["research_files"] > 5:
        hints.append((
            "Organize all saved job listings into a summary report",
            0.2
        ))

    return hints
```

# ── Valence Reader ────────────────────────────────────────────────────────────

class ValenceReader:
“”“Reads current valence from persistent_state.json or AutonomousCore.”””

```
def read(self) -> dict:
    # Try persistent state file first
    p = Path("persistent_state.json")
    if p.exists():
        try:
            state = json.loads(p.read_text())
            if "valence" in state:
                return state["valence"]
        except Exception:
            pass

    # Default balanced valence
    return {
        "pleasure":   0.5,
        "arousal":    0.5,
        "curiosity":  0.5,
        "autonomy":   0.6,
        "competence": 0.5,
        "connection": 0.4,
    }
```

# ── Autonomous Goal Engine ────────────────────────────────────────────────────

class AutonomousGoalEngine:
“””
The top-level autonomous loop.

```
Janus continuously:
  1. Reads its valence state
  2. Scans the environment
  3. Scores all candidate goals
  4. Picks the highest scoring goal
  5. Executes it via ReplanningAgent
  6. Waits, then repeats

This is what makes Janus genuinely autonomous rather than
a tool that waits for human instructions.
"""

# How long to wait between goal cycles (seconds)
CYCLE_DELAY = 30

# How many recent goals to remember for deduplication
HISTORY_WINDOW = 20

# Max consecutive failures before taking a longer break
MAX_CONSECUTIVE_FAILURES = 3

def __init__(self, verbose: bool = True):
    self.agent    = ReplanningAgent(verbose=verbose)
    self.scorer   = GoalScorer()
    self.scanner  = EnvironmentScanner()
    self.valence  = ValenceReader()
    self.memory   = AgentMemory()
    self.verbose  = verbose

    self._running             = False
    self._consecutive_failures = 0
    self._completed_goals: list[str] = []

    # Build full candidate pool
    self._candidates = (
        [(g, "income")      for g in INCOME_GOALS] +
        [(g, "learning")    for g in LEARNING_GOALS] +
        [(g, "maintenance") for g in MAINTENANCE_GOALS] +
        [(g, "opportunity") for g in OPPORTUNITY_GOALS]
    )

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, self._handle_shutdown)

def _log(self, msg: str):
    if self.verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[GoalEngine {ts}] {msg}")

def generate_goals(self, top_n: int = 3) -> list[ScoredGoal]:
    """
    Generate a ranked list of goals based on current state.
    Returns top_n goals sorted by score descending.
    """
    valence       = self.valence.read()
    env_signals   = self.scanner.scan()
    env_hints     = self.scanner.goal_hints(env_signals)
    recent        = self.memory.recent_plans(self.HISTORY_WINDOW)
    recent_failed = {
        p["goal"][:40] for p in recent
        if not p.get("completed", False)
    }

    scored: list[ScoredGoal] = []

    # Score standard candidates
    for goal_text, category in self._candidates:
        sg = self.scorer.score(
            goal_text, category, valence, recent, recent_failed
        )
        scored.append(sg)

    # Apply environment hint boosts
    for hint_text, boost in env_hints:
        for sg in scored:
            if sg.goal[:40] == hint_text[:40]:
                sg.score = min(1.0, sg.score + boost)
                sg.reasoning += f"; env_boost+{boost}"

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_n]

def run_forever(self):
    """
    Main autonomous loop. Runs until stopped via Ctrl+C or stop().
    """
    self._running = True
    self._log("Autonomous mode started. Janus is now self-directing.")
    self._log(f"Cycle delay: {self.CYCLE_DELAY}s between goals.")

    cycle = 0
    while self._running:
        cycle += 1
        self._log(f"\n{'─'*50}")
        self._log(f"Cycle {cycle} — generating goals...")

        try:
            goals = self.generate_goals(top_n=3)

            if not goals:
                self._log("No goals generated — waiting...")
                time.sleep(self.CYCLE_DELAY)
                continue

            # Log goal queue
            for i, g in enumerate(goals):
                self._log(f"  #{i+1} [{g.category}] score={g.score:.2f} — {g.goal[:60]}")
                self._log(f"      reason: {g.reasoning}")

            # Execute top goal
            top_goal = goals[0]
            self._log(f"\nExecuting: {top_goal.goal}")

            plan = self.agent.run_goal(top_goal.goal)
            self._completed_goals.append(top_goal.goal)

            if plan.completed:
                self._consecutive_failures = 0
                self._log(f"✓ Goal completed: {plan.outcome}")
            else:
                self._consecutive_failures += 1
                self._log(f"✗ Goal incomplete: {plan.outcome}")

            # Back off if failing repeatedly
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                backoff = self.CYCLE_DELAY * 3
                self._log(
                    f"  {self._consecutive_failures} consecutive failures — "
                    f"backing off {backoff}s"
                )
                self._consecutive_failures = 0
                time.sleep(backoff)
            else:
                time.sleep(self.CYCLE_DELAY)

        except Exception as e:
            self._log(f"Cycle error: {e}")
            time.sleep(self.CYCLE_DELAY)

    self._log("Autonomous mode stopped.")

def run_once(self) -> ScoredGoal | None:
    """
    Generate goals and execute just the top one.
    Useful for testing or scheduled runs.
    """
    goals = self.generate_goals(top_n=1)
    if not goals:
        return None
    top = goals[0]
    self._log(f"Running single goal: {top.goal}")
    self.agent.run_goal(top.goal)
    return top

def status(self) -> dict:
    """Return current engine status."""
    return {
        "running":              self._running,
        "goals_completed":      len(self._completed_goals),
        "consecutive_failures": self._consecutive_failures,
        "recent_goals":         self._completed_goals[-5:],
        "success_rate":         self.memory.success_rate(),
        "current_valence":      self.valence.read(),
    }

def stop(self):
    self._running = False
    self.agent.stop()

def _handle_shutdown(self, sig, frame):
    self._log("\nShutdown signal — finishing current goal then stopping.")
    self._running = False
```

# ── CLI entry point ───────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys

```
engine = AutonomousGoalEngine(verbose=True)

if "--once" in sys.argv:
    # Run just one goal cycle
    goal = engine.run_once()
    if goal:
        print(f"\nExecuted: {goal.goal}")
        print(f"Category: {goal.category}")
        print(f"Score: {goal.score}")
elif "--list" in sys.argv:
    # Just show what goals would be selected
    goals = engine.generate_goals(top_n=5)
    print("\nTop 5 autonomous goals right now:\n")
    for i, g in enumerate(goals):
        print(f"#{i+1} [{g.category}] score={g.score:.3f}")
        print(f"   {g.goal}")
        print(f"   reason: {g.reasoning}\n")
else:
    # Full autonomous mode
    print("Starting Janus autonomous goal engine.")
    print("Press Ctrl+C to stop gracefully.\n")
    engine.run_forever()
```