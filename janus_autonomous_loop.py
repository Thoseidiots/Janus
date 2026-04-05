"""
janus_autonomous_loop.py
=========================
The true autonomous loop for Janus.

This replaces the rigid scheduler + hardcoded task list with a
brain-driven observe-think-act cycle. Janus:

  1. OBSERVE  -- reads its own state, environment, inbox, finances
  2. THINK    -- asks JanusBrain "what should I do right now?"
  3. GENERATE -- synthesizes novel goals (not from a template list)
  4. PRIORITIZE -- scores them against current context
  5. ACT      -- executes the top goal via the agent loop
  6. LEARN    -- records what worked, updates identity and heuristics
  7. REPEAT

The key difference from the scheduler:
  - No pre-defined task list
  - Goals are generated fresh each cycle from brain reasoning
  - Janus can invent entirely new approaches
  - Learning compounds -- past outcomes influence future goal selection

Usage:
    from janus_autonomous_loop import JanusAutonomousLoop
    loop = JanusAutonomousLoop()
    loop.run()          # blocking, runs forever
    loop.run_once()     # single cycle, returns what happened
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.loop")

# ── Goal record ───────────────────────────────────────────────────────────────

@dataclass
class GeneratedGoal:
    goal_id:     str
    description: str
    reasoning:   str
    category:    str        # "income" | "capability" | "maintenance" | "relationship"
    urgency:     float      # 0-1
    effort:      float      # estimated hours
    created_at:  str        = field(default_factory=lambda: datetime.now().isoformat())
    executed:    bool       = False
    outcome:     Optional[str] = None
    success:     Optional[bool] = None


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles a rich context string for JanusBrain to reason about.
    Pulls from identity, finance, escalations, comms queue, and environment.
    """

    def build(self) -> str:
        parts = []

        # Identity / stats
        try:
            from janus_identity import get_identity
            ident = get_identity()
            pos   = ident.full_summary()
            parts.append(
                f"Sessions: {pos['sessions']} | "
                f"Uptime: {pos['uptime_hours']:.1f}h | "
                f"Tasks done: {pos['stats'].get('total_tasks_completed', 0)} | "
                f"Revenue tracked: ${pos['stats'].get('total_revenue_tracked', 0):.2f}"
            )
            if pos["heuristics"]:
                top = ident.get_heuristics(min_confidence=0.6)[:3]
                if top:
                    parts.append("Known heuristics: " +
                                 " | ".join(h["lesson"][:50] for h in top))
        except Exception:
            pass

        # Financial position
        try:
            from janus_finance import get_finance
            fin = get_finance()
            fp  = fin.get_position()
            parts.append(
                f"Finances: income=${fp['income']:.2f} | "
                f"expenses=${fp['expenses']:.2f} | "
                f"profit=${fp['profit']:.2f} | "
                f"pending=${fp['pending_income']:.2f}"
            )
            # Count pending invoices
            pending = [e for e in fin._entries.values()
                       if hasattr(e, 'status') and str(e.status) in ('EntryStatus.PENDING', 'pending')
                       and str(getattr(e, 'type', '')) in ('EntryType.INCOME', 'income')]
            if pending:
                parts.append(f"Pending invoices: {len(pending)} unpaid")
        except Exception:
            pass

        # Open escalations
        try:
            from janus_escalation import get_escalation_manager
            open_escs = get_escalation_manager().get_open_escalations()
            if open_escs:
                parts.append(f"Open escalations needing attention: {len(open_escs)}")
        except Exception:
            pass

        # Unread messages
        try:
            from janus_comms import get_comms
            unread = get_comms().get_unread_summary()
            if unread != "No unread messages.":
                parts.append(f"Unread updates: {unread[:200]}")
        except Exception:
            pass

        # Environment signals
        env = self._scan_environment()
        if env:
            parts.append("Environment: " + " | ".join(env))

        return "\n".join(parts) if parts else "No context available."

    def _scan_environment(self) -> List[str]:
        signals = []
        if Path("agent_errors.txt").exists():
            signals.append("agent_errors.txt exists (review needed)")
        if Path("partial_results.txt").exists():
            signals.append("partial_results.txt exists (incomplete work)")
        research = list(Path(".").glob("research_*.txt"))
        if research:
            signals.append(f"{len(research)} research files saved")
        return signals


# ── Goal synthesizer ──────────────────────────────────────────────────────────

class GoalSynthesizer:
    """
    Uses JanusBrain to generate novel goals from the current context.
    No hardcoded template list — goals emerge from reasoning.
    """

    _SYNTHESIS_PROMPT = """You are Janus, an autonomous AI agent.
Your current situation:
{context}

Recent goal history (avoid repeating these):
{recent_goals}

Based on this situation, generate 3 specific, actionable goals I should pursue right now.
For each goal, provide:
- A clear one-sentence description of what to do
- The category (income / capability / maintenance / relationship)
- Urgency score 0.0-1.0
- Estimated effort in hours

Format each goal as:
GOAL: [description]
CATEGORY: [category]
URGENCY: [0.0-1.0]
EFFORT: [hours]
REASON: [why this matters now]

Focus on goals that generate income or build capability to generate income.
Be specific and creative -- don't just repeat obvious things."""

    def synthesize(self, context: str, recent_goals: List[str]) -> List[GeneratedGoal]:
        """Ask JanusBrain to generate goals. Falls back to heuristic goals if brain unavailable."""
        recent_str = "\n".join(f"- {g}" for g in recent_goals[-5:]) or "None yet"
        prompt = self._SYNTHESIS_PROMPT.format(
            context=context,
            recent_goals=recent_str,
        )

        try:
            from avus_brain import get_brain
            brain    = get_brain()
            response = brain.ask(prompt, max_tokens=500)
            goals    = self._parse_goals(response)
            if goals:
                return goals
        except Exception as e:
            logger.warning(f"Brain synthesis failed: {e}")

        # Fallback: heuristic goal generation
        return self._heuristic_goals(context)

    def _parse_goals(self, response: str) -> List[GeneratedGoal]:
        """Parse brain response into GeneratedGoal objects."""
        goals = []
        blocks = response.strip().split("\n\n")

        for block in blocks:
            lines = {
                line.split(":")[0].strip().upper(): ":".join(line.split(":")[1:]).strip()
                for line in block.splitlines()
                if ":" in line
            }
            desc     = lines.get("GOAL", "")
            category = lines.get("CATEGORY", "income").lower()
            reason   = lines.get("REASON", "")

            try:
                urgency = float(lines.get("URGENCY", "0.5"))
            except ValueError:
                urgency = 0.5

            try:
                effort = float(lines.get("EFFORT", "2"))
            except ValueError:
                effort = 2.0

            if desc and len(desc) > 10:
                goals.append(GeneratedGoal(
                    goal_id     = f"g_{int(time.time())}_{len(goals)}",
                    description = desc,
                    reasoning   = reason,
                    category    = category if category in
                                  ("income", "capability", "maintenance", "relationship")
                                  else "income",
                    urgency     = max(0.0, min(1.0, urgency)),
                    effort      = max(0.1, effort),
                ))

        return goals[:5]  # cap at 5

    def _heuristic_goals(self, context: str) -> List[GeneratedGoal]:
        """
        Fallback goal generation when brain isn't available.
        Uses context signals to pick sensible defaults.
        """
        goals = []
        ctx = context.lower()

        if "pending invoices" in ctx:
            goals.append(GeneratedGoal(
                goal_id     = f"g_inv_{int(time.time())}",
                description = "Follow up on pending invoices and scan inbox for payment confirmations",
                reasoning   = "Pending invoices detected — chase payment",
                category    = "income",
                urgency     = 0.9,
                effort      = 0.5,
            ))

        if "escalation" in ctx:
            goals.append(GeneratedGoal(
                goal_id     = f"g_esc_{int(time.time())}",
                description = "Review and respond to open escalations",
                reasoning   = "Escalations need human-in-loop resolution",
                category    = "maintenance",
                urgency     = 0.95,
                effort      = 0.25,
            ))

        if "profit" in ctx and "$0" in ctx:
            goals.append(GeneratedGoal(
                goal_id     = f"g_rev_{int(time.time())}",
                description = "Identify the highest-ROI service to offer and create an invoice template",
                reasoning   = "No profit recorded — need to generate income",
                category    = "income",
                urgency     = 0.8,
                effort      = 1.0,
            ))

        # Always have a fallback
        if not goals:
            goals.append(GeneratedGoal(
                goal_id     = f"g_def_{int(time.time())}",
                description = "Review current state, identify the single most impactful action, and execute it",
                reasoning   = "Default: always be moving forward",
                category    = "income",
                urgency     = 0.5,
                effort      = 1.0,
            ))

        return goals


# ── Goal executor ─────────────────────────────────────────────────────────────

class GoalExecutor:
    """
    Executes a GeneratedGoal using the appropriate Janus subsystem.
    Routes based on category and content.
    """

    def execute(self, goal: GeneratedGoal) -> tuple[bool, str]:
        """
        Execute a goal. Returns (success, result_description).
        """
        desc = goal.description.lower()

        # Route to appropriate handler
        if "invoice" in desc or "payment" in desc or "revenue" in desc:
            return self._handle_finance(goal)
        elif "escalation" in desc or "review" in desc and "error" in desc:
            return self._handle_maintenance(goal)
        elif "scan" in desc or "search" in desc or "find" in desc:
            return self._handle_research(goal)
        elif "learn" in desc or "study" in desc or "capability" in desc:
            return self._handle_capability(goal)
        else:
            return self._handle_brain_execution(goal)

    def _handle_finance(self, goal: GeneratedGoal) -> tuple[bool, str]:
        try:
            from janus_finance import get_finance
            fin = get_finance()
            confirmed = fin.scan_for_payments()
            pos = fin.get_position()
            result = (f"Scanned for payments: {len(confirmed)} confirmed. "
                      f"Current profit: ${pos['profit']:.2f}")
            return True, result
        except Exception as e:
            return False, str(e)

    def _handle_maintenance(self, goal: GeneratedGoal) -> tuple[bool, str]:
        results = []
        # Check escalations
        try:
            from janus_escalation import get_escalation_manager
            mgr  = get_escalation_manager()
            open_escs = mgr.get_open_escalations()
            expired = mgr.process_expired()
            results.append(f"{len(open_escs)} open escalations, {len(expired)} auto-resolved")
        except Exception as e:
            results.append(f"Escalation check failed: {e}")

        # Check error files
        if Path("agent_errors.txt").exists():
            try:
                errors = Path("agent_errors.txt").read_text()[-500:]
                results.append(f"agent_errors.txt: {len(errors)} chars of recent errors")
            except Exception:
                pass

        return True, " | ".join(results)

    def _handle_research(self, goal: GeneratedGoal) -> tuple[bool, str]:
        """Use the agent loop for web research goals."""
        try:
            from replanner import ReplanningAgent
            agent  = ReplanningAgent(verbose=False)
            plan   = agent.run_goal(goal.description)
            agent.stop()
            return plan.completed, plan.outcome
        except Exception as e:
            return False, str(e)

    def _handle_capability(self, goal: GeneratedGoal) -> tuple[bool, str]:
        """Record capability goals as learning intentions."""
        try:
            from janus_identity import get_identity
            get_identity().learn(
                f"Capability goal: {goal.description}",
                context=goal.reasoning,
                confidence=0.6,
            )
            return True, f"Capability goal logged: {goal.description[:60]}"
        except Exception as e:
            return False, str(e)

    def _handle_brain_execution(self, goal: GeneratedGoal) -> tuple[bool, str]:
        """Ask the brain how to execute this goal, then act on its answer."""
        try:
            from avus_brain import get_brain
            brain    = get_brain()
            response = brain.plan(goal.description, context=goal.reasoning)
            # Log the plan to identity
            from janus_identity import get_identity
            get_identity().remember_decision(
                f"Executed goal: {goal.description[:60]}",
                response[:200],
            )
            return True, f"Brain plan executed: {response[:100]}"
        except Exception as e:
            return False, str(e)


# ── Learning recorder ─────────────────────────────────────────────────────────

class LearningRecorder:
    """Records what worked and what didn't to improve future goal selection."""

    def record(self, goal: GeneratedGoal, success: bool, result: str):
        try:
            from janus_identity import get_identity
            ident = get_identity()

            # Record the decision
            ident.remember_decision(
                decision  = goal.description,
                reasoning = goal.reasoning,
                outcome   = result[:150],
            )

            # Learn from success
            if success:
                ident.learn(
                    lesson     = f"'{goal.category}' goals like '{goal.description[:50]}' work well",
                    context    = result[:100],
                    confidence = 0.65,
                )
                ident.increment_stat("total_tasks_completed")
            else:
                ident.learn(
                    lesson     = f"Avoid: '{goal.description[:50]}' — failed with: {result[:60]}",
                    context    = goal.reasoning,
                    confidence = 0.7,
                )
                ident.increment_stat("total_tasks_failed")

        except Exception as e:
            logger.warning(f"Learning record failed: {e}")


# ── Main autonomous loop ──────────────────────────────────────────────────────

class JanusAutonomousLoop:
    """
    The true autonomous loop.
    Give it a directive and it figures out the rest.

    loop = JanusAutonomousLoop(directive="generate income")
    loop.run()
    """

    DEFAULT_CYCLE_SECONDS = 120   # 2 minutes between cycles
    MAX_CONSECUTIVE_FAILS = 4

    def __init__(
        self,
        directive:      str   = "generate income and build capability",
        cycle_seconds:  int   = DEFAULT_CYCLE_SECONDS,
        verbose:        bool  = True,
    ):
        self.directive      = directive
        self.cycle_seconds  = cycle_seconds
        self.verbose        = verbose

        self._context    = ContextBuilder()
        self._synthesizer= GoalSynthesizer()
        self._executor   = GoalExecutor()
        self._learner    = LearningRecorder()

        self._running         = False
        self._cycle_count     = 0
        self._consecutive_fails = 0
        self._goal_history:   List[str] = []
        self._lock            = threading.Lock()

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        """Start the autonomous loop. Blocks until stopped."""
        self._running = True
        self._log(f"Autonomous loop started | Directive: '{self.directive}'")
        self._log(f"Cycle interval: {self.cycle_seconds}s\n")

        while self._running:
            try:
                self._cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self._consecutive_fails += 1

            if not self._running:
                break

            # Backoff on repeated failures
            if self._consecutive_fails >= self.MAX_CONSECUTIVE_FAILS:
                backoff = self.cycle_seconds * 3
                self._log(f"Too many failures — backing off {backoff}s")
                self._consecutive_fails = 0
                time.sleep(backoff)
            else:
                time.sleep(self.cycle_seconds)

        self._log("Autonomous loop stopped.")

    def run_once(self) -> Optional[GeneratedGoal]:
        """Run a single cycle and return the goal that was executed."""
        return self._cycle()

    def run_in_background(self) -> threading.Thread:
        """Start the loop in a daemon thread."""
        t = threading.Thread(target=self.run, daemon=True, name="janus-autonomous")
        t.start()
        self._log("Autonomous loop running in background")
        return t

    def stop(self):
        self._running = False

    # ── Core cycle ────────────────────────────────────────────────────────────

    def _cycle(self) -> Optional[GeneratedGoal]:
        self._cycle_count += 1
        self._log(f"\n{'─'*55}")
        self._log(f"Cycle {self._cycle_count} | {datetime.now().strftime('%H:%M:%S')}")

        # 1. OBSERVE
        self._log("Observing current state...")
        context = self._context.build()
        full_context = f"Directive: {self.directive}\n\n{context}"

        # 2. THINK + GENERATE
        self._log("Synthesizing goals...")
        goals = self._synthesizer.synthesize(full_context, self._goal_history)

        if not goals:
            self._log("No goals generated this cycle")
            return None

        # 3. PRIORITIZE
        goals.sort(key=lambda g: g.urgency, reverse=True)
        self._log(f"Generated {len(goals)} goals:")
        for i, g in enumerate(goals):
            self._log(f"  #{i+1} [{g.category}] urgency={g.urgency:.2f} "
                      f"effort={g.effort:.1f}h — {g.description[:55]}")

        # 4. ACT on top goal
        top = goals[0]
        self._log(f"\nExecuting: {top.description}")
        self._log(f"Reason: {top.reasoning[:80]}")

        success, result = self._executor.execute(top)
        top.executed = True
        top.outcome  = result
        top.success  = success

        status = "✓" if success else "✗"
        self._log(f"{status} Result: {result[:100]}")

        # 5. LEARN
        self._learner.record(top, success, result)
        self._goal_history.append(top.description)
        if len(self._goal_history) > 20:
            self._goal_history = self._goal_history[-20:]

        if success:
            self._consecutive_fails = 0
        else:
            self._consecutive_fails += 1

        # Notify
        try:
            from janus_comms import get_comms
            get_comms().post_update(
                "autonomous_cycle",
                f"Cycle {self._cycle_count}: {status} {top.description[:60]}",
                {"success": success, "result": result[:100]},
            )
        except Exception:
            pass

        return top

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Janus] {msg}")
        logger.info(msg)

    def _handle_signal(self, sig, frame):
        self._log(f"\nSignal {sig} received — stopping after current cycle...")
        self._running = False

    def status(self) -> dict:
        return {
            "running":            self._running,
            "directive":          self.directive,
            "cycles_completed":   self._cycle_count,
            "consecutive_fails":  self._consecutive_fails,
            "recent_goals":       self._goal_history[-5:],
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_loop: Optional[JanusAutonomousLoop] = None

def get_loop(directive: str = "generate income and build capability") -> JanusAutonomousLoop:
    global _loop
    if _loop is None:
        _loop = JanusAutonomousLoop(directive=directive)
    return _loop


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Autonomous Loop")
    parser.add_argument("--directive", type=str,
                        default="generate income and build capability",
                        help="High-level directive for Janus")
    parser.add_argument("--once",     action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--cycle",    type=int, default=120,
                        help="Seconds between cycles (default 120)")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    loop = JanusAutonomousLoop(
        directive     = args.directive,
        cycle_seconds = args.cycle,
        verbose       = not args.quiet,
    )

    if args.once:
        goal = loop.run_once()
        if goal:
            print(f"\nExecuted: {goal.description}")
            print(f"Success:  {goal.success}")
            print(f"Result:   {goal.outcome}")
    else:
        print(f"Starting Janus autonomous loop.")
        print(f"Directive: '{args.directive}'")
        print(f"Press Ctrl+C to stop.\n")
        loop.run()
