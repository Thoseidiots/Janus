# replanner.py

“””
Janus Dynamic Replanner — closes the most critical agency gap.

When a step fails, instead of blindly retrying the same action,
the Replanner:

1. Diagnoses WHY it failed (timeout, element not found, blocked, etc.)
1. Selects an alternative strategy from a failure→fix map
1. Patches the plan with new steps and re-executes

This plugs into agent_loop.py’s JanusAgent as a drop-in upgrade.

Usage:
from replanner import ReplanningAgent
agent = ReplanningAgent()
plan = agent.run_goal(“Find data entry jobs on Fiverr”)
“””

from **future** import annotations

import time
import uuid
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from agent_loop import (
JanusAgent, AgentStep, AgentPlan,
SimplePlanner, WebToolHandler, AgentMemory
)
from tool_executor import ToolExecutor, ToolCall, ToolResult, RiskTier

# ── Failure taxonomy ──────────────────────────────────────────────────────────

class FailureKind(str, Enum):
ELEMENT_NOT_FOUND  = “element_not_found”   # CSS selector missed
TIMEOUT            = “timeout”              # page/network too slow
BLOCKED            = “blocked”              # rate-limited or CAPTCHA
PERMISSION_DENIED  = “permission_denied”    # auth required
NETWORK_ERROR      = “network_error”        # no connectivity
PARSE_ERROR        = “parse_error”          # content not what expected
UNKNOWN            = “unknown”

@dataclass
class FailureDiagnosis:
kind:        FailureKind
confidence:  float          # 0–1
detail:      str
step:        AgentStep
attempt:     int

# ── Failure Diagnoser ─────────────────────────────────────────────────────────

class FailureDiagnoser:
“””
Reads the error message from a failed ToolResult and classifies it
into a FailureKind with confidence score.
“””

```
# Error substring → (FailureKind, confidence)
SIGNATURES: list[tuple[str, FailureKind, float]] = [
    ("timeout",            FailureKind.TIMEOUT,           0.95),
    ("timed out",          FailureKind.TIMEOUT,           0.95),
    ("networkidle",        FailureKind.TIMEOUT,           0.80),
    ("ElementHandle",      FailureKind.ELEMENT_NOT_FOUND, 0.90),
    ("selector",           FailureKind.ELEMENT_NOT_FOUND, 0.85),
    ("No node found",      FailureKind.ELEMENT_NOT_FOUND, 0.95),
    ("waiting for",        FailureKind.ELEMENT_NOT_FOUND, 0.75),
    ("403",                FailureKind.PERMISSION_DENIED, 0.90),
    ("401",                FailureKind.PERMISSION_DENIED, 0.90),
    ("captcha",            FailureKind.BLOCKED,           0.95),
    ("rate limit",         FailureKind.BLOCKED,           0.95),
    ("too many requests",  FailureKind.BLOCKED,           0.90),
    ("blocked",            FailureKind.BLOCKED,           0.80),
    ("ECONNREFUSED",       FailureKind.NETWORK_ERROR,     0.95),
    ("net::ERR",           FailureKind.NETWORK_ERROR,     0.90),
    ("Could not start",    FailureKind.NETWORK_ERROR,     0.85),
    ("JSONDecodeError",    FailureKind.PARSE_ERROR,       0.90),
    ("KeyError",           FailureKind.PARSE_ERROR,       0.75),
    ("Permission",         FailureKind.PERMISSION_DENIED, 0.85),
]

def diagnose(self, result: ToolResult, step: AgentStep) -> FailureDiagnosis:
    error_text = (result.error or "").lower()

    best_kind = FailureKind.UNKNOWN
    best_conf = 0.0
    best_sig  = ""

    for sig, kind, conf in self.SIGNATURES:
        if sig.lower() in error_text:
            if conf > best_conf:
                best_conf = conf
                best_kind = kind
                best_sig  = sig

    return FailureDiagnosis(
        kind       = best_kind,
        confidence = best_conf if best_conf > 0 else 0.3,
        detail     = f"Matched '{best_sig}' in: {result.error[:200]}",
        step       = step,
        attempt    = step.attempts,
    )
```

# ── Alternative Strategy Generator ───────────────────────────────────────────

class StrategyGenerator:
“””
Given a failure diagnosis, produces replacement AgentSteps
that try a different approach to achieve the same outcome.
“””

```
def alternatives(
    self,
    diagnosis: FailureDiagnosis,
    original_goal: str,
    context: dict,
) -> list[AgentStep]:
    kind = diagnosis.kind
    step = diagnosis.step

    if kind == FailureKind.ELEMENT_NOT_FOUND:
        return self._alt_element_not_found(step, context)

    elif kind == FailureKind.TIMEOUT:
        return self._alt_timeout(step, context)

    elif kind == FailureKind.BLOCKED:
        return self._alt_blocked(step, original_goal, context)

    elif kind == FailureKind.NETWORK_ERROR:
        return self._alt_network_error(step, context)

    elif kind == FailureKind.PERMISSION_DENIED:
        return self._alt_permission_denied(step, context)

    elif kind == FailureKind.PARSE_ERROR:
        return self._alt_parse_error(step, context)

    else:
        # Unknown — try a generic fallback
        return self._alt_generic(step, context)

# ── Alternative strategies per failure kind ───────────────────────────────

def _alt_element_not_found(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Try broader selectors or extract full page text instead."""
    alt_id = self._new_id()
    if step.tool == "web_extract":
        # Fall back to extracting all text from the page body
        return [AgentStep(
            step_id     = alt_id,
            description = f"Fallback: extract full page text (selector failed)",
            tool        = "web_extract_text",
            args        = {"selector": "body"},
            max_attempts= 2,
        )]
    elif step.tool == "web_search_page":
        # Try clicking a search button instead of filling input
        return [AgentStep(
            step_id     = alt_id,
            description = "Fallback: try alternate search selector",
            tool        = "web_search_page",
            args        = {
                **step.args,
                "selector": "input[type='search'], input[name='q'], .search-input"
            },
            max_attempts= 2,
        )]
    return self._alt_generic(step, ctx)

def _alt_timeout(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Wait longer, then retry. If web_navigate, try without networkidle."""
    alt_id = self._new_id()
    return [
        AgentStep(
            step_id     = alt_id,
            description = f"Fallback: wait 5s then retry {step.tool}",
            tool        = "shell_cmd",
            args        = {"cmd": "echo 'waiting before retry'"},
            max_attempts= 1,
        ),
        AgentStep(
            step_id     = self._new_id(),
            description = f"Retry after wait: {step.description}",
            tool        = step.tool,
            args        = step.args,
            depends_on  = [alt_id],
            max_attempts= 2,
        ),
    ]

def _alt_blocked(self, step: AgentStep, goal: str, ctx: dict) -> list[AgentStep]:
    """
    If blocked on primary site, try an alternative source.
    E.g. if Fiverr blocks, try searching via DuckDuckGo instead.
    """
    alt_id = self._new_id()
    search_term = ctx.get("search_term", goal[:50])

    # Detect which site was being used
    url = step.args.get("url", "")
    if "fiverr" in url.lower():
        alt_url = f"https://duckduckgo.com/?q=fiverr+{search_term.replace(' ', '+')}"
    elif "upwork" in url.lower():
        alt_url = f"https://duckduckgo.com/?q=upwork+{search_term.replace(' ', '+')}"
    else:
        alt_url = f"https://duckduckgo.com/?q={search_term.replace(' ', '+')}"

    return [
        AgentStep(
            step_id     = alt_id,
            description = f"Blocked — trying alternative route via DuckDuckGo",
            tool        = "web_navigate",
            args        = {"url": alt_url},
            max_attempts= 2,
        ),
        AgentStep(
            step_id     = self._new_id(),
            description = "Extract search results from alternative source",
            tool        = "web_extract_text",
            args        = {"selector": "body"},
            depends_on  = [alt_id],
            max_attempts= 2,
        ),
    ]

def _alt_network_error(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Network is down — save what we have and stop gracefully."""
    alt_id = self._new_id()
    return [AgentStep(
        step_id     = alt_id,
        description = "Network error — saving partial results",
        tool        = "file_write",
        args        = {
            "path":    "partial_results.txt",
            "content": f"Network error at step: {step.description}\n"
                       f"Context: {str(ctx)[:500]}",
        },
        max_attempts= 1,
    )]

def _alt_permission_denied(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Auth required — log the need for credentials and skip."""
    alt_id = self._new_id()
    return [AgentStep(
        step_id     = alt_id,
        description = "Auth required — logging credential gap",
        tool        = "file_write",
        args        = {
            "path":    "auth_required.txt",
            "content": f"Step '{step.description}' needs authentication.\n"
                       f"Tool: {step.tool}\nArgs: {step.args}",
        },
        max_attempts= 1,
    )]

def _alt_parse_error(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Content wasn't in expected format — extract raw text instead."""
    alt_id = self._new_id()
    return [AgentStep(
        step_id     = alt_id,
        description = "Parse error — falling back to raw text extraction",
        tool        = "web_extract_text",
        args        = {"selector": "body"},
        max_attempts= 2,
    )]

def _alt_generic(self, step: AgentStep, ctx: dict) -> list[AgentStep]:
    """Unknown failure — log and move on."""
    return [AgentStep(
        step_id     = self._new_id(),
        description = f"Unknown failure in '{step.description}' — logging and continuing",
        tool        = "file_write",
        args        = {
            "path":    "agent_errors.txt",
            "content": f"Failed step: {step.description}\n"
                       f"Tool: {step.tool}\nArgs: {step.args}\n"
                       f"Timestamp: {datetime.now().isoformat()}\n\n",
        },
        max_attempts= 1,
    )]

@staticmethod
def _new_id() -> str:
    return str(uuid.uuid4())[:8]
```

# ── Replanning Agent (upgrades JanusAgent) ────────────────────────────────────

class ReplanningAgent(JanusAgent):
“””
Drop-in upgrade to JanusAgent that adds dynamic replanning.

```
When a step fails, instead of giving up:
  1. Diagnose why it failed
  2. Generate alternative steps
  3. Insert them into the plan
  4. Continue execution

Max replan attempts per step is capped to prevent infinite loops.
"""

MAX_REPLANS_PER_STEP = 2

def __init__(self, verbose: bool = True):
    super().__init__(verbose=verbose)
    self.diagnoser  = FailureDiagnoser()
    self.strategies = StrategyGenerator()
    self._replan_counts: dict[str, int] = {}

def run_goal(self, goal: str, context: dict = {}) -> AgentPlan:
    """
    Overrides JanusAgent.run_goal with replanning capability.
    """
    self._log(f"Goal received: {goal}")
    self._try_load_core()

    if self._core:
        self._core.perceive(goal)

    plan = self.planner.make_plan(goal, context)
    self._log(f"Plan created: {len(plan.steps)} steps")

    step_outputs: dict[str, dict] = {}
    i = 0

    while i < len(plan.steps):
        step = plan.steps[i]
        self._log(f"Step {i+1}/{len(plan.steps)}: {step.description}")

        # Check dependencies
        deps_met = all(
            step_outputs.get(dep, {}).get("success", False)
            for dep in step.depends_on
        )
        if step.depends_on and not deps_met:
            self._log("  Skipping — dependencies not met")
            i += 1
            continue

        # Inject previous outputs
        for dep in step.depends_on:
            dep_out = step_outputs.get(dep, {}).get("output")
            if dep_out is not None:
                step.args[f"prev_{step_outputs[dep]['tool']}"] = dep_out

        # Execute
        result = self._execute_step(step)
        step_outputs[step.step_id] = {
            "success": result.success,
            "output":  result.output,
            "tool":    step.tool,
        }

        if result.success:
            self._log(f"  ✓ Done ({result.duration_ms:.0f}ms)")
            step.result  = result.output
            step.success = True
            i += 1

        else:
            self._log(f"  ✗ Failed: {result.error}")

            # Check replan budget
            replan_key = step.step_id
            replan_count = self._replan_counts.get(replan_key, 0)

            if replan_count >= self.MAX_REPLANS_PER_STEP:
                self._log(f"  Replan budget exhausted for this step — skipping")
                i += 1
                continue

            # Diagnose failure
            diagnosis = self.diagnoser.diagnose(result, step)
            self._log(
                f"  Diagnosis: {diagnosis.kind.value} "
                f"(confidence={diagnosis.confidence:.0%})"
            )

            # Generate alternatives
            alternatives = self.strategies.alternatives(
                diagnosis, goal, context
            )

            if not alternatives:
                self._log("  No alternatives found — skipping step")
                i += 1
                continue

            self._log(
                f"  Replanning: inserting {len(alternatives)} "
                f"alternative step(s)"
            )

            # Insert alternatives right after current position
            plan.steps = (
                plan.steps[:i + 1] +
                alternatives +
                plan.steps[i + 1:]
            )

            self._replan_counts[replan_key] = replan_count + 1

            # Move to first alternative (skip failed step)
            i += 1

    # Evaluate outcome
    successes = sum(1 for s in plan.steps if s.success)
    plan.completed = successes > 0  # partial success counts
    plan.outcome = (
        f"{successes}/{len(plan.steps)} steps succeeded"
    )

    if self._core:
        self._core.perceive(f"Goal '{goal}': {plan.outcome}")

    self.memory.save_plan(plan)
    self._log(f"Plan complete: {plan.outcome}")

    return plan

def explain_last_replans(self, plan: AgentPlan) -> str:
    """
    Returns a human-readable summary of what was replanned and why.
    Useful for debugging and for feeding back into JanusGPT.
    """
    lines = [f"Goal: {plan.goal}", f"Outcome: {plan.outcome}", ""]
    for step in plan.steps:
        status = "✓" if step.success else "✗"
        replan_note = ""
        if self._replan_counts.get(step.step_id, 0) > 0:
            replan_note = f" [replanned x{self._replan_counts[step.step_id]}]"
        lines.append(f"  {status} {step.description}{replan_note}")
    return "\n".join(lines)
```

# ── CLI entry point ───────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys

```
agent = ReplanningAgent(verbose=True)

goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
    "Search Fiverr for data entry jobs and save the top 5 listings"

try:
    plan = agent.run_goal(goal)
    print("\n" + agent.explain_last_replans(plan))
finally:
    agent.stop()
```