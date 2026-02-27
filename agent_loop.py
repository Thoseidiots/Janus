# agent_loop.py

“””
Janus Agent Loop — the missing piece that ties everything together.

Takes a high-level goal (e.g. “find a data entry job on Fiverr and apply”)
and executes it end-to-end using:

- goal_planner.py   → breaks goal into steps
- tool_executor.py  → executes each step safely
- web_autonomy.py   → browser actions
- memory_manager.py → remembers what worked
- core.py           → valence/mood awareness

Usage:
agent = JanusAgent()
agent.run_goal(“Search Fiverr for data entry jobs and collect the top 5 listings”)
“””

from **future** import annotations

import json
import time
import uuid
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tool_executor import ToolExecutor, ToolCall, RiskTier

# ── Step & Plan dataclasses ───────────────────────────────────────────────────

@dataclass
class AgentStep:
“”“A single action step in a plan.”””
step_id:     str
description: str
tool:        str
args:        dict
depends_on:  list[str] = field(default_factory=list)
result:      Any       = None
success:     bool      = False
attempts:    int       = 0
max_attempts: int      = 3

@dataclass
class AgentPlan:
“”“A sequence of steps to achieve a goal.”””
plan_id:   str
goal:      str
steps:     list[AgentStep]
created:   str = field(default_factory=lambda: datetime.now().isoformat())
completed: bool = False
outcome:   str  = “”

# ── Planner ───────────────────────────────────────────────────────────────────

class SimplePlanner:
“””
Converts a goal string into a sequence of AgentSteps.
Uses keyword matching to select the right plan template.
When JanusGPT is available, it can generate plans dynamically.
“””

```
PLAN_TEMPLATES = {
    "fiverr": [
        AgentStep("1", "Navigate to Fiverr", "web_navigate",
                  {"url": "https://www.fiverr.com"}),
        AgentStep("2", "Search for jobs", "web_search_page",
                  {"selector": "input[placeholder*='search']",
                   "query": "{search_term}"},
                  depends_on=["1"]),
        AgentStep("3", "Extract listings", "web_extract",
                  {"selector": ".gig-card-layout", "limit": 5},
                  depends_on=["2"]),
        AgentStep("4", "Save results to file", "file_write",
                  {"path": "fiverr_results.json", "content": "{extracted}"},
                  depends_on=["3"]),
    ],
    "web_research": [
        AgentStep("1", "Navigate to target", "web_navigate",
                  {"url": "{url}"}),
        AgentStep("2", "Extract page text", "web_extract_text",
                  {"selector": "body"}, depends_on=["1"]),
        AgentStep("3", "Save research", "file_write",
                  {"path": "research_{timestamp}.txt",
                   "content": "{extracted_text}"},
                  depends_on=["2"]),
    ],
    "file_task": [
        AgentStep("1", "Read input file", "file_read",
                  {"path": "{input_path}"}),
        AgentStep("2", "Process and write output", "code_exec",
                  {"code": "result = args['prev_file_read'][:1000]"},
                  depends_on=["1"]),
        AgentStep("3", "Save output", "file_write",
                  {"path": "{output_path}", "content": "{processed}"},
                  depends_on=["2"]),
    ],
    "default": [
        AgentStep("1", "Execute goal directly", "shell_cmd",
                  {"cmd": "echo 'Goal: {goal}'"}),
    ],
}

def make_plan(self, goal: str, context: dict = {}) -> AgentPlan:
    goal_lower = goal.lower()

    # Select template
    if any(w in goal_lower for w in ["fiverr", "gig", "freelance", "job"]):
        template_key = "fiverr"
        context.setdefault("search_term", self._extract_search_term(goal))
    elif any(w in goal_lower for w in ["research", "search", "find", "browse", "web"]):
        template_key = "web_research"
        context.setdefault("url", self._extract_url(goal))
    elif any(w in goal_lower for w in ["file", "read", "write", "process", "organize"]):
        template_key = "file_task"
    else:
        template_key = "default"

    # Clone template and interpolate args
    import copy
    steps = copy.deepcopy(self.PLAN_TEMPLATES[template_key])
    context["goal"] = goal
    context["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    for step in steps:
        step.step_id = str(uuid.uuid4())[:8]
        step.args = self._interpolate(step.args, context)

    return AgentPlan(
        plan_id = str(uuid.uuid4())[:8],
        goal    = goal,
        steps   = steps,
    )

def _interpolate(self, args: dict, context: dict) -> dict:
    result = {}
    for k, v in args.items():
        if isinstance(v, str):
            for ck, cv in context.items():
                v = v.replace(f"{{{ck}}}", str(cv))
        result[k] = v
    return result

def _extract_search_term(self, goal: str) -> str:
    keywords = ["data entry", "content writing", "research", "translation",
                "virtual assistant", "social media", "design"]
    for kw in keywords:
        if kw in goal.lower():
            return kw
    return "data entry"

def _extract_url(self, goal: str) -> str:
    import re
    urls = re.findall(r'https?://\S+', goal)
    return urls[0] if urls else "https://www.google.com"
```

# ── Web Tool Extensions ───────────────────────────────────────────────────────

class WebToolHandler:
“””
Extends ToolExecutor with web-specific tools using JanusWebAutonomy.
Lazy-loads Playwright only when actually needed.
“””

```
def __init__(self):
    self._web = None

def _ensure_web(self):
    if self._web is None:
        try:
            from web_autonomy import JanusWebAutonomy
            self._web = JanusWebAutonomy(headless=True)
            self._web.start()
        except Exception as e:
            raise RuntimeError(f"Could not start web autonomy: {e}")
    return self._web

def handle(self, tool: str, args: dict) -> Any:
    if tool == "web_navigate":
        web = self._ensure_web()
        content = web.navigate(args["url"])
        return {"url": args["url"], "content_length": len(content)}

    elif tool == "web_search_page":
        web = self._ensure_web()
        try:
            web.fill(args["selector"], args["query"])
            web.page.keyboard.press("Enter")
            web.page.wait_for_load_state("networkidle")
            return {"searched": args["query"]}
        except Exception as e:
            return {"error": str(e)}

    elif tool == "web_extract":
        web = self._ensure_web()
        try:
            elements = web.page.query_selector_all(args["selector"])
            limit = int(args.get("limit", 10))
            results = []
            for el in elements[:limit]:
                results.append(el.inner_text()[:500])
            return results
        except Exception as e:
            return {"error": str(e)}

    elif tool == "web_extract_text":
        web = self._ensure_web()
        try:
            text = web.page.inner_text(args.get("selector", "body"))
            return text[:5000]
        except Exception as e:
            return {"error": str(e)}

    elif tool == "web_screenshot":
        web = self._ensure_web()
        path = args.get("path", f"screenshot_{int(time.time())}.png")
        web.screenshot(path)
        return {"saved": path}

    return None

def stop(self):
    if self._web:
        try:
            self._web.stop()
        except Exception:
            pass
        self._web = None
```

# ── Agent Memory ──────────────────────────────────────────────────────────────

class AgentMemory:
“”“Persists plan history and learned patterns.”””

```
def __init__(self, path: str = "agent_memory.jsonl"):
    self.path = Path(path)

def save_plan(self, plan: AgentPlan):
    record = {
        "plan_id":   plan.plan_id,
        "goal":      plan.goal,
        "completed": plan.completed,
        "outcome":   plan.outcome,
        "steps":     len(plan.steps),
        "timestamp": datetime.now().isoformat(),
    }
    with self.path.open("a") as f:
        f.write(json.dumps(record) + "\n")

def recent_plans(self, n: int = 10) -> list[dict]:
    if not self.path.exists():
        return []
    lines = self.path.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]

def success_rate(self) -> float:
    plans = self.recent_plans(50)
    if not plans:
        return 0.0
    return sum(1 for p in plans if p["completed"]) / len(plans)
```

# ── Main Agent Loop ───────────────────────────────────────────────────────────

class JanusAgent:
“””
The top-level agent that takes a goal and executes it.

```
This is the missing piece — the loop that ties:
  goal_planner → tool_executor → web_autonomy → memory → valence
into a single coherent system.
"""

def __init__(self, verbose: bool = True):
    self.executor   = ToolExecutor()
    self.executor.register_defaults()
    self.planner    = SimplePlanner()
    self.web_tools  = WebToolHandler()
    self.memory     = AgentMemory()
    self.verbose    = verbose
    self._core      = None  # lazy load AutonomousCore

    # Register web tools into executor
    self._register_web_tools()

def _register_web_tools(self):
    from tool_executor import ToolSpec
    web_tool_names = [
        "web_navigate", "web_search_page", "web_extract",
        "web_extract_text", "web_screenshot"
    ]
    for name in web_tool_names:
        self.executor.registry.register(ToolSpec(
            name=name,
            description=f"Web tool: {name}",
            risk=RiskTier.HIGH,
            parameters={},
            handler=lambda args, n=name: self.web_tools.handle(n, args),
        ))

def _log(self, msg: str):
    if self.verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[Janus {ts}] {msg}")

def _try_load_core(self):
    """Optionally load AutonomousCore for valence-aware planning."""
    if self._core is not None:
        return
    try:
        from core import AutonomousCore
        self._core = AutonomousCore()
        self._log("AutonomousCore loaded — valence-aware mode active.")
    except Exception:
        self._log("Running without AutonomousCore (no valence).")

def run_goal(self, goal: str, context: dict = {}) -> AgentPlan:
    """
    Main entry point. Takes a goal string, plans, executes, returns results.
    """
    self._log(f"Goal received: {goal}")
    self._try_load_core()

    # Perceive goal through homeostasis if core available
    if self._core:
        self._core.perceive(goal)

    # Plan
    plan = self.planner.make_plan(goal, context)
    self._log(f"Plan created: {len(plan.steps)} steps")

    # Execute steps
    step_outputs = {}
    for step in plan.steps:
        self._log(f"Step {step.step_id}: {step.description}")

        # Check dependencies
        deps_met = all(
            step_outputs.get(dep, {}).get("success", False)
            for dep in step.depends_on
        )
        if step.depends_on and not deps_met:
            self._log(f"  Skipping — dependencies not met")
            continue

        # Inject outputs from previous steps
        for dep in step.depends_on:
            dep_output = step_outputs.get(dep, {}).get("output")
            if dep_output is not None:
                step.args[f"prev_{step_outputs[dep]['tool']}"] = dep_output

        # Execute with retry
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
        else:
            self._log(f"  ✗ Failed: {result.error}")

    # Evaluate outcome
    successes = sum(1 for s in plan.steps if s.success)
    plan.completed = successes == len(plan.steps)
    plan.outcome = (
        f"{successes}/{len(plan.steps)} steps completed successfully"
    )

    self._log(f"Plan complete: {plan.outcome}")

    # Update valence based on outcome
    if self._core:
        feedback = f"Goal '{goal}' result: {plan.outcome}"
        self._core.perceive(feedback)

    # Save to memory
    self.memory.save_plan(plan)

    return plan

def _execute_step(self, step: AgentStep) -> Any:
    """Execute a single step with retry logic."""
    last_result = None
    while step.attempts < step.max_attempts:
        step.attempts += 1
        call = ToolCall(
            call_id = f"{step.step_id}_{step.attempts}",
            tool    = step.tool,
            args    = step.args,
        )
        result = self.executor.execute(call)
        last_result = result
        if result.success:
            return result
        self._log(f"  Retry {step.attempts}/{step.max_attempts}...")
        time.sleep(1)
    return last_result

def run_continuous(self, goals: list[str], delay: float = 5.0):
    """
    Run a list of goals sequentially with a delay between each.
    This is the autonomous mode — Janus keeps working through its goal list.
    """
    self._log(f"Continuous mode: {len(goals)} goals queued")
    for i, goal in enumerate(goals):
        self._log(f"\n── Goal {i+1}/{len(goals)} ──")
        try:
            self.run_goal(goal)
        except Exception as e:
            self._log(f"Goal failed with exception: {e}")
            traceback.print_exc()
        if i < len(goals) - 1:
            time.sleep(delay)

    success_rate = self.memory.success_rate()
    self._log(f"\nAll goals processed. Historical success rate: {success_rate:.0%}")

def status(self) -> dict:
    """Return current agent status."""
    return {
        "recent_plans":   self.memory.recent_plans(5),
        "success_rate":   self.memory.success_rate(),
        "valence":        self._core.valence_context_string() if self._core else "no core",
    }

def stop(self):
    self.web_tools.stop()
    self._log("Agent stopped.")
```

# ── CLI entry point ───────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys

```
agent = JanusAgent(verbose=True)

if len(sys.argv) > 1:
    goal = " ".join(sys.argv[1:])
else:
    goal = "Search Fiverr for data entry jobs and save the top 5 listings"

try:
    plan = agent.run_goal(goal)
    print(f"\nResult: {plan.outcome}")
    for step in plan.steps:
        status = "✓" if step.success else "✗"
        print(f"  {status} {step.description}")
        if step.result:
            preview = str(step.result)[:200]
            print(f"    → {preview}")
finally:
    agent.stop()
```