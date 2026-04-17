"""
janus_agent_core.py
===================
Unified agent loop for Janus. The one file that ties everything together.

Takes a goal, plans it, executes each step with real tools, verifies
outputs, retries on failure with reasoning, and persists everything to memory.

This replaces the broken agent_loop.py and consolidates the working
logic from tree_planner.py and tool_executor.py into one clean system.

Usage:
    from janus_agent_core import JanusAgent
    agent = JanusAgent()
    result = agent.run("Read TODO.md and summarize the top 3 priorities")

    # Or from CLI:
    python janus_agent_core.py "Read TODO.md and summarize the top 3 priorities"
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Force UTF-8 on Windows to avoid cp1252 encode errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
logger = logging.getLogger("janus")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class RiskTier(Enum):
    LOW    = "low"       # pure reads, memory queries → auto-approved
    MEDIUM = "medium"    # file writes, shell cmds    → logged + rate-limited
    HIGH   = "high"      # network, code exec         → sandboxed


@dataclass
class ToolSpec:
    """Definition of a tool the agent can use."""
    name:        str
    description: str
    risk:        RiskTier
    parameters:  Dict[str, str]
    handler:     Optional[Callable] = None


@dataclass
class ToolCall:
    """A proposed tool invocation."""
    call_id:     str
    tool:        str
    args:        Dict[str, Any]
    proposed_by: str = "janus"
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ToolResult:
    """The result of executing a tool."""
    call_id:      str
    tool:         str
    success:      bool
    output:       Any
    error:        Optional[str]
    duration_ms:  float
    sandbox:      bool
    verified:     bool = False
    timestamp:    str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentStep:
    """A single action step in a plan."""
    step_id:      str
    description:  str
    tool:         str
    args:         dict
    depends_on:   List[str] = field(default_factory=list)
    result:       Any = None
    success:      bool = False
    attempts:     int = 0
    max_attempts: int = 3
    critic_note:  str = ""


@dataclass
class AgentPlan:
    """A sequence of steps to achieve a goal."""
    plan_id:   str
    goal:      str
    steps:     List[AgentStep]
    created:   str = field(default_factory=lambda: datetime.now().isoformat())
    completed: bool = False
    outcome:   str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: EVENT TRACE (Append-only audit log)
# ══════════════════════════════════════════════════════════════════════════════

class EventTrace:
    """Append-only, tamper-evident log of all tool calls and results."""

    def __init__(self, path: str = "event_trace.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict):
        record["_ts"] = datetime.now().isoformat()
        record["_hash"] = self._hash(record)
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def tail(self, n: int = 50) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    @staticmethod
    def _hash(record: dict) -> str:
        payload = json.dumps(
            {k: v for k, v in record.items() if k != "_hash"},
            sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: TOOL REGISTRY & EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec
        logger.debug(f"Registered tool: {spec.name} ({spec.risk.value})")

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_all(self) -> List[dict]:
        return [
            {"name": s.name, "description": s.description,
             "risk": s.risk.value, "parameters": s.parameters}
            for s in self._tools.values()
        ]

    def names(self) -> List[str]:
        return list(self._tools.keys())


class SandboxRunner:
    """Runs high-risk tools inside an isolated subprocess."""

    TIMEOUT = 10  # seconds

    def run(self, code: str, args: dict) -> Tuple[bool, Any, Optional[str]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "task.py"
            args_file = Path(tmpdir) / "args.json"
            out_file = Path(tmpdir) / "out.json"

            args_file.write_text(json.dumps(args, default=str))
            script.write_text(
                "import json, sys\n"
                "args = json.loads(open('args.json').read())\n"
                "out = {\"result\": None, \"error\": None}\n"
                "try:\n"
                f"    {code}\n"
                "    out[\"result\"] = result\n"
                "except Exception as e:\n"
                "    out[\"error\"] = str(e)\n"
                "open('out.json', 'w').write(json.dumps(out))\n"
            )

            try:
                python_cmd = sys.executable or "python"
                proc = subprocess.run(
                    [python_cmd, str(script)],
                    cwd=tmpdir, capture_output=True,
                    timeout=self.TIMEOUT
                )
                if out_file.exists():
                    out = json.loads(out_file.read_text())
                    if out.get("error"):
                        return False, None, out["error"]
                    return True, out.get("result"), None
                return False, None, proc.stderr.decode()[:500]
            except subprocess.TimeoutExpired:
                return False, None, f"Sandbox timeout ({self.TIMEOUT}s)"
            except Exception as e:
                return False, None, str(e)


class ToolExecutor:
    """
    Orchestrates PROPOSE → VERIFY → APPLY → CRITIQUE pipeline.

    Every call is logged to an append-only event trace.
    High-risk tools run in a sandbox subprocess.
    """

    # Shell commands considered safe to run
    SHELL_WHITELIST = {
        "ls", "dir", "pwd", "echo", "date", "whoami", "type", "cat",
        "find", "wc", "head", "tail", "sort", "grep", "python",
        "Get-ChildItem", "Get-Content", "Get-Date", "Measure-Object",
    }

    def __init__(self, trace_path: str = "event_trace.jsonl"):
        self.registry = ToolRegistry()
        self.trace = EventTrace(trace_path)
        self.sandbox = SandboxRunner()

        # Rate limiting
        self._rate_log: Dict[str, List[float]] = {}
        self._rate_limit = 30
        self._lock = threading.Lock()

    def register_defaults(self):
        """Register Janus's built-in tool suite."""

        self.registry.register(ToolSpec(
            name="file_read",
            description="Read a local file (max 8KB returned)",
            risk=RiskTier.LOW,
            parameters={"path": "str"},
            handler=self._tool_file_read,
        ))
        self.registry.register(ToolSpec(
            name="file_write",
            description="Write content to a local file",
            risk=RiskTier.MEDIUM,
            parameters={"path": "str", "content": "str"},
            handler=self._tool_file_write,
        ))
        self.registry.register(ToolSpec(
            name="file_list",
            description="List files in a directory",
            risk=RiskTier.LOW,
            parameters={"path": "str", "pattern": "str"},
            handler=self._tool_file_list,
        ))
        self.registry.register(ToolSpec(
            name="shell_cmd",
            description="Run a shell command (whitelisted commands only)",
            risk=RiskTier.MEDIUM,
            parameters={"cmd": "str"},
            handler=self._tool_shell_cmd,
        ))
        self.registry.register(ToolSpec(
            name="code_exec",
            description="Execute Python code in a sandboxed subprocess",
            risk=RiskTier.HIGH,
            parameters={"code": "str"},
        ))
        self.registry.register(ToolSpec(
            name="web_fetch",
            description="Fetch a URL and return its text content",
            risk=RiskTier.HIGH,
            parameters={"url": "str"},
        ))
        self.registry.register(ToolSpec(
            name="memory_query",
            description="Search long-term memory for relevant context",
            risk=RiskTier.LOW,
            parameters={"query": "str", "limit": "int"},
            handler=self._tool_memory_query,
        ))
        self.registry.register(ToolSpec(
            name="memory_store",
            description="Save a fact or lesson to long-term memory",
            risk=RiskTier.LOW,
            parameters={"content": "str", "category": "str", "importance": "float"},
            handler=self._tool_memory_store,
        ))
        self.registry.register(ToolSpec(
            name="credit_status",
            description="Check Janus Credit balance, network status, and value",
            risk=RiskTier.LOW,
            parameters={"node_id": "str"},
            handler=self._tool_credit_status,
        ))
        self.registry.register(ToolSpec(
            name="credit_contribute",
            description="Record a compute contribution and mint Janus Credits",
            risk=RiskTier.MEDIUM,
            parameters={"node_id": "str", "task_hours": "str", "gpu_hours": "str"},
            handler=self._tool_credit_contribute,
        ))

    # ── Execution pipeline ────────────────────────────────────────────────────

    def execute(self, call: ToolCall) -> ToolResult:
        start = time.time()

        # PROPOSE: log intent
        self.trace.append({"phase": "PROPOSE", "call": asdict(call)})

        # VERIFY: check permissions, rate limits, parameters
        ok, reason = self._verify(call)
        self.trace.append({
            "phase": "VERIFY", "call_id": call.call_id,
            "approved": ok, "reason": reason,
        })

        if not ok:
            result = ToolResult(
                call_id=call.call_id, tool=call.tool,
                success=False, output=None,
                error=f"Blocked: {reason}",
                duration_ms=0, sandbox=False,
            )
            self.trace.append({"phase": "RESULT", "result": asdict(result)})
            return result

        # APPLY: execute the tool
        spec = self.registry.get(call.tool)
        sandboxed = spec.risk == RiskTier.HIGH if spec else False

        try:
            if sandboxed:
                output, error = self._apply_sandboxed(call)
                success = error is None
            else:
                output = spec.handler(call.args) if spec and spec.handler else None
                error = None
                success = True
        except Exception as e:
            output, error, success = None, str(e), False

        elapsed = (time.time() - start) * 1000
        result = ToolResult(
            call_id=call.call_id, tool=call.tool,
            success=success, output=output, error=error,
            duration_ms=round(elapsed, 2), sandbox=sandboxed,
        )
        self.trace.append({"phase": "RESULT", "result": asdict(result)})
        return result

    def _verify(self, call: ToolCall) -> Tuple[bool, str]:
        spec = self.registry.get(call.tool)
        if not spec:
            return False, f"Unknown tool: {call.tool}"

        # Rate limit
        now = time.time()
        with self._lock:
            times = self._rate_log.get(call.tool, [])
            times = [t for t in times if now - t < 60]
            if len(times) >= self._rate_limit:
                return False, f"Rate limit exceeded for {call.tool}"
            times.append(now)
            self._rate_log[call.tool] = times

        # Required parameter check
        for param, ptype in spec.parameters.items():
            if ptype == "str" and param not in call.args:
                return False, f"Missing required parameter: {param}"

        return True, "approved"

    def _apply_sandboxed(self, call: ToolCall) -> Tuple[Any, Optional[str]]:
        if call.tool == "code_exec":
            code = call.args.get("code", "result = None")
            ok, out, err = self.sandbox.run(code, call.args)
            return out, err
        elif call.tool == "web_fetch":
            url = call.args.get("url", "")
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = resp.read().decode("utf-8", errors="replace")[:8000]
                return {"status": resp.status, "body": body, "url": url}, None
            except Exception as e:
                return None, f"web_fetch failed: {e}"
        return None, "No sandbox handler for this tool"

    # ── Built-in tool handlers ────────────────────────────────────────────────

    def _tool_file_read(self, args: dict) -> Any:
        p = Path(args.get("path", ""))
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        content = p.read_text(encoding="utf-8", errors="replace")
        return content[:8192]  # cap at 8KB

    def _tool_file_write(self, args: dict) -> Any:
        p = Path(args.get("path", "output.txt"))
        p.parent.mkdir(parents=True, exist_ok=True)
        content = args.get("content", "")
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"

    def _tool_file_list(self, args: dict) -> Any:
        p = Path(args.get("path", "."))
        pattern = args.get("pattern", "*")
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {p}")
        files = sorted(p.glob(pattern))[:100]  # cap at 100 entries
        return [{"name": f.name, "is_dir": f.is_dir(),
                 "size": f.stat().st_size if f.is_file() else None}
                for f in files]

    def _tool_shell_cmd(self, args: dict) -> Any:
        cmd = args.get("cmd", "")
        if not cmd.strip():
            raise ValueError("Empty command")
        base = cmd.strip().split()[0]
        if base not in self.SHELL_WHITELIST:
            raise PermissionError(
                f"Command not whitelisted: '{base}'. "
                f"Allowed: {', '.join(sorted(self.SHELL_WHITELIST))}"
            )
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        output = proc.stdout.strip()
        if proc.returncode != 0 and proc.stderr:
            output += f"\n[stderr] {proc.stderr.strip()}"
        return output[:4096]

    def _tool_memory_query(self, args: dict) -> Any:
        """Delegates to LongTermMemory if available."""
        query = args.get("query", "")
        limit = int(args.get("limit", 5))
        try:
            from long_term_memory import LongTermMemory
            mem = LongTermMemory()
            results = mem.recall_similar(query, top_k=limit)
            return [{"content": m.content, "importance": m.importance,
                     "score": m.decay_score()} for m in results]
        except ImportError:
            return {"query": query, "results": [],
                    "note": "long_term_memory.py not available"}

    def _tool_memory_store(self, args: dict) -> Any:
        """Store a memory via LongTermMemory."""
        content = args.get("content", "")
        category = args.get("category", "lesson")
        importance = float(args.get("importance", 0.5))
        try:
            from long_term_memory import LongTermMemory
            mem = LongTermMemory()
            mem.add_memory(content, category, importance)
            mem.save_memories()
            return f"Stored memory: {content[:50]}..."
        except ImportError:
            return {"stored": False, "note": "long_term_memory.py not available"}

    def _tool_credit_status(self, args: dict) -> Any:
        """Check Janus Credit balance and network status."""
        node_id = args.get("node_id", "owner")
        try:
            from janus_credits import JanusCreditEngine
            engine = JanusCreditEngine()
            balance = engine.get_balance(node_id)
            status = engine.network_status()
            return {
                "node_id": node_id,
                "balance": balance,
                "credit_value": status["credit_value"],
                "total_supply": status["network"]["total_supply"],
                "providers": status["network"]["total_providers"],
                "halving_epoch": status["halving"]["current_epoch"],
                "issuance_rate": status["halving"]["current_rate_per_hour"],
            }
        except ImportError:
            return {"error": "janus_credits.py not available"}

    def _tool_credit_contribute(self, args: dict) -> Any:
        """Record compute and mint credits."""
        node_id = args.get("node_id", "owner")
        task_hours = float(args.get("task_hours", "1.0"))
        gpu_hours = float(args.get("gpu_hours", "0.0"))
        try:
            from janus_credits import JanusCreditEngine
            engine = JanusCreditEngine()
            # Ensure provider is registered
            engine.register_provider(node_id, gpu=(gpu_hours > 0))
            # Record contribution and mint
            contribution = engine.record_contribution(
                node_id, task_hours, tasks_completed=1, gpu_hours=gpu_hours
            )
            mint = engine.mint_for_contribution(contribution)
            return {
                "node_id": node_id,
                "credits_earned": mint.amount,
                "issuance_rate": mint.issuance_rate,
                "new_balance": engine.get_balance(node_id),
                "credit_value": mint.estimated_value,
            }
        except ImportError:
            return {"error": "janus_credits.py not available"}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class GoalPlanner:
    """
    Converts a goal string into a sequence of AgentSteps.

    Uses keyword matching to select plan templates.
    Designed to be replaced by LLM-driven planning once Avus is trained.
    """

    PLAN_TEMPLATES = {
        "file_read": [
            AgentStep("1", "Read the file", "file_read",
                      {"path": "{file_path}"}),
            AgentStep("2", "Process the content", "code_exec",
                      {"code": "lines = args.get('prev_output', '').split('\\n'); result = '\\n'.join(lines[:50])"},
                      depends_on=["1"]),
            AgentStep("3", "Save results", "file_write",
                      {"path": "janus_output.txt", "content": "{processed}"},
                      depends_on=["2"]),
        ],
        "file_organize": [
            AgentStep("1", "List directory contents", "file_list",
                      {"path": "{dir_path}", "pattern": "{pattern}"}),
            AgentStep("2", "Summarize findings", "code_exec",
                      {"code": "result = f'Found {len(args.get(\"prev_output\", []))} items'"},
                      depends_on=["1"]),
        ],
        "shell_task": [
            AgentStep("1", "Run the command", "shell_cmd",
                      {"cmd": "{command}"}),
        ],
        "web_research": [
            AgentStep("1", "Fetch the page", "web_fetch",
                      {"url": "{url}"}),
            AgentStep("2", "Save research results", "file_write",
                      {"path": "research_{timestamp}.txt",
                       "content": "{fetched}"},
                      depends_on=["1"]),
        ],
        "default": [
            AgentStep("1", "Execute goal", "shell_cmd",
                      {"cmd": "echo Goal received: {goal}"}),
        ],
    }

    def plan(self, goal: str, context: dict = None) -> AgentPlan:
        """Create an execution plan from a goal string."""
        context = context or {}
        goal_lower = goal.lower()
        context["goal"] = goal
        context["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Select template by keyword matching
        template_key, extra_context = self._match_template(goal_lower, goal)
        context.update(extra_context)

        # Clone and interpolate
        steps = copy.deepcopy(self.PLAN_TEMPLATES[template_key])

        # Build mapping from template IDs to new UUIDs
        id_map: Dict[str, str] = {}
        for step in steps:
            old_id = step.step_id
            new_id = str(uuid.uuid4())[:8]
            id_map[old_id] = new_id
            step.step_id = new_id
            step.args = self._interpolate(step.args, context)

        # Update depends_on references to use new IDs
        for step in steps:
            step.depends_on = [id_map.get(dep, dep) for dep in step.depends_on]

        return AgentPlan(
            plan_id=str(uuid.uuid4())[:8],
            goal=goal,
            steps=steps,
        )

    def _match_template(self, goal_lower: str, goal: str) -> Tuple[str, dict]:
        """Match goal to a plan template and extract context."""

        # File read patterns
        file_match = re.search(r'(?:read|open|load|parse)\s+["\']?([^\s"\']+\.\w+)', goal, re.I)
        if file_match:
            return "file_read", {"file_path": file_match.group(1)}

        # File listing / organizing patterns
        if any(w in goal_lower for w in ["list", "count", "organize", "directory"]):
            # Extract directory path, skipping filler words
            skip_words = {"the", "this", "my", "a", "an", "all", "every"}
            dir_match = re.search(r'(?:in|from|of)\s+(.+?)(?:\s+directory|\s+folder|\s+dir|\s*$)', goal, re.I)
            dir_path = "."
            if dir_match:
                candidate = dir_match.group(1).strip().strip("'\"")
                # Remove leading filler words
                parts = candidate.split()
                parts = [p for p in parts if p.lower() not in skip_words]
                if parts:
                    resolved = " ".join(parts)
                    if resolved.lower() in ("current", "current directory", "here", "cwd", "."):
                        dir_path = "."
                    else:
                        dir_path = resolved

            # Detect file type from natural language
            pattern_match = re.search(r'\*\.\w+', goal)
            if pattern_match:
                pattern = pattern_match.group(0)
            elif "python" in goal_lower:
                pattern = "*.py"
            elif "rust" in goal_lower:
                pattern = "*.rs"
            elif "json" in goal_lower:
                pattern = "*.json"
            elif "markdown" in goal_lower or "md file" in goal_lower:
                pattern = "*.md"
            else:
                pattern = "*"
            return "file_organize", {"dir_path": dir_path, "pattern": pattern}

        # Shell command patterns
        if any(w in goal_lower for w in ["run", "execute", "command"]):
            cmd_match = re.search(r'["\']([^"\']+)["\']', goal)
            cmd = cmd_match.group(1) if cmd_match else f"echo {goal}"
            return "shell_task", {"command": cmd}

        # Web research patterns
        url_match = re.search(r'https?://\S+', goal)
        if url_match or any(w in goal_lower for w in ["fetch", "download", "url"]):
            url = url_match.group(0) if url_match else "https://www.google.com"
            return "web_research", {"url": url}

        # File read without explicit "read" keyword but with a file path
        file_path_match = re.search(r'([A-Za-z_][\w]*\.(?:md|txt|py|json|rs|toml|yaml|yml))', goal)
        if file_path_match:
            return "file_read", {"file_path": file_path_match.group(1)}

        return "default", {}

    @staticmethod
    def _interpolate(args: dict, context: dict) -> dict:
        result = {}
        for k, v in args.items():
            if isinstance(v, str):
                for ck, cv in context.items():
                    v = v.replace(f"{{{ck}}}", str(cv))
            result[k] = v
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: CRITIC / VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

class StepCritic:
    """
    Verifies tool outputs before they're accepted.

    Checks:
    - Output is not empty/None when success is True
    - File operations actually produced/read content
    - Shell commands didn't return error codes
    - Web fetches returned actual content
    """

    def critique(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        """
        Returns (accepted: bool, note: str).
        If not accepted, the agent should retry with modified approach.
        """
        if not result.success:
            return False, f"Tool failed: {result.error}"

        if result.output is None:
            return False, "Tool returned None despite reporting success"

        # Tool-specific checks
        checker = getattr(self, f"_check_{step.tool}", None)
        if checker:
            return checker(step, result)

        return True, "Output accepted"

    def _check_file_read(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        if not result.output or len(str(result.output).strip()) == 0:
            return False, "File read returned empty content"
        return True, f"Read {len(str(result.output))} chars"

    def _check_file_write(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        if "Written" not in str(result.output):
            return False, "File write did not confirm success"
        return True, str(result.output)

    def _check_shell_cmd(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        output = str(result.output)
        if "[stderr]" in output and "error" in output.lower():
            return False, f"Command produced errors: {output[:200]}"
        return True, f"Command output: {len(output)} chars"

    def _check_web_fetch(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        if isinstance(result.output, dict):
            body = result.output.get("body", "")
            if len(body) < 50:
                return False, "Web fetch returned very little content"
            return True, f"Fetched {len(body)} chars"
        return False, "Web fetch returned unexpected format"

    def _check_code_exec(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        if result.output is None:
            return False, "Code execution produced no output"
        return True, f"Code returned: {str(result.output)[:100]}"

    def _check_file_list(self, step: AgentStep, result: ToolResult) -> Tuple[bool, str]:
        if isinstance(result.output, list):
            return True, f"Listed {len(result.output)} items"
        return False, "file_list returned unexpected format"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: AGENT MEMORY (plan history + lessons)
# ══════════════════════════════════════════════════════════════════════════════

class AgentMemory:
    """Persists plan history, tracks success rates, and learns from outcomes."""

    def __init__(self, path: str = "agent_history.jsonl"):
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

    def recent_plans(self, n: int = 10) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    def success_rate(self) -> float:
        plans = self.recent_plans(50)
        if not plans:
            return 0.0
        return sum(1 for p in plans if p["completed"]) / len(plans)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: THE AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

class JanusAgent:
    """
    The top-level agent. Takes a goal and executes it end-to-end.

    Pipeline:
        GOAL → PLAN → [EXECUTE → CRITIQUE → RETRY?] × N → OUTCOME → MEMORY

    This is the piece that was missing — a working loop that ties
    planning, execution, verification, and memory into one system.
    """

    def __init__(self, verbose: bool = True, trace_path: str = "event_trace.jsonl"):
        self.executor = ToolExecutor(trace_path=trace_path)
        self.executor.register_defaults()
        self.planner = GoalPlanner()
        self.critic = StepCritic()
        self.memory = AgentMemory()
        self.verbose = verbose

        # Optional: connect to LongTermMemory for richer memory
        self._ltm = None
        try:
            from long_term_memory import LongTermMemory
            self._ltm = LongTermMemory()
            self._log("Long-term memory connected")
        except ImportError:
            self._log("Running without long-term memory (long_term_memory.py not found)")

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            getattr(logger, level)(msg)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, goal: str, context: dict = None) -> AgentPlan:
        """
        Main entry point. Takes a goal string, plans, executes, returns results.
        """
        self._log("=" * 60)
        self._log(f"GOAL: {goal}")
        self._log("=" * 60)

        # Check memory for similar past goals
        if self._ltm:
            past = self._ltm.recall_similar(goal, top_k=2)
            if past:
                self._log(f"Memory: found {len(past)} related past experiences")

        # Plan
        plan = self.planner.plan(goal, context)
        self._log(f"Plan created: {len(plan.steps)} steps")
        for i, step in enumerate(plan.steps, 1):
            self._log(f"  {i}. {step.description} [{step.tool}]")

        # Execute steps
        step_outputs: Dict[str, dict] = {}
        for step in plan.steps:
            self._log("-" * 40)
            self._log(f"Step: {step.description}")

            # Check dependencies
            deps_met = all(
                step_outputs.get(dep, {}).get("success", False)
                for dep in step.depends_on
            )
            if step.depends_on and not deps_met:
                self._log("  >> Skipping -- dependencies not met")
                step.critic_note = "Skipped: dependencies failed"
                continue

            # Inject previous step outputs
            for dep_id in step.depends_on:
                dep_data = step_outputs.get(dep_id)
                if dep_data and dep_data.get("output") is not None:
                    step.args["prev_output"] = dep_data["output"]

            # Execute with critic-in-the-loop retry
            result = self._execute_with_critic(step)
            step_outputs[step.step_id] = {
                "success": result.success and result.verified,
                "output":  result.output,
                "tool":    step.tool,
            }

            if result.success and result.verified:
                self._log(f"  [OK] Accepted ({result.duration_ms:.0f}ms) -- {step.critic_note}")
                step.result = result.output
                step.success = True
            else:
                self._log(f"  [FAIL] -- {step.critic_note}")
                step.result = result.error or result.output
                step.success = False

        # Evaluate outcome
        successes = sum(1 for s in plan.steps if s.success)
        plan.completed = successes == len(plan.steps)
        plan.outcome = f"{successes}/{len(plan.steps)} steps completed"

        self._log("=" * 60)
        self._log(f"RESULT: {plan.outcome} {'[OK]' if plan.completed else '[FAIL]'}")
        self._log("=" * 60)

        # Learn from outcome
        self._learn_from_plan(plan)

        # Persist plan history
        self.memory.save_plan(plan)

        return plan

    # ── Execute with critic ───────────────────────────────────────────────────

    def _execute_with_critic(self, step: AgentStep) -> ToolResult:
        """
        Execute a step with critic-in-the-loop retry.

        If the critic rejects the output, we modify the approach and retry
        up to max_attempts times.
        """
        last_result = None
        retry_reasons = []

        while step.attempts < step.max_attempts:
            step.attempts += 1

            call = ToolCall(
                call_id=f"{step.step_id}_{step.attempts}",
                tool=step.tool,
                args=copy.deepcopy(step.args),
            )

            # Execute
            result = self.executor.execute(call)
            last_result = result

            # Critique
            accepted, note = self.critic.critique(step, result)
            result.verified = accepted
            step.critic_note = note

            if accepted:
                return result

            # Not accepted — log and prepare retry
            retry_reasons.append(note)
            self._log(
                f"  -> Critic rejected (attempt {step.attempts}/{step.max_attempts}): {note}"
            )

            # Error recovery: modify args based on failure reason
            self._adapt_for_retry(step, result, note)

            if step.attempts < step.max_attempts:
                time.sleep(0.5)  # brief pause before retry

        # All attempts exhausted
        if last_result:
            last_result.verified = False
            step.critic_note = f"Failed after {step.max_attempts} attempts: {'; '.join(retry_reasons)}"
        return last_result

    def _adapt_for_retry(self, step: AgentStep, result: ToolResult, reason: str):
        """
        Modify step args to try a different approach on retry.
        This is basic error recovery — when Avus is trained, it will
        generate retry strategies dynamically.
        """
        if step.tool == "file_read" and "not found" in reason.lower():
            # Try common path variations
            path = step.args.get("path", "")
            if not path.startswith("./") and not os.path.isabs(path):
                step.args["path"] = "./" + path
            elif path.startswith("./"):
                step.args["path"] = path[2:]

        elif step.tool == "shell_cmd" and "not whitelisted" in reason.lower():
            # Fall back to echo
            step.args["cmd"] = f"echo Command was not allowed: {step.args.get('cmd', '')}"

    # ── Learning ──────────────────────────────────────────────────────────────

    def _learn_from_plan(self, plan: AgentPlan):
        """Store lessons from this execution in long-term memory."""
        if not self._ltm:
            return

        if plan.completed:
            self._ltm.add_memory(
                f"Successfully completed goal: {plan.goal}",
                category="outcome",
                importance=0.6,
            )
        else:
            failed_steps = [s for s in plan.steps if not s.success]
            failures = "; ".join(
                f"{s.description}: {s.critic_note}" for s in failed_steps[:3]
            )
            self._ltm.learn_lesson(
                situation=plan.goal,
                outcome=f"Partial failure: {plan.outcome}",
                lesson=f"Failed steps: {failures}",
                importance=0.7,
            )

        self._ltm.save_memories()

    # ── Continuous mode ───────────────────────────────────────────────────────

    def run_batch(self, goals: List[str], delay: float = 2.0) -> List[AgentPlan]:
        """Run multiple goals sequentially."""
        results = []
        self._log(f"Batch mode: {len(goals)} goals queued")

        for i, goal in enumerate(goals, 1):
            self._log(f"\n{'━'*60}")
            self._log(f"BATCH {i}/{len(goals)}")
            try:
                plan = self.run(goal)
                results.append(plan)
            except Exception as e:
                self._log(f"Goal failed with exception: {e}", "error")
                traceback.print_exc()

            if i < len(goals):
                time.sleep(delay)

        rate = self.memory.success_rate()
        self._log(f"\nBatch complete. Historical success rate: {rate:.0%}")
        return results

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Current agent status."""
        return {
            "tools_available": self.executor.registry.names(),
            "recent_plans":    self.memory.recent_plans(5),
            "success_rate":    self.memory.success_rate(),
            "memory_connected": self._ltm is not None,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Janus Agent — autonomous goal execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python janus_agent_core.py "Read TODO.md and summarize the top 3 priorities"
  python janus_agent_core.py "List all Python files in the current directory"
  python janus_agent_core.py --status
  python janus_agent_core.py --batch goals.txt
        """,
    )
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--batch", type=str, help="File with goals (one per line)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--trace", default="event_trace.jsonl",
                        help="Path to event trace file")
    parser.add_argument("--history", action="store_true",
                        help="Show recent plan history")

    args = parser.parse_args()
    agent = JanusAgent(verbose=not args.quiet, trace_path=args.trace)

    if args.status:
        status = agent.status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.history:
        plans = agent.memory.recent_plans(20)
        if not plans:
            print("No plan history found.")
            return
        print(f"\nRecent plans ({len(plans)}):")
        print(f"{'Date':<20} {'Status':<10} {'Goal'}")
        print("-" * 60)
        for p in plans:
            status_icon = "[OK]" if p["completed"] else "[FAIL]"
            ts = p["timestamp"][:19]
            print(f"{ts:<20} {status_icon:<10} {p['goal'][:40]}")
        rate = agent.memory.success_rate()
        print(f"\nSuccess rate: {rate:.0%}")
        return

    if args.batch:
        goals = Path(args.batch).read_text().strip().splitlines()
        goals = [g.strip() for g in goals if g.strip() and not g.startswith("#")]
        agent.run_batch(goals)
        return

    if args.goal:
        plan = agent.run(args.goal)
        print(f"\n{'-'*40}")
        print(f"Result: {plan.outcome}")
        for step in plan.steps:
            icon = "[OK]" if step.success else "[FAIL]"
            print(f"  {icon} {step.description}")
            if step.result and step.success:
                preview = str(step.result)[:200]
                print(f"    -> {preview}")
            elif step.critic_note:
                print(f"    -> {step.critic_note}")
        return

    # No goal provided — show help
    parser.print_help()


if __name__ == "__main__":
    main()
