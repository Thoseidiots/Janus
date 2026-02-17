“””
tool_executor.py
────────────────────────────────────────────────────────────
Dynamic tool invocation for Janus’s PROPOSE → VERIFY → APPLY loop.

Tools are registered with a risk tier.  High-risk tools run inside a
subprocess sandbox (simulating the WASM wrapper from janus-wasm).
Every call is appended to an append-only event trace on disk.
“””

import os
import json
import time
import hashlib
import threading
import subprocess
import tempfile
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

class RiskTier(Enum):
LOW    = “low”    # pure reads, memory queries  → auto-approved
MEDIUM = “medium” # file writes, shell cmds      → logged + rate-limited
HIGH   = “high”   # network, code exec           → sandboxed process

@dataclass
class ToolSpec:
name:        str
description: str
risk:        RiskTier
parameters:  Dict[str, str]          # name → type hint string
handler:     Optional[Callable] = None

@dataclass
class ToolCall:
call_id:   str
tool:      str
args:      Dict[str, Any]
proposed_by: str = “janus”
timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ToolResult:
call_id:    str
tool:       str
success:    bool
output:     Any
error:      Optional[str]
duration_ms: float
sandbox:    bool
timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())

class EventTrace:
“”“Append-only, tamper-evident log of all tool calls and results.”””

```
def __init__(self, path: str = "event_trace.jsonl"):
    self.path = Path(path)
    self._lock = threading.Lock()

def append(self, record: dict):
    record["_hash"] = self._hash(record)
    with self._lock:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

def tail(self, n: int = 50) -> List[dict]:
    if not self.path.exists():
        return []
    lines = self.path.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]

@staticmethod
def _hash(record: dict) -> str:
    payload = json.dumps({k: v for k, v in record.items() if k != "_hash"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

class SandboxRunner:
“”“Runs high-risk tools inside an isolated subprocess (WASM proxy).”””

```
TIMEOUT = 10  # seconds

def run(self, code: str, args: dict) -> Tuple[bool, Any, Optional[str]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "task.py"
        args_file = Path(tmpdir) / "args.json"
        out_file  = Path(tmpdir) / "out.json"

        args_file.write_text(json.dumps(args))
        script.write_text(f"""
```

import json, sys
args = json.loads(open(‘args.json’).read())
out  = {{“result”: None, “error”: None}}
try:
{code}
out[“result”] = result
except Exception as e:
out[“error”] = str(e)
open(‘out.json’, ‘w’).write(json.dumps(out))
“””)
try:
proc = subprocess.run(
[“python3”, str(script)],
cwd=tmpdir, capture_output=True,
timeout=self.TIMEOUT
)
if out_file.exists():
out = json.loads(out_file.read_text())
if out.get(“error”):
return False, None, out[“error”]
return True, out.get(“result”), None
return False, None, proc.stderr.decode()[:300]
except subprocess.TimeoutExpired:
return False, None, f”Sandbox timeout ({self.TIMEOUT}s)”
except Exception as e:
return False, None, str(e)

class ToolRegistry:
“”“Holds all registered tools.”””

```
def __init__(self):
    self._tools: Dict[str, ToolSpec] = {}

def register(self, spec: ToolSpec):
    self._tools[spec.name] = spec

def get(self, name: str) -> Optional[ToolSpec]:
    return self._tools.get(name)

def list_all(self) -> List[dict]:
    return [
        {
            "name": s.name,
            "description": s.description,
            "risk": s.risk.value,
            "parameters": s.parameters,
        }
        for s in self._tools.values()
    ]
```

class ToolExecutor:
“””
Orchestrates the PROPOSE → VERIFY → APPLY pipeline for tool calls.

```
Usage
─────
executor = ToolExecutor()
executor.register_defaults()
result = executor.execute(ToolCall(call_id="1", tool="memory_query",
                                   args={"query": "person"}))
"""

def __init__(self, trace_path: str = "event_trace.jsonl"):
    self.registry = ToolRegistry()
    self.trace    = EventTrace(trace_path)
    self.sandbox  = SandboxRunner()

    # Rate limiting: tool → list of recent call timestamps
    self._rate_log: Dict[str, List[float]] = {}
    self._rate_limit = 20   # calls per minute per tool
    self._lock = threading.Lock()

    # External callbacks
    self.on_result: Optional[Callable[[ToolResult], None]] = None

# ── Registration ──────────────────────────────────────────────────────────
def register_defaults(self):
    """Register Janus's built-in tools."""

    self.registry.register(ToolSpec(
        name="memory_query",
        description="Search episodic and semantic memory",
        risk=RiskTier.LOW,
        parameters={"query": "str", "limit": "int"},
        handler=self._tool_memory_query,
    ))
    self.registry.register(ToolSpec(
        name="file_read",
        description="Read a local file",
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
        name="shell_cmd",
        description="Run a whitelisted shell command",
        risk=RiskTier.MEDIUM,
        parameters={"cmd": "str"},
        handler=self._tool_shell_cmd,
    ))
    self.registry.register(ToolSpec(
        name="code_exec",
        description="Execute Python code in sandbox",
        risk=RiskTier.HIGH,
        parameters={"code": "str"},
    ))
    self.registry.register(ToolSpec(
        name="web_fetch",
        description="Fetch a URL (offline: returns cached stub)",
        risk=RiskTier.HIGH,
        parameters={"url": "str"},
    ))
    self.registry.register(ToolSpec(
        name="calendar_add",
        description="Add an event to local calendar store",
        risk=RiskTier.MEDIUM,
        parameters={"title": "str", "time": "str", "duration_min": "int"},
        handler=self._tool_calendar_add,
    ))
    self.registry.register(ToolSpec(
        name="valence_query",
        description="Read current homeostasis valence state",
        risk=RiskTier.LOW,
        parameters={},
        handler=self._tool_valence_query,
    ))

# ── Main entry point ───────────────────────────────────────────────────────
def execute(self, call: ToolCall) -> ToolResult:
    start = time.time()

    # PROPOSE: log intent
    self.trace.append({"phase": "PROPOSE", "call": asdict(call)})

    # VERIFY
    ok, reason = self._verify(call)
    self.trace.append({"phase": "VERIFY", "call_id": call.call_id, "approved": ok, "reason": reason})

    if not ok:
        result = ToolResult(
            call_id=call.call_id, tool=call.tool,
            success=False, output=None,
            error=f"Blocked: {reason}",
            duration_ms=0, sandbox=False,
        )
        self.trace.append({"phase": "APPLY", "result": asdict(result)})
        return result

    # APPLY
    spec = self.registry.get(call.tool)
    sandboxed = spec.risk == RiskTier.HIGH if spec else False

    try:
        if sandboxed:
            output, error = self._apply_sandboxed(call)
            success = error is None
        else:
            output = spec.handler(call.args) if spec and spec.handler else None
            error  = None
            success = True
    except Exception as e:
        output, error, success = None, str(e), False

    elapsed = (time.time() - start) * 1000
    result = ToolResult(
        call_id=call.call_id, tool=call.tool,
        success=success, output=output, error=error,
        duration_ms=round(elapsed, 2), sandbox=sandboxed,
    )
    self.trace.append({"phase": "APPLY", "result": asdict(result)})
    if self.on_result:
        self.on_result(result)
    return result

def chain(self, calls: List[ToolCall]) -> List[ToolResult]:
    """Execute a sequence of tool calls, passing output of each to the next."""
    results = []
    context = {}
    for call in calls:
        # Inject previous outputs into args
        call.args = {**context, **call.args}
        r = self.execute(call)
        results.append(r)
        if r.success and r.output is not None:
            context[f"prev_{call.tool}"] = r.output
    return results

# ── VERIFY ────────────────────────────────────────────────────────────────
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

    # Parameter validation
    for param in spec.parameters:
        if param not in call.args:
            # Only fail for required params (not int with default)
            if spec.parameters[param] == "str":
                return False, f"Missing required parameter: {param}"

    return True, "approved"

# ── APPLY helpers ─────────────────────────────────────────────────────────
def _apply_sandboxed(self, call: ToolCall) -> Tuple[Any, Optional[str]]:
    if call.tool == "code_exec":
        code = call.args.get("code", "result = None")
        ok, out, err = self.sandbox.run(code, call.args)
        return out, err
    elif call.tool == "web_fetch":
        # Offline stub
        return {"status": 200, "body": f"[offline stub for {call.args.get('url','')}]"}, None
    return None, "No sandbox handler"

# ── Built-in tool handlers ─────────────────────────────────────────────────
def _tool_memory_query(self, args: dict) -> Any:
    query = args.get("query", "")
    limit = int(args.get("limit", 5))
    # Returns a stub; real impl delegates to memory.py
    return {"query": query, "results": [], "note": "wire to memory.py"}

def _tool_file_read(self, args: dict) -> Any:
    p = Path(args.get("path", ""))
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p.read_text()[:4096]

def _tool_file_write(self, args: dict) -> Any:
    p = Path(args.get("path", "/tmp/janus_out.txt"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args.get("content", ""))
    return f"Written {len(args.get('content',''))} chars to {p}"

def _tool_shell_cmd(self, args: dict) -> Any:
    ALLOWED = {"ls", "pwd", "echo", "date", "whoami", "uname"}
    cmd = args.get("cmd", "")
    base = cmd.strip().split()[0] if cmd.strip() else ""
    if base not in ALLOWED:
        raise PermissionError(f"Command not whitelisted: {base}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
    return out.stdout.strip()

def _tool_calendar_add(self, args: dict) -> Any:
    cal_path = Path("calendar_events.jsonl")
    entry = {"title": args.get("title"), "time": args.get("time"),
             "duration_min": args.get("duration_min", 60),
             "created": datetime.now().isoformat()}
    with cal_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Event '{args.get('title')}' saved."

def _tool_valence_query(self, args: dict) -> Any:
    # Returns last known valence from persistent_state.json
    p = Path("persistent_state.json")
    if p.exists():
        state = json.loads(p.read_text())
        return state.get("valence", {})
    return {}
```