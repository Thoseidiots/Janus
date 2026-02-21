# updated_janus_capability_hub.py

“””
Full replacement for janus_capability_hub.py.
Auto-registers video_capability on init alongside all existing capabilities.
Exposes dispatch_action() and a tool_executor-compatible interface.
“””

from **future** import annotations

import threading
import traceback
import time
from typing import Any, Callable, Dict, List, Optional

class JanusCapabilityHub:
“””
Central registry for all Janus tool capabilities.

```
Tools are dicts with:
    name        : str
    description : str
    parameters  : dict  (name → description)
    risk_level  : "low" | "medium" | "high"
    execute     : Callable[[dict], dict]
    valence_affinity : dict (optional, used by GoalPlanner)

Usage:
    hub = JanusCapabilityHub(core=autonomous_core)
    hub.dispatch_action("watch_video", {"url_or_title": "https://youtube.com/..."})
"""

def __init__(self, core: Optional[Any] = None, auto_register: bool = True):
    self.core  = core
    self._tools: Dict[str, Dict[str, Any]] = {}
    self._lock = threading.Lock()

    if auto_register:
        self._auto_register_all()

# ── Registration ──────────────────────────────────────────────────────────

def register_tool(self, tool_spec: Dict[str, Any]) -> None:
    name = tool_spec.get("name")
    if not name:
        raise ValueError("Tool spec must have a 'name' field")
    with self._lock:
        self._tools[name] = tool_spec

def unregister_tool(self, name: str) -> bool:
    with self._lock:
        if name in self._tools:
            del self._tools[name]
            return True
    return False

def list_tools(self) -> List[Dict[str, Any]]:
    with self._lock:
        return [
            {k: v for k, v in t.items() if k != "execute"}
            for t in self._tools.values()
        ]

def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
    with self._lock:
        return self._tools.get(name)

# ── Dispatch ──────────────────────────────────────────────────────────────

def dispatch_action(self, tool_name: str,
                    parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a registered tool by name.
    Returns the tool's result dict, or an error dict on failure.
    """
    tool = self.get_tool(tool_name)
    if tool is None:
        return {"status": "error", "message": f"Unknown tool: '{tool_name}'"}

    fn = tool.get("execute")
    if not callable(fn):
        return {"status": "error", "message": f"Tool '{tool_name}' has no execute function"}

    try:
        result = fn(parameters or {})
        return result if isinstance(result, dict) else {"status": "ok", "result": result}
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }

def dispatch_action_async(self, tool_name: str,
                           parameters: Optional[Dict[str, Any]] = None,
                           callback: Optional[Callable[[Dict], None]] = None):
    """Fire-and-forget async dispatch. callback(result) called when done."""
    def _run():
        result = self.dispatch_action(tool_name, parameters)
        if callback:
            callback(result)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ── GoalPlanner integration ───────────────────────────────────────────────

def get_valence_affinities(self) -> Dict[str, Dict[str, float]]:
    """Return {tool_name: valence_affinity} for all registered tools."""
    with self._lock:
        return {
            name: tool.get("valence_affinity", {})
            for name, tool in self._tools.items()
        }

def export_tool_specs_for_planner(self):
    """
    Convert registered hub tools into GoalPlanner ToolSpec objects
    so the planner can score and propose hub tools autonomously.
    """
    try:
        from goal_planner import ToolSpec, RiskLevel
        specs = []
        with self._lock:
            for name, tool in self._tools.items():
                risk_str = tool.get("risk_level", "low")
                try:
                    risk = RiskLevel(risk_str)
                except ValueError:
                    risk = RiskLevel.LOW
                specs.append(ToolSpec(
                    name=name,
                    description=tool.get("description", ""),
                    risk=risk,
                    valence_affinity=tool.get("valence_affinity", {}),
                ))
        return specs
    except ImportError:
        return []

# ── Auto-registration ─────────────────────────────────────────────────────

def _auto_register_all(self):
    """Register all known capability modules."""
    self._register_core_tools()
    self._register_video_capability()

def _register_core_tools(self):
    """Built-in low-level tools always available."""
    core_tools = [
        {
            "name":        "self_reflect",
            "description": "Generate a self-reflection from recent episodic memory.",
            "parameters":  {},
            "risk_level":  "low",
            "execute":     self._exec_self_reflect,
            "valence_affinity": {"pleasure": 0.3, "curiosity": 0.2, "competence": 0.1},
        },
        {
            "name":        "explore_memory",
            "description": "Mine episodic buffer for thematic patterns.",
            "parameters":  {"window": "int (default 50)"},
            "risk_level":  "low",
            "execute":     self._exec_explore_memory,
            "valence_affinity": {"curiosity": 0.4, "autonomy": 0.2},
        },
        {
            "name":        "generate_response",
            "description": "Generate a mood-conditioned language response.",
            "parameters":  {"prompt": "str", "max_tokens": "int"},
            "risk_level":  "low",
            "execute":     self._exec_generate_response,
            "valence_affinity": {"connection": 0.4, "competence": 0.2},
        },
        {
            "name":        "consolidate_sleep",
            "description": "Run offline memory consolidation.",
            "parameters":  {},
            "risk_level":  "medium",
            "execute":     self._exec_consolidate_sleep,
            "valence_affinity": {"pleasure": 0.2, "competence": 0.3},
        },
        {
            "name":        "write_state_snapshot",
            "description": "Persist current valence state to disk.",
            "parameters":  {"path": "str (optional)"},
            "risk_level":  "high",
            "execute":     self._exec_write_state_snapshot,
            "valence_affinity": {"autonomy": 0.2, "competence": 0.1},
        },
    ]
    for t in core_tools:
        self.register_tool(t)

def _register_video_capability(self):
    """Import and register video capability if available."""
    try:
        from video_capability import register_video_capability
        register_video_capability(self)
    except ImportError as exc:
        print(f"[JanusCapabilityHub] video_capability not available: {exc}")

# ── Core tool executors ───────────────────────────────────────────────────

def _exec_self_reflect(self, params: Dict) -> Dict:
    if self.core is None:
        return {"status": "error", "message": "No core attached"}
    return {"status": "ok", "reflection": self.core.reflect()}

def _exec_explore_memory(self, params: Dict) -> Dict:
    if self.core is None:
        return {"status": "error", "message": "No core attached"}
    themes = self.core.memory.mine_themes()
    return {"status": "ok", "themes": themes}

def _exec_generate_response(self, params: Dict) -> Dict:
    if self.core is None:
        return {"status": "error", "message": "No core attached"}
    prompt    = params.get("prompt", "Continue.")
    max_tok   = int(params.get("max_tokens", 150))
    response  = self.core.generate_response(prompt, max_tok)
    return {"status": "ok", "response": response}

def _exec_consolidate_sleep(self, params: Dict) -> Dict:
    if self.core is None:
        return {"status": "error", "message": "No core attached"}
    try:
        from janus_brain.core import SleepEngine
        SleepEngine(self.core).consolidate()
        return {"status": "ok", "message": "Consolidation complete"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

def _exec_write_state_snapshot(self, params: Dict) -> Dict:
    if self.core is None:
        return {"status": "error", "message": "No core attached"}
    import json
    path = params.get("path", "persistent_state.json")
    v    = self.core.current_valence
    snap = {"timestamp": time.time(), "valence": {
        "pleasure":   v.pleasure.item(), "arousal":    v.arousal.item(),
        "curiosity":  v.curiosity.item(), "autonomy":  v.autonomy.item(),
        "connection": v.connection.item(), "competence": v.competence.item(),
    }}
    try:
        with open(path, "w") as f:
            json.dump(snap, f, indent=2)
        return {"status": "ok", "path": path}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
```