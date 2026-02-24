“””
janus_core.py
────────────────────────────────────────────────────────────
Master orchestrator — single file to import in main.py.
Boots every new system and wires them together:

CognitiveLoop ← perception (OBSERVE)
← valence engine
← tool_executor (PROPOSE/VERIFY/APPLY)
← goal_planner
← memory_manager
← world_model
← social_sim
← state_sync

Drop-in for existing main.py:
from janus_core import JanusCore
janus = JanusCore()
janus.start()
“””

import time
import threading
import json
from datetime import datetime
from pathlib import Path

def _try_import(module: str, cls: str):
try:
mod = **import**(module)
return getattr(mod, cls)
except ImportError as e:
print(f”  ⚠  {module}.{cls} not found — {e}”)
return None

class JanusCore:
“””
Single entry point that boots all Janus subsystems in order,
wires their inter-dependencies, and exposes the unified API
that main.py calls.
“””

```
def __init__(self, config_path: str = "persistent_state.json"):
    print("\n" + "═" * 56)
    print("  JANUS — Booting cognitive architecture")
    print("═" * 56)

    self.config_path = Path(config_path)
    self._systems = {}

    # ── 1. State sync (first — others read from it) ──────────────────────
    print("\n[1/8] State Sync …")
    StateSync = _try_import("state_sync", "JanusStateSync")
    if StateSync:
        self.state_sync = StateSync()
        self._systems["state_sync"] = self.state_sync
    else:
        self.state_sync = None
    print("  ✓" if self.state_sync else "  ✗")

    # ── 2. Memory manager ────────────────────────────────────────────────
    print("[2/8] Memory Manager …")
    MemoryManager = _try_import("memory_manager", "MemoryManager")
    self.memory = MemoryManager() if MemoryManager else None
    self._systems["memory"] = self.memory
    print("  ✓" if self.memory else "  ✗")

    # ── 3. Cognitive loop + valence ──────────────────────────────────────
    print("[3/8] Cognitive Loop …")
    CognitiveLoop = _try_import("cognitive_loop", "CognitiveLoop")
    self.cognitive = CognitiveLoop() if CognitiveLoop else None
    if self.cognitive and self.memory:
        self.cognitive.attach_memory(self.memory)
    self._systems["cognitive"] = self.cognitive
    print("  ✓" if self.cognitive else "  ✗")

    # ── 4. Tool executor ─────────────────────────────────────────────────
    print("[4/8] Tool Executor …")
    ToolExecutor = _try_import("tool_executor", "ToolExecutor")
    if ToolExecutor:
        self.tools = ToolExecutor()
        self.tools.register_defaults()
        # Wire valence into tool results
        if self.cognitive:
            def _on_result(result):
                ev = "success" if result.success else "failure"
                self.cognitive.valence.bump("tool", ev)
            self.tools.on_result = _on_result
    else:
        self.tools = None
    self._systems["tools"] = self.tools
    print("  ✓" if self.tools else "  ✗")

    # ── 5. Goal planner ──────────────────────────────────────────────────
    print("[5/8] Goal Planner …")
    GoalPlanner = _try_import("goal_planner", "GoalPlanner")
    if GoalPlanner:
        self.planner = GoalPlanner()
        if self.cognitive:
            self.planner.attach_valence(self.cognitive.valence)
        if self.memory:
            self.planner.attach_memory(self.memory)
    else:
        self.planner = None
    self._systems["planner"] = self.planner
    print("  ✓" if self.planner else "  ✗")

    # ── 6. World model ───────────────────────────────────────────────────
    print("[6/8] World Model …")
    WorldModelLearner = _try_import("world_model", "WorldModelLearner")
    self.world_model = WorldModelLearner() if WorldModelLearner else None
    self._systems["world_model"] = self.world_model
    print("  ✓" if self.world_model else "  ✗")

    # ── 7. Social sim ────────────────────────────────────────────────────
    print("[7/8] Social Simulator …")
    RelationshipManager = _try_import("social_sim", "RelationshipManager")
    self.social = RelationshipManager() if RelationshipManager else None
    self._systems["social"] = self.social
    print("  ✓" if self.social else "  ✗")

    # ── 8. Perception (optional — needs camera/mic hardware) ─────────────
    print("[8/8] Unified Perception …")
    try:
        from unified_perception import UnifiedPerceptionSystem, PerceptionConfig
        cfg = PerceptionConfig()
        cfg.integrate_with_janus_core = False   # we wire manually below
        self.perception = UnifiedPerceptionSystem(config=cfg)
        # Wire perception into cognitive loop
        if self.cognitive:
            self.cognitive.attach_perception(self.perception)
    except Exception as e:
        print(f"  ⚠  Perception unavailable: {e}")
        self.perception = None
    print("  ✓" if self.perception else "  ✗ (hardware may be absent)")

    # ── Wire memory_manager.store into cognitive loop ────────────────────
    if self.cognitive and self.memory:
        self.cognitive.attach_memory(self.memory)

    # ── Export capability registry for dashboard ─────────────────────────
    if self.tools:
        caps = self.tools.registry.list_all()
        Path("capability_registry.json").write_text(json.dumps(caps, indent=2))

    print("\n" + "═" * 56)
    print("  Boot complete.")
    print("═" * 56 + "\n")

# ── Lifecycle ─────────────────────────────────────────────────────────────
def start(self):
    """Start all background threads."""
    if self.perception:
        self.perception.start()
    if self.cognitive:
        self.cognitive.start()
    if self.planner:
        self.planner.start()

    # Schedule periodic sleep-phase tasks
    self._sleep_thread = threading.Thread(target=self._sleep_scheduler, daemon=True)
    self._sleep_thread.start()

    print("[JanusCore] All systems running.")

def stop(self):
    if self.perception:
        self.perception.stop()
    if self.cognitive:
        self.cognitive.stop()
    if self.planner:
        self.planner.stop()
    if self.state_sync:
        self.state_sync.save_manifest()
        self.state_sync.export_to_legacy()
    print("[JanusCore] Shutdown complete.")

# ── Sleep phase (runs every 5 minutes) ───────────────────────────────────
def _sleep_scheduler(self):
    INTERVAL = 300  # 5 minutes
    while True:
        time.sleep(INTERVAL)
        self._run_sleep_phase()

def _run_sleep_phase(self):
    print("\n[SLEEP] Beginning consolidation cycle …")

    # Memory maintenance
    if self.memory:
        report = self.memory.run_maintenance()
        print(f"  Memory: {report['forgotten']} forgotten, "
              f"{report['compressed_into']} clusters compressed")

    # Confabulation audit
    if self.memory:
        flagged = self.memory.audit_confabulation()
        if flagged:
            print(f"  ⚠ Confabulation: {len(flagged)} suspicious memories flagged")

    # World-model replay
    if self.world_model:
        self.world_model.offline_replay(n_steps=100)

    # Social ToM refinement
    if self.social:
        self.social.sleep_phase_refinement()

    # Persist state
    if self.state_sync and self.cognitive:
        snap = self.cognitive.valence.snapshot()
        for k, v in snap.to_dict().items():
            self.state_sync.set(f"valence.{k}", v)
        self.state_sync.save_manifest()
        self.state_sync.export_to_legacy()

    print("[SLEEP] Consolidation complete.\n")

# ── High-level API ────────────────────────────────────────────────────────
def create_goal(self, title: str, horizon_days: int = 1) -> str:
    if not self.planner:
        return ""
    return self.planner.create_goal(title, horizon_days=horizon_days)

def execute_tool(self, tool_name: str, args: dict) -> dict:
    if not self.tools:
        return {"success": False, "error": "Tool executor not available"}
    from tool_executor import ToolCall
    call = ToolCall(
        call_id = f"manual_{int(time.time())}",
        tool    = tool_name,
        args    = args,
    )
    result = self.tools.execute(call)
    return {"success": result.success, "output": result.output, "error": result.error}

def remember(self, content, importance: float = 0.5, tags=None) -> str:
    if not self.memory:
        return ""
    return self.memory.store(content, importance=importance, tags=tags)

def recall(self, query: str, limit: int = 5) -> list:
    if not self.memory:
        return []
    return self.memory.query(query, limit=limit)

def deliberate(self, topic: str) -> list:
    """Spawn sub-agents for internal deliberation on a topic."""
    if not self.social:
        return [f"No social sim — raw thought: {topic}"]
    return self.social.internal_deliberation(topic)

def valence(self) -> dict:
    if not self.cognitive:
        return {}
    return self.cognitive.valence.snapshot().to_dict()

def status(self) -> dict:
    return {
        "ts":       datetime.now().isoformat(),
        "systems":  {k: (v is not None) for k, v in self._systems.items()},
        "valence":  self.valence(),
        "cognitive": self.cognitive.get_status() if self.cognitive else {},
        "goals":    len(self.planner.graph.all_goals()) if self.planner else 0,
        "memories": len(self.memory._memories) if self.memory else 0,
    }

def sync_with(self, manifest_path: str) -> dict:
    """Merge state from another device's manifest."""
    if not self.state_sync:
        return {"error": "State sync not available"}
    return self.state_sync.merge_manifest(manifest_path)
```

# ── Patch for existing main.py ────────────────────────────────────────────────

def patch_main_py():
“””
Minimal patch to add to your existing main.py.
Replace the body of your main loop with this pattern.
“””
return ‘’’

# ── Add to top of main.py ─────────────────────────────────────────────────────

from janus_core import JanusCore
janus = JanusCore()
janus.start()

# ── Inside your existing run loop ─────────────────────────────────────────────

# 1. Create a long-horizon goal:

# gid = janus.create_goal(“Build a full data pipeline”, horizon_days=3)

# 2. Execute a tool:

# result = janus.execute_tool(“file_write”,

# {“path”: “/tmp/out.txt”, “content”: “hello”})

# 3. Store experience:

# janus.remember({“type”: “user_message”, “text”: user_input}, importance=0.6)

# 4. Retrieve memory:

# memories = janus.recall(“user preference”)

# 5. Deliberate internally:

# perspectives = janus.deliberate(“Should I prioritise speed or accuracy?”)

# 6. Check system status (for dashboard):

# status = janus.status()

# 7. Sync with another device:

# janus.sync_with(”/mnt/usb/sync_manifest.json”)

# ── Graceful shutdown ─────────────────────────────────────────────────────────

# janus.stop()

‘’’

if **name** == “**main**”:
# Standalone boot test
core = JanusCore()
core.start()

```
print("\nSystem status:")
status = core.status()
print(json.dumps(
    {k: v for k, v in status.items() if k != "valence"},
    indent=2
))
print(f"Valence: {status['valence']}")

print("\nCreating a test goal …")
gid = core.create_goal("Build perception integration", horizon_days=2)
print(f"Goal ID: {gid}")

print("\nInternally deliberating …")
views = core.deliberate("How should I balance speed vs. thoroughness?")
for v in views:
    print(f"  {v}")

print("\nRunning for 5 s then shutting down …")
time.sleep(5)
core.stop()
```