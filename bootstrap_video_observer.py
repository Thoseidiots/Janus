# bootstrap_video_observer.py

“””
One-time bootstrap: imports all updated modules, wires them into a running
Janus instance (or creates one), registers video capability, and starts the
background observer thread.

Run this once at Janus startup or hot-patch into a running session:
python bootstrap_video_observer.py
“””

import sys
import os
import importlib
import time

# ── Ensure project root is on path ────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(**file**))
if _HERE not in sys.path:
sys.path.insert(0, _HERE)

def bootstrap(existing_core=None, existing_hub=None, existing_loop=None):
“””
Wire up the full video observer stack.

```
Parameters
----------
existing_core : AutonomousCore | None
    Pass an already-running core to reuse it.  If None, a fresh one is created.
existing_hub  : JanusCapabilityHub | None
    Pass an existing hub to extend.  If None, a fresh one is created.
existing_loop : CognitiveLoop | None
    Pass an existing loop to upgrade.  If None, a fresh one is created.

Returns
-------
dict with keys: core, hub, loop, observer
"""

print("[bootstrap] Starting Janus Video Observer bootstrap...")

# ── 1. Import (or reload) all updated modules ─────────────────────────────
modules_to_load = [
    "video_observer",
    "updated_enhanced_vision",
    "video_capability",
    "updated_janus_capability_hub",
    "updated_cognitive_loop",
]
loaded = {}
for mod_name in modules_to_load:
    try:
        if mod_name in sys.modules:
            loaded[mod_name] = importlib.reload(sys.modules[mod_name])
        else:
            loaded[mod_name] = importlib.import_module(mod_name)
        print(f"[bootstrap]   ✓ {mod_name}")
    except Exception as exc:
        print(f"[bootstrap]   ✗ {mod_name} — {exc}")

# ── 2. Build or reuse core ────────────────────────────────────────────────
core = existing_core
if core is None:
    try:
        from janus_brain.core import AutonomousCore
        core = AutonomousCore()
        print("[bootstrap]   ✓ AutonomousCore created")
    except Exception as exc:
        print(f"[bootstrap]   ✗ Could not create AutonomousCore: {exc}")
        core = None

# ── 3. Build or reuse hub ─────────────────────────────────────────────────
hub = existing_hub
if hub is None:
    try:
        from updated_janus_capability_hub import JanusCapabilityHub
        hub = JanusCapabilityHub(core=core, auto_register=True)
        print(f"[bootstrap]   ✓ JanusCapabilityHub ready ({len(hub.list_tools())} tools)")
    except Exception as exc:
        print(f"[bootstrap]   ✗ Could not create JanusCapabilityHub: {exc}")
        hub = None
else:
    # Hot-patch video capability into existing hub
    try:
        from video_capability import register_video_capability
        register_video_capability(hub)
        print("[bootstrap]   ✓ Video capability patched into existing hub")
    except Exception as exc:
        print(f"[bootstrap]   ✗ Could not patch video capability: {exc}")

# ── 4. Build or upgrade cognitive loop ────────────────────────────────────
loop = existing_loop
if loop is None:
    try:
        from updated_cognitive_loop import CognitiveLoop
        from goal_planner import GoalPlanner
        loop = CognitiveLoop(
            core=core,
            planner=GoalPlanner(),
            hub=hub,
            auto_video_learning=True,
        )
        print("[bootstrap]   ✓ CognitiveLoop (with video hooks) created")
    except Exception as exc:
        print(f"[bootstrap]   ✗ Could not create CognitiveLoop: {exc}")
        loop = None

# ── 5. Get/configure the VideoObserver singleton ──────────────────────────
observer = None
try:
    from video_observer import get_observer
    observer = get_observer()
    # Wire memory if available
    if core is not None and hasattr(core, "memory"):
        observer.memory = core.memory
    print("[bootstrap]   ✓ VideoObserver singleton configured")
except Exception as exc:
    print(f"[bootstrap]   ✗ VideoObserver: {exc}")

# ── 6. Export tool specs to GoalPlanner ───────────────────────────────────
if hub is not None and loop is not None:
    try:
        specs = hub.export_tool_specs_for_planner()
        if specs and loop.planner:
            for spec in specs:
                loop.planner.register_tool(spec)
            print(f"[bootstrap]   ✓ {len(specs)} hub tools exported to GoalPlanner")
    except Exception as exc:
        print(f"[bootstrap]   ✗ GoalPlanner export: {exc}")

# ── 7. Verify everything is wired ────────────────────────────────────────
print("\n[bootstrap] ── Status ──────────────────────────────────────────")
print(f"  Core:     {'✓' if core     else '✗ (not available)'}")
print(f"  Hub:      {'✓ (' + str(len(hub.list_tools())) + ' tools)' if hub else '✗'}")
print(f"  Loop:     {'✓ (video hooks active)' if loop else '✗'}")
print(f"  Observer: {'✓ (idle, ready to watch)' if observer else '✗'}")
print("[bootstrap] ────────────────────────────────────────────────────")
print()
print("Video observer enabled — Janus can now watch autonomously.")
print()
print("Quick-start examples:")
print("  observer.watch_video('https://www.youtube.com/watch?v=...')")
print("  hub.dispatch_action('watch_video', {'url_or_title': 'YouTube'})")
print("  loop.step()  # cognitive loop will auto-start tutorials when curious")
print()

return {
    "core":     core,
    "hub":      hub,
    "loop":     loop,
    "observer": observer,
}
```

# ── Standalone entry point ────────────────────────────────────────────────────

if **name** == “**main**”:
result = bootstrap()
# Keep alive so observer thread stays running if called directly
if result.get(“observer”):
print(”[bootstrap] Running — press Ctrl+C to stop.”)
try:
while True:
time.sleep(5)
obs = result[“observer”]
if obs.is_playing():
summary = obs.get_current_summary()
print(f”[observer] {summary[:120]}”)
except KeyboardInterrupt:
print(”\n[bootstrap] Shutting down observer…”)
result[“observer”].stop()
print(”[bootstrap] Done.”)
