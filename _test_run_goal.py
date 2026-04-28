"""
Minimal live test: tell JanusOrchestrator to open Notepad and type hello.
Runs 3 cycles max. Watch your screen.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from janus_reasoning_engine.agents.orchestrator_bridge import OrchestratorBridge

print("Loading orchestrator...")
bridge = OrchestratorBridge()

if bridge._orchestrator is None:
    print("ERROR: Orchestrator not loaded")
    exit(1)

print("\nOrchestrator ready. Running goal: 'Open Notepad and type hello'\n")
result = bridge.run_goal("Open Notepad and type hello", max_cycles=3)

print(f"\n--- Result ---")
print(f"Success : {result.success}")
print(f"Cycles  : {result.cycle_num}")
print(f"Output  : {result.output[:200]}")
if result.error:
    print(f"Error   : {result.error}")
