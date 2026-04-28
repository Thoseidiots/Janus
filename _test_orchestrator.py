import logging
logging.basicConfig(level=logging.WARNING)  # suppress verbose output

from janus_reasoning_engine.agents.orchestrator_bridge import OrchestratorBridge

print("Loading orchestrator (this loads Avus weights, may take ~10s)...")
bridge = OrchestratorBridge()
print("Orchestrator loaded:", bridge._orchestrator is not None)
if bridge._orchestrator is not None:
    status = bridge._orchestrator._status()
    for k, v in status.items():
        print(f"  {k:15s}: {'✅' if v else '❌'}")
else:
    print("  (running in stub mode)")
