“””
bridge.py — Janus Brain Bridge (patched)

Changes vs original:

- Added “loop_step” command: runs one cognitive loop cycle and returns
  the CycleResult as JSON.  This is what the Rust janus-cli should call
  on each tick instead of driving perceive/generate manually.
- Added “valence” command: returns the current ValenceState so the Rust
  layer can inject it into cloud LLM system prompts (bottleneck 3 fix).
- Error responses now include a stack trace field for easier debugging.
  “””

import sys
import json
import traceback
import torch

from .core import AutonomousCore
from .cognitive_loop import CognitiveLoop
from .goal_planner import GoalPlanner

class JanusBrainBridge:
def **init**(self):
self.core  = AutonomousCore()
self.loop  = CognitiveLoop(
core=self.core,
planner=GoalPlanner(),
)

```
def handle_request(self, request_json: str) -> str:
    try:
        request = json.loads(request_json)
        cmd = request.get("cmd")

        # ── Original commands (unchanged) ────────────────────────
        if cmd == "perceive":
            self.core.perceive(request["text"])
            return json.dumps({"status": "ok"})

        elif cmd == "generate":
            response = self.core.generate_response(request["prompt"])
            return json.dumps({"status": "ok", "response": response})

        elif cmd == "reflect":
            reflection = self.core.reflect()
            return json.dumps({"status": "ok", "reflection": reflection})

        # ── New: single cognitive cycle ──────────────────────────
        elif cmd == "loop_step":
            stimulus = request.get("stimulus")   # optional
            result   = self.loop.step(stimulus)
            return json.dumps({"status": "ok", "cycle": json.loads(result.to_json())})

        # ── New: current valence for system prompt injection ──────
        elif cmd == "valence":
            v = self.core.current_valence
            return json.dumps({
                "status": "ok",
                "valence": {
                    "pleasure":   v.pleasure.item(),
                    "arousal":    v.arousal.item(),
                    "curiosity":  v.curiosity.item(),
                    "autonomy":   v.autonomy.item(),
                    "connection": v.connection.item(),
                    "competence": v.competence.item(),
                },
                "context_string": self.core.valence_context_string(),
            })

        # ── New: run N loop cycles ────────────────────────────────
        elif cmd == "loop_run":
            cycles   = request.get("cycles", 1)
            stimuli  = request.get("stimuli")     # optional list
            results  = self.loop.run(stimuli=stimuli, cycles=cycles)
            return json.dumps({
                "status": "ok",
                "cycles": [json.loads(r.to_json()) for r in results]
            })

        else:
            return json.dumps({"status": "error", "message": f"Unknown command: {cmd}"})

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(limit=5),
        })
```

if **name** == “**main**”:
bridge = JanusBrainBridge()
for line in sys.stdin:
if not line.strip():
continue
print(bridge.handle_request(line.strip()), flush=True)