import sys
import json
import torch
from .core import AutonomousCore

class JanusBrainBridge:
    def __init__(self):
        self.core = AutonomousCore()

    def handle_request(self, request_json: str) -> str:
        try:
            request = json.loads(request_json)
            cmd = request.get("cmd")
            
            if cmd == "perceive":
                self.core.perceive(request["text"])
                return json.dumps({"status": "ok"})
            
            elif cmd == "generate":
                response = self.core.generate_response(request["prompt"])
                return json.dumps({"status": "ok", "response": response})
            
            elif cmd == "reflect":
                reflection = self.core.reflect()
                return json.dumps({"status": "ok", "reflection": reflection})
            
            else:
                return json.dumps({"status": "error", "message": f"Unknown command: {cmd}"})
        
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

if __name__ == "__main__":
    bridge = JanusBrainBridge()
    # Simple line-based protocol for communication with Rust
    for line in sys.stdin:
        if not line.strip():
            continue
        print(bridge.handle_request(line.strip()), flush=True)
