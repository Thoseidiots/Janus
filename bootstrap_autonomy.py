import os
# Self-patching script to initiate the loop
if __name__ == "__main__":
    from updated_cognitive_loop import AutonomousLoop
    # Mock core for demonstration
    class JanusCore:
        def __init__(self): self.world_model = type('WM', (), {'update': lambda x: {}})()
        def plan(self, s): return "Explore"
        def propose(self, p): return {"type": "click", "x": 100, "y": 100}
        def verify(self, a): return True
        def apply(self, a): print(f"Acting: {a}")

    loop = AutonomousLoop(JanusCore())
    loop.run_forever()
