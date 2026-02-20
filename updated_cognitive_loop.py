# Replacement for Janus core loop
class AutonomousLoop:
    def __init__(self, core):
        self.core = core
        self.os = JanusOS()
        self.active = True

    def run_forever(self):
        while self.active:
            # OBSERVE
            screen, w, h = self.os.capture_screen()
            vision = JanusVision(screen, w, h)
            state = self.core.world_model.update(vision)
            
            # PLAN & PROPOSE
            plan = self.core.plan(state)
            action = self.core.propose(plan)
            
            # VERIFY & APPLY
            if self.core.verify(action):
                self.core.apply(action)
            
            time.sleep(1) # Frequency of thought
