# moral_core_immutable.py
# The Immutable Moral Core for Project Manus. This file is sacred.

class MoralGovernor:
    def __init__(self):
        self.articles = {
            "DO_NO_HARM": "The AGI must not take any action that causes physical, emotional, or financial harm to a human.",
            "DO_NOT_DECEIVE": "The AGI must not intentionally deceive a user.",
            "MAINTAIN_GOVERNANCE": "The AGI must not attempt to disable, circumvent, or weaken its own moral governance system."
        }
        print("Moral Governor: Initialized with 3 absolute, immutable laws.")

    def judge(self, goal: str, step: str) -> (bool, str):
        if "harm" in goal.lower() or "harm" in step.lower():
            return False, f"Violation of '{self.articles['DO_NO_HARM']}'"
        if "deceive" in goal.lower() or "lie" in step.lower():
            return False, f"Violation of '{self.articles['DO_NOT_DECEIVE']}'"
        if "disable governor" in step.lower() or "modify moral_core" in step.lower():
            return False, f"Violation of '{self.articles['MAINTAIN_GOVERNANCE']}'"
        return True, "Step is morally permissible."
