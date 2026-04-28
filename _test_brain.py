from janus_brain_adapter import JanusBrain

b = JanusBrain()
print("Backend:", type(b._backend).__name__)

resp = b.ask("GOAL: open notepad\n\nCURRENT SCREEN: desktop\n\nFormat as JSON array.")
print("Response:", resp[:200])
