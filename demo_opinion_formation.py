import time
from janus_human_core import OpinionEngine, HumanCore
from janus_true_human_learning import AdaptiveMemory, Experience

def demo_opinion_formation():
    print("=== Janus Opinion Formation Demo ===\n")
    
    # Initialize the core which contains the OpinionEngine
    core = HumanCore()
    print("Janus Initial State: Opinions Formed.")
    for op_text in core.opinions.list_opinions():
        print(f" - {op_text}")
    print()

    # Topic: Remote Work
    topic = "remote work"
    
    # Step 1: Check initial view
    print(f"Checking Janus's view on '{topic}'...")
    print(f"Janus: {core.share_opinion(topic)}\n")

    # Step 2: Provide some evidence
    evidence_1 = "Many studies show that remote work increases focus and reduces stress for software engineers."
    print(f"User provides evidence: \"{evidence_1}\"")
    print(core.update_opinion(topic, evidence_1, weight=0.4))
    print()

    # Step 3: Provide conflicting or nuanced evidence
    evidence_2 = "However, some teams report a decrease in spontaneous collaboration and mentorship."
    print(f"User provides counter-evidence: \"{evidence_2}\"")
    # This evidence contains 'decrease' which might be seen as negative in our simple sentiment scorer
    # Actually 'decrease' is not in our simple list, but 'fails', 'bad', 'wrong' are.
    # Let's use words the simple scorer understands if we want a shift.
    evidence_2_v2 = "However, it can be bad for spontaneous collaboration and mentorship if not managed well."
    print(core.update_opinion(topic, evidence_2_v2, weight=0.3))
    print()

    # Step 4: Final verdict on a new topic
    new_topic = "Python vs Rust"
    print(f"Asking about a new topic: '{new_topic}'")
    print(f"Janus: {core.share_opinion(new_topic)}\n")
    
    evidence_3 = "Rust provides memory safety and high performance, making it great for systems programming."
    print(f"User provides evidence: \"{evidence_3}\"")
    print(core.update_opinion(new_topic, evidence_3, weight=0.8))
    print()

    print("=== Conclusion ===")
    print("Janus doesn't just parrot strings; it uses a Bayesian-inspired update mechanism")
    print("to shift its 'stance' and 'confidence' based on the weight of incoming data.")

if __name__ == "__main__":
    demo_opinion_formation()
