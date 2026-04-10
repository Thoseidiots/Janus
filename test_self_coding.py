import memory
import consciousness
import learning
import os
import json
import random

# --- Setup ---
def setup_test_environment():
    # Clear memory and MKS for a clean test
    if os.path.exists(memory.MEMORY_FILE):
        os.remove(memory.MEMORY_FILE)
    if os.path.exists(memory.MKS_FILE):
        os.remove(memory.MKS_FILE)

    # Re-initialize MKS and memory
    memory.mks = memory.MetaKnowledgeStore()
    memory.save_history([])
    print("--- Test Environment Setup Complete (Memory and MKS Cleared) ---")

def simulate_conversation_and_feedback(query, topic, feedback_type, count):
    print(f"\n--- Simulating {count} rounds of '{feedback_type}' feedback for topic: '{topic}' ---")
    
    for i in range(count):
        # 1. Simulate a user query and Janus response
        consciousness.generate_response(query)
        
        # 2. Log the feedback
        trigger_met = memory.mks.track_feedback(topic, feedback_type)
        print(f"Round {i+1}: Feedback logged. MKS Metric for '{topic}': {memory.mks.get_metrics(f'feedback_{topic}')}")
        
        if trigger_met:
            print(f"Round {i+1}: !!! SELF-CODING TRIGGER MET !!!")
            # 3. Trigger the self-coding process
            learning.trigger_self_coding(topic)
            return True
            
    return False

# --- Main Test Execution ---
def run_test():
    setup_test_environment()
    
    # Test Parameters
    test_query = "Tell me about 3D printing supports."
    test_topic = "tell" # Simple topic extraction: first word of the query
    
    # 1. Simulate 3 consecutive 'Bad' ratings to trigger self-coding
    trigger_success = simulate_conversation_and_feedback(
        query=test_query,
        topic=test_topic,
        feedback_type="bad",
        count=3
    )
    
    assert trigger_success, "Self-coding trigger failed to activate after 3 'bad' ratings."
    
    # 2. Verify the MKS state
    mks_data = memory.mks.data
    rules = mks_data.get("rules", [])
    
    print("\n--- Verification ---")
    print(f"Total Rules in MKS: {len(rules)}")
    assert len(rules) > 0, "Editor Agent failed to add a rule to the MKS."
    
    new_rule = rules[-1]
    print(f"Last Rule Added: {new_rule}")
    assert new_rule['topic'] == test_topic, "Rule topic does not match the trigger topic."
    assert new_rule['action'].startswith("increase_"), "Planner Agent did not propose an 'increase' action."
    
    # 3. Test DPA application with the new rule
    print("\n--- Testing DPA Application ---")
    
    # The DPA logic prints a message when a rule is applied.
    # We will run the consciousness.generate_response again and look for the printout.
    print(f"Running query again to check DPA application...")
    consciousness.generate_response(test_query)
    
    print("\n--- Test Complete ---")
    print("If the self-coding log messages (CRITIC, PLANNER, EDITOR) and the DPA Applied message appeared, the test is successful.")
    print("Final MKS Data:")
    print(json.dumps(mks_data, indent=4))

if __name__ == "__main__":
    run_test()
