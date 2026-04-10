import consciousness
import memory
import learning
import os
import json
from datetime import datetime, timedelta

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

def simulate_self_coding_for_reflection(topic):
    """Simulates the self-coding process until a 'reflection' rule is generated."""
    
    # We will force the Planner to propose a reflection rule for this test
    # by temporarily modifying the planner_agent function.
    original_planner = learning.planner_agent
    
    def forced_planner(critique):
        new_rule = {
            "topic": critique["topic"],
            "action": "increase_reflection",
            "value": 0.3 # Use a larger value for a more noticeable change
        }
        print(f"[PLANNER - FORCED] Proposed Rule: Increase reflection by {new_rule['value']} for topic '{new_rule['topic']}'.")
        return new_rule
        
    learning.planner_agent = forced_planner
    
    # Trigger the self-coding process
    learning.trigger_self_coding(topic)
    
    # Restore the original planner
    learning.planner_agent = original_planner
    
    print(f"--- Forced Self-Coding Complete for Reflection ---")

# --- Main Test Execution ---
def run_test():
    setup_test_environment()
    
    test_topic = "life"
    
    # 1. Force a self-coding event to create a 'reflection' rule
    simulate_self_coding_for_reflection(test_topic)
    
    # 2. Verify the MKS state
    mks_data = memory.mks.data
    rules = mks_data.get("rules", [])
    
    print("\n--- Verification of MKS Rule ---")
    assert len(rules) == 1, "Forced self-coding failed to add exactly one rule."
    new_rule = rules[0]
    print(f"Rule Added: {new_rule}")
    assert new_rule['action'] == "increase_reflection", "Rule is not 'increase_reflection'."
    
    # 3. Test DPA application and prompt update
    print("\n--- Testing DPA Application and Prompt Update ---")
    
    test_query = "What is the meaning of life?"
    
    # The DPA logic prints the applied rule. We will check the final prompt.
    dynamic_prompt = consciousness.get_dynamic_system_prompt(test_query)
    
    # Check for the DPA application printout
    # The default reflection_level is 0.4. The rule adds 0.3, so it should be 0.7.
    expected_level = 0.4 + 0.3
    
    # Check if the new level is in the prompt
    expected_prompt_line = f"* Reflection Level: {expected_level:.2f}"
    
    if expected_prompt_line in dynamic_prompt:
        print(f"\n--- TEST SUCCESS ---")
        print(f"DPA successfully applied the 'increase_reflection' rule.")
        print(f"New Reflection Level ({expected_level:.2f}) found in the system prompt.")
        print("The new personality depth is now integrated and dynamic.")
    else:
        print(f"\n--- TEST FAILURE ---")
        print(f"Expected prompt line not found: '{expected_prompt_line}'")
        print("DPA failed to apply the new 'reflection_level' parameter.")
        
    # Print the final prompt for visual inspection
    print("\n--- Final Dynamic System Prompt (Snippet) ---")
    print(dynamic_prompt.split("Your personality is currently tuned with the following dynamic parameters:")[1].split("Based on these parameters, your communication style should be:")[0])

if __name__ == "__main__":
    run_test()
