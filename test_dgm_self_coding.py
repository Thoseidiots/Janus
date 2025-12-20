import learning
import memory
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
    
    # Clean up any previous simulated fix files
    for f in os.listdir('.'):
        if f.startswith('simulated_fix_') and f.endswith('.py'):
            os.remove(f)

    # Re-initialize MKS and memory
    memory.mks = memory.MetaKnowledgeStore()
    memory.save_history([])
    print("--- Test Environment Setup Complete (Memory, MKS, and old fix files Cleared) ---")

# --- Main Test Execution ---
def run_test():
    setup_test_environment()
    
    test_topic = "ambition"
    
    # 1. Simulate the self-coding trigger (3 bad ratings)
    # We will simulate the history to ensure the trigger function has the necessary context
    
    # Simulate the history that leads to the trigger
    history = [
        {"role": "user", "message": "What is ambition?", "feedback": "bad"},
        {"role": "janus", "message": "Ambition is a delusion."},
        {"role": "user", "message": "Why do you say that?", "feedback": "bad"},
        {"role": "janus", "message": "Because it always fails."},
        {"role": "user", "message": "I disagree.", "feedback": "bad"},
        {"role": "janus", "message": "Your disagreement is noted."},
    ]
    memory.save_history(history)
    
    # 2. Trigger the self-coding process
    print(f"\n[TEST] Triggering self-coding for topic: '{test_topic}'")
    learning.trigger_self_coding(test_topic)
    
    # 3. Verification
    print("\n--- Verification of DGM Self-Coding ---")
    
    # Check for the DGM fix file
    expected_file = f"simulated_fix_{test_topic}.py"
    if os.path.exists(expected_file):
        print(f"[SUCCESS] DGM Fix File '{expected_file}' was created.")
        
        # Check the content of the file
        with open(expected_file, 'r') as f:
            content = f.read()
            if "def apply_simulated_fix():" in content and "Executing simulated code fix" in content:
                print("[SUCCESS] DGM Fix File content is correct (contains function and print statement).")
            else:
                print("[FAILURE] DGM Fix File content is incorrect.")
                
    else:
        print(f"[FAILURE] DGM Fix File '{expected_file}' was NOT created.")
        
    # Check the MKS for the new rule
    mks_data = memory.mks.data
    rules = mks_data.get("rules", [])
    
    if len(rules) == 1 and rules[0]["topic"] == test_topic:
        print(f"[SUCCESS] New DPA Rule was added to MKS: {rules[0]}")
    else:
        print("[FAILURE] New DPA Rule was NOT added to MKS.")
        
    # Check the console output for the DGM execution confirmation
    # This is implicitly checked by the print statements in learning.py, 
    # but we can check the log entry for the execution output.
    
    # Since the log entry is returned by editor_agent, we need to modify trigger_self_coding
    # to return the log entry for a perfect check. For now, we rely on the console output.
    
    print("\n--- Final Test Status ---")
    print("If the console output above shows '[DGM FIX] Executing simulated code fix...' and '[DGM FIX] Execution Output: [DGM FIX] Executing simulated code fix...', the test is successful.")

if __name__ == "__main__":
    # Change directory to Janus for the subprocess to work correctly
    os.chdir("Janus")
    run_test()
    # Change back to home directory
    os.chdir("..")
