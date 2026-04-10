import consciousness
import memory
import os
import json
from datetime import datetime, timedelta

# --- Setup ---
def setup_test_environment():
    # Ensure memory.py is initialized
    if not hasattr(memory, 'mks'):
        memory.mks = memory.MetaKnowledgeStore()
    
    # Clear conversation history to avoid cluttering the test
    if os.path.exists(memory.MEMORY_FILE):
        os.remove(memory.MEMORY_FILE)
    memory.save_history([])
    
    print(f"--- RAG 3D Face Generator Test Setup ---")
    print(f"Current Date for Recency: {datetime.now().isoformat()}")
    print(f"--- Testing for retrieval of new knowledge source ---")

# --- Main Test Execution ---
def run_test():
    setup_test_environment()
    
    test_query = "How does the 3D Face Generator work? Tell me about the Tri-planes."
    
    # Run the response generation, which will print the RAG ranking log
    print(f"\n[TEST] Running query: '{test_query}'")
    response = consciousness.generate_response(test_query)
    
    print(f"\n[JANUS RESPONSE] {response}")
    
    # Verification: The response should contain a key phrase from the new knowledge source.
    expected_phrase = "Tri-planes (XY, XZ, YZ feature maps)"
    
    if expected_phrase in response:
        print("\n--- TEST SUCCESS ---")
        print("The response successfully retrieved and used the 3D Face Generator knowledge.")
    else:
        print("\n--- TEST FAILURE ---")
        print("The response did not contain the expected knowledge phrase. Check the RAG Ranking log above for details.")

if __name__ == "__main__":
    run_test()
