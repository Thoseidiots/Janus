import consciousness
import memory
import os
import json
from datetime import datetime, timedelta

# --- Setup ---
def setup_test_environment():
    # Ensure memory.py is initialized to get the current date for recency calculation
    if not hasattr(memory, 'mks'):
        memory.mks = memory.MetaKnowledgeStore()
    
    # Clear conversation history to avoid cluttering the test
    if os.path.exists(memory.MEMORY_FILE):
        os.remove(memory.MEMORY_FILE)
    memory.save_history([])
    
    print(f"--- RAG Recency Test Setup ---")
    print(f"Current Date for Recency: {datetime.now().isoformat()}")
    print(f"Recency Half-Life: {memory.RECENCY_HALF_LIFE_DAYS} days")
    print(f"--- Expected Ranking Logic: (Semantic * 0.8) + (Recency * 0.2) ---")

# --- Main Test Execution ---
def run_test():
    setup_test_environment()
    
    test_query = "Tell me about 3D printing."
    
    # Run the response generation, which will print the RAG ranking log
    print(f"\n[TEST] Running query: '{test_query}'")
    response = consciousness.generate_response(test_query)
    
    print(f"\n[JANUS RESPONSE] {response}")
    
    # Verification: The highest-ranked context should be the one with the medium semantic score
    # but the high recency score (Entry 2: 'New high-temp resin...').
    
    # Since the ranking is printed to stdout in consciousness.py, we rely on the log
    # to confirm the correct prioritization. The expected top context is:
    expected_top_context_snippet = "New high-temp resin 'Janus-Resin-X' is now the standard"
    
    # To programmatically verify, we can check the final response content, which should
    # be based on the highest-ranked context.
    if expected_top_context_snippet in response:
        print("\n--- TEST SUCCESS ---")
        print("The response is based on the newer, medium-semantic-score context, confirming Recency Scoring successfully boosted its rank above the older, high-semantic-score context.")
    else:
        print("\n--- TEST FAILURE ---")
        print("The response did not contain the expected newer context. Check the RAG Ranking log above for details.")

if __name__ == "__main__":
    run_test()
