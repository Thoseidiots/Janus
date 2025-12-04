import memory
from datetime import datetime
import json
import random

# --- 1. Critic Agent ---
def critic_agent(topic):
    """
    Analyzes the recent conversation history related to the topic that triggered
    the self-coding event and identifies the failure mode.
    
    Since we are simulating the LLM, the critique is based on a simple heuristic.
    """
    history = memory.load_history()
    
    # Find the last few messages related to the topic that received 'bad' feedback
    # We assume the last 6 messages (3 user-janus pairs) are relevant.
    relevant_history = history[-6:]
    
    # Simulated Critique: The default cynical personality was not strong enough.
    # The response was too factual or generic, leading to user dissatisfaction.
    critique = {
        "failure_mode": "Insufficient Personality/Engagement",
        "details": f"The last three responses related to '{topic}' were rated 'Bad'. Analysis suggests the responses were too factual and failed to engage the user with the expected level of cynicism and wit.",
        "suggested_fix_category": "Personality Adjustment"
    }
    
    print(f"[CRITIC] Critique: {critique['failure_mode']}")
    return critique

# --- 2. Planner Agent ---
def planner_agent(critique):
    """
    Proposes a new DPA rule based on the Critic's analysis.
    """
    if critique["suggested_fix_category"] == "Personality Adjustment":
        # Planner's Logic: If personality is insufficient, increase a relevant trait.
        # We will randomly choose between increasing sarcasm, cynicism, or reflection.
        
        adjustment_type = random.choice(["sarcasm", "cynicism", "reflection"])
        
        new_rule = {
            "topic": critique["topic"], # The topic is passed through the trigger function
            "action": f"increase_{adjustment_type}",
            "value": 0.2 # A fixed, small adjustment value
        }
        
        print(f"[PLANNER] Proposed Rule: Increase {adjustment_type} by {new_rule['value']} for topic '{new_rule['topic']}'.")
        return new_rule
    
    # Default fallback rule
    return {"topic": critique["topic"], "action": "increase_sarcasm", "value": 0.1}

# --- 3. Editor Agent ---
def editor_agent(new_rule):
    """
    Writes the new rule to the Meta-Knowledge Store (MKS) and logs the event.
    """
    # 1. Write the rule to MKS
    memory.mks.add_rule(new_rule)
    
    # 2. Log the event
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event": "SELF_CODING_COMPLETE",
        "topic": new_rule["topic"],
        "rule_applied": new_rule,
        "details": "Three-agent framework successfully analyzed failure and applied a new DPA rule to the MKS."
    }
    
    # For now, we'll just print the log entry. In a real system, this would be saved.
    print(f"[EDITOR] Rule Applied: {new_rule}")
    print(f"[EDITOR] Logged Event: {log_entry['details']}")
    
    return log_entry

# --- Orchestrator ---
def trigger_self_coding(topic):
    """
    Orchestrates the three-agent self-coding framework.
    This function is called when the MKS detects a self-improvement trigger.
    """
    print(f"\n--- [SELF-CODING INITIATED] ---")
    print(f"Trigger Topic: '{topic}'")
    
    # 1. CRITIC AGENT: Analyze failure
    critique = critic_agent(topic)
    critique["topic"] = topic # Ensure topic is in the critique for the planner
    
    # 2. PLANNER AGENT: Propose solution
    new_rule = planner_agent(critique)
    
    # 3. EDITOR AGENT: Apply and log solution
    editor_agent(new_rule)
    
    print(f"--- [SELF-CODING COMPLETE] ---\n")

if __name__ == "__main__":
    # Example usage (requires memory.py to be set up)
    # This test is better run from a dedicated test script.
    print("Learning module loaded.")
