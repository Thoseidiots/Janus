import json
import os
from datetime import datetime, timedelta

# --- Configuration ---
MEMORY_FILE = "conversation_history.json"
MKS_FILE = "meta_knowledge.json"

# --- Recency Scoring Function ---

RECENCY_HALF_LIFE_DAYS = 30 # Knowledge decays by half every 30 days

def calculate_recency_score(timestamp_str):
    """
    Calculates a recency score (0.0 to 1.0) based on how old the timestamp is.
    Uses an exponential decay model (half-life).
    """
    try:
        knowledge_date = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return 0.5 # Neutral score for invalid timestamps

    time_difference = datetime.now() - knowledge_date
    days_old = time_difference.total_seconds() / (60 * 60 * 24)
    
    # Exponential decay: score = 2^(-days_old / half_life)
    # The score will be between 0.0 and 1.0
    decay_factor = 0.5 ** (days_old / RECENCY_HALF_LIFE_DAYS)
    
    return max(0.0, min(1.0, decay_factor))

# --- Conversation History Functions ---

def load_history():
    """Loads the entire conversation history from the JSON file."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_history(history):
    """Saves the entire conversation history to the JSON file."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def add_message(role, content, feedback=None):
    """Adds a new message (user or Janus) to the history, with optional feedback."""
    history = load_history()
    timestamp = datetime.now().isoformat()
    message = {
        "timestamp": timestamp,
        "role": role,
        "content": content
    }
    if feedback:
        message["feedback"] = feedback
    history.append(message)
    save_history(history)

def get_recent_history(limit=10):
    """Retrieves the most recent messages for context."""
    history = load_history()
    # Return the last 'limit' messages
    return history[-limit:]

# --- Meta-Knowledge Store (MKS) Class ---

class MetaKnowledgeStore:
    def __init__(self, file_path=MKS_FILE):
        self.file_path = file_path
        self.data = self._load()

    def _load(self):
        """Loads the MKS data from the JSON file."""
        if not os.path.exists(self.file_path):
            return {"rules": [], "metrics": {}}
        with open(self.file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"rules": [], "metrics": {}}

    def _save(self):
        """Saves the MKS data to the JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_rules(self):
        """Returns the list of learned rules for DPA."""
        return self.data.get("rules", [])

    def add_rule(self, rule):
        """Adds a new rule to the MKS."""
        self.data["rules"].append(rule)
        self._save()

    def update_metrics(self, key, value):
        """Updates a performance metric."""
        self.data["metrics"][key] = value
        self._save()

    def get_metrics(self, key=None):
        """Returns all metrics or a specific one."""
        if key:
            return self.data["metrics"].get(key)
        return self.data["metrics"]

    def track_feedback(self, topic, feedback_type):
        """
        Tracks user feedback (Good/Bad) for a given topic.
        Returns True if a self-improvement trigger condition is met.
        """
        key = f"feedback_{topic}"
        current_count = self.data["metrics"].get(key, 0)

        if feedback_type.lower() == "bad":
            # Decrement for 'Bad' feedback
            current_count -= 1
        elif feedback_type.lower() == "good":
            # Increment for 'Good' feedback
            current_count += 1
        
        self.data["metrics"][key] = current_count
        self._save()

        # Self-improvement trigger: 3 consecutive 'Bad' ratings on a topic
        # (Simulated by a score of -3 or less)
        if current_count <= -3:
            # Reset the counter after triggering
            self.data["metrics"][key] = 0
            self._save()
            return True
        return False

# Initialize the history file if it doesn't exist
if not os.path.exists(MEMORY_FILE):
    save_history([])

# Initialize the MKS instance
mks = MetaKnowledgeStore()

if __name__ == "__main__":
    # Example usage for testing
    print("--- MKS Test ---")
    mks.track_feedback("art", "bad")
    mks.track_feedback("art", "bad")
    print(f"MKS Metrics after 2 'bad' on art: {mks.get_metrics('feedback_art')}")
    if mks.track_feedback("art", "bad"):
        print("Self-improvement trigger activated for 'art'!")
        mks.add_rule({"topic": "art", "action": "increase_sarcasm", "value": 0.2})
    print(f"MKS Rules: {mks.get_rules()}")
    print(f"MKS Metrics after trigger: {mks.get_metrics('feedback_art')}")
