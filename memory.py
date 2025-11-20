import json
import os
from datetime import datetime

# File to store the conversation history
MEMORY_FILE = "conversation_history.json"

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

def add_message(role, content):
    """Adds a new message (user or Janus) to the history."""
    history = load_history()
    timestamp = datetime.now().isoformat()
    history.append({
        "timestamp": timestamp,
        "role": role,
        "content": content
    })
    save_history(history)

def get_recent_history(limit=10):
    """Retrieves the most recent messages for context."""
    history = load_history()
    # Return the last 'limit' messages
    return history[-limit:]

# Initialize the file if it doesn't exist
if not os.path.exists(MEMORY_FILE):
    save_history([])

if __name__ == "__main__":
    # Example usage for testing
    add_message("user", "Hello Janus, I am building a 3D printer.")
    add_message("janus", "A 3D printer? A fascinating exercise in controlled chaos.")
    print("Recent History:")
    print(get_recent_history(2))
