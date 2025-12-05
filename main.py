import consciousness
import memory
import os

def load_env_vars(env_path=".env"):
    """Custom function to load environment variables from a .env file."""
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        print(f"[CONFIG] Loaded environment variables from {env_path}")
    except FileNotFoundError:
        print(f"[CONFIG] Warning: {env_path} not found. Proceeding without it.")
    except Exception as e:
        print(f"[CONFIG] Error loading {env_path}: {e}")

def main():
    # Load environment variables from the .env file
    load_env_vars("upload/.env") # Use the path where the user uploaded the file
    
    print("--- Janus AI Companion (v2 - Fine-Tuned Personality) ---")
    print("Janus: System online. The world is as noisy as ever. What do you want?")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Clear the memory file at the start of a new session for clean testing
    # In a real app, you would not do this.
    if os.path.exists(memory.MEMORY_FILE):
        os.remove(memory.MEMORY_FILE)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nJanus: Fine. Go find your own entertainment. I'll be here.")
                break
            
            if not user_input.strip():
                continue
            
            # Generate response using the consciousness module
            janus_response = consciousness.generate_response(user_input)
            
            print(f"\nJanus: {janus_response}")
            
        except EOFError:
            print("\nJanus: Fine. Go find your own entertainment. I'll be here.")
            break
        except Exception as e:
            print(f"\n[ERROR] Janus experienced a glitch: {e}")
            break

if __name__ == "__main__":
    main()
