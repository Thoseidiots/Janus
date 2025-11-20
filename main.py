import consciousness
import memory
import os

def main():
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
