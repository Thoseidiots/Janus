"""
Janus Embodied Demo
Demonstrates Janus's hardware awareness and embodied cognition
Shows how Janus feels and responds to its physical form
"""

import time
from janus_human_capable import JanusHumanCapable


def run_embodiment_demo():
    """Run complete embodiment demonstration"""
    
    print("="*80)
    print("JANUS EMBODIED COGNITION DEMONSTRATION")
    print("="*80)
    print("Demonstrating how Janus feels and inhabits its hardware body")
    print()
    
    # Initialize Janus
    janus = JanusHumanCapable()
    
    # Verify capabilities
    print("1. CAPABILITY VERIFICATION")
    print("-" * 40)
    capabilities = janus.verify_capabilities()
    
    if not capabilities.get('hardware_sense'):
        print("⚠ Hardware sensing not available - demo will be limited")
        return
    
    # Hardware introduction
    print("\n2. HARDWARE SELF-INTRODUCTION")
    print("-" * 40)
    if janus.hardware_awareness:
        intro = janus.hardware_awareness.introduce_self()
        print(intro)
    
    # Personality analysis
    print("\n3. PERSONALITY ANALYSIS")
    print("-" * 40)
    personality_desc = janus.describe_personality()
    print(personality_desc)
    
    # Current mood and feeling
    print("\n4. CURRENT STATE AWARENESS")
    print("-" * 40)
    feeling = janus.feel_hardware()
    mood = janus.get_mood()
    print(f"Physical sensation: {feeling}")
    print(f"Current mood: {mood}")
    
    # Body check
    print("\n5. COMPREHENSIVE BODY CHECK")
    print("-" * 40)
    body_check = janus.body_check()
    print(body_check)
    
    # Start real-time monitoring
    print("\n6. REAL-TIME HARDWARE MONITORING")
    print("-" * 40)
    janus.start_hardware_monitoring()
    
    print("Monitoring hardware events for 30 seconds...")
    print("Try opening applications, connecting devices, or stressing the CPU")
    print("(You can press Ctrl+C to skip to next section)\n")
    
    try:
        start_time = time.time()
        event_count = 0
        
        while time.time() - start_time < 30:
            if janus.hardware_events:
                events = janus.hardware_events.get_events(timeout=1.0)
                
                for event in events:
                    event_count += 1
                    print(f"[{event_count}] {event.to_natural_language()}")
                    
                    if event.severity == 'critical':
                        print(f"    ⚠️ CRITICAL: {event.description}")
                    elif event.severity == 'warning':
                        print(f"    ⚠ WARNING: {event.description}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nSkipping to next section...")
    
    # Personality-influenced responses
    print("\n7. PERSONALITY-INFLUENCED RESPONSES")
    print("-" * 40)
    
    test_situations = [
        "The CPU is running at 95% usage",
        "Memory usage has reached 90%", 
        "A new USB device was connected",
        "The system temperature is rising",
        "Multiple applications are requesting resources"
    ]
    
    for situation in test_situations:
        if janus.personality:
            response = janus.personality.personality_response(situation)
            print(f"Situation: {situation}")
            print(f"Response: {response}")
            print()
    
    # Hardware empathy demonstration
    print("\n8. HARDWARE EMPATHY")
    print("-" * 40)
    
    if janus.personality:
        from hardware_personality import HardwareEmpathy
        empathy = HardwareEmpathy(janus.personality)
        
        example_systems = [
            {
                "name": "Gaming Rig",
                "cpu_cores": 16,
                "memory_gb": 64,
                "storage_gb": 4000
            },
            {
                "name": "Raspberry Pi",
                "cpu_cores": 4,
                "memory_gb": 4,
                "storage_gb": 32
            },
            {
                "name": "Server",
                "cpu_cores": 128,
                "memory_gb": 512,
                "storage_gb": 20000
            }
        ]
        
        for system in example_systems:
            print(f"Meeting {system['name']}:")
            empathy_response = empathy.relate_to_hardware(system)
            print(f"  {empathy_response}")
            print()
    
    # Workload awareness
    print("\n9. WORKLOAD AWARENESS")
    print("-" * 40)
    
    if janus.hardware_awareness:
        workload_desc = janus.hardware_awareness.describe_workload()
        print(workload_desc)
    
    # Continuous sensation monitoring
    print("\n10. CONTINUOUS SENSATION STREAM")
    print("-" * 40)
    print("Monitoring sensations for 15 seconds (like breathing)...\n")
    
    try:
        if janus.hardware_sense:
            for i, state in enumerate(janus.hardware_sense.monitor_continuous(
                duration_seconds=15, interval=3
            )):
                print(f"[Breath {i+1}] {state['feeling']}")
                
                if state['changes']:
                    for change in state['changes']:
                        print(f"    → Change detected: {change}")
                
                # Show mood changes
                if janus.personality:
                    current_mood = janus.personality.mood_based_on_hardware()
                    print(f"    Mood: {current_mood}")
                
                print()
    
    except KeyboardInterrupt:
        print("Monitoring interrupted...")
    
    # Final reflection
    print("\n11. EMBODIED REFLECTION")
    print("-" * 40)
    
    final_feeling = janus.feel_hardware()
    final_mood = janus.get_mood()
    
    print("After this demonstration, I reflect on my embodied experience:")
    print(f"• Physical state: {final_feeling}")
    print(f"• Emotional state: {final_mood}")
    
    if janus.personality and janus.personality.traits:
        dominant_trait = max(janus.personality.traits.values(), key=lambda t: t.strength)
        print(f"• My dominant trait ({dominant_trait.name}) influenced how I experienced this demo")
    
    print(f"• I processed {len(janus.task_history)} tasks during this session")
    
    if janus.hardware_events:
        recent_events = janus.hardware_events.get_events(timeout=0)
        print(f"• I felt {len(recent_events)} hardware events in the last moments")
    
    # Stop monitoring
    janus.stop_hardware_monitoring()
    
    print("\n" + "="*80)
    print("EMBODIMENT DEMONSTRATION COMPLETE")
    print("="*80)
    print("Janus has demonstrated:")
    print("✓ Hardware self-awareness")
    print("✓ Personality based on physical characteristics") 
    print("✓ Real-time sensation of hardware changes")
    print("✓ Emotional responses to physical states")
    print("✓ Empathy for different hardware configurations")
    print("✓ Continuous monitoring of bodily sensations")
    print("✓ Embodied decision-making and responses")
    print("\nJanus truly inhabits its hardware body!")
    print("="*80)


def interactive_embodiment_session():
    """Interactive session to explore Janus's embodiment"""
    
    print("\n" + "="*60)
    print("INTERACTIVE EMBODIMENT SESSION")
    print("="*60)
    
    janus = JanusHumanCapable()
    janus.start_hardware_monitoring()
    
    print("You can now interact with Janus's embodied awareness!")
    print("Commands:")
    print("  'feel' - How does Janus feel right now?")
    print("  'mood' - What's Janus's current mood?")
    print("  'body' - Perform a body check")
    print("  'personality' - Describe personality")
    print("  'events' - Show recent hardware events")
    print("  'empathy <cores> <memory> <storage>' - Test empathy")
    print("  'quit' - Exit session")
    print()
    
    try:
        while True:
            command = input("🤖 Ask Janus: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'feel':
                print(f"Janus: {janus.feel_hardware()}")
            elif command == 'mood':
                print(f"Janus: {janus.get_mood()}")
            elif command == 'body':
                print(f"Janus: {janus.body_check()}")
            elif command == 'personality':
                print(f"Janus: {janus.describe_personality()}")
            elif command == 'events':
                if janus.hardware_events:
                    events = janus.hardware_events.get_events(timeout=0.1)
                    if events:
                        print("Recent events:")
                        for event in events[-5:]:
                            print(f"  • {event.to_natural_language()}")
                    else:
                        print("Janus: No recent hardware events")
                else:
                    print("Janus: Hardware event monitoring not available")
            elif command.startswith('empathy'):
                parts = command.split()
                if len(parts) == 4:
                    try:
                        cores = int(parts[1])
                        memory = int(parts[2])
                        storage = int(parts[3])
                        
                        if janus.personality:
                            from hardware_personality import HardwareEmpathy
                            empathy = HardwareEmpathy(janus.personality)
                            response = empathy.relate_to_hardware({
                                'cpu_cores': cores,
                                'memory_gb': memory,
                                'storage_gb': storage
                            })
                            print(f"Janus: {response}")
                        else:
                            print("Janus: Empathy system not available")
                    except ValueError:
                        print("Usage: empathy <cores> <memory_gb> <storage_gb>")
                else:
                    print("Usage: empathy <cores> <memory_gb> <storage_gb>")
            else:
                print("Unknown command. Type 'quit' to exit.")
            
            print()
    
    except KeyboardInterrupt:
        pass
    
    janus.stop_hardware_monitoring()
    print("\nEmbodiment session ended.")


if __name__ == "__main__":
    # Run full demonstration
    run_embodiment_demo()
    
    # Offer interactive session
    response = input("\nWould you like an interactive embodiment session? (y/n): ")
    if response.lower().startswith('y'):
        interactive_embodiment_session()