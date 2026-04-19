"""
Demo: AI Fault Detection System for Common Web Development Issues

This demonstrates how the fault detection system catches common AI-generated code problems
like characters moving off-screen, broken dialogue systems, and accessibility issues.
"""

from janus_fault_integration import JanusAIGuard

def demo_character_positioning_issues():
    """Demo: Character positioned off-screen (common AI mistake)"""
    print("=" * 60)
    print("DEMO 1: Character Positioned Off-Screen")
    print("=" * 60)
    
    ai_guard = JanusAIGuard()
    
    # Common AI mistake: character positioned way outside viewport
    faulty_character = """
    <div class="game-character" style="position: fixed; left: -1500px; top: -800px; pointer-events: none;">
        <img src="character.png" alt="Character">
        <div class="character-dialogue">Hello!</div>
    </div>
    <style>
    .game-character {
        z-index: 99999;
        opacity: 0.5;
        transform: translateX(-2000px);
    }
    </style>
    """
    
    result = ai_guard.validate_ai_generation(
        faulty_character, 
        "html", 
        context={"is_game_character": True}
    )
    
    print(f"Quality Score: {result['quality_score']}/100")
    print(f"Faults Detected: {result['total_faults']}")
    print(f"Critical Faults: {result['critical_faults']}")
    print(f"Code Allowed: {result['is_allowed']}")
    
    if result['fault_reports']:
        print("\nDetected Issues:")
        for fault in result['fault_reports']:
            print(f"  [{fault['severity'].upper()}] {fault['title']}")
            print(f"    {fault['description']}")
            if fault['suggested_fix']:
                print(f"    Fix: {fault['suggested_fix']}")
    
    if result['suggestions']:
        print("\nSuggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    
    return result

def demo_dialogue_system_issues():
    """Demo: Non-clickable dialogue system (common AI mistake)"""
    print("\n" + "=" * 60)
    print("DEMO 2: Broken Dialogue System")
    print("=" * 60)
    
    ai_guard = JanusAIGuard()
    
    # Common AI mistake: dialogue becomes unclickable or invisible
    faulty_dialogue = """
    <div class="dialogue-system" style="position: fixed; opacity: 0; pointer-events: auto;">
        <div class="dialogue-box" style="background: black; color: black;">
            <p class="dialogue-text">Welcome to the game!</p>
            <button onclick="continueDialogue()" style="display: none; cursor: pointer;">Continue</button>
        </div>
    </div>
    <style>
    .dialogue-system {
        z-index: 999999;
        pointer-events: none;
    }
    .dialogue-box {
        opacity: 0;
        cursor: pointer;
    }
    </style>
    <script>
    function continueDialogue() {
        eval('alert("Continuing...")');
        document.querySelector('.dialogue-text').innerHTML = user_input;
    }
    </script>
    """
    
    result = ai_guard.validate_ai_generation(
        faulty_dialogue, 
        "html", 
        context={"is_dialogue_system": True}
    )
    
    print(f"Quality Score: {result['quality_score']}/100")
    print(f"Faults Detected: {result['total_faults']}")
    print(f"Critical Faults: {result['critical_faults']}")
    print(f"Code Allowed: {result['is_allowed']}")
    
    if result['fault_reports']:
        print("\nDetected Issues:")
        for fault in result['fault_reports']:
            print(f"  [{fault['severity'].upper()}] {fault['title']}")
            print(f"    {fault['description']}")
            if fault['suggested_fix']:
                print(f"    Fix: {fault['suggested_fix']}")
    
    if result['suggestions']:
        print("\nSuggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    
    return result

def demo_good_code_example():
    """Demo: Well-structured code that passes validation"""
    print("\n" + "=" * 60)
    print("DEMO 3: Good Code Example")
    print("=" * 60)
    
    ai_guard = JanusAIGuard()
    
    # Good code that should pass validation
    good_code = """
    <div class="game-container" style="background: black; width: 100vw; height: 100vh;">
        <div class="character" style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);">
            <img src="character.png" alt="Game character">
        </div>
        <div class="dialogue-system" style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);">
            <div class="dialogue-box" style="background: rgba(0,0,0,0.8); color: white; padding: 20px;">
                <p class="dialogue-text">Hello! Welcome to the game.</p>
                <button onclick="continueDialogue()" aria-label="Continue dialogue">Continue</button>
            </div>
        </div>
    </div>
    <style>
    .character {
        width: 100px;
        height: 100px;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .character:hover {
        transform: translate(-50%, -50%) scale(1.1);
    }
    .dialogue-system {
        z-index: 100;
        opacity: 1;
        pointer-events: auto;
    }
    </style>
    <script>
    function continueDialogue() {
        const dialogueText = document.querySelector('.dialogue-text');
        dialogueText.textContent = 'Great to meet you!';
        
        // Safe DOM manipulation
        const newElement = document.createElement('div');
        newElement.textContent = 'Dialogue continued';
        document.querySelector('.dialogue-box').appendChild(newElement);
    }
    </script>
    """
    
    result = ai_guard.validate_ai_generation(
        good_code, 
        "html", 
        context={"is_game_character": True, "is_dialogue_system": True}
    )
    
    print(f"Quality Score: {result['quality_score']}/100")
    print(f"Faults Detected: {result['total_faults']}")
    print(f"Critical Faults: {result['critical_faults']}")
    print(f"Code Allowed: {result['is_allowed']}")
    
    if result['fault_reports']:
        print("\nDetected Issues:")
        for fault in result['fault_reports']:
            print(f"  [{fault['severity'].upper()}] {fault['title']}")
            print(f"    {fault['description']}")
    
    if result['suggestions']:
        print("\nSuggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    
    return result

def demo_real_time_monitoring():
    """Demo: Real-time monitoring capabilities"""
    print("\n" + "=" * 60)
    print("DEMO 4: Real-Time Monitoring")
    print("=" * 60)
    
    ai_guard = JanusAIGuard()
    
    # Start monitoring session
    session_id = ai_guard.start_monitoring_session("./demo_project", "ai_code_demo")
    print(f"Started monitoring session: {session_id}")
    
    # Simulate multiple code generations
    test_cases = [
        ("Good code", "<div class='character'>Character</div>"),
        ("Bad positioning", "<div style='position: fixed; left: -1000px;'>Off-screen</div>"),
        ("Security issue", "<div onclick='eval(\"alert()\")'>Click me</div>"),
        ("Accessibility issue", "<div onclick='doSomething()'>Interactive div</div>")
    ]
    
    for name, code in test_cases:
        print(f"\nTesting: {name}")
        result = ai_guard.validate_ai_generation(code, "html")
        print(f"  Score: {result['quality_score']}, Allowed: {result['is_allowed']}")
    
    # Get quality metrics
    metrics = ai_guard.get_quality_metrics()
    print(f"\nQuality Metrics:")
    print(f"  Total Generations: {metrics['total_code_generations']}")
    print(f"  Blocked Generations: {metrics['blocked_generations']}")
    print(f"  Block Rate: {metrics['block_rate']:.1f}%")
    print(f"  Average Quality: {metrics['average_quality_score']:.1f}/100")
    
    # Generate quality report
    print(f"\n{ai_guard.generate_quality_report()}")
    
    # Stop monitoring
    ai_guard.stop_monitoring_session(session_id)
    print(f"\nStopped monitoring session: {session_id}")

def main():
    """Run all demos"""
    print("AI Fault Detection System Demo")
    print("This system catches common AI-generated code issues before they affect users")
    print()
    
    # Run all demos
    demo_character_positioning_issues()
    demo_dialogue_system_issues()
    demo_good_code_example()
    demo_real_time_monitoring()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Benefits of the AI Fault Detection System:")
    print("1. Prevents characters from moving off-screen")
    print("2. Ensures dialogue systems remain clickable and visible")
    print("3. Blocks security vulnerabilities like eval() usage")
    print("4. Enforces accessibility standards")
    print("5. Provides real-time quality monitoring")
    print("6. Offers actionable suggestions for code improvement")
    print("7. Integrates seamlessly with existing Janus AI systems")

if __name__ == "__main__":
    main()
