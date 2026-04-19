"""
Test Suite for AI Fault Detector

Comprehensive tests demonstrating fault detection capabilities for common AI-generated code issues.
"""

import unittest
from ai_fault_detector import AIFaultDetector, FaultSeverity, FaultCategory
from fault_monitor import JanusFaultIntegration, RealTimeFaultMonitor

class TestAIFaultDetector(unittest.TestCase):
    """Test cases for AI Fault Detector"""
    
    def setUp(self):
        self.detector = AIFaultDetector()
    
    def test_off_screen_positioning(self):
        """Test detection of elements positioned off-screen"""
        html = """
        <div style="position: fixed; left: -1000px; top: -500px;">
            Character element
        </div>
        """
        
        result = self.detector.analyze_code(html)
        
        # Should detect positioning issues
        positioning_faults = [f for f in result.fault_reports if f.category == FaultCategory.POSITIONING]
        self.assertGreater(len(positioning_faults), 0, "Should detect off-screen positioning")
        
        # Check severity
        for fault in positioning_faults:
            self.assertIn(fault.severity, [FaultSeverity.HIGH, FaultSeverity.CRITICAL])
    
    def test_broken_interactivity(self):
        """Test detection of broken interactivity issues"""
        css = """
        .character {
            pointer-events: none;
            cursor: pointer;
            opacity: 0;
        }
        """
        
        result = self.detector.analyze_code("", css)
        
        # Should detect interactivity issues
        interactivity_faults = [f for f in result.fault_reports if f.category == FaultCategory.INTERACTIVITY]
        self.assertGreater(len(interactivity_faults), 0, "Should detect interactivity issues")
        
        # Should have at least one critical fault
        critical_faults = [f for f in interactivity_faults if f.severity == FaultSeverity.CRITICAL]
        self.assertGreater(len(critical_faults), 0, "Should detect critical interactivity faults")
    
    def test_accessibility_violations(self):
        """Test detection of accessibility violations"""
        html = """
        <div onclick="doSomething()">
            <img src="character.png">
            <button>Click me</button>
        </div>
        """
        
        result = self.detector.analyze_code(html)
        
        # Should detect accessibility issues
        accessibility_faults = [f for f in result.fault_reports if f.category == FaultCategory.ACCESSIBILITY]
        self.assertGreater(len(accessibility_faults), 0, "Should detect accessibility violations")
    
    def test_javascript_security_issues(self):
        """Test detection of JavaScript security issues"""
        js = """
        eval(user_input);
        element.innerHTML = malicious_content;
        setTimeout("alert('test')", 0);
        """
        
        result = self.detector.analyze_code("", "", js)
        
        # Should detect JavaScript issues
        js_faults = [f for f in result.fault_reports if f.category == FaultCategory.JAVASCRIPT]
        self.assertGreater(len(js_faults), 0, "Should detect JavaScript security issues")
        
        # Should detect eval usage
        eval_faults = [f for f in js_faults if "eval" in f.title]
        self.assertGreater(len(eval_faults), 0, "Should detect eval() usage")
    
    def test_quality_score_calculation(self):
        """Test code quality score calculation"""
        # Clean code should have high score
        clean_html = """
        <div class="character-container">
            <img src="character.png" alt="Game character">
            <button aria-label="Interact with character">Interact</button>
        </div>
        """
        
        result = self.detector.analyze_code(clean_html)
        self.assertGreater(result.code_quality_score, 80, "Clean code should have high quality score")
        
        # Faulty code should have low score
        faulty_html = """
        <div onclick="eval('alert()')" style="position: fixed; left: -2000px; pointer-events: none;">
            <img src="character.png">
        </div>
        """
        
        result = self.detector.analyze_code(faulty_html)
        self.assertLess(result.code_quality_score, 50, "Faulty code should have low quality score")
    
    def test_recommendations_generation(self):
        """Test generation of actionable recommendations"""
        faulty_html = """
        <div style="position: fixed; left: -1000px;" onclick="doSomething()">
            <img src="character.png">
        </div>
        """
        
        result = self.detector.analyze_code(faulty_html)
        self.assertGreater(len(result.recommendations), 0, "Should generate recommendations")
        
        # Should have positioning recommendation
        positioning_rec = any("positioning" in rec.lower() for rec in result.recommendations)
        self.assertTrue(positioning_rec, "Should recommend positioning improvements")

class TestFaultMonitor(unittest.TestCase):
    """Test cases for Fault Monitor"""
    
    def setUp(self):
        self.monitor = RealTimeFaultMonitor(check_interval=0.1)  # Fast for testing
        self.integration = JanusFaultIntegration()
    
    def test_session_management(self):
        """Test monitoring session management"""
        # Start session
        session_id = self.monitor.start_monitoring("./test_project")
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.monitor.active_sessions)
        
        # Get session stats
        stats = self.monitor.get_session_stats(session_id)
        self.assertIsNotNone(stats)
        self.assertEqual(stats['session_id'], session_id)
        
        # Stop session
        session = self.monitor.stop_monitoring(session_id)
        self.assertIsNotNone(session)
        self.assertNotIn(session_id, self.monitor.active_sessions)
    
    def test_fault_prevention(self):
        """Test fault prevention system"""
        # Safe code should pass
        safe_code = """
        <div class="character">
            <img src="character.png" alt="Character">
        </div>
        """
        
        is_safe, issues = self.integration.validate_ai_output(safe_code, "html")
        self.assertTrue(is_safe, "Safe code should pass validation")
        self.assertEqual(len(issues), 0, "Safe code should have no issues")
        
        # Unsafe code should be blocked
        unsafe_code = """
        <div style="position: fixed; left: -2000px; pointer-events: none;" onclick="eval('test')">
            <img src="character.png">
        </div>
        """
        
        is_safe, issues = self.integration.validate_ai_output(unsafe_code, "html")
        self.assertFalse(is_safe, "Unsafe code should be blocked")
        self.assertGreater(len(issues), 0, "Unsafe code should have issues")

class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world AI code generation scenarios"""
    
    def setUp(self):
        self.detector = AIFaultDetector()
    
    def test_character_positioning_scenario(self):
        """Test scenario: AI positions character off-screen"""
        # Common AI mistake - character positioned outside viewport
        faulty_scenario = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        .game-container {
            background: black;
            width: 100vw;
            height: 100vh;
        }
        .character {
            position: absolute;
            left: -1500px;  # AI mistake - too far left
            top: -800px;    # AI mistake - too far up
            width: 100px;
            height: 100px;
            background: red;
        }
        </style>
        </head>
        <body>
        <div class="game-container">
            <div class="character" onclick="interact()">Character</div>
        </div>
        </body>
        </html>
        """
        
        result = self.detector.analyze_code(faulty_scenario)
        
        # Should detect positioning issues
        positioning_faults = [f for f in result.fault_reports if f.category == FaultCategory.POSITIONING]
        self.assertGreater(len(positioning_faults), 0, "Should detect character positioning issues")
        
        # Quality should be low
        self.assertLess(result.code_quality_score, 70, "Poor positioning should lower quality score")
    
    def test_dialogue_system_scenario(self):
        """Test scenario: AI creates non-clickable dialogue system"""
        # Common AI mistake - dialogue becomes unclickable
        faulty_dialogue = """
        <div class="dialogue-box" style="position: fixed; opacity: 0; pointer-events: auto;">
            <div class="dialogue-text">Hello!</div>
            <button onclick="continueDialogue()" style="opacity: 0;">Continue</button>
        </div>
        <style>
        .dialogue-box {
            z-index: 99999;  # Extremely high z-index
            background: rgba(0,0,0,0.8);
            color: black;     # Same as background - invisible text
        }
        </style>
        """
        
        result = self.detector.analyze_code(faulty_dialogue)
        
        # Should detect multiple issues
        self.assertGreater(len(result.fault_reports), 2, "Should detect multiple dialogue issues")
        
        # Should detect interactivity and styling issues
        categories = {f.category for f in result.fault_reports}
        self.assertIn(FaultCategory.INTERACTIVITY, categories)
        self.assertIn(FaultCategory.STYLING, categories)
    
    def test_responsive_design_scenario(self):
        """Test scenario: AI creates non-responsive design"""
        non_responsive = """
        <div style="width: 1920px; height: 1080px; position: fixed;">
            <div style="width: 500px; height: 500px; position: absolute; left: 1500px;">
                Character
            </div>
        </div>
        """
        
        result = self.detector.analyze_code(non_responsive)
        
        # Should detect responsive design issues
        positioning_faults = [f for f in result.fault_reports if "viewport" in f.description.lower() 
                            or "1920" in str(f.code_snippet or "")]
        self.assertGreater(len(positioning_faults), 0, "Should detect responsive design issues")

def run_demo():
    """Run demonstration of fault detection capabilities"""
    print("=== AI Fault Detector Demo ===\n")
    
    detector = AIFaultDetector()
    
    # Demo 1: Character positioning issue
    print("Demo 1: Character positioned off-screen")
    print("-" * 40)
    
    faulty_html = """
    <div class="game-character" style="position: fixed; left: -1000px; top: -500px;">
        Character sprite
    </div>
    """
    
    result = detector.analyze_code(faulty_html)
    print(f"Quality Score: {result.code_quality_score}/100")
    print(f"Faults Detected: {len(result.fault_reports)}")
    
    for fault in result.fault_reports:
        print(f"  [{fault.severity.value.upper()}] {fault.title}")
        print(f"    {fault.description}")
        if fault.suggested_fix:
            print(f"    Fix: {fault.suggested_fix}")
    
    print("\n" + "="*50 + "\n")
    
    # Demo 2: Dialogue system issues
    print("Demo 2: Non-clickable dialogue system")
    print("-" * 40)
    
    dialogue_html = """
    <div class="dialogue" style="opacity: 0; pointer-events: auto;">
        <button onclick="continue()" style="display: none; cursor: pointer;">Continue</button>
    </div>
    """
    
    result = detector.analyze_code(dialogue_html)
    print(f"Quality Score: {result.code_quality_score}/100")
    print(f"Faults Detected: {len(result.fault_reports)}")
    
    for fault in result.fault_reports:
        print(f"  [{fault.severity.value.upper()}] {fault.title}")
        print(f"    {fault.description}")
    
    print("\n" + "="*50 + "\n")
    
    # Demo 3: Integration with monitoring
    print("Demo 3: Real-time monitoring setup")
    print("-" * 40)
    
    integration = JanusFaultIntegration()
    integration.setup_janus_integration()
    
    session_id = integration.monitor.start_monitoring("./demo_project")
    print(f"Started monitoring session: {session_id}")
    
    # Test validation
    test_code = """
    <div style="position: fixed; left: -2000px;" onclick="eval('test')">
        Test element
    </div>
    """
    
    is_safe, issues = integration.validate_ai_output(test_code, "html")
    print(f"Code validation: {'PASSED' if is_safe else 'BLOCKED'}")
    if not is_safe:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Get session stats
    stats = integration.monitor.get_session_stats(session_id)
    if stats:
        print(f"Session stats: {stats['total_checks']} checks, {stats['faults_detected']} faults")
    
    integration.monitor.stop_monitoring(session_id)
    print("Monitoring stopped")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    # Run tests
    print("Running AI Fault Detector Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run demo
    print("\n" + "="*60)
    run_demo()
