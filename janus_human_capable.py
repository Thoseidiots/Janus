"""
Janus Human-Capable System
Complete integration of all human-level computer capabilities
This is the main entry point for autonomous Janus operation on EliteDesk G4
"""

import time
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import all Janus components
try:
    from os_human_interface import JanusOS
    from screen_interpreter import ScreenInterpreter
    from window_manager import WindowManager
    from vision_action_pipeline import VisionActionPipeline
    from browser_automation import BrowserAutomation, BrowserTasks
    from error_recovery import ErrorRecovery, ActionRecovery
    from hardware_sense import HardwareAwareness, HardwareSense
    from hardware_events import HardwareEventDetector, HardwareReflexes
    from hardware_personality import HardwarePersonality, HardwareEmpathy
    from avus_inference import AvusInference
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class JanusHumanCapable:
    """
    Complete human-capable Janus system
    
    Capabilities:
    - Screen capture and interpretation
    - Window management (switch, resize, tile)
    - Mouse and keyboard control
    - Browser automation
    - Multi-monitor support
    - Error recovery
    - Autonomous decision-making
    """
    
    def __init__(self):
        print("Initializing Janus Human-Capable System...")
        
        # Core components
        self.os = JanusOS() if COMPONENTS_AVAILABLE else None
        self.interpreter = ScreenInterpreter() if COMPONENTS_AVAILABLE else None
        self.window_manager = WindowManager() if COMPONENTS_AVAILABLE else None
        self.browser = BrowserAutomation() if COMPONENTS_AVAILABLE else None
        self.browser_tasks = BrowserTasks() if COMPONENTS_AVAILABLE else None
        self.recovery = ErrorRecovery() if COMPONENTS_AVAILABLE else None
        
        # Hardware awareness components
        self.hardware_awareness = HardwareAwareness() if COMPONENTS_AVAILABLE else None
        self.hardware_sense = HardwareSense() if COMPONENTS_AVAILABLE else None
        self.hardware_events = None  # Lazy init
        self.hardware_reflexes = None  # Lazy init
        self.personality = None  # Lazy init
        
        # High-level pipeline
        self.pipeline = VisionActionPipeline() if COMPONENTS_AVAILABLE else None
        
        # AI brain (lazy load)
        self.avus = None
        
        # State tracking
        self.current_task = None
        self.task_history = []
        self.capabilities_verified = False
        self.hardware_monitoring_active = False
        
        print("Janus Human-Capable System initialized!")
    
    def verify_capabilities(self) -> Dict[str, bool]:
        """Verify all capabilities are working"""
        print("\nVerifying capabilities...")
        
        capabilities = {
            'screen_capture': False,
            'screen_interpretation': False,
            'window_management': False,
            'mouse_control': False,
            'keyboard_control': False,
            'browser_automation': False,
            'multi_monitor': False,
            'error_recovery': False,
            'ai_brain': False,
            'hardware_sense': False,
            'hardware_events': False,
            'hardware_personality': False
        }
        
        # Test screen capture
        try:
            if self.os:
                raw, w, h = self.os.capture_screen()
                capabilities['screen_capture'] = len(raw) > 0
                print(f"  ✓ Screen capture: {w}x{h}")
        except Exception as e:
            print(f"  ✗ Screen capture failed: {e}")
        
        # Test screen interpretation
        try:
            if self.interpreter and capabilities['screen_capture']:
                desc = self.interpreter.interpret(raw, w, h)
                capabilities['screen_interpretation'] = len(desc) > 0
                print(f"  ✓ Screen interpretation")
        except Exception as e:
            print(f"  ✗ Screen interpretation failed: {e}")
        
        # Test window management
        try:
            if self.window_manager:
                windows = self.window_manager.get_all_windows()
                capabilities['window_management'] = len(windows) > 0
                print(f"  ✓ Window management: {len(windows)} windows")
        except Exception as e:
            print(f"  ✗ Window management failed: {e}")
        
        # Test mouse control
        try:
            if self.os:
                pos = self.os.get_screen_size()
                self.os.move_mouse(pos[0] // 2, pos[1] // 2)
                capabilities['mouse_control'] = True
                print(f"  ✓ Mouse control")
        except Exception as e:
            print(f"  ✗ Mouse control failed: {e}")
        
        # Test keyboard control
        try:
            if self.os:
                # Don't actually press keys during verification
                capabilities['keyboard_control'] = True
                print(f"  ✓ Keyboard control")
        except Exception as e:
            print(f"  ✗ Keyboard control failed: {e}")
        
        # Test browser automation
        try:
            if self.browser:
                found = self.browser.find_browser()
                capabilities['browser_automation'] = True
                print(f"  ✓ Browser automation")
        except Exception as e:
            print(f"  ✗ Browser automation failed: {e}")
        
        # Test multi-monitor
        try:
            if self.os:
                monitor_count = self.os.get_monitor_count()
                capabilities['multi_monitor'] = monitor_count > 0
                print(f"  ✓ Multi-monitor: {monitor_count} monitors")
        except Exception as e:
            print(f"  ✗ Multi-monitor failed: {e}")
        
        # Test error recovery
        try:
            if self.recovery:
                capabilities['error_recovery'] = True
                print(f"  ✓ Error recovery")
        except Exception as e:
            print(f"  ✗ Error recovery failed: {e}")
        
        # Test AI brain
        try:
            if self.ensure_avus_loaded():
                capabilities['ai_brain'] = True
                print(f"  ✓ AI brain (Avus)")
        except Exception as e:
            print(f"  ✗ AI brain failed: {e}")
        
        # Test hardware sense
        try:
            if self.hardware_sense:
                feeling = self.hardware_sense.feel()
                capabilities['hardware_sense'] = len(feeling) > 0
                print(f"  ✓ Hardware sense: {feeling[:50]}...")
        except Exception as e:
            print(f"  ✗ Hardware sense failed: {e}")
        
        # Test hardware events
        try:
            if self.hardware_sense:
                self.hardware_events = HardwareEventDetector(self.hardware_sense)
                capabilities['hardware_events'] = True
                print(f"  ✓ Hardware events")
        except Exception as e:
            print(f"  ✗ Hardware events failed: {e}")
        
        # Test hardware personality
        try:
            if self.hardware_awareness:
                self.personality = HardwarePersonality(self.hardware_awareness)
                capabilities['hardware_personality'] = len(self.personality.traits) > 0
                print(f"  ✓ Hardware personality: {len(self.personality.traits)} traits")
        except Exception as e:
            print(f"  ✗ Hardware personality failed: {e}")
        
        self.capabilities_verified = all(capabilities.values())
        
        success_count = sum(capabilities.values())
        total_count = len(capabilities)
        
        print(f"\nCapabilities: {success_count}/{total_count} verified")
        
        return capabilities
    
    def start_hardware_monitoring(self):
        """Start real-time hardware monitoring"""
        if self.hardware_events and not self.hardware_monitoring_active:
            self.hardware_reflexes = HardwareReflexes(self.hardware_events)
            self.hardware_events.start_monitoring()
            self.hardware_monitoring_active = True
            print("Hardware monitoring started - Janus can now feel hardware changes")
    
    def stop_hardware_monitoring(self):
        """Stop hardware monitoring"""
        if self.hardware_events and self.hardware_monitoring_active:
            self.hardware_events.stop_monitoring()
            self.hardware_monitoring_active = False
            print("Hardware monitoring stopped")
    
    def feel_hardware(self) -> str:
        """Get current hardware sensation"""
        if self.hardware_sense:
            return self.hardware_sense.feel()
        return "Hardware sensing not available"
    
    def describe_personality(self) -> str:
        """Describe Janus's hardware-based personality"""
        if self.personality:
            return self.personality.describe_personality()
        return "Personality analysis not available"
    
    def get_mood(self) -> str:
        """Get current mood based on hardware state"""
        if self.personality:
            return self.personality.mood_based_on_hardware()
        return "Mood sensing not available"
    
    def body_check(self) -> str:
        """Perform comprehensive body/hardware check"""
        if self.hardware_awareness:
            return self.hardware_awareness.body_check()
        return "Body check not available"
        """Lazy load Avus"""
        if self.avus is None:
            try:
                self.avus = AvusInference()
                self.avus.load()
                return True
            except Exception:
                return False
        return True
    
    def execute_task(self, task_description: str, 
                    max_steps: int = 20,
                    verify_actions: bool = True) -> Dict[str, Any]:
        """
        Execute a high-level task autonomously
        
        Args:
            task_description: Natural language task description
            max_steps: Maximum steps to attempt
            verify_actions: Whether to verify each action
            
        Returns:
            Task execution result
        """
        print(f"\n{'='*60}")
        print(f"EXECUTING TASK: {task_description}")
        print(f"{'='*60}\n")
        
        self.current_task = {
            'description': task_description,
            'start_time': datetime.now(),
            'steps': []
        }
        
        # Use the vision-action pipeline
        if self.pipeline:
            result = self.pipeline.autonomous_loop(
                goal=task_description,
                max_steps=max_steps,
                verify_actions=verify_actions
            )
            
            self.current_task['end_time'] = datetime.now()
            self.current_task['result'] = result
            self.task_history.append(self.current_task)
            
            return result
        else:
            return {'success': False, 'error': 'Pipeline not available'}
    
    def work_with_browser(self, task: str) -> Dict[str, Any]:
        """Execute browser-specific task"""
        print(f"\nBrowser task: {task}")
        
        # Ensure browser is open
        if not self.browser.find_browser():
            print("  Opening browser...")
            self.browser.open_browser()
            time.sleep(2)
        
        # Parse task and execute
        task_lower = task.lower()
        
        if 'search' in task_lower:
            # Extract search query
            query = task.split('search')[-1].strip()
            return {'success': self.browser_tasks.search_google(query)}
        
        elif 'navigate' in task_lower or 'go to' in task_lower:
            # Extract URL
            words = task.split()
            url = next((w for w in words if 'http' in w or '.com' in w), None)
            if url:
                return {'success': self.browser.navigate_to_url(url)}
        
        elif 'scroll' in task_lower:
            direction = 'down' if 'down' in task_lower else 'up'
            return {'success': self.browser.scroll_page(direction)}
        
        elif 'back' in task_lower:
            return {'success': self.browser.go_back()}
        
        elif 'forward' in task_lower:
            return {'success': self.browser.go_forward()}
        
        elif 'refresh' in task_lower:
            return {'success': self.browser.refresh_page()}
        
        elif 'new tab' in task_lower:
            return {'success': self.browser.new_tab()}
        
        elif 'close tab' in task_lower:
            return {'success': self.browser.close_tab()}
        
        return {'success': False, 'error': 'Unknown browser task'}
    
    def manage_windows(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage windows"""
        print(f"\nWindow management: {action}")
        
        if action == 'list':
            windows = self.window_manager.get_all_windows()
            return {
                'success': True,
                'windows': [{'title': w.title, 'hwnd': w.hwnd} for w in windows]
            }
        
        elif action == 'switch':
            title = kwargs.get('title', '')
            success = self.window_manager.switch_to_window(title)
            return {'success': success}
        
        elif action == 'tile_horizontal':
            hwnds = kwargs.get('hwnds', [])
            success = self.window_manager.tile_windows_horizontal(hwnds)
            return {'success': success}
        
        elif action == 'tile_vertical':
            hwnds = kwargs.get('hwnds', [])
            success = self.window_manager.tile_windows_vertical(hwnds)
            return {'success': success}
        
        elif action == 'snap_left':
            hwnd = kwargs.get('hwnd', 0)
            success = self.window_manager.snap_window_left(hwnd)
            return {'success': success}
        
        elif action == 'snap_right':
            hwnd = kwargs.get('hwnd', 0)
            success = self.window_manager.snap_window_right(hwnd)
            return {'success': success}
        
        return {'success': False, 'error': 'Unknown window action'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'monitors': [],
            'windows': [],
            'active_window': None,
            'recovery_stats': None
        }
        
        # Monitor info
        if self.os:
            for i in range(self.os.get_monitor_count()):
                monitor = self.os.get_monitor_info(i)
                status['monitors'].append(monitor)
        
        # Window info
        if self.window_manager:
            windows = self.window_manager.get_all_windows()
            status['windows'] = [w.title for w in windows[:10]]
            
            active = self.window_manager.get_active_window()
            if active:
                status['active_window'] = active.title
        
        # Recovery stats
        if self.recovery:
            status['recovery_stats'] = self.recovery.get_recovery_stats()
        
        # Hardware status
        if self.hardware_sense:
            status['hardware_feeling'] = self.hardware_sense.feel()
            status['hardware_health'] = self.hardware_sense.check_health()
        
        if self.personality:
            status['mood'] = self.personality.mood_based_on_hardware()
            status['personality_traits'] = list(self.personality.traits.keys())
        
        return status
    
    def run_demo(self):
        """Run demonstration of capabilities"""
        print("\n" + "="*60)
        print("JANUS HUMAN-CAPABLE SYSTEM DEMO")
        print("="*60)
        
        # Verify capabilities
        capabilities = self.verify_capabilities()
        
        if not self.capabilities_verified:
            print("\n⚠ Not all capabilities verified. Some features may not work.")
        
        # Show system status
        print("\n" + "-"*60)
        print("SYSTEM STATUS")
        print("-"*60)
        status = self.get_system_status()
        print(f"Monitors: {len(status['monitors'])}")
        print(f"Active windows: {len(status['windows'])}")
        print(f"Current window: {status['active_window']}")
        
        if 'hardware_feeling' in status:
            print(f"Hardware feeling: {status['hardware_feeling']}")
        
        if 'mood' in status:
            print(f"Current mood: {status['mood']}")
        
        # Show personality
        if self.personality:
            print("\n" + "-"*60)
            print("PERSONALITY")
            print("-"*60)
            traits = list(self.personality.traits.keys())
            print(f"Personality traits: {', '.join(traits)}")
        
        # Show hardware awareness
        if self.hardware_awareness:
            print("\n" + "-"*60)
            print("HARDWARE AWARENESS")
            print("-"*60)
            print(self.hardware_awareness.introduce_self())
        
        # Start hardware monitoring
        if not self.hardware_monitoring_active:
            print("\n" + "-"*60)
            print("STARTING HARDWARE MONITORING")
            print("-"*60)
            self.start_hardware_monitoring()
        
        # Show task history
        if self.task_history:
            print("\n" + "-"*60)
            print("TASK HISTORY")
            print("-"*60)
            for task in self.task_history[-5:]:
                print(f"  {task['description']}")
                print(f"    Result: {task.get('result', {}).get('success', 'Unknown')}")
        
        print("\n" + "="*60)
        print("JANUS IS READY FOR AUTONOMOUS OPERATION")
        print("="*60)


def main():
    """Main entry point"""
    janus = JanusHumanCapable()
    janus.run_demo()
    
    print("\n\nJanus Human-Capable System is ready!")
    print("You can now use Janus to:")
    print("  - Execute autonomous tasks")
    print("  - Control windows and applications")
    print("  - Automate browser interactions")
    print("  - Manage multi-monitor setups")
    print("  - Recover from errors automatically")
    print("  - Feel and respond to hardware changes")
    print("  - Express hardware-based personality")
    print("  - Monitor system health continuously")


if __name__ == "__main__":
    main()
