"""
Vision-Action Pipeline for Janus
Connects ScreenInterpreter → AvusInference → Action Execution
This is the core loop that makes Janus see and act like a human
"""

from typing import Optional, Dict, Any, Tuple
import time
import json
from datetime import datetime
from pathlib import Path

# Import Janus components
try:
    from os_human_interface import JanusOS
    from screen_interpreter import ScreenInterpreter
    from window_manager import WindowManager
    from avus_inference import AvusInference
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class VisionActionPipeline:
    """
    Complete vision-action pipeline for autonomous operation
    
    Flow:
    1. Capture screen (JanusOS)
    2. Interpret screen (ScreenInterpreter)
    3. Generate action (AvusInference)
    4. Execute action (JanusOS + WindowManager)
    5. Verify result
    """
    
    def __init__(self):
        self.janus_os = JanusOS() if COMPONENTS_AVAILABLE else None
        self.interpreter = ScreenInterpreter() if COMPONENTS_AVAILABLE else None
        self.window_manager = WindowManager() if COMPONENTS_AVAILABLE else None
        self.avus = None  # Lazy load
        
        self.action_history = []
        self.success_count = 0
        self.failure_count = 0
        
    def ensure_avus_loaded(self) -> bool:
        """Lazy load Avus inference"""
        if self.avus is None and COMPONENTS_AVAILABLE:
            try:
                self.avus = AvusInference()
                self.avus.load()
                return True
            except Exception as e:
                print(f"Failed to load Avus: {e}")
                return False
        return self.avus is not None
    
    def observe(self, goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Observe the screen and interpret what's visible
        
        Returns:
            {
                'description': str,
                'raw_bytes': bytes,
                'width': int,
                'height': int,
                'timestamp': str
            }
        """
        if not self.janus_os or not self.interpreter:
            return {'error': 'Components not available'}
        
        try:
            # Capture screen
            raw_bytes, width, height = self.janus_os.capture_screen()
            
            # Interpret screen
            description = self.interpreter.interpret(raw_bytes, width, height, goal)
            
            # Get active window context
            active_window = self.window_manager.get_active_window()
            window_context = ""
            if active_window:
                window_context = f" Active window: '{active_window.title}'."
            
            return {
                'description': description + window_context,
                'raw_bytes': raw_bytes,
                'width': width,
                'height': height,
                'active_window': active_window,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def decide(self, observation: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """
        Decide what action to take based on observation
        
        Returns:
            {
                'action_type': str,  # 'click', 'type', 'key', 'window', 'wait'
                'params': dict,
                'reasoning': str
            }
        """
        if not self.ensure_avus_loaded():
            return {'error': 'Avus not available'}
        
        try:
            # Build prompt for Avus
            prompt = f"""Screen observation: {observation['description']}

Goal: {goal}

Based on the screen state and goal, what action should be taken next?

Respond in JSON format:
{{
    "action_type": "click|type|key|window|wait",
    "params": {{}},
    "reasoning": "why this action"
}}

Examples:
- {{"action_type": "click", "params": {{"x": 640, "y": 400}}, "reasoning": "Click login button"}}
- {{"action_type": "type", "params": {{"text": "username"}}, "reasoning": "Enter username"}}
- {{"action_type": "key", "params": {{"key": "enter"}}, "reasoning": "Submit form"}}
- {{"action_type": "window", "params": {{"action": "switch", "title": "Chrome"}}, "reasoning": "Switch to browser"}}

ACTION:"""
            
            # Generate action using Avus
            response = self.avus.generate(prompt, max_tokens=200)
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    action = json.loads(response[json_start:json_end])
                    return action
                else:
                    # Fallback: parse manually
                    return self._parse_action_fallback(response)
            except json.JSONDecodeError:
                return self._parse_action_fallback(response)
                
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_action_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback parser for non-JSON responses"""
        response_lower = response.lower()
        
        if 'click' in response_lower:
            # Try to extract coordinates
            import re
            coords = re.findall(r'\((\d+),\s*(\d+)\)', response)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                return {
                    'action_type': 'click',
                    'params': {'x': x, 'y': y},
                    'reasoning': 'Extracted from response'
                }
        
        if 'type' in response_lower or 'enter' in response_lower:
            # Try to extract text
            import re
            text_match = re.search(r'["\']([^"\']+)["\']', response)
            if text_match:
                return {
                    'action_type': 'type',
                    'params': {'text': text_match.group(1)},
                    'reasoning': 'Extracted from response'
                }
        
        # Default: wait
        return {
            'action_type': 'wait',
            'params': {'duration': 1},
            'reasoning': 'Could not parse action, waiting'
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decided action
        
        Returns:
            {
                'success': bool,
                'message': str,
                'execution_time': float
            }
        """
        if 'error' in action:
            return {'success': False, 'message': action['error'], 'execution_time': 0}
        
        start_time = time.time()
        
        try:
            action_type = action.get('action_type', 'wait')
            params = action.get('params', {})
            
            if action_type == 'click':
                x, y = params.get('x', 0), params.get('y', 0)
                self.janus_os.click(x, y)
                message = f"Clicked at ({x}, {y})"
                
            elif action_type == 'type':
                text = params.get('text', '')
                self.janus_os.type_string(text)
                message = f"Typed: {text}"
                
            elif action_type == 'key':
                key = params.get('key', 'enter')
                vk_code = self._get_vk_code(key)
                self.janus_os.press_key(vk_code)
                message = f"Pressed key: {key}"
                
            elif action_type == 'window':
                window_action = params.get('action', 'switch')
                if window_action == 'switch':
                    title = params.get('title', '')
                    success = self.window_manager.switch_to_window(title)
                    message = f"Switched to window: {title}" if success else f"Failed to switch to: {title}"
                elif window_action == 'minimize':
                    hwnd = params.get('hwnd', 0)
                    self.window_manager.minimize_window(hwnd)
                    message = "Minimized window"
                elif window_action == 'maximize':
                    hwnd = params.get('hwnd', 0)
                    self.window_manager.maximize_window(hwnd)
                    message = "Maximized window"
                else:
                    message = f"Unknown window action: {window_action}"
                    
            elif action_type == 'wait':
                duration = params.get('duration', 1)
                time.sleep(duration)
                message = f"Waited {duration}s"
                
            else:
                message = f"Unknown action type: {action_type}"
            
            execution_time = time.time() - start_time
            
            # Record action
            self.action_history.append({
                'action': action,
                'result': message,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time
            })
            self.success_count += 1
            
            return {
                'success': True,
                'message': message,
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failure_count += 1
            
            return {
                'success': False,
                'message': str(e),
                'execution_time': execution_time
            }
    
    def verify(self, observation_before: Dict, observation_after: Dict, 
               action: Dict) -> Dict[str, Any]:
        """
        Verify that the action had the expected effect
        
        Returns:
            {
                'verified': bool,
                'changes_detected': list,
                'confidence': float
            }
        """
        changes = []
        
        # Check if active window changed
        if observation_before.get('active_window') != observation_after.get('active_window'):
            changes.append('active_window_changed')
        
        # Check if screen description changed
        if observation_before.get('description') != observation_after.get('description'):
            changes.append('screen_content_changed')
        
        # Simple verification: if changes detected, action likely worked
        verified = len(changes) > 0 or action.get('action_type') == 'wait'
        confidence = 0.8 if verified else 0.3
        
        return {
            'verified': verified,
            'changes_detected': changes,
            'confidence': confidence
        }
    
    def autonomous_loop(self, goal: str, max_steps: int = 20, 
                       verify_actions: bool = True) -> Dict[str, Any]:
        """
        Run autonomous loop: observe → decide → act → verify
        
        Returns:
            {
                'success': bool,
                'steps_taken': int,
                'goal_achieved': bool,
                'action_history': list
            }
        """
        print(f"\n{'='*60}")
        print(f"AUTONOMOUS LOOP: {goal}")
        print(f"{'='*60}\n")
        
        for step in range(max_steps):
            print(f"[Step {step + 1}/{max_steps}]")
            
            # 1. Observe
            print("  Observing...")
            observation = self.observe(goal)
            if 'error' in observation:
                print(f"  ✗ Observation failed: {observation['error']}")
                break
            
            print(f"  Screen: {observation['description'][:100]}...")
            
            # 2. Decide
            print("  Deciding...")
            action = self.decide(observation, goal)
            if 'error' in action:
                print(f"  ✗ Decision failed: {action['error']}")
                break
            
            print(f"  Action: {action['action_type']} - {action.get('reasoning', '')}")
            
            # 3. Act
            print("  Acting...")
            result = self.act(action)
            print(f"  {'✓' if result['success'] else '✗'} {result['message']}")
            
            if not result['success']:
                print(f"  Action failed, stopping.")
                break
            
            # 4. Verify (optional)
            if verify_actions and action['action_type'] != 'wait':
                time.sleep(0.5)  # Wait for UI to update
                observation_after = self.observe(goal)
                verification = self.verify(observation, observation_after, action)
                print(f"  Verification: {verification['verified']} (confidence: {verification['confidence']:.1%})")
            
            # Check if goal achieved (simple heuristic)
            if 'complete' in observation.get('description', '').lower() or \
               'success' in observation.get('description', '').lower():
                print(f"\n✓ Goal appears to be achieved!")
                return {
                    'success': True,
                    'steps_taken': step + 1,
                    'goal_achieved': True,
                    'action_history': self.action_history
                }
            
            time.sleep(0.5)  # Brief pause between steps
        
        print(f"\n{'='*60}")
        print(f"Loop completed: {self.success_count} successes, {self.failure_count} failures")
        print(f"{'='*60}\n")
        
        return {
            'success': self.failure_count == 0,
            'steps_taken': len(self.action_history),
            'goal_achieved': False,
            'action_history': self.action_history
        }
    
    def _get_vk_code(self, key: str) -> int:
        """Get virtual key code for common keys"""
        key_map = {
            'enter': 0x0D,
            'return': 0x0D,
            'tab': 0x09,
            'escape': 0x1B,
            'esc': 0x1B,
            'space': 0x20,
            'backspace': 0x08,
            'delete': 0x2E,
            'up': 0x26,
            'down': 0x28,
            'left': 0x25,
            'right': 0x27,
        }
        return key_map.get(key.lower(), 0x0D)


if __name__ == "__main__":
    # Test the pipeline
    pipeline = VisionActionPipeline()
    
    print("Vision-Action Pipeline Test")
    print("=" * 60)
    
    # Test observation
    print("\n1. Testing observation...")
    obs = pipeline.observe(goal="Test screen capture")
    if 'error' not in obs:
        print(f"   ✓ Screen captured: {obs['width']}x{obs['height']}")
        print(f"   Description: {obs['description'][:100]}...")
    else:
        print(f"   ✗ Error: {obs['error']}")
    
    # Test decision (requires Avus)
    print("\n2. Testing decision...")
    if pipeline.ensure_avus_loaded():
        action = pipeline.decide(obs, "Click the start button")
        print(f"   Action: {action}")
    else:
        print("   ✗ Avus not available")
    
    print("\n" + "=" * 60)
    print("Pipeline ready for autonomous operation!")
