"""
Vision-based Screen Automation Module for Janus
Enables the agent to see and interact with the desktop autonomously.
"""

import pyautogui
import time
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import mss
import numpy as np

class ScreenAutomationAgent:
    """Autonomous agent that can see and control the desktop."""
    
    def __init__(self, task_history_file: str = "task_history.json"):
        self.task_history_file = task_history_file
        self.task_history: List[Dict] = self.load_history()
        self.screen_width, self.screen_height = pyautogui.size()
        self.recording = False
        self.current_task_steps = []
        
    def load_history(self) -> List[Dict]:
        """Load previous task executions."""
        if Path(self.task_history_file).exists():
            with open(self.task_history_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_history(self):
        """Save task history for learning."""
        with open(self.task_history_file, 'w') as f:
            json.dump(self.task_history, f, indent=2)
    
    def capture_screen(self) -> np.ndarray:
        """Capture current screen as numpy array."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            return np.array(screenshot)
    
    def save_screenshot(self, filename: Optional[str] = None) -> str:
        """Save screenshot to file."""
        if filename is None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        screenshot = self.capture_screen()
        import cv2
        cv2.imwrite(filename, screenshot)
        return filename
    
    def screenshot_to_base64(self) -> str:
        """Convert current screenshot to base64 for API calls."""
        screenshot = self.capture_screen()
        import cv2
        _, buffer = cv2.imencode('.png', screenshot)
        return base64.b64encode(buffer).decode('utf-8')
    
    def click(self, x: int, y: int, duration: float = 0.1):
        """Click at coordinates."""
        pyautogui.click(x, y, duration=duration)
        self.record_action("click", {"x": x, "y": y})
        time.sleep(0.2)
    
    def type_text(self, text: str, interval: float = 0.05):
        """Type text with natural speed."""
        pyautogui.typewrite(text, interval=interval)
        self.record_action("type", {"text": text})
        time.sleep(0.2)
    
    def write_text(self, text: str):
        """Write text using clipboard (handles special chars)."""
        import subprocess
        process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
        pyautogui.hotkey('ctrl', 'v')
        self.record_action("write", {"text": text})
        time.sleep(0.2)
    
    def press_key(self, key: str):
        """Press a single key."""
        pyautogui.press(key)
        self.record_action("press", {"key": key})
        time.sleep(0.1)
    
    def hotkey(self, *keys):
        """Press multiple keys simultaneously."""
        pyautogui.hotkey(*keys)
        self.record_action("hotkey", {"keys": list(keys)})
        time.sleep(0.2)
    
    def move_mouse(self, x: int, y: int, duration: float = 0.5):
        """Move mouse to coordinates."""
        pyautogui.moveTo(x, y, duration=duration)
        self.record_action("move", {"x": x, "y": y})
    
    def scroll(self, x: int, y: int, clicks: int = 5):
        """Scroll at coordinates."""
        pyautogui.scroll(clicks, x=x, y=y)
        self.record_action("scroll", {"x": x, "y": y, "clicks": clicks})
        time.sleep(0.3)
    
    def double_click(self, x: int, y: int):
        """Double-click at coordinates."""
        pyautogui.doubleClick(x, y)
        self.record_action("double_click", {"x": x, "y": y})
        time.sleep(0.3)
    
    def record_action(self, action_type: str, params: Dict):
        """Record an action for learning."""
        if self.recording:
            self.current_task_steps.append({
                "type": action_type,
                "params": params,
                "timestamp": datetime.now().isoformat()
            })
    
    def start_recording(self, task_name: str):
        """Start recording a task."""
        self.recording = True
        self.current_task_steps = []
        self.current_task_name = task_name
        print(f"Recording task: {task_name}")
    
    def stop_recording(self):
        """Stop recording and save task."""
        self.recording = False
        task_entry = {
            "name": self.current_task_name,
            "steps": self.current_task_steps,
            "timestamp": datetime.now().isoformat()
        }
        self.task_history.append(task_entry)
        self.save_history()
        print(f"Task recorded: {len(self.current_task_steps)} steps")
        return task_entry
    
    def get_similar_tasks(self, task_name: str) -> List[Dict]:
        """Retrieve similar tasks from history for learning."""
        similar = [t for t in self.task_history if task_name.lower() in t['name'].lower()]
        return similar
    
    def replay_task(self, task_name: str):
        """Replay a recorded task."""
        tasks = self.get_similar_tasks(task_name)
        if not tasks:
            print(f"No tasks found matching: {task_name}")
            return False
        
        task = tasks[-1]  # Use most recent
        print(f"Replaying task: {task['name']}")
        
        for step in task['steps']:
            action_type = step['type']
            params = step['params']
            
            try:
                if action_type == "click":
                    self.click(params['x'], params['y'])
                elif action_type == "type":
                    self.type_text(params['text'])
                elif action_type == "write":
                    self.write_text(params['text'])
                elif action_type == "press":
                    self.press_key(params['key'])
                elif action_type == "hotkey":
                    self.hotkey(*params['keys'])
                elif action_type == "move":
                    self.move_mouse(params['x'], params['y'])
                elif action_type == "scroll":
                    self.scroll(params['x'], params['y'], params['clicks'])
                elif action_type == "double_click":
                    self.double_click(params['x'], params['y'])
            except Exception as e:
                print(f"Error executing step {action_type}: {e}")
        
        return True
    
    def autonomous_loop(self, vision_analyzer, goal: str, max_steps: int = 20):
        """
        Main autonomous loop: observe -> plan -> act -> verify
        Requires a vision_analyzer with analyze_screen() method.
        """
        step = 0
        while step < max_steps:
            # Observe: capture and analyze screen
            screenshot = self.capture_screen()
            print(f"\n[Step {step+1}] Observing screen...")
            
            analysis = vision_analyzer.analyze_screen(screenshot, goal)
            print(f"Analysis: {analysis.get('description', 'No description')}")
            
            # Check if goal is achieved
            if analysis.get('goal_achieved', False):
                print(f"\n✓ Goal achieved: {goal}")
                return True
            
            # Plan: decide next action
            next_action = analysis.get('recommended_action')
            if not next_action:
                print("No action recommended, stopping.")
                return False
            
            print(f"Action: {next_action}")
            
            # Act: execute action
            try:
                if 'click' in next_action.lower():
                    coords = analysis.get('click_coords', (self.screen_width//2, self.screen_height//2))
                    self.click(coords[0], coords[1])
                elif 'type' in next_action.lower():
                    text = analysis.get('text_to_type', '')
                    self.write_text(text)
                elif 'scroll' in next_action.lower():
                    self.scroll(self.screen_width//2, self.screen_height//2, clicks=3)
                elif 'wait' in next_action.lower():
                    time.sleep(2)
                else:
                    print(f"Unknown action: {next_action}")
            except Exception as e:
                print(f"Error executing action: {e}")
            
            step += 1
            time.sleep(1)
        
        print(f"\nMax steps ({max_steps}) reached, stopping.")
        return False


class SimpleVisionAnalyzer:
    """Placeholder vision analyzer. Replace with Claude/GPT-4V integration."""
    
    def analyze_screen(self, screenshot: np.ndarray, goal: str) -> Dict[str, Any]:
        """
        Analyze screenshot and recommend next action.
        This is a stub - integrate with Claude/GPT-4V for real vision understanding.
        """
        return {
            "description": f"Screen analyzed. Goal: {goal}",
            "recommended_action": "wait",  # Default safe action
            "goal_achieved": False,
            "confidence": 0.5
        }


if __name__ == "__main__":
    # Example usage
    agent = ScreenAutomationAgent()
    
    # Take a screenshot
    print("Capturing screen...")
    filename = agent.save_screenshot()
    print(f"Screenshot saved: {filename}")
    
    # Record a simple task
    print("\nRecording task: Open Notepad and type hello")
    agent.start_recording("open_notepad_and_type")
    
    # Example actions (customize based on your desktop)
    agent.hotkey('win', 'r')
    time.sleep(1)
    agent.write_text("notepad")
    agent.press_key('enter')
    time.sleep(2)
    agent.write_text("Hello, Janus!")
    
    agent.stop_recording()
    print(f"Recorded {len(agent.current_task_steps)} steps")
    
    print(f"\nTask history saved to {agent.task_history_file}")
