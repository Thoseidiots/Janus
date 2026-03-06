"""
Video Learning Module for Janus
Watches tutorial videos, learns workflows, builds skill library
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class VideoLearner:
    """Learn workflows from tutorial videos."""
    
    def __init__(self, skill_library_file: str = "skill_library.json"):
        self.skill_library_file = skill_library_file
        self.skill_library = self.load_skill_library()
        self.learned_skills = []
    
    def load_skill_library(self) -> Dict[str, Any]:
        """Load existing skill library."""
        if Path(self.skill_library_file).exists():
            with open(self.skill_library_file, 'r') as f:
                return json.load(f)
        return {"skills": [], "metadata": {"created": datetime.now().isoformat()}}
    
    def save_skill_library(self):
        """Save skill library to disk."""
        self.skill_library["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.skill_library_file, 'w') as f:
            json.dump(self.skill_library, f, indent=2)
    
    def extract_frames_from_video(self, video_path: str, sample_rate: int = 30) -> List[np.ndarray]:
        """
        Extract frames from video at sample_rate interval.
        sample_rate=30 means every 30th frame (roughly 1 frame per second at 30fps)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {Path(video_path).name}")
        print(f"  FPS: {fps}, Total Frames: {total_frames}")
        print(f"  Sampling every {sample_rate} frames (~{sample_rate/fps:.1f}s interval)")
        
        frame_idx = 0
        captured = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frames.append(frame)
                captured += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"  Extracted {captured} frames")
        return frames
    
    def analyze_frame_sequence(self, frames: List[np.ndarray], vision_analyzer) -> Dict[str, Any]:
        """
        Analyze a sequence of frames to detect changes and infer actions.
        Requires vision_analyzer with analyze_frame_change() method.
        """
        actions = []
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Detect changes between frames
            diff = cv2.absdiff(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY))
            change_magnitude = np.sum(diff) / (diff.shape[0] * diff.shape[1])
            
            if change_magnitude > 5:  # Threshold for significant change
                # Extract key regions that changed
                contours, _ = cv2.findContours(cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1],
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Use vision analyzer to infer action
                    action = vision_analyzer.infer_action(current_frame, next_frame, contours)
                    if action:
                        actions.append({
                            "frame_index": i,
                            "change_magnitude": float(change_magnitude),
                            "action": action,
                            "timestamp": f"{i/30:.1f}s"  # Assuming 30fps
                        })
        
        return {
            "total_frames": len(frames),
            "detected_actions": actions,
            "confidence": len(actions) / len(frames) if frames else 0
        }
    
    def create_skill_from_video(self, video_path: str, skill_name: str, 
                                vision_analyzer, description: str = "") -> Dict[str, Any]:
        """Learn a skill from a video and add to library."""
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path, sample_rate=30)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Analyze frames
        analysis = self.analyze_frame_sequence(frames, vision_analyzer)
        
        # Create skill
        skill = {
            "name": skill_name,
            "description": description or f"Learned from {Path(video_path).name}",
            "source_video": str(video_path),
            "video_hash": self._hash_file(video_path),
            "created_at": datetime.now().isoformat(),
            "frames_analyzed": analysis["total_frames"],
            "actions": analysis["detected_actions"],
            "confidence": analysis["confidence"],
            "status": "ready_for_execution"
        }
        
        # Add to library
        self.skill_library["skills"].append(skill)
        self.save_skill_library()
        self.learned_skills.append(skill_name)
        
        print(f"\n✓ Skill created: {skill_name}")
        print(f"  Actions learned: {len(skill['actions'])}")
        print(f"  Confidence: {skill['confidence']:.2%}")
        
        return skill
    
    def _hash_file(self, filepath: str) -> str:
        """Create hash of file for versioning."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a skill from library."""
        for skill in self.skill_library["skills"]:
            if skill["name"] == skill_name:
                return skill
        return None
    
    def list_skills(self) -> List[str]:
        """List all learned skills."""
        return [skill["name"] for skill in self.skill_library["skills"]]
    
    def get_skill_summary(self) -> str:
        """Get summary of learned skills."""
        if not self.skill_library["skills"]:
            return "No skills learned yet"
        
        summary = f"Skill Library ({len(self.skill_library['skills'])} skills):\n"
        for skill in self.skill_library["skills"]:
            summary += f"  • {skill['name']}: {len(skill['actions'])} actions\n"
        return summary


class SimpleVisionAnalyzer:
    """Placeholder vision analyzer for frame-to-action conversion."""
    
    def infer_action(self, frame1: np.ndarray, frame2: np.ndarray, 
                    contours: List) -> Optional[Dict[str, Any]]:
        """
        Infer an action from frame changes.
        This is a stub - integrate with Claude/GPT-4V for real understanding.
        """
        if not contours:
            return None
        
        # For now, classify by position
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Heuristic: small change in top-left = mouse move, etc.
        if w < 50 and h < 50:
            return {
                "type": "mouse_move",
                "x": x,
                "y": y,
                "confidence": 0.6
            }
        elif h > w:
            return {
                "type": "scroll",
                "direction": "down" if y > frame1.shape[0] // 2 else "up",
                "confidence": 0.5
            }
        else:
            return {
                "type": "click",
                "x": x + w // 2,
                "y": y + h // 2,
                "confidence": 0.7
            }
    
    def analyze_frame_change(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, Any]:
        """Analyze what changed between two frames."""
        return {
            "change_type": "unknown",
            "confidence": 0.5
        }


class SkillExecutor:
    """Execute learned skills autonomously."""
    
    def __init__(self, vision_learner: VideoLearner):
        self.vision_learner = vision_learner
        self.execution_history = []
    
    def execute_skill(self, skill_name: str, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a learned skill."""
        
        skill = self.vision_learner.get_skill(skill_name)
        if not skill:
            return {"success": False, "error": f"Skill not found: {skill_name}"}
        
        print(f"\nExecuting skill: {skill_name}")
        print(f"Actions to execute: {len(skill['actions'])}")
        
        results = {
            "skill": skill_name,
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "dry_run": dry_run
        }
        
        for action in skill['actions']:
            action_type = action['action']['type']
            print(f"  → {action_type}: {action['action']}")
            
            if not dry_run:
                # Execute action (placeholder)
                try:
                    result = self._execute_action(action['action'])
                    results["actions"].append(result)
                except Exception as e:
                    results["actions"].append({
                        "action": action,
                        "success": False,
                        "error": str(e)
                    })
        
        self.execution_history.append(results)
        results["success"] = True
        return results
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        action_type = action['type']
        
        # Placeholder implementations
        if action_type == "mouse_move":
            return {"action": action_type, "status": "simulated"}
        elif action_type == "click":
            return {"action": action_type, "status": "simulated"}
        elif action_type == "scroll":
            return {"action": action_type, "status": "simulated"}
        else:
            return {"action": action_type, "status": "unknown"}
    
    def get_execution_history(self) -> List[Dict]:
        """Get history of skill executions."""
        return self.execution_history


if __name__ == "__main__":
    # Example usage
    print("Janus Video Learning Module")
    print("=" * 50)
    
    learner = VideoLearner()
    analyzer = SimpleVisionAnalyzer()
    
    # Example: Learn from a video (if you have one)
    video_path = "tutorial_video.mp4"
    if Path(video_path).exists():
        skill = learner.create_skill_from_video(
            video_path,
            "example_workflow",
            analyzer,
            description="Example workflow learned from video"
        )
        print(f"\nSkill created: {skill['name']}")
        print(f"Learned {len(skill['actions'])} actions")
    else:
        print(f"\nNo video found at {video_path}")
        print("To use: place a tutorial video and call:")
        print("  learner.create_skill_from_video('your_video.mp4', 'skill_name', analyzer)")
    
    print(f"\n{learner.get_skill_summary()}")
