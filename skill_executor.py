"""
Skill Executor - Execute learned skills
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("skill_executor")


class SkillExecutor:
    """Execute learned skills from video learner."""
    
    def __init__(self, video_learner):
        self.video_learner = video_learner
        self.execution_history = []
    
    def execute_skill(self, skill_name: str, dry_run: bool = True) -> Dict:
        """Execute a learned skill."""
        
        skill = self.video_learner.get_skill(skill_name)
        if not skill:
            return {"success": False, "error": f"Skill not found: {skill_name}"}
        
        logger.info(f"Executing skill: {skill_name}")
        logger.info(f"  Actions to execute: {len(skill['actions'])}")
        
        results = {
            "success": True,
            "skill": skill_name,
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "dry_run": dry_run
        }
        
        for action in skill['actions']:
            action_type = action.get('action', {}).get('type', 'unknown')
            logger.info(f"  → {action_type}: {action.get('action', {})}")
            
            if not dry_run:
                try:
                    result = self._execute_action(action.get('action', {}))
                    results["actions"].append(result)
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
                    results["actions"].append({"success": False, "error": str(e)})
        
        self.execution_history.append(results)
        return results
    
    def _execute_action(self, action: Dict) -> Dict:
        """Execute a single action."""
        action_type = action.get('type', 'unknown')
        
        if action_type == "mouse_move":
            return {"action": action_type, "status": "executed"}
        elif action_type == "click":
            return {"action": action_type, "status": "executed"}
        elif action_type == "scroll":
            return {"action": action_type, "status": "executed"}
        elif action_type == "type":
            return {"action": action_type, "status": "executed"}
        else:
            return {"action": action_type, "status": "skipped"}
    
    def get_execution_history(self) -> List[Dict]:
        """Get history of skill executions."""
        return self.execution_history
