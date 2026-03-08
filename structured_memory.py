import json
import time
from typing import Any, Dict, List, Optional

class StructuredMemory:
    def __init__(self):
        self.episodic = []
        self.semantic = {}
        self.working = {}

    def add_episode(self, task_id: str, action: str, result: Any, success: bool):
        entry = {
            "timestamp": time.time(),
            "task_id": task_id,
            "action": action,
            "result": result,
            "success": success
        }
        self.episodic.append(entry)

    def update_semantic(self, key: str, value: Any, confidence: float = 1.0):
        self.semantic[key] = {
            "value": value,
            "confidence": confidence,
            "last_updated": time.time()
        }

    def get_semantic(self, key: str):
        return self.semantic.get(key)

    def set_working_context(self, key: str, value: Any):
        self.working[key] = value

    def get_working_context(self, key: str):
        return self.working.get(key)
