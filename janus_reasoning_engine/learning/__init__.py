"""
Learning and Skill Acquisition system for the Janus Reasoning Engine.

Provides:
- Skill gap detection (compare opportunity requirements to skill inventory)
- Autonomous learning (search tutorials, watch videos, extract knowledge)
- Skill inventory management (track skills, confidence, domains)
"""

from janus_reasoning_engine.learning.skill_gap_detector import (
    SkillGap,
    SkillGapDetector,
)
from janus_reasoning_engine.learning.autonomous_learner import (
    LearningResult,
    AutonomousLearner,
)
from janus_reasoning_engine.learning.skill_inventory import (
    SkillInventory,
)

__all__ = [
    "SkillGap",
    "SkillGapDetector",
    "LearningResult",
    "AutonomousLearner",
    "SkillInventory",
]
