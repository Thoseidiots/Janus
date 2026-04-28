"""
Memory layer for the Janus Reasoning Engine.

Provides unified memory interface integrating:
- HolographicBrainMemory for associative recall
- SQLite for structured data storage
- File system for artifacts and checkpoints
- Episodic memory for experience storage and replay
- Semantic memory for skills, knowledge, and procedures
- Working memory for active context and attention management
"""

from janus_reasoning_engine.memory.unified_memory import UnifiedMemory
from janus_reasoning_engine.memory.interfaces import (
    MemoryBackend,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)
from janus_reasoning_engine.memory.episodic_memory import (
    EpisodicMemory,
    Experience,
    OutcomeType,
)
from janus_reasoning_engine.memory.semantic_memory import (
    SemanticMemory,
    Skill,
    SkillLevel,
    Procedure,
    Knowledge,
    KnowledgeType,
)
from janus_reasoning_engine.memory.working_memory import (
    WorkingMemory,
    ContextItem,
    ThoughtThread,
    InterruptionCheckpoint,
    AttentionLevel,
    ThreadStatus,
)

__all__ = [
    "UnifiedMemory",
    "MemoryBackend",
    "MemoryQuery",
    "MemoryResult",
    "MemoryType",
    "EpisodicMemory",
    "Experience",
    "OutcomeType",
    "SemanticMemory",
    "Skill",
    "SkillLevel",
    "Procedure",
    "Knowledge",
    "KnowledgeType",
    "WorkingMemory",
    "ContextItem",
    "ThoughtThread",
    "InterruptionCheckpoint",
    "AttentionLevel",
    "ThreadStatus",
]
