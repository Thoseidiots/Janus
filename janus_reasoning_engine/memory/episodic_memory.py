"""
Episodic Memory System for the Janus Reasoning Engine.

Stores experiences as (context, action, outcome) tuples with indexing
by skills, platforms, and outcomes. Implements similarity-based retrieval
using HBM and experience replay for learning.

**Validates: Requirements REQ-6.1, REQ-6.4**
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

from janus_reasoning_engine.memory.interfaces import MemoryType
from janus_reasoning_engine.memory.unified_memory import UnifiedMemory


class OutcomeType(Enum):
    """Types of experience outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class Experience:
    """
    Represents a single experience/episode.
    
    Stores (context, action, outcome) tuples with rich metadata
    for indexing and retrieval.
    """
    experience_id: str
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    timestamp: datetime
    
    # Indexing metadata
    skills: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    outcome_type: OutcomeType = OutcomeType.UNKNOWN
    
    # Additional metadata
    earnings: Optional[float] = None
    time_spent: Optional[float] = None
    difficulty: Optional[float] = None
    learning_value: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary."""
        return {
            "experience_id": self.experience_id,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "skills": self.skills,
            "platforms": self.platforms,
            "outcome_type": self.outcome_type.value,
            "earnings": self.earnings,
            "time_spent": self.time_spent,
            "difficulty": self.difficulty,
            "learning_value": self.learning_value,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create experience from dictionary."""
        return cls(
            experience_id=data["experience_id"],
            context=data["context"],
            action=data["action"],
            outcome=data["outcome"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            skills=data.get("skills", []),
            platforms=data.get("platforms", []),
            outcome_type=OutcomeType(data.get("outcome_type", "unknown")),
            earnings=data.get("earnings"),
            time_spent=data.get("time_spent"),
            difficulty=data.get("difficulty"),
            learning_value=data.get("learning_value"),
            tags=data.get("tags", []),
        )


class EpisodicMemory:
    """
    Episodic Memory System for storing and retrieving experiences.
    
    Features:
    - Store experiences as (context, action, outcome) tuples
    - Index by skills, platforms, and outcomes
    - Similarity-based retrieval using HBM
    - Experience replay for learning
    - Statistical analysis of experiences
    """
    
    def __init__(self, unified_memory: UnifiedMemory):
        """
        Initialize episodic memory system.
        
        Args:
            unified_memory: Unified memory interface for storage
        """
        self.unified_memory = unified_memory
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_experiences = 0
        self.success_count = 0
        self.failure_count = 0
    
    def store_experience(
        self,
        context: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        skills: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
        outcome_type: OutcomeType = OutcomeType.UNKNOWN,
        earnings: Optional[float] = None,
        time_spent: Optional[float] = None,
        difficulty: Optional[float] = None,
        learning_value: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a new experience.
        
        Args:
            context: Context in which action was taken
            action: Action that was performed
            outcome: Result of the action
            skills: Skills used/learned
            platforms: Platforms involved
            outcome_type: Type of outcome
            earnings: Money earned (if applicable)
            time_spent: Time spent in hours
            difficulty: Difficulty rating (0.0-1.0)
            learning_value: Learning value (0.0-1.0)
            tags: Additional tags for categorization
            
        Returns:
            Experience ID
        """
        # Create experience
        experience = Experience(
            experience_id=str(uuid.uuid4()),
            context=context,
            action=action,
            outcome=outcome,
            timestamp=datetime.utcnow(),
            skills=skills or [],
            platforms=platforms or [],
            outcome_type=outcome_type,
            earnings=earnings,
            time_spent=time_spent,
            difficulty=difficulty,
            learning_value=learning_value,
            tags=tags or [],
        )
        
        # Store in unified memory
        content = experience.to_dict()
        metadata = {
            "skills": experience.skills,
            "platforms": experience.platforms,
            "outcome_type": experience.outcome_type.value,
            "success": outcome_type == OutcomeType.SUCCESS,
            "tags": experience.tags,
        }
        
        memory_id = self.unified_memory.store(
            MemoryType.EPISODIC,
            content,
            metadata,
            use_both=True  # Store in both HBM and SQLite
        )
        
        # Update statistics
        self.total_experiences += 1
        if outcome_type == OutcomeType.SUCCESS:
            self.success_count += 1
        elif outcome_type == OutcomeType.FAILURE:
            self.failure_count += 1
        
        self.logger.info(
            f"Stored experience {experience.experience_id} "
            f"(outcome: {outcome_type.value})"
        )
        
        return experience.experience_id
    
    def retrieve_similar_experiences(
        self,
        query_context: str,
        limit: int = 10,
        similarity_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Experience]:
        """
        Retrieve experiences similar to a query context using HBM.
        
        Args:
            query_context: Context description to search for
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional filters (skills, platforms, outcome_type, etc.)
            
        Returns:
            List of similar experiences
        """
        # Use semantic search with HBM
        results = self.unified_memory.semantic_search(
            query_context,
            MemoryType.EPISODIC,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
        
        # Convert results to Experience objects
        experiences = []
        for result in results:
            # Apply filters if provided
            if filters:
                if not self._matches_filters(result.metadata, filters):
                    continue
            
            try:
                experience = Experience.from_dict(result.content)
                experiences.append(experience)
            except Exception as e:
                self.logger.warning(f"Failed to parse experience: {e}")
                continue
        
        return experiences
    
    def retrieve_by_skill(
        self,
        skill: str,
        limit: int = 10,
        outcome_type: Optional[OutcomeType] = None,
    ) -> List[Experience]:
        """
        Retrieve experiences involving a specific skill.
        
        Args:
            skill: Skill to search for
            limit: Maximum number of results
            outcome_type: Filter by outcome type (optional)
            
        Returns:
            List of experiences
        """
        filters = {"skills": skill}
        if outcome_type:
            filters["outcome_type"] = outcome_type.value
        
        return self._retrieve_structured(filters, limit)
    
    def retrieve_by_platform(
        self,
        platform: str,
        limit: int = 10,
        outcome_type: Optional[OutcomeType] = None,
    ) -> List[Experience]:
        """
        Retrieve experiences on a specific platform.
        
        Args:
            platform: Platform to search for
            limit: Maximum number of results
            outcome_type: Filter by outcome type (optional)
            
        Returns:
            List of experiences
        """
        filters = {"platforms": platform}
        if outcome_type:
            filters["outcome_type"] = outcome_type.value
        
        return self._retrieve_structured(filters, limit)
    
    def retrieve_by_outcome(
        self,
        outcome_type: OutcomeType,
        limit: int = 10,
    ) -> List[Experience]:
        """
        Retrieve experiences by outcome type.
        
        Args:
            outcome_type: Type of outcome to filter by
            limit: Maximum number of results
            
        Returns:
            List of experiences
        """
        filters = {"outcome_type": outcome_type.value}
        return self._retrieve_structured(filters, limit)
    
    def retrieve_successful_experiences(
        self,
        limit: int = 10,
        min_earnings: Optional[float] = None,
    ) -> List[Experience]:
        """
        Retrieve successful experiences.
        
        Args:
            limit: Maximum number of results
            min_earnings: Minimum earnings filter (optional)
            
        Returns:
            List of successful experiences
        """
        filters = {"success": True}
        experiences = self._retrieve_structured(filters, limit)
        
        # Apply earnings filter if provided
        if min_earnings is not None:
            experiences = [
                exp for exp in experiences
                if exp.earnings is not None and exp.earnings >= min_earnings
            ]
        
        return experiences
    
    def get_experience_replay_batch(
        self,
        batch_size: int = 10,
        prioritize_failures: bool = True,
        prioritize_recent: bool = False,
    ) -> List[Experience]:
        """
        Get a batch of experiences for replay-based learning.
        
        Args:
            batch_size: Number of experiences to retrieve
            prioritize_failures: Prioritize learning from failures
            prioritize_recent: Prioritize recent experiences
            
        Returns:
            Batch of experiences for replay
        """
        experiences = []
        
        if prioritize_failures:
            # Get failures first
            failures = self.retrieve_by_outcome(
                OutcomeType.FAILURE,
                limit=batch_size // 2
            )
            experiences.extend(failures)
            
            # Fill remaining with successes
            remaining = batch_size - len(experiences)
            if remaining > 0:
                successes = self.retrieve_by_outcome(
                    OutcomeType.SUCCESS,
                    limit=remaining
                )
                experiences.extend(successes)
        else:
            # Get mixed batch
            experiences = self._retrieve_structured({}, batch_size)
        
        # Sort by recency if requested
        if prioritize_recent:
            experiences.sort(key=lambda e: e.timestamp, reverse=True)
        
        return experiences[:batch_size]
    
    def analyze_skill_performance(
        self,
        skill: str
    ) -> Dict[str, Any]:
        """
        Analyze performance for a specific skill.
        
        Args:
            skill: Skill to analyze
            
        Returns:
            Performance statistics
        """
        experiences = self.retrieve_by_skill(skill, limit=1000)
        
        if not experiences:
            return {
                "skill": skill,
                "total_experiences": 0,
                "success_rate": 0.0,
                "average_earnings": 0.0,
                "average_time": 0.0,
                "total_earnings": 0.0,
            }
        
        successes = sum(1 for e in experiences if e.outcome_type == OutcomeType.SUCCESS)
        total_earnings = sum(e.earnings or 0.0 for e in experiences)
        total_time = sum(e.time_spent or 0.0 for e in experiences)
        
        return {
            "skill": skill,
            "total_experiences": len(experiences),
            "success_rate": successes / len(experiences) if experiences else 0.0,
            "average_earnings": total_earnings / len(experiences) if experiences else 0.0,
            "average_time": total_time / len(experiences) if experiences else 0.0,
            "total_earnings": total_earnings,
        }
    
    def analyze_platform_performance(
        self,
        platform: str
    ) -> Dict[str, Any]:
        """
        Analyze performance on a specific platform.
        
        Args:
            platform: Platform to analyze
            
        Returns:
            Performance statistics
        """
        experiences = self.retrieve_by_platform(platform, limit=1000)
        
        if not experiences:
            return {
                "platform": platform,
                "total_experiences": 0,
                "success_rate": 0.0,
                "average_earnings": 0.0,
                "total_earnings": 0.0,
            }
        
        successes = sum(1 for e in experiences if e.outcome_type == OutcomeType.SUCCESS)
        total_earnings = sum(e.earnings or 0.0 for e in experiences)
        
        return {
            "platform": platform,
            "total_experiences": len(experiences),
            "success_rate": successes / len(experiences) if experiences else 0.0,
            "average_earnings": total_earnings / len(experiences) if experiences else 0.0,
            "total_earnings": total_earnings,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall episodic memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_experiences": self.total_experiences,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": (
                self.success_count / self.total_experiences
                if self.total_experiences > 0
                else 0.0
            ),
        }
    
    def _retrieve_structured(
        self,
        filters: Dict[str, Any],
        limit: int
    ) -> List[Experience]:
        """
        Retrieve experiences using structured query.
        
        Args:
            filters: Filters to apply
            limit: Maximum number of results
            
        Returns:
            List of experiences
        """
        results = self.unified_memory.structured_query(
            MemoryType.EPISODIC,
            filters=filters,
            limit=limit,
        )
        
        experiences = []
        for result in results:
            try:
                experience = Experience.from_dict(result.content)
                experiences.append(experience)
            except Exception as e:
                self.logger.warning(f"Failed to parse experience: {e}")
                continue
        
        return experiences
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filters.
        
        Args:
            metadata: Metadata to check
            filters: Filters to apply
            
        Returns:
            True if matches, False otherwise
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            # Handle list filters (e.g., skills, platforms)
            if isinstance(metadata[key], list):
                if value not in metadata[key]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
