"""
Semantic Memory System for the Janus Reasoning Engine.

Stores general knowledge, skills, and procedures as structured data with
hierarchical organization. Implements knowledge retrieval by topic/skill
and update/refinement mechanisms.

**Validates: Requirements REQ-6.2, REQ-3.3**
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

from janus_reasoning_engine.memory.interfaces import MemoryType
from janus_reasoning_engine.memory.unified_memory import UnifiedMemory


class KnowledgeType(Enum):
    """Types of semantic knowledge."""
    SKILL = "skill"
    FACT = "fact"
    PROCEDURE = "procedure"
    CONCEPT = "concept"
    RULE = "rule"


class SkillLevel(Enum):
    """Skill proficiency levels."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Skill:
    """
    Represents a skill with proficiency tracking.
    """
    skill_id: str
    name: str
    description: str
    level: SkillLevel
    confidence: float  # 0.0-1.0
    
    # Hierarchical organization
    parent_skill: Optional[str] = None
    sub_skills: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    
    # Learning metadata
    learned_date: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    success_count: int = 0
    
    # Knowledge links
    related_procedures: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Tags and categorization
    tags: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "confidence": self.confidence,
            "parent_skill": self.parent_skill,
            "sub_skills": self.sub_skills,
            "related_skills": self.related_skills,
            "learned_date": self.learned_date.isoformat() if self.learned_date else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "related_procedures": self.related_procedures,
            "related_concepts": self.related_concepts,
            "tags": self.tags,
            "domains": self.domains,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create skill from dictionary."""
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            level=SkillLevel(data["level"]),
            confidence=data["confidence"],
            parent_skill=data.get("parent_skill"),
            sub_skills=data.get("sub_skills", []),
            related_skills=data.get("related_skills", []),
            learned_date=datetime.fromisoformat(data["learned_date"]) if data.get("learned_date") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            use_count=data.get("use_count", 0),
            success_count=data.get("success_count", 0),
            related_procedures=data.get("related_procedures", []),
            related_concepts=data.get("related_concepts", []),
            tags=data.get("tags", []),
            domains=data.get("domains", []),
        )


@dataclass
class Procedure:
    """
    Represents a procedure or how-to knowledge.
    """
    procedure_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    
    # Hierarchical organization
    parent_procedure: Optional[str] = None
    sub_procedures: List[str] = field(default_factory=list)
    
    # Metadata
    required_skills: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_time: Optional[float] = None
    difficulty: Optional[float] = None
    
    # Usage tracking
    use_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    
    # Tags
    tags: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert procedure to dictionary."""
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "parent_procedure": self.parent_procedure,
            "sub_procedures": self.sub_procedures,
            "required_skills": self.required_skills,
            "prerequisites": self.prerequisites,
            "estimated_time": self.estimated_time,
            "difficulty": self.difficulty,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "tags": self.tags,
            "domains": self.domains,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Procedure":
        """Create procedure from dictionary."""
        return cls(
            procedure_id=data["procedure_id"],
            name=data["name"],
            description=data["description"],
            steps=data["steps"],
            parent_procedure=data.get("parent_procedure"),
            sub_procedures=data.get("sub_procedures", []),
            required_skills=data.get("required_skills", []),
            prerequisites=data.get("prerequisites", []),
            estimated_time=data.get("estimated_time"),
            difficulty=data.get("difficulty"),
            use_count=data.get("use_count", 0),
            success_count=data.get("success_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            tags=data.get("tags", []),
            domains=data.get("domains", []),
        )


@dataclass
class Knowledge:
    """
    Represents general knowledge (facts, concepts, rules).
    """
    knowledge_id: str
    knowledge_type: KnowledgeType
    name: str
    content: Dict[str, Any]
    
    # Hierarchical organization
    parent_knowledge: Optional[str] = None
    related_knowledge: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0  # 0.0-1.0
    source: Optional[str] = None
    learned_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Tags
    tags: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge to dictionary."""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "name": self.name,
            "content": self.content,
            "parent_knowledge": self.parent_knowledge,
            "related_knowledge": self.related_knowledge,
            "confidence": self.confidence,
            "source": self.source,
            "learned_date": self.learned_date.isoformat() if self.learned_date else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "tags": self.tags,
            "domains": self.domains,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Knowledge":
        """Create knowledge from dictionary."""
        return cls(
            knowledge_id=data["knowledge_id"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            name=data["name"],
            content=data["content"],
            parent_knowledge=data.get("parent_knowledge"),
            related_knowledge=data.get("related_knowledge", []),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            learned_date=datetime.fromisoformat(data["learned_date"]) if data.get("learned_date") else None,
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
            tags=data.get("tags", []),
            domains=data.get("domains", []),
        )


class SemanticMemory:
    """
    Semantic Memory System for storing and retrieving general knowledge.
    
    Features:
    - Store skills with proficiency tracking
    - Store procedures as step-by-step knowledge
    - Store facts, concepts, and rules
    - Hierarchical knowledge organization
    - Retrieval by topic, skill, domain
    - Knowledge update and refinement
    - Skill inventory management
    """
    
    def __init__(self, unified_memory: UnifiedMemory):
        """
        Initialize semantic memory system.
        
        Args:
            unified_memory: Unified memory interface for storage
        """
        self.unified_memory = unified_memory
        self.logger = logging.getLogger(__name__)
        
        # In-memory caches for fast access
        self._skill_cache: Dict[str, Skill] = {}
        self._procedure_cache: Dict[str, Procedure] = {}
        self._knowledge_cache: Dict[str, Knowledge] = {}
        
        # Statistics
        self.total_skills = 0
        self.total_procedures = 0
        self.total_knowledge = 0
    
    # ==================== Skill Management ====================
    
    def add_skill(
        self,
        name: str,
        description: str,
        level: SkillLevel = SkillLevel.NOVICE,
        confidence: float = 0.5,
        parent_skill: Optional[str] = None,
        tags: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new skill to semantic memory.
        
        Args:
            name: Skill name
            description: Skill description
            level: Proficiency level
            confidence: Confidence in skill (0.0-1.0)
            parent_skill: Parent skill ID (for hierarchy)
            tags: Tags for categorization
            domains: Domains this skill belongs to
            
        Returns:
            Skill ID
        """
        skill = Skill(
            skill_id=str(uuid.uuid4()),
            name=name,
            description=description,
            level=level,
            confidence=confidence,
            parent_skill=parent_skill,
            learned_date=datetime.utcnow(),
            tags=tags or [],
            domains=domains or [],
        )
        
        # Update parent skill's sub_skills if parent exists
        if parent_skill and parent_skill in self._skill_cache:
            parent = self._skill_cache[parent_skill]
            if skill.skill_id not in parent.sub_skills:
                parent.sub_skills.append(skill.skill_id)
                self._update_skill_in_storage(parent)
        
        # Store in unified memory
        content = skill.to_dict()
        metadata = {
            "name": skill.name,
            "level": skill.level.value,
            "tags": skill.tags,
            "domains": skill.domains,
            "parent_skill": skill.parent_skill,
        }
        
        self.unified_memory.store(
            MemoryType.SEMANTIC,
            content,
            metadata,
            use_both=True
        )
        
        # Cache
        self._skill_cache[skill.skill_id] = skill
        self.total_skills += 1
        
        self.logger.info(f"Added skill: {name} (level: {level.value})")
        return skill.skill_id
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """
        Get a skill by ID.
        
        Args:
            skill_id: Skill ID
            
        Returns:
            Skill or None if not found
        """
        # Check cache first
        if skill_id in self._skill_cache:
            return self._skill_cache[skill_id]
        
        # Query from storage
        results = self.unified_memory.structured_query(
            MemoryType.SEMANTIC,
            filters={"skill_id": skill_id},
            limit=1
        )
        
        if results:
            skill = Skill.from_dict(results[0].content)
            self._skill_cache[skill_id] = skill
            return skill
        
        return None
    
    def update_skill(
        self,
        skill_id: str,
        level: Optional[SkillLevel] = None,
        confidence: Optional[float] = None,
        increment_use: bool = False,
        increment_success: bool = False,
    ) -> None:
        """
        Update a skill's proficiency and usage.
        
        Args:
            skill_id: Skill ID
            level: New proficiency level
            confidence: New confidence level
            increment_use: Increment use count
            increment_success: Increment success count
        """
        skill = self.get_skill(skill_id)
        if not skill:
            self.logger.warning(f"Skill {skill_id} not found for update")
            return
        
        # Update fields
        if level is not None:
            skill.level = level
        if confidence is not None:
            skill.confidence = max(0.0, min(1.0, confidence))
        if increment_use:
            skill.use_count += 1
            skill.last_used = datetime.utcnow()
        if increment_success:
            skill.success_count += 1
        
        # Update in storage
        self._update_skill_in_storage(skill)
        
        self.logger.info(f"Updated skill: {skill.name}")
    
    def get_skills_by_domain(
        self,
        domain: str,
        min_level: Optional[SkillLevel] = None,
    ) -> List[Skill]:
        """
        Get all skills in a domain.
        
        Args:
            domain: Domain name
            min_level: Minimum skill level filter
            
        Returns:
            List of skills
        """
        # Get all semantic memories and filter manually
        # (SQLite doesn't handle list membership well)
        results = self.unified_memory.structured_query(
            MemoryType.SEMANTIC,
            filters={},
            limit=10000
        )
        
        skills = []
        for result in results:
            try:
                # Check if this is a skill (has 'level' field)
                if "level" not in result.content:
                    continue
                
                skill = Skill.from_dict(result.content)
                
                # Check if domain matches
                if domain not in skill.domains:
                    continue
                
                # Apply level filter
                if min_level and self._skill_level_value(skill.level) < self._skill_level_value(min_level):
                    continue
                
                skills.append(skill)
                self._skill_cache[skill.skill_id] = skill
            except Exception as e:
                self.logger.warning(f"Failed to parse skill: {e}")
                continue
        
        return skills
    
    def search_skills(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Skill]:
        """
        Search for skills using semantic search.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching skills
        """
        results = self.unified_memory.semantic_search(
            query,
            MemoryType.SEMANTIC,
            limit=limit,
            similarity_threshold=0.3
        )
        
        skills = []
        for result in results:
            try:
                # Check if this is a skill
                if "level" not in result.content:
                    continue
                
                skill = Skill.from_dict(result.content)
                skills.append(skill)
                self._skill_cache[skill.skill_id] = skill
            except Exception as e:
                self.logger.warning(f"Failed to parse skill: {e}")
                continue
        
        return skills
    
    def get_skill_hierarchy(self, skill_id: str) -> Dict[str, Any]:
        """
        Get the full hierarchy for a skill (parent and children).
        
        Args:
            skill_id: Skill ID
            
        Returns:
            Hierarchy dictionary
        """
        skill = self.get_skill(skill_id)
        if not skill:
            return {}
        
        hierarchy = {
            "skill": skill.to_dict(),
            "parent": None,
            "children": [],
        }
        
        # Get parent
        if skill.parent_skill:
            parent = self.get_skill(skill.parent_skill)
            if parent:
                hierarchy["parent"] = parent.to_dict()
        
        # Get children
        for sub_skill_id in skill.sub_skills:
            sub_skill = self.get_skill(sub_skill_id)
            if sub_skill:
                hierarchy["children"].append(sub_skill.to_dict())
        
        return hierarchy
    
    # ==================== Procedure Management ====================
    
    def add_procedure(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        required_skills: Optional[List[str]] = None,
        parent_procedure: Optional[str] = None,
        tags: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new procedure to semantic memory.
        
        Args:
            name: Procedure name
            description: Procedure description
            steps: List of steps (each step is a dict with 'action', 'description', etc.)
            required_skills: Skills required to execute
            parent_procedure: Parent procedure ID
            tags: Tags for categorization
            domains: Domains this procedure belongs to
            
        Returns:
            Procedure ID
        """
        procedure = Procedure(
            procedure_id=str(uuid.uuid4()),
            name=name,
            description=description,
            steps=steps,
            required_skills=required_skills or [],
            parent_procedure=parent_procedure,
            tags=tags or [],
            domains=domains or [],
        )
        
        # Store in unified memory
        content = procedure.to_dict()
        metadata = {
            "name": procedure.name,
            "tags": procedure.tags,
            "domains": procedure.domains,
            "required_skills": procedure.required_skills,
        }
        
        self.unified_memory.store(
            MemoryType.SEMANTIC,
            content,
            metadata,
            use_both=True
        )
        
        # Cache
        self._procedure_cache[procedure.procedure_id] = procedure
        self.total_procedures += 1
        
        self.logger.info(f"Added procedure: {name}")
        return procedure.procedure_id
    
    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """
        Get a procedure by ID.
        
        Args:
            procedure_id: Procedure ID
            
        Returns:
            Procedure or None if not found
        """
        # Check cache
        if procedure_id in self._procedure_cache:
            return self._procedure_cache[procedure_id]
        
        # Query from storage
        results = self.unified_memory.structured_query(
            MemoryType.SEMANTIC,
            filters={"procedure_id": procedure_id},
            limit=1
        )
        
        if results:
            procedure = Procedure.from_dict(results[0].content)
            self._procedure_cache[procedure_id] = procedure
            return procedure
        
        return None
    
    def search_procedures(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Procedure]:
        """
        Search for procedures using semantic search.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching procedures
        """
        results = self.unified_memory.semantic_search(
            query,
            MemoryType.SEMANTIC,
            limit=limit,
            similarity_threshold=0.3
        )
        
        procedures = []
        for result in results:
            try:
                # Check if this is a procedure
                if "steps" not in result.content:
                    continue
                
                procedure = Procedure.from_dict(result.content)
                procedures.append(procedure)
                self._procedure_cache[procedure.procedure_id] = procedure
            except Exception as e:
                self.logger.warning(f"Failed to parse procedure: {e}")
                continue
        
        return procedures
    
    def update_procedure_usage(
        self,
        procedure_id: str,
        success: bool = True,
    ) -> None:
        """
        Update procedure usage statistics.
        
        Args:
            procedure_id: Procedure ID
            success: Whether execution was successful
        """
        procedure = self.get_procedure(procedure_id)
        if not procedure:
            return
        
        procedure.use_count += 1
        procedure.last_used = datetime.utcnow()
        if success:
            procedure.success_count += 1
        
        # Update in storage
        content = procedure.to_dict()
        self.unified_memory.update(
            procedure_id,
            MemoryType.SEMANTIC,
            content=content
        )
        
        self.logger.info(f"Updated procedure usage: {procedure.name}")
    
    # ==================== General Knowledge Management ====================
    
    def add_knowledge(
        self,
        knowledge_type: KnowledgeType,
        name: str,
        content: Dict[str, Any],
        confidence: float = 1.0,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Add general knowledge (fact, concept, rule).
        
        Args:
            knowledge_type: Type of knowledge
            name: Knowledge name
            content: Knowledge content
            confidence: Confidence in knowledge (0.0-1.0)
            source: Source of knowledge
            tags: Tags for categorization
            domains: Domains this knowledge belongs to
            
        Returns:
            Knowledge ID
        """
        knowledge = Knowledge(
            knowledge_id=str(uuid.uuid4()),
            knowledge_type=knowledge_type,
            name=name,
            content=content,
            confidence=confidence,
            source=source,
            learned_date=datetime.utcnow(),
            tags=tags or [],
            domains=domains or [],
        )
        
        # Store in unified memory
        content_dict = knowledge.to_dict()
        metadata = {
            "name": knowledge.name,
            "knowledge_type": knowledge.knowledge_type.value,
            "tags": knowledge.tags,
            "domains": knowledge.domains,
        }
        
        self.unified_memory.store(
            MemoryType.SEMANTIC,
            content_dict,
            metadata,
            use_both=True
        )
        
        # Cache
        self._knowledge_cache[knowledge.knowledge_id] = knowledge
        self.total_knowledge += 1
        
        self.logger.info(f"Added knowledge: {name} (type: {knowledge_type.value})")
        return knowledge.knowledge_id
    
    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """
        Get knowledge by ID.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge or None if not found
        """
        # Check cache
        if knowledge_id in self._knowledge_cache:
            return self._knowledge_cache[knowledge_id]
        
        # Query from storage
        results = self.unified_memory.structured_query(
            MemoryType.SEMANTIC,
            filters={"knowledge_id": knowledge_id},
            limit=1
        )
        
        if results:
            knowledge = Knowledge.from_dict(results[0].content)
            self._knowledge_cache[knowledge_id] = knowledge
            return knowledge
        
        return None
    
    def search_knowledge(
        self,
        query: str,
        knowledge_type: Optional[KnowledgeType] = None,
        limit: int = 10,
    ) -> List[Knowledge]:
        """
        Search for knowledge using semantic search.
        
        Args:
            query: Search query
            knowledge_type: Filter by knowledge type
            limit: Maximum results
            
        Returns:
            List of matching knowledge
        """
        results = self.unified_memory.semantic_search(
            query,
            MemoryType.SEMANTIC,
            limit=limit,
            similarity_threshold=0.3
        )
        
        knowledge_list = []
        for result in results:
            try:
                # Check if this is knowledge (has 'knowledge_type' field)
                if "knowledge_type" not in result.content:
                    continue
                
                knowledge = Knowledge.from_dict(result.content)
                
                # Apply type filter
                if knowledge_type and knowledge.knowledge_type != knowledge_type:
                    continue
                
                knowledge_list.append(knowledge)
                self._knowledge_cache[knowledge.knowledge_id] = knowledge
            except Exception as e:
                self.logger.warning(f"Failed to parse knowledge: {e}")
                continue
        
        return knowledge_list
    
    def update_knowledge(
        self,
        knowledge_id: str,
        content: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Update existing knowledge.
        
        Args:
            knowledge_id: Knowledge ID
            content: New content
            confidence: New confidence level
        """
        knowledge = self.get_knowledge(knowledge_id)
        if not knowledge:
            return
        
        if content is not None:
            knowledge.content = content
            knowledge.last_updated = datetime.utcnow()
        if confidence is not None:
            knowledge.confidence = max(0.0, min(1.0, confidence))
        
        # Update in storage
        content_dict = knowledge.to_dict()
        self.unified_memory.update(
            knowledge_id,
            MemoryType.SEMANTIC,
            content=content_dict
        )
        
        self.logger.info(f"Updated knowledge: {knowledge.name}")
    
    # ==================== Utility Methods ====================
    
    def get_skill_inventory(self) -> Dict[str, Any]:
        """
        Get complete skill inventory with statistics.
        
        Returns:
            Skill inventory dictionary
        """
        # Get all skills
        results = self.unified_memory.structured_query(
            MemoryType.SEMANTIC,
            filters={},
            limit=10000
        )
        
        skills = []
        for result in results:
            try:
                if "level" in result.content:
                    skill = Skill.from_dict(result.content)
                    skills.append(skill)
            except Exception:
                continue
        
        # Organize by level
        by_level = {level: [] for level in SkillLevel}
        for skill in skills:
            by_level[skill.level].append(skill.name)
        
        # Organize by domain
        by_domain: Dict[str, List[str]] = {}
        for skill in skills:
            for domain in skill.domains:
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(skill.name)
        
        return {
            "total_skills": len(skills),
            "by_level": {level.value: names for level, names in by_level.items()},
            "by_domain": by_domain,
            "top_skills": sorted(
                skills,
                key=lambda s: (self._skill_level_value(s.level), s.confidence),
                reverse=True
            )[:10],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get semantic memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_skills": self.total_skills,
            "total_procedures": self.total_procedures,
            "total_knowledge": self.total_knowledge,
            "cached_skills": len(self._skill_cache),
            "cached_procedures": len(self._procedure_cache),
            "cached_knowledge": len(self._knowledge_cache),
        }
    
    # ==================== Private Helper Methods ====================
    
    def _update_skill_in_storage(self, skill: Skill) -> None:
        """Update skill in storage."""
        content = skill.to_dict()
        self.unified_memory.update(
            skill.skill_id,
            MemoryType.SEMANTIC,
            content=content
        )
    
    @staticmethod
    def _skill_level_value(level: SkillLevel) -> int:
        """Convert skill level to numeric value for comparison."""
        level_map = {
            SkillLevel.NOVICE: 1,
            SkillLevel.BEGINNER: 2,
            SkillLevel.INTERMEDIATE: 3,
            SkillLevel.ADVANCED: 4,
            SkillLevel.EXPERT: 5,
        }
        return level_map.get(level, 0)
