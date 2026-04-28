"""
Memory layer interfaces for the Janus Reasoning Engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class MemoryType(Enum):
    """Types of memory storage."""
    EPISODIC = "episodic"  # Experiences and events
    SEMANTIC = "semantic"  # Knowledge and facts
    WORKING = "working"  # Active context
    ARTIFACT = "artifact"  # Files and checkpoints


@dataclass
class MemoryQuery:
    """Query for retrieving memories."""
    query_type: MemoryType
    query_text: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    similarity_threshold: float = 0.5
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryResult:
    """Result from memory retrieval."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    similarity_score: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemoryBackend(ABC):
    """
    Abstract base class for memory backends.
    
    Different backends handle different types of memory:
    - HBM backend for associative recall
    - SQLite backend for structured data
    - File system backend for artifacts
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the memory backend."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the memory backend."""
        pass
    
    @abstractmethod
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory.
        
        Args:
            memory_type: Type of memory to store
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: Memory query
            
        Returns:
            List of matching memories
        """
        pass
    
    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            content: New content (if provided)
            metadata: New metadata (if provided)
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get backend statistics.
        
        Returns:
            Dictionary with statistics
        """
        pass
