"""
Unified memory interface for the Janus Reasoning Engine.

Integrates HBM, SQLite, and file system backends into a single API.
"""

from typing import Any, Dict, List, Optional
import logging

from janus_reasoning_engine.memory.interfaces import (
    MemoryBackend,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)
from janus_reasoning_engine.memory.hbm_backend import HBMBackend
from janus_reasoning_engine.memory.sqlite_backend import SQLiteBackend
from janus_reasoning_engine.memory.filesystem_backend import FileSystemBackend


class UnifiedMemory:
    """
    Unified memory interface integrating multiple backends.
    
    Routes memory operations to the appropriate backend:
    - HBM for associative recall (episodic, semantic, working memory)
    - SQLite for structured data queries
    - File system for artifacts and checkpoints
    """
    
    def __init__(
        self,
        hbm_dimension: int = 10000,
        hbm_sparsity: float = 0.1,
        sqlite_path: str = "janus_reasoning.db",
        artifacts_dir: str = "janus_artifacts",
        enable_hbm: bool = True,
        enable_sqlite: bool = True,
        enable_filesystem: bool = True,
    ):
        """
        Initialize unified memory.
        
        Args:
            hbm_dimension: Dimension for HBM vectors
            hbm_sparsity: Sparsity for HBM encoding
            sqlite_path: Path to SQLite database
            artifacts_dir: Directory for artifacts
            enable_hbm: Enable HBM backend
            enable_sqlite: Enable SQLite backend
            enable_filesystem: Enable file system backend
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize backends
        self.hbm_backend: Optional[HBMBackend] = None
        self.sqlite_backend: Optional[SQLiteBackend] = None
        self.filesystem_backend: Optional[FileSystemBackend] = None
        
        if enable_hbm:
            self.hbm_backend = HBMBackend(
                dimension=hbm_dimension,
                sparsity=hbm_sparsity
            )
        
        if enable_sqlite:
            self.sqlite_backend = SQLiteBackend(db_path=sqlite_path)
        
        if enable_filesystem:
            self.filesystem_backend = FileSystemBackend(base_dir=artifacts_dir)
        
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize all memory backends."""
        if self.initialized:
            self.logger.warning("Unified memory already initialized")
            return
        
        self.logger.info("Initializing unified memory...")
        
        # Initialize backends
        if self.hbm_backend:
            self.hbm_backend.initialize()
            self.logger.info("HBM backend initialized")
        
        if self.sqlite_backend:
            self.sqlite_backend.initialize()
            self.logger.info("SQLite backend initialized")
        
        if self.filesystem_backend:
            self.filesystem_backend.initialize()
            self.logger.info("FileSystem backend initialized")
        
        self.initialized = True
        self.logger.info("Unified memory initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown all memory backends."""
        if not self.initialized:
            return
        
        self.logger.info("Shutting down unified memory...")
        
        # Shutdown backends
        if self.hbm_backend:
            self.hbm_backend.shutdown()
        
        if self.sqlite_backend:
            self.sqlite_backend.shutdown()
        
        if self.filesystem_backend:
            self.filesystem_backend.shutdown()
        
        self.initialized = False
        self.logger.info("Unified memory shutdown complete")
    
    def _get_backend(self, memory_type: MemoryType) -> MemoryBackend:
        """
        Get the appropriate backend for a memory type.
        
        Args:
            memory_type: Type of memory
            
        Returns:
            Appropriate backend
        """
        if memory_type == MemoryType.ARTIFACT:
            if self.filesystem_backend is None:
                raise RuntimeError("FileSystem backend not enabled")
            return self.filesystem_backend
        
        # For episodic, semantic, and working memory, use both HBM and SQLite
        # HBM for associative recall, SQLite for structured queries
        # Default to SQLite if HBM not available
        if self.hbm_backend:
            return self.hbm_backend
        elif self.sqlite_backend:
            return self.sqlite_backend
        else:
            raise RuntimeError("No backend available for memory type")
    
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        use_both: bool = True,
    ) -> str:
        """
        Store a memory.
        
        Args:
            memory_type: Type of memory to store
            content: Memory content
            metadata: Optional metadata
            use_both: Store in both HBM and SQLite (for non-artifact memories)
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        # For artifacts, use file system only
        if memory_type == MemoryType.ARTIFACT:
            return self.filesystem_backend.store(memory_type, content, metadata)
        
        # For other memory types, store in SQLite (primary) and optionally HBM
        memory_id = None
        
        if self.sqlite_backend:
            memory_id = self.sqlite_backend.store(memory_type, content, metadata)
        
        if use_both and self.hbm_backend:
            # Store in HBM for associative recall (use same ID if available)
            # Note: HBM backend generates its own ID, but we track it separately
            self.hbm_backend.store(memory_type, content, metadata)
        
        if memory_id is None and self.hbm_backend:
            # Fallback to HBM if SQLite not available
            memory_id = self.hbm_backend.store(memory_type, content, metadata)
        
        if memory_id is None:
            raise RuntimeError("No backend available to store memory")
        
        return memory_id
    
    def retrieve(
        self,
        query: MemoryQuery,
        use_semantic_search: bool = True,
    ) -> List[MemoryResult]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: Memory query
            use_semantic_search: Use HBM for semantic search (if available)
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        # For artifacts, use file system
        if query.query_type == MemoryType.ARTIFACT:
            if self.filesystem_backend:
                return self.filesystem_backend.retrieve(query)
            return []
        
        # For semantic search with query text, prefer HBM
        if use_semantic_search and query.query_text and self.hbm_backend:
            return self.hbm_backend.retrieve(query)
        
        # Otherwise use SQLite for structured queries
        if self.sqlite_backend:
            return self.sqlite_backend.retrieve(query)
        
        # Fallback to HBM if SQLite not available
        if self.hbm_backend:
            return self.hbm_backend.retrieve(query)
        
        return []
    
    def update(
        self,
        memory_id: str,
        memory_type: MemoryType,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            memory_type: Type of memory
            content: New content (if provided)
            metadata: New metadata (if provided)
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        # Update in SQLite (primary storage)
        if memory_type == MemoryType.ARTIFACT:
            if self.filesystem_backend:
                self.filesystem_backend.update(memory_id, content, metadata)
        else:
            if self.sqlite_backend:
                self.sqlite_backend.update(memory_id, content, metadata)
    
    def delete(self, memory_id: str, memory_type: MemoryType) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
            memory_type: Type of memory
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        # Delete from appropriate backend
        if memory_type == MemoryType.ARTIFACT:
            if self.filesystem_backend:
                self.filesystem_backend.delete(memory_id)
        else:
            if self.sqlite_backend:
                self.sqlite_backend.delete(memory_id)
    
    def semantic_search(
        self,
        query_text: str,
        memory_type: MemoryType,
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[MemoryResult]:
        """
        Perform semantic search using HBM.
        
        Args:
            query_text: Text to search for
            memory_type: Type of memory to search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        if not self.hbm_backend:
            raise RuntimeError("HBM backend not available for semantic search")
        
        query = MemoryQuery(
            query_type=memory_type,
            query_text=query_text,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
        
        return self.hbm_backend.retrieve(query)
    
    def structured_query(
        self,
        memory_type: MemoryType,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemoryResult]:
        """
        Perform structured query using SQLite.
        
        Args:
            memory_type: Type of memory to query
            filters: Field filters
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("Unified memory not initialized")
        
        if not self.sqlite_backend:
            raise RuntimeError("SQLite backend not available for structured queries")
        
        query = MemoryQuery(
            query_type=memory_type,
            filters=filters,
            limit=limit,
        )
        
        return self.sqlite_backend.retrieve(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all backends.
        
        Returns:
            Dictionary with statistics from all backends
        """
        stats = {
            "initialized": self.initialized,
            "backends": {},
        }
        
        if self.hbm_backend:
            stats["backends"]["hbm"] = self.hbm_backend.get_statistics()
        
        if self.sqlite_backend:
            stats["backends"]["sqlite"] = self.sqlite_backend.get_statistics()
        
        if self.filesystem_backend:
            stats["backends"]["filesystem"] = self.filesystem_backend.get_statistics()
        
        return stats
