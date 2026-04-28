"""
HolographicBrainMemory backend for associative recall.

Integrates the existing HBM system for pattern-based memory retrieval.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json

from holographic_brain_memory.core import HolographicBrainMemory
from janus_reasoning_engine.memory.interfaces import (
    MemoryBackend,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)


class HBMBackend(MemoryBackend):
    """
    HolographicBrainMemory backend for associative recall.
    
    Uses complex-valued holographic representations for efficient
    pattern-based memory storage and retrieval.
    """
    
    def __init__(self, dimension: int = 10000, sparsity: float = 0.1):
        """
        Initialize HBM backend.
        
        Args:
            dimension: Dimension of holographic vectors
            sparsity: Sparsity level for encoding
        """
        self.dimension = dimension
        self.sparsity = sparsity
        self.hbm = HolographicBrainMemory(dim=dimension)
        
        # Memory index: maps memory IDs to metadata
        self.memory_index: Dict[str, Dict[str, Any]] = {}
        
        # Encoding cache: maps content to holographic vectors
        self.encoding_cache: Dict[str, torch.Tensor] = {}
        
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the HBM backend."""
        if self.initialized:
            return
        
        # HBM is already initialized in __init__
        self.initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the HBM backend."""
        if not self.initialized:
            return
        
        # Clear caches
        self.encoding_cache.clear()
        self.memory_index.clear()
        
        # Reset HBM
        self.hbm.reset()
        
        self.initialized = False
    
    def _encode_content(self, content: Dict[str, Any]) -> torch.Tensor:
        """
        Encode content as a holographic vector.
        
        Args:
            content: Content to encode
            
        Returns:
            Complex-valued holographic vector
        """
        # Convert content to string representation
        content_str = json.dumps(content, sort_keys=True)
        
        # Check cache
        if content_str in self.encoding_cache:
            return self.encoding_cache[content_str]
        
        # Create sparse random vector based on content hash
        np.random.seed(hash(content_str) % (2**32))
        
        # Generate sparse complex vector
        real_part = np.zeros(self.dimension)
        imag_part = np.zeros(self.dimension)
        
        num_active = int(self.dimension * self.sparsity)
        active_indices = np.random.choice(self.dimension, num_active, replace=False)
        
        real_part[active_indices] = np.random.randn(num_active)
        imag_part[active_indices] = np.random.randn(num_active)
        
        # Normalize
        magnitude = np.sqrt(real_part**2 + imag_part**2).sum()
        if magnitude > 0:
            real_part /= magnitude
            imag_part /= magnitude
        
        # Convert to complex tensor
        vector = torch.tensor(real_part + 1j * imag_part, dtype=torch.complex64)
        
        # Cache encoding
        self.encoding_cache[content_str] = vector
        
        return vector
    
    def _compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute similarity between two holographic vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Compute dot product of complex vectors
        dot_product = torch.dot(torch.conj(vec1), vec2)
        
        # Take absolute value and normalize
        similarity = torch.abs(dot_product).item()
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))
    
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory in HBM.
        
        Args:
            memory_type: Type of memory
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            raise RuntimeError("HBM backend not initialized")
        
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Encode content as holographic vector
        content_vector = self._encode_content(content)
        
        # Store in HBM (write to holographic memory)
        self.hbm.write(content_vector, content_vector, strength=1.0)
        
        # Store metadata in index
        self.memory_index[memory_id] = {
            "memory_type": memory_type.value,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow(),
            "vector": content_vector,
        }
        
        return memory_id
    
    def retrieve(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve memories using associative recall.
        
        Args:
            query: Memory query
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("HBM backend not initialized")
        
        if query.query_text is None:
            # No query text - return all memories of the specified type
            results = []
            for memory_id, memory_data in self.memory_index.items():
                if memory_data["memory_type"] == query.query_type.value:
                    results.append(MemoryResult(
                        memory_id=memory_id,
                        memory_type=query.query_type,
                        content=memory_data["content"],
                        similarity_score=1.0,
                        timestamp=memory_data["timestamp"],
                        metadata=memory_data["metadata"],
                    ))
            return results[:query.limit]
        
        # Encode query as holographic vector
        query_content = {"query": query.query_text}
        query_vector = self._encode_content(query_content)
        
        # Retrieve from HBM
        retrieved_vector = self.hbm.read(query_vector)
        
        # Find similar memories in index
        results = []
        for memory_id, memory_data in self.memory_index.items():
            # Filter by memory type
            if memory_data["memory_type"] != query.query_type.value:
                continue
            
            # Compute similarity
            similarity = self._compute_similarity(
                retrieved_vector,
                memory_data["vector"]
            )
            
            # Filter by threshold
            if similarity >= query.similarity_threshold:
                results.append(MemoryResult(
                    memory_id=memory_id,
                    memory_type=query.query_type,
                    content=memory_data["content"],
                    similarity_score=similarity,
                    timestamp=memory_data["timestamp"],
                    metadata=memory_data["metadata"],
                ))
        
        # Sort by similarity and limit
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:query.limit]
    
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
        if not self.initialized:
            raise RuntimeError("HBM backend not initialized")
        
        if memory_id not in self.memory_index:
            raise KeyError(f"Memory {memory_id} not found")
        
        memory_data = self.memory_index[memory_id]
        
        # Update content if provided
        if content is not None:
            # Re-encode and store in HBM
            new_vector = self._encode_content(content)
            self.hbm.write(new_vector, new_vector, strength=1.0)
            
            memory_data["content"] = content
            memory_data["vector"] = new_vector
        
        # Update metadata if provided
        if metadata is not None:
            memory_data["metadata"].update(metadata)
        
        memory_data["timestamp"] = datetime.utcnow()
    
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
        """
        if not self.initialized:
            raise RuntimeError("HBM backend not initialized")
        
        if memory_id in self.memory_index:
            del self.memory_index[memory_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get HBM backend statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "backend_type": "HBM",
            "dimension": self.dimension,
            "sparsity": self.sparsity,
            "total_memories": len(self.memory_index),
            "hbm_access_count": self.hbm.access_count,
            "hbm_magnitude": self.hbm.get_magnitude(),
            "cache_size": len(self.encoding_cache),
        }
