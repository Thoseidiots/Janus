"""
File system backend for artifacts and checkpoints.

Handles storage and retrieval of files, checkpoints, and large artifacts.
"""

import os
import json
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import uuid

from janus_reasoning_engine.memory.interfaces import (
    MemoryBackend,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)


class FileSystemBackend(MemoryBackend):
    """
    File system backend for artifacts and checkpoints.
    
    Stores files, checkpoints, and large artifacts on disk with
    metadata tracking in a JSON index.
    """
    
    def __init__(self, base_dir: str = "janus_artifacts"):
        """
        Initialize file system backend.
        
        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "index.json"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        # In-memory index
        self.index: Dict[str, Dict[str, Any]] = {}
        
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the file system backend."""
        if self.initialized:
            return
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index if it exists
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        
        self.initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the file system backend."""
        if not self.initialized:
            return
        
        # Save index
        self._save_index()
        
        self.initialized = False
    
    def _save_index(self) -> None:
        """Save the index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, default=str)
    
    def _get_artifact_path(self, memory_id: str, filename: str) -> Path:
        """Get the full path for an artifact file."""
        return self.artifacts_dir / memory_id / filename
    
    def _get_checkpoint_path(self, memory_id: str) -> Path:
        """Get the full path for a checkpoint directory."""
        return self.checkpoints_dir / memory_id
    
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an artifact or checkpoint.
        
        Args:
            memory_type: Type of memory (should be ARTIFACT)
            content: Content with 'data' or 'file_path' key
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            raise RuntimeError("FileSystem backend not initialized")
        
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Determine storage location
        if "checkpoint" in (metadata or {}).get("tags", []):
            storage_path = self._get_checkpoint_path(memory_id)
        else:
            filename = content.get("filename", "artifact.bin")
            storage_path = self._get_artifact_path(memory_id, filename)
        
        # Create directory
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store content
        if "data" in content:
            # Store raw data
            if isinstance(content["data"], (str, bytes)):
                mode = 'wb' if isinstance(content["data"], bytes) else 'w'
                with open(storage_path, mode) as f:
                    f.write(content["data"])
            else:
                # Store as JSON
                with open(storage_path, 'w') as f:
                    json.dump(content["data"], f, indent=2)
        
        elif "file_path" in content:
            # Copy existing file
            source_path = Path(content["file_path"])
            if source_path.exists():
                if source_path.is_file():
                    shutil.copy2(source_path, storage_path)
                else:
                    shutil.copytree(source_path, storage_path)
        
        # Update index
        self.index[memory_id] = {
            "memory_type": memory_type.value,
            "storage_path": str(storage_path),
            "content": {k: v for k, v in content.items() if k not in ["data", "file_path"]},
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Save index
        self._save_index()
        
        return memory_id
    
    def retrieve(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve artifacts or checkpoints.
        
        Args:
            query: Memory query
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("FileSystem backend not initialized")
        
        results = []
        
        for memory_id, memory_data in self.index.items():
            # Filter by memory type
            if memory_data["memory_type"] != query.query_type.value:
                continue
            
            # Apply filters
            if query.filters:
                match = True
                for key, value in query.filters.items():
                    if memory_data.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Text search in metadata
            if query.query_text:
                metadata_str = json.dumps(memory_data.get("metadata", {}))
                if query.query_text.lower() not in metadata_str.lower():
                    continue
            
            # Add to results
            results.append(MemoryResult(
                memory_id=memory_id,
                memory_type=query.query_type,
                content={
                    **memory_data["content"],
                    "storage_path": memory_data["storage_path"],
                },
                similarity_score=1.0,
                timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                metadata=memory_data["metadata"],
            ))
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        
        return results[:query.limit]
    
    def update(
        self,
        memory_id: str,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update artifact metadata.
        
        Args:
            memory_id: ID of memory to update
            content: New content (if provided)
            metadata: New metadata (if provided)
        """
        if not self.initialized:
            raise RuntimeError("FileSystem backend not initialized")
        
        if memory_id not in self.index:
            raise KeyError(f"Memory {memory_id} not found")
        
        memory_data = self.index[memory_id]
        
        # Update content metadata (not the file itself)
        if content is not None:
            memory_data["content"].update({k: v for k, v in content.items() if k not in ["data", "file_path"]})
        
        # Update metadata
        if metadata is not None:
            memory_data["metadata"].update(metadata)
        
        memory_data["timestamp"] = datetime.utcnow().isoformat()
        
        # Save index
        self._save_index()
    
    def delete(self, memory_id: str) -> None:
        """
        Delete an artifact or checkpoint.
        
        Args:
            memory_id: ID of memory to delete
        """
        if not self.initialized:
            raise RuntimeError("FileSystem backend not initialized")
        
        if memory_id not in self.index:
            return
        
        memory_data = self.index[memory_id]
        storage_path = Path(memory_data["storage_path"])
        
        # Delete file or directory
        if storage_path.exists():
            if storage_path.is_file():
                storage_path.unlink()
            else:
                shutil.rmtree(storage_path)
        
        # Remove from index
        del self.index[memory_id]
        
        # Save index
        self._save_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get file system backend statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.initialized:
            return {"backend_type": "FileSystem", "initialized": False}
        
        # Calculate total size
        total_size = 0
        file_count = 0
        
        for memory_data in self.index.values():
            storage_path = Path(memory_data["storage_path"])
            if storage_path.exists():
                if storage_path.is_file():
                    total_size += storage_path.stat().st_size
                    file_count += 1
                else:
                    for file_path in storage_path.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
        
        return {
            "backend_type": "FileSystem",
            "base_dir": str(self.base_dir),
            "total_artifacts": len(self.index),
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
