"""
SQLite backend for structured data storage.

Handles goals, opportunities, experiences, and other structured data.
"""

import sqlite3
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
from pathlib import Path

from janus_reasoning_engine.memory.interfaces import (
    MemoryBackend,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)


class SQLiteBackend(MemoryBackend):
    """
    SQLite backend for structured data storage.
    
    Provides persistent storage for goals, opportunities, experiences,
    and other structured data that needs to be queried efficiently.
    """
    
    def __init__(self, db_path: str = "janus_reasoning.db"):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the SQLite backend."""
        if self.initialized:
            return
        
        # Create database directory if needed
        db_dir = Path(self.db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
        
        self.initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the SQLite backend."""
        if not self.initialized:
            return
        
        if self.conn:
            self.conn.close()
            self.conn = None
        
        self.initialized = False
    
    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.conn.cursor()
        
        # Memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indices for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type 
            ON memories(memory_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON memories(timestamp)
        """)
        
        self.conn.commit()
    
    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory in SQLite.
        
        Args:
            memory_type: Type of memory
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            raise RuntimeError("SQLite backend not initialized")
        
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Serialize content and metadata
        content_json = json.dumps(content)
        metadata_json = json.dumps(metadata or {})
        
        # Get timestamp
        timestamp = datetime.utcnow().isoformat()
        
        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO memories (memory_id, memory_type, content, metadata, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, memory_type.value, content_json, metadata_json, timestamp, timestamp))
        
        self.conn.commit()
        
        return memory_id
    
    def retrieve(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Retrieve memories from SQLite.
        
        Args:
            query: Memory query
            
        Returns:
            List of matching memories
        """
        if not self.initialized:
            raise RuntimeError("SQLite backend not initialized")
        
        cursor = self.conn.cursor()
        
        # Build SQL query
        sql = "SELECT * FROM memories WHERE memory_type = ?"
        params = [query.query_type.value]
        
        # Add filters (check both content and metadata)
        if query.filters:
            for key, value in query.filters.items():
                # Check in both content and metadata JSON fields
                sql += f" AND (json_extract(content, '$.{key}') = ? OR json_extract(metadata, '$.{key}') = ?)"
                params.extend([value, value])
        
        # Add text search if provided
        if query.query_text:
            sql += " AND (content LIKE ? OR metadata LIKE ?)"
            search_term = f"%{query.query_text}%"
            params.extend([search_term, search_term])
        
        # Order by timestamp (most recent first)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(query.limit)
        
        # Execute query
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # Convert to MemoryResult objects
        results = []
        for row in rows:
            results.append(MemoryResult(
                memory_id=row["memory_id"],
                memory_type=MemoryType(row["memory_type"]),
                content=json.loads(row["content"]),
                similarity_score=1.0,  # SQLite doesn't compute similarity
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
            ))
        
        return results
    
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
            raise RuntimeError("SQLite backend not initialized")
        
        cursor = self.conn.cursor()
        
        # Build update query
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(json.dumps(content))
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if not updates:
            return
        
        # Update timestamp
        updates.append("timestamp = ?")
        params.append(datetime.utcnow().isoformat())
        
        # Add memory_id to params
        params.append(memory_id)
        
        # Execute update
        sql = f"UPDATE memories SET {', '.join(updates)} WHERE memory_id = ?"
        cursor.execute(sql, params)
        
        self.conn.commit()
    
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
        """
        if not self.initialized:
            raise RuntimeError("SQLite backend not initialized")
        
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get SQLite backend statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.initialized:
            return {"backend_type": "SQLite", "initialized": False}
        
        cursor = self.conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        total_count = cursor.fetchone()["count"]
        
        # Get counts by type
        cursor.execute("""
            SELECT memory_type, COUNT(*) as count 
            FROM memories 
            GROUP BY memory_type
        """)
        type_counts = {row["memory_type"]: row["count"] for row in cursor.fetchall()}
        
        # Get database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()["size"]
        
        return {
            "backend_type": "SQLite",
            "db_path": self.db_path,
            "total_memories": total_count,
            "memories_by_type": type_counts,
            "db_size_bytes": db_size,
        }
