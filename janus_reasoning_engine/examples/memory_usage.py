"""
Example demonstrating the unified memory layer.

Shows how to store and retrieve different types of memories using
HBM, SQLite, and file system backends.
"""

from janus_reasoning_engine.memory import UnifiedMemory, MemoryQuery, MemoryType


def main():
    """Demonstrate memory layer usage."""
    print("=== Janus Reasoning Engine - Memory Layer Demo ===\n")
    
    # Initialize unified memory
    print("Initializing unified memory...")
    memory = UnifiedMemory(
        hbm_dimension=10000,
        sqlite_path="demo_memory.db",
        artifacts_dir="demo_artifacts"
    )
    memory.initialize()
    print("Memory initialized!\n")
    
    # 1. Store episodic memories (experiences)
    print("1. Storing episodic memories (experiences)...")
    
    experience1 = memory.store(
        MemoryType.EPISODIC,
        {
            "context": "Applied for web development job on Upwork",
            "action": "Submitted proposal with portfolio",
            "outcome": "Got hired",
            "earnings": 500.0,
        },
        metadata={
            "platform": "upwork",
            "skill": "web_development",
            "success": True,
        }
    )
    print(f"  Stored experience: {experience1}")
    
    experience2 = memory.store(
        MemoryType.EPISODIC,
        {
            "context": "Applied for data analysis job on Fiverr",
            "action": "Submitted proposal",
            "outcome": "Rejected",
        },
        metadata={
            "platform": "fiverr",
            "skill": "data_analysis",
            "success": False,
        }
    )
    print(f"  Stored experience: {experience2}\n")
    
    # 2. Store semantic memories (knowledge)
    print("2. Storing semantic memories (knowledge)...")
    
    knowledge1 = memory.store(
        MemoryType.SEMANTIC,
        {
            "skill": "React",
            "topic": "Hooks",
            "content": "useState and useEffect are fundamental React hooks",
            "confidence": 0.9,
        }
    )
    print(f"  Stored knowledge: {knowledge1}")
    
    knowledge2 = memory.store(
        MemoryType.SEMANTIC,
        {
            "skill": "Python",
            "topic": "Decorators",
            "content": "Decorators modify function behavior",
            "confidence": 0.85,
        }
    )
    print(f"  Stored knowledge: {knowledge2}\n")
    
    # 3. Store artifacts
    print("3. Storing artifacts...")
    
    artifact1 = memory.store(
        MemoryType.ARTIFACT,
        {
            "filename": "checkpoint.json",
            "data": '{"state": "saved", "step": 100}',
        },
        metadata={
            "type": "checkpoint",
            "tags": ["checkpoint", "state"],
        }
    )
    print(f"  Stored artifact: {artifact1}\n")
    
    # 4. Retrieve memories using structured queries
    print("4. Retrieving memories using structured queries...")
    
    # Get successful experiences
    print("  Successful experiences:")
    success_results = memory.structured_query(
        MemoryType.EPISODIC,
        filters={"success": True},
        limit=10
    )
    for result in success_results:
        print(f"    - {result.content.get('context')}")
    
    # Get all semantic knowledge
    print("\n  All knowledge:")
    knowledge_results = memory.structured_query(
        MemoryType.SEMANTIC,
        limit=10
    )
    for result in knowledge_results:
        print(f"    - {result.content.get('skill')}: {result.content.get('topic')}")
    
    # 5. Semantic search with HBM
    print("\n5. Semantic search (HBM-based)...")
    print("  Searching for 'web development'...")
    semantic_results = memory.semantic_search(
        "web development",
        MemoryType.EPISODIC,
        limit=5,
        similarity_threshold=0.0
    )
    print(f"  Found {len(semantic_results)} results")
    
    # 6. Get statistics
    print("\n6. Memory statistics:")
    stats = memory.get_statistics()
    print(f"  Initialized: {stats['initialized']}")
    print(f"  Backends:")
    for backend_name, backend_stats in stats['backends'].items():
        print(f"    {backend_name}:")
        if backend_name == 'hbm':
            print(f"      Total memories: {backend_stats.get('total_memories', 0)}")
            print(f"      HBM access count: {backend_stats.get('hbm_access_count', 0)}")
        elif backend_name == 'sqlite':
            print(f"      Total memories: {backend_stats.get('total_memories', 0)}")
            print(f"      DB size: {backend_stats.get('db_size_bytes', 0)} bytes")
        elif backend_name == 'filesystem':
            print(f"      Total artifacts: {backend_stats.get('total_artifacts', 0)}")
            print(f"      Total size: {backend_stats.get('total_size_mb', 0)} MB")
    
    # Shutdown
    print("\nShutting down memory...")
    memory.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
