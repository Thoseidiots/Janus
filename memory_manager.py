import chromadb
from chromadb.config import Settings
import os
import json
import time

class JanusMemoryManager:
    def __init__(self, persist_directory="/home/ubuntu/Janus/memory_db"):
        self.persist_directory = persist_directory
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Collection for conversation history
        self.conversation_col = self.client.get_or_create_collection(
            name="conversation_history",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for meta-knowledge and heuristics
        self.knowledge_col = self.client.get_or_create_collection(
            name="meta_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def add_message(self, role, content, metadata=None):
        timestamp = str(time.time())
        doc_id = f"msg_{timestamp}"
        
        if metadata is None:
            metadata = {}
        metadata.update({"role": role, "timestamp": timestamp})
        
        self.conversation_col.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Added message to memory: {doc_id}")

    def search_memory(self, query, n_results=5):
        results = self.conversation_col.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def add_knowledge(self, topic, content, metadata=None):
        timestamp = str(time.time())
        doc_id = f"kn_{topic}_{timestamp}"
        
        if metadata is None:
            metadata = {}
        metadata.update({"topic": topic, "timestamp": timestamp})
        
        self.knowledge_col.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Added knowledge to memory: {doc_id}")

    def search_knowledge(self, query, n_results=3):
        results = self.knowledge_col.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    # Test the memory manager
    mem = JanusMemoryManager()
    mem.add_message("user", "How do I build a recursive self-improving AI?")
    mem.add_message("assistant", "That is a complex task involving meta-cognitive loops and secure execution substrates.")
    
    print("\nSearching memory for 'recursive AI'...")
    results = mem.search_memory("recursive AI")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"[{meta['role']}]: {doc}")
