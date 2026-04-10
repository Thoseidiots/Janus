import torch
import glob
import os
import pickle
import re
import logging
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union
from holographic_memory import (
    HolographicMemoryCore,
    _encode_text,
    _make_theta,
    _text_seed
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridRetriever")

class HybridHolographicRetriever:
    """
    Hybrid retriever optimized for long-term intelligence accumulation in Janus.
    
    Architecture:
    - Primary Path: Holographic similarity (associative, fixed-size, infinite superposition).
    - Fallback Path: Lightweight keyword index for precise retrieval in noisy regimes.
    
    Features:
    - No decay, no forgetting: Older memories remain accessible.
    - Multi-seed probing: Increases recall by checking multiple semantic variations.
    - Hybrid Fusion: Prioritizes holographic results but fills gaps with keyword hits.
    """

    def __init__(self, memory_dir: str = "pocket_dimension", dim: int = 2048):
        self.memory_dir = os.path.abspath(memory_dir)
        self.dim = dim
        self.cores: List[HolographicMemoryCore] = []
        self.keyword_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Ensure memory directory exists
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
            logger.info(f"Created memory directory: {self.memory_dir}")
            
        self.load_all_cores()

    def _extract_text_for_indexing(self, data: Any) -> str:
        """Extracts readable text from various data formats for keyword indexing."""
        if isinstance(data, dict):
            # Prioritize known Janus schema keys
            parts = []
            for key in ["text", "result", "value_preview", "action", "stimulus", "thought", "goal"]:
                if key in data and data[key]:
                    parts.append(str(data[key]))
            
            # If no specific keys found, join all string values
            if not parts:
                parts = [str(v) for v in data.values() if isinstance(v, (str, int, float))]
                
            return " | ".join(parts)
        return str(data)

    def load_all_cores(self):
        """Loads all holographic cores from disk and builds the keyword index."""
        self.cores.clear()
        self.keyword_index.clear()
        
        # Support multiple extensions used in Janus
        patterns = ["*.holo", "*.bin", "*.pkl", "*.pt"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.memory_dir, pattern)))
        
        files = sorted(files, key=os.path.getmtime) # Sort by modification time
        
        for file_idx, file_path in enumerate(files):
            try:
                # Use weights_only=True for security with torch.load
                if file_path.endswith(('.pkl', '.bin')):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    data = torch.load(file_path, weights_only=True, map_location='cpu')
                
                # Initialize and load core
                core = HolographicMemoryCore(dim=self.dim)
                
                # Handle different Janus memory serialization formats
                memory_data = None
                if isinstance(data, dict):
                    if "core" in data and isinstance(data["core"], dict) and "memory" in data["core"]:
                        memory_data = data["core"]["memory"]
                    elif "memory" in data:
                        memory_data = data["memory"]
                else:
                    memory_data = data
                
                if memory_data is not None:
                    if isinstance(memory_data, torch.Tensor):
                        core.memory.data = memory_data.to(torch.cfloat)
                    else:
                        core.memory.data = torch.tensor(memory_data, dtype=torch.cfloat)
                    
                    self.cores.append(core)
                    
                    # Build keyword index for fallback
                    text = self._extract_text_for_indexing(data)
                    if text.strip():
                        # Tokenize and filter small words
                        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
                        for word in words:
                            self.keyword_index[word].append({
                                "core_idx": len(self.cores) - 1,
                                "snippet": text[:280],
                                "file": os.path.basename(file_path),
                                "timestamp": os.path.getmtime(file_path)
                            })
            except Exception as e:
                logger.warning(f"Failed to load memory core {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.cores)} holographic cores. Keyword index: {len(self.keyword_index)} terms.")

    def _keyword_fallback(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs a keyword-based search as a reliable fallback."""
        words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        hits = defaultdict(float)
        
        for word in words:
            if word in self.keyword_index:
                for entry in self.keyword_index[word]:
                    # Simple scoring: count matches
                    hits[entry["core_idx"]] += 1.0
        
        ranked = sorted(
            [{"core_idx": idx, "score": score, "type": "keyword"} 
             for idx, score in hits.items()],
            key=lambda x: x["score"], reverse=True
        )
        return ranked[:top_k]

    def retrieve(self, query: str, top_k: int = 8, min_similarity: float = 0.15) -> List[Dict[str, Any]]:
        """
        Main retrieval interface. Uses holographic search with keyword fallback.
        """
        if not self.cores:
            return []
        
        # Encode query once
        q_vec = _encode_text(query, self.dim)
        holographic_results = []
        
        # Multi-seed probing to handle semantic variations
        probe_variations = [
            query,
            f"{query}_ep",   # Episodic context
            f"{query}_sem",  # Semantic context
            f"{query}_ref"   # Reflection context
        ]
        
        seeds = [_text_seed(v) for v in probe_variations]
        
        # Search across all cores
        for core_idx, core in enumerate(self.cores):
            best_sim = -1.0
            best_vec = None
            best_seed = None
            
            for seed in seeds:
                theta = _make_theta(seed, self.dim)
                retrieved = core.read(theta)
                
                # Calculate similarity
                sim = F.cosine_similarity(q_vec.unsqueeze(0), retrieved.unsqueeze(0)).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_vec = retrieved
                    best_seed = seed
            
            if best_sim >= min_similarity:
                holographic_results.append({
                    "score": best_sim,
                    "core_idx": core_idx,
                    "type": "holographic",
                    "vector": best_vec,
                    "seed": best_seed
                })
        
        # Sort by similarity score
        holographic_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Hybrid Fusion
        final = holographic_results[:top_k]
        
        # If we don't have enough results, use keyword fallback
        if len(final) < top_k:
            keyword_hits = self._keyword_fallback(query, top_k=top_k - len(final))
            for hit in keyword_hits:
                # Avoid duplicates
                if not any(r["core_idx"] == hit["core_idx"] for r in final):
                    final.append(hit)
        
        # Final sort (holographic usually scores higher than keyword match counts)
        final.sort(key=lambda x: x["score"], reverse=True)
        return final[:top_k]

    def refresh(self):
        """Re-scans the memory directory for new cores."""
        self.load_all_cores()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the retriever."""
        return {
            "cores_loaded": len(self.cores),
            "memory_directory": self.memory_dir,
            "keyword_terms": len(self.keyword_index),
            "dimension": self.dim,
            "total_estimated_size_kb": len(self.cores) * (self.dim * 8 / 1024) # complex64 = 8 bytes/element
        }

if __name__ == "__main__":
    # Quick test/demo
    retriever = HybridHolographicRetriever()
    print(f"Status: {retriever.get_status()}")
    results = retriever.retrieve("test query")
    print(f"Found {len(results)} results.")
