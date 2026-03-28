import torch
import unittest
import sys
import os

# Add the parent directory to sys.path to import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from holographic_brain_memory.core import HolographicBrainMemory, PhaseBrainLayer
from holographic_brain_memory.real_valued import RealHolographicMemory

class TestHolographicMemory(unittest.TestCase):
    def test_initialization(self):
        dim = 512
        memory = HolographicBrainMemory(dim=dim)
        self.assertEqual(memory.dim, dim)
        self.assertEqual(memory.memory.shape, (dim,))
        self.assertTrue(memory.memory.is_complex())

    def test_bind_unbind_cycle(self):
        dim = 1024
        memory = HolographicBrainMemory(dim=dim)
        signal = torch.randn(dim)
        theta = torch.rand(dim) * 2 * torch.pi
        
        # Write to memory
        memory.write(signal, theta)
        
        # Read back
        retrieved = memory.read(theta)
        
        # Check similarity (should be highly correlated)
        similarity = torch.cosine_similarity(signal.unsqueeze(0), retrieved.unsqueeze(0)).item()
        print(f"Retrieval Similarity: {similarity:.4f}")
        self.assertGreater(similarity, 0.5) # Threshold for a single item

    def test_superposition_capacity(self):
        dim = 2048
        memory = HolographicBrainMemory(dim=dim, decay=1.0) # No decay for this test
        
        num_items = 10
        signals = [torch.randn(dim) for _ in range(num_items)]
        thetas = [torch.rand(dim) * 2 * torch.pi for _ in range(num_items)]
        
        # Store all items
        for s, t in zip(signals, thetas):
            memory.write(s, t)
            
        # Retrieve all items
        avg_similarity = 0
        for s, t in zip(signals, thetas):
            retrieved = memory.read(t)
            sim = torch.cosine_similarity(s.unsqueeze(0), retrieved.unsqueeze(0)).item()
            avg_similarity += sim
            
        avg_similarity /= num_items
        print(f"Average Similarity for {num_items} items: {avg_similarity:.4f}")
        # With superposition, similarity drops as 1/sqrt(N). For N=10, 0.1-0.2 is reasonable
        self.assertGreater(avg_similarity, 0.05)

    def test_byte_compression(self):
        dim = 1024
        memory = HolographicBrainMemory(dim=dim)
        target_bytes = 512
        packed = memory.compress_bytes(target_bytes)
        self.assertLessEqual(packed.numel(), target_bytes)

class TestRealHolographicMemory(unittest.TestCase):
    def test_real_bind_unbind(self):
        dim = 1024
        memory = RealHolographicMemory(dim=dim)
        signal = torch.randn(dim)
        key = torch.randn(dim)
        
        # Write to memory
        memory.write(signal, key)
        
        # Read back
        retrieved = memory.read(key)
        
        # Check similarity
        similarity = torch.cosine_similarity(signal.unsqueeze(0), retrieved.unsqueeze(0)).item()
        print(f"Real Retrieval Similarity: {similarity:.4f}")
        self.assertGreater(similarity, 0.4)

if __name__ == "__main__":
    unittest.main()
