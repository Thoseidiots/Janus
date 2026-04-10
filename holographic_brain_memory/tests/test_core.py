"""
Unit tests for core HBM components and real-valued HRR
"""

import torch
import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from holographic_brain_memory.core import HolographicBrainMemory, PhaseBrainLayer
from holographic_brain_memory.real_valued import RealHolographicMemory, RealPhaseBrainLayer


class TestHolographicBrainMemory(unittest.TestCase):
    """Test cases for HolographicBrainMemory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_dim = 128
        self.memory = HolographicBrainMemory(dim=self.memory_dim)
    
    def test_initialization(self):
        """Test memory initialization."""
        self.assertEqual(self.memory.dim, self.memory_dim)
        self.assertEqual(self.memory.memory.shape[0], self.memory_dim)
        self.assertEqual(self.memory.memory.dtype, torch.complex64)
    
    def test_bind_unbind(self):
        """Test bind and unbind operations."""
        a = torch.randn(self.memory_dim, dtype=torch.complex64)
        b = torch.randn(self.memory_dim, dtype=torch.complex64)
        
        # Bind and unbind should approximately recover b
        bound = self.memory.bind(a, b)
        unbound = self.memory.unbind(bound, a)
        
        # Check that unbound is close to b
        similarity = torch.abs(torch.dot(unbound, b.conj())).item()
        self.assertGreater(similarity, 0.1)  # Should have some similarity
    
    def test_write_read(self):
        """Test write and read operations."""
        key = torch.randn(self.memory_dim, dtype=torch.complex64)
        value = torch.randn(self.memory_dim, dtype=torch.complex64)
        
        # Write to memory
        self.memory.write(key, value, strength=1.0)
        
        # Read from memory
        retrieved = self.memory.read(key)
        
        # Check that retrieved has non-zero magnitude
        magnitude = torch.abs(retrieved).mean().item()
        self.assertGreater(magnitude, 0.0)
    
    def test_reset(self):
        """Test memory reset."""
        key = torch.randn(self.memory_dim, dtype=torch.complex64)
        value = torch.randn(self.memory_dim, dtype=torch.complex64)
        
        self.memory.write(key, value)
        self.assertGreater(self.memory.get_magnitude(), 0.0)
        
        self.memory.reset()
        self.assertEqual(self.memory.get_magnitude(), 0.0)


class TestPhaseBrainLayer(unittest.TestCase):
    """Test cases for PhaseBrainLayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_dim = 32
        self.memory_dim = 128
        self.memory = HolographicBrainMemory(dim=self.memory_dim)
        self.layer = PhaseBrainLayer(in_dim=self.in_dim, memory=self.memory)
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.in_dim, self.in_dim)
        self.assertEqual(self.layer.out_dim, self.memory_dim)
    
    def test_encode(self):
        """Test encoding operation."""
        x = torch.randn(1, self.in_dim)
        encoded = self.layer.encode(x)
        
        self.assertEqual(encoded.shape, (1, self.memory_dim))
        self.assertEqual(encoded.dtype, torch.complex64)
    
    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(1, self.in_dim)
        output = self.layer(x)
        
        self.assertEqual(output.shape, (1, self.layer.out_dim))
        self.assertEqual(output.dtype, torch.float32)


class TestRealHolographicMemory(unittest.TestCase):
    """Test cases for RealHolographicMemory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_dim = 128
        self.memory = RealHolographicMemory(dim=self.memory_dim)
    
    def test_initialization(self):
        """Test memory initialization."""
        self.assertEqual(self.memory.dim, self.memory_dim)
        self.assertEqual(self.memory.memory.shape[0], self.memory_dim)
        self.assertEqual(self.memory.memory.dtype, torch.float32)
    
    def test_bind_unbind(self):
        """Test bind and unbind operations."""
        a = torch.randn(self.memory_dim, dtype=torch.float32)
        b = torch.randn(self.memory_dim, dtype=torch.float32)
        
        # Bind and unbind
        bound = self.memory.bind(a, b)
        unbound = self.memory.unbind(bound, a)
        
        # Check shapes
        self.assertEqual(bound.shape, (self.memory_dim,))
        self.assertEqual(unbound.shape, (self.memory_dim,))
    
    def test_write_read(self):
        """Test write and read operations."""
        key = torch.randn(self.memory_dim, dtype=torch.float32)
        value = torch.randn(self.memory_dim, dtype=torch.float32)
        
        self.memory.write(key, value, strength=1.0)
        retrieved = self.memory.read(key)
        
        self.assertEqual(retrieved.shape, (self.memory_dim,))
        self.assertGreater(torch.abs(retrieved).mean().item(), 0.0)


class TestRealPhaseBrainLayer(unittest.TestCase):
    """Test cases for RealPhaseBrainLayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_dim = 32
        self.memory_dim = 128
        self.memory = RealHolographicMemory(dim=self.memory_dim)
        self.layer = RealPhaseBrainLayer(in_dim=self.in_dim, memory=self.memory)
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.in_dim, self.in_dim)
        self.assertEqual(self.layer.out_dim, self.memory_dim)
    
    def test_encode(self):
        """Test encoding operation."""
        x = torch.randn(1, self.in_dim)
        encoded = self.layer.encode(x)
        
        self.assertEqual(encoded.shape, (1, self.memory_dim))
        self.assertEqual(encoded.dtype, torch.float32)
    
    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(1, self.in_dim)
        output = self.layer(x)
        
        self.assertEqual(output.shape, (1, self.layer.out_dim))
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
