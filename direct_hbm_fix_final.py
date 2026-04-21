#!/usr/bin/env python3
"""
Direct HBM Complex Number Fix - FINAL
====================================

This script directly modifies the HBM core.py file to fix the complex number issue.
"""

import os
import sys
import shutil
from pathlib import Path

def direct_hbm_fix():
    """Directly patch HBM core.py file to handle complex numbers."""
    
    print("=== DIRECT HBM COMPLEX NUMBER FIX ===")
    
    # HBM core file path
    hbm_core_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/holographic_brain_memory/core.py"
    
    if not os.path.exists(hbm_core_path):
        print(f"HBM core not found: {hbm_core_path}")
        return False
    
    print(f"Found HBM core: {hbm_core_path}")
    
    # Read the current HBM core
    with open(hbm_core_path, 'r') as f:
        hbm_code = f.read()
    
    # Find the problematic encode method
    original_encode = '''    def encode(self, x):
        """
        Encode input tensor into holographic memory space.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Encoded tensor of shape (batch_size, memory_dim)
        """
        # x: (batch, in_dim) -> (batch, memory.dim)
        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
        return encoded'''
    
    # Fixed encode method with complex number handling
    fixed_encode = '''    def encode(self, x):
        """
        Encode input tensor into holographic memory space.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Encoded tensor of shape (batch_size, memory_dim)
        """
        # x: (batch, in_dim) -> (batch, memory.dim)
        # Handle complex phase_weights by using real part for matmul
        if torch.is_complex(self.phase_weights):
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
        else:
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
        return encoded'''
    
    # Apply the fix
    if original_encode in hbm_code:
        fixed_code = hbm_code.replace(original_encode, fixed_encode)
        print("Applied complex number fix to HBM encode method")
    else:
        # Try alternative approach - find just the problematic line
        problematic_line = "        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
        fixed_line = "        # Handle complex phase_weights by using real part for matmul\n        if torch.is_complex(self.phase_weights):\n            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)\n        else:\n            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
        
        if problematic_line in hbm_code:
            fixed_code = hbm_code.replace(problematic_line, fixed_line)
            print("Applied complex number fix to problematic line")
        else:
            print("Could not find encode method or problematic line")
            return False
    
    # Create backup of original file
    backup_path = hbm_core_path + ".backup"
    shutil.copy2(hbm_core_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Write the fixed code back
    try:
        with open(hbm_core_path, 'w') as f:
            f.write(fixed_code)
        print(f"Fixed HBM core written: {hbm_core_path}")
        return True
    except Exception as e:
        print(f"Failed to write fixed HBM core: {e}")
        # Restore from backup
        shutil.copy2(backup_path, hbm_core_path)
        return False

def create_runtime_hbm_fix():
    """Create a runtime fix that can be applied before HBM training."""
    
    runtime_fix = '''
import sys
import torch

# Add repo to path
sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")

# Apply HBM complex number fix
def apply_hbm_complex_fix():
    """Apply runtime fix for HBM complex numbers."""
    try:
        import holographic_brain_memory.core as hbm_core
        
        def safe_encode(self, x):
            """Safe encode function that handles complex numbers."""
            if torch.is_complex(self.phase_weights):
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
                print(f"[HBM] Used real part of complex weights")
            else:
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            return encoded
        
        # Apply the patch
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("HBM complex number fix applied at runtime")
        return True
    except Exception as e:
        print(f"Failed to apply HBM runtime fix: {e}")
        return False

# Apply the fix immediately
apply_hbm_complex_fix()'''
    
    with open("/kaggle/working/hbm_runtime_fix.py", 'w') as f:
        f.write(runtime_fix)
    
    print("Runtime HBM fix created: /kaggle/working/hbm_runtime_fix.py")
    return True

def verify_hbm_fix():
    """Verify the HBM fix is working."""
    
    print("=== VERIFYING HBM FIX ===")
    
    try:
        # Test the fixed HBM module
        sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")
        import holographic_brain_memory.core as hbm_core
        import torch
        
        # Create test HBM instance
        test_hbm = hbm_core.HolographicBrainMemory(
            in_dim=10, 
            memory_dim=20, 
            capacity=100
        )
        
        # Test with complex weights
        test_hbm.phase_weights = torch.complex(torch.randn(20, 10), torch.randn(20, 10))
        
        # Test encoding
        test_input = torch.randn(5, 10)
        result = test_hbm.encode(test_input)
        
        print(f"HBM fix verification PASSED!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")
        return True
        
    except Exception as e:
        print(f"HBM fix verification FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Applying DIRECT HBM complex number fix...")
    
    # Try direct file fix first
    direct_fix_success = direct_hbm_fix()
    
    # Create runtime fix as backup
    runtime_fix_created = create_runtime_hbm_fix()
    
    # Verify the fix
    if direct_fix_success:
        fix_verified = verify_hbm_fix()
    else:
        print("Direct fix failed, trying runtime fix...")
        exec(open("/kaggle/working/hbm_runtime_fix.py").read())
        fix_verified = verify_hbm_fix()
    
    print(f"\n{'='*60}")
    print("HBM FIX SUMMARY:")
    print(f"{'='*60}")
    print(f"Direct file fix: {'SUCCESS' if direct_fix_success else 'FAILED'}")
    print(f"Runtime fix created: {'YES' if runtime_fix_created else 'NO'}")
    print(f"Fix verification: {'PASSED' if fix_verified else 'FAILED'}")
    
    if direct_fix_success:
        print(f"\nHBM core.py has been directly patched!")
        print(f"The complex number error should be resolved.")
    elif runtime_fix_created:
        print(f"\nUse runtime fix before HBM training:")
        print(f"exec(open('/kaggle/working/hbm_runtime_fix.py').read())")
    
    print(f"\n{'='*60}")
