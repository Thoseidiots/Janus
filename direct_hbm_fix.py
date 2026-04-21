#!/usr/bin/env python3
"""
Direct HBM Complex Number Fix
===============================

This script directly patches the HBM module to fix complex number issues.
"""

import os
import sys

def fix_hbm_complex_numbers():
    """Directly patch HBM core.py to handle complex numbers."""
    
    print("=== DIRECT HBM COMPLEX NUMBER FIX ===")
    
    # Find the HBM core file
    hbm_core_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/holographic_brain_memory/core.py"
    
    if not os.path.exists(hbm_core_path):
        print(f"HBM core not found at: {hbm_core_path}")
        return False
    
    # Read the current HBM core
    with open(hbm_core_path, 'r') as f:
        hbm_code = f.read()
    
    # Find the problematic line and fix it
    original_line = "        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
    fixed_line = "        # Handle complex phase weights in matrix multiplication\n        if torch.is_complex(self.phase_weights):\n            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)\n        else:\n            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
    
    if original_line in hbm_code:
        fixed_code = hbm_code.replace(original_line, fixed_line)
        print("✅ Applied complex number fix to HBM core")
        
        # Write the fixed code back
        with open(hbm_core_path, 'w') as f:
            f.write(fixed_code)
        
        print(f"✅ HBM core patched: {hbm_core_path}")
        return True
    else:
        print("❌ Could not find problematic line in HBM core")
        return False

if __name__ == "__main__":
    if fix_hbm_complex_numbers():
        print("\n🎉 HBM COMPLEX NUMBER FIX APPLIED!")
        print("The RuntimeError should now be resolved.")
    else:
        print("❌ Failed to apply HBM fix")
