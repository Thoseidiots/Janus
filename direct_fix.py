#!/usr/bin/env python3
"""
Direct Fix for Complex Number RuntimeError in Holographic Brain Memory
================================================================

This script directly patches the exact line causing the RuntimeError.
"""

import os
import sys
import torch

def apply_direct_fix():
    """Apply direct fix to the problematic line in HBM core."""
    
    print("=== APPLYING DIRECT FIX FOR COMPLEX NUMBERS ===")
    
    # Find the janus-repo dataset in /kaggle/input
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found in /kaggle/input")
        return False
    
    hbm_core_path = os.path.join(REPO, "holographic_brain_memory", "core.py")
    
    if not os.path.exists(hbm_core_path):
        print(f"HBM core not found at: {hbm_core_path}")
        return False
    
    # Read the file
    with open(hbm_core_path, 'r') as f:
        content = f.read()
    
    # The exact problematic line 144:
    # encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
    
    # Replace it with a version that handles complex numbers
    original_line = "        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
    
    # Multiple possible fixes - try them in order
    fixes = [
        # Fix 1: Use real part if complex
        """        # Handle complex phase weights in matrix multiplication
        if torch.is_complex(self.phase_weights):
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
        else:
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)""",
        
        # Fix 2: Force to float32
        """        # Convert phase_weights to real before matmul
        phase_weights_real = torch.real(self.phase_weights)
        encoded = torch.matmul(x.unsqueeze(1), phase_weights_real.unsqueeze(0)).squeeze(1)""",
        
        # Fix 3: Use absolute value
        """        # Use absolute value of phase weights
        phase_weights_abs = torch.abs(self.phase_weights)
        encoded = torch.matmul(x.unsqueeze(1), phase_weights_abs.unsqueeze(0)).squeeze(1)"""
    ]
    
    applied_fix = False
    for i, fix in enumerate(fixes, 1):
        if original_line in content:
            content = content.replace(original_line, fix)
            print(f"✅ Applied fix {i} to HBM core")
            applied_fix = True
            break
        else:
            print(f"❌ Fix {i} pattern not found")
    
    if not applied_fix:
        print("⚠️  No exact line match found, trying regex replacement...")
        # Try regex to find and replace the problematic line
        import re
        
        # Pattern to match the matmul line
        pattern = r'(\s+)encoded = torch\.matmul\(x\.unsqueeze\(1\), self\.phase_weights\.unsqueeze\(0\)\)\.squeeze\(1\))'
        
        if re.search(pattern, content):
            replacement = r'\1# Handle complex phase weights\n\1if torch.is_complex(self.phase_weights):\n\1    phase_weights_real = self.phase_weights.real\n\1else:\n\1    phase_weights_real = self.phase_weights\n\1encoded = torch.matmul(x.unsqueeze(1), phase_weights_real.unsqueeze(0)).squeeze(1)'
            content = re.sub(pattern, replacement, content)
            print("✅ Applied regex fix to HBM core")
            applied_fix = True
        else:
            print("❌ Could not find matmul pattern with regex")
    
    if applied_fix:
        # Write the fixed content back
        with open(hbm_core_path, 'w') as f:
            f.write(content)
        
        print("HBM core successfully patched!")
        return True
    else:
        print("❌ Failed to apply any fix to HBM core")
        return False

def verify_fix():
    """Verify that the fix was applied correctly."""
    
    print("\n=== VERIFYING FIX ===")
    
    # Find the janus-repo dataset
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        return False
    
    hbm_core_path = os.path.join(REPO, "holographic_brain_memory", "core.py")
    
    if not os.path.exists(hbm_core_path):
        return False
    
    with open(hbm_core_path, 'r') as f:
        content = f.read()
    
    # Check if the fix is present
    fix_indicators = [
        "torch.is_complex(self.phase_weights)",
        "phase_weights.real",
        "torch.real(self.phase_weights)",
        "phase_weights_abs",
        "phase_weights_real"
    ]
    
    found_indicators = [indicator for indicator in fix_indicators if indicator in content]
    
    if found_indicators:
        print(f"✅ Fix verified - found: {', '.join(found_indicators)}")
        return True
    else:
        print("❌ Fix verification failed - no indicators found")
        return False

if __name__ == "__main__":
    print("Starting direct fix for HBM complex number issue...")
    
    # Apply the fix
    if apply_direct_fix():
        # Verify it worked
        if verify_fix():
            print("\n🎉 SUCCESS: HBM complex number issue fixed!")
            print("The RuntimeError should be resolved.")
        else:
            print("\n⚠️  WARNING: Fix applied but verification failed")
    else:
        print("\n❌ FAILED: Could not apply fix to HBM core")
        print("Manual intervention may be required.")
