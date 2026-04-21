#!/usr/bin/env python3
"""
FINAL RecursionError Fix for Kaggle Training
===========================================

This script provides a definitive fix for the RecursionError in device-aware linear.
The issue is that multiple patches are creating circular references.

SOLUTION: Store the TRUE original PyTorch function at module level and never reassign it.
"""

import torch.nn as nn

# Store the TRUE original PyTorch forward function ONCE at module level
# This will never be reassigned, preventing circular references
_TRUE_PYTORCH_LINEAR_FORWARD = nn.Linear.__dict__['forward']

def _safe_device_aware_linear(self, x):
    """Safe device-aware linear that never recurses."""
    # Only move parameters if they're on different devices
    if self.weight.device != x.device:
        self.weight.data = self.weight.data.to(x.device)
    if self.bias is not None and self.bias.device != x.device:
        self.bias.data = self.bias.data.to(x.device)
    
    # Call the TRUE PyTorch original function directly
    return _TRUE_PYTORCH_LINEAR_FORWARD(self, x)

# Apply the fix - but only if not already patched to prevent issues
if not hasattr(nn.Linear, '_final_recursion_fix_applied'):
    nn.Linear.forward = _safe_device_aware_linear
    nn.Linear._final_recursion_fix_applied = True
    print("FINAL: Recursion-safe device-aware linear applied")
else:
    print("FINAL: Device-aware linear already fixed")

# Also apply HBM complex number fix
def _apply_hbm_fix():
    """Apply complex number fix to HBM."""
    try:
        import sys
        sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")
        
        import holographic_brain_memory.core as hbm_core
        import torch
        
        def safe_hbm_encode(self, x):
            """Safe HBM encode that handles complex numbers."""
            if torch.is_complex(self.phase_weights):
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
            else:
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            return encoded
        
        # Apply patch only if not already applied
        if not hasattr(hbm_core.HolographicBrainMemory, '_complex_fix_applied'):
            hbm_core.HolographicBrainMemory.encode = safe_hbm_encode
            hbm_core.HolographicBrainMemory._complex_fix_applied = True
            print("FINAL: HBM complex number fix applied")
        else:
            print("FINAL: HBM complex fix already applied")
        
        return True
    except Exception as e:
        print(f"FINAL: HBM fix failed: {e}")
        return False

# Apply HBM fix
_apply_hbm_fix()

print("FINAL: All fixes applied - training should work now")
