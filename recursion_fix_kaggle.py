#!/usr/bin/env python3
"""
Fix RecursionError for Kaggle Environment
======================================

This script fixes the infinite recursion in device-aware linear for Kaggle.
"""

import torch.nn as nn

# Store the original forward function BEFORE any patching
_orig_linear_forward = nn.Linear.forward

# Create a non-recursive version that calls the original directly
def _device_aware_linear(self, x):
    # Only move parameters if they're on different devices
    if self.weight.device != x.device:
        self.weight = nn.Parameter(self.weight.to(x.device), requires_grad=self.weight.requires_grad)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(x.device), requires_grad=self.bias.requires_grad)
    
    # Call the ORIGINAL forward directly, not the patched one
    return _orig_linear_forward(self, x)

# Apply the fix
nn.Linear.forward = _device_aware_linear

print("RecursionError fixed in device-aware linear")
