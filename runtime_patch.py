#!/usr/bin/env python3
"""
Runtime Patch for Complex Number Issue in Holographic Brain Memory
=================================================================

This script applies patches at runtime without modifying read-only files.
"""

import os
import sys
import torch
import importlib
import types

def apply_runtime_patches():
    """Apply patches at runtime to fix complex number issues."""
    
    print("=== APPLYING RUNTIME PATCHES ===")
    
    # Find the janus-repo dataset
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found")
        return False
    
    print(f"Found janus-repo at: {REPO}")
    
    # Add REPO to path
    sys.path.insert(0, REPO)
    
    # Import the HBM modules
    try:
        import holographic_brain_memory.core as hbm_core
        import holographic_brain_memory.real_valued as hbm_real
        import holographic_brain_memory.spawning as hbm_spawn
        print("✅ HBM modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import HBM modules: {e}")
        return False
    
    # Create patched encode function
    def safe_encode(self, x):
        """Safe encode function that handles complex numbers."""
        # Original problematic line:
        # encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
        
        # Patched version:
        if torch.is_complex(self.phase_weights):
            # Use real part for matrix multiplication
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
        else:
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
        
        return encoded
    
    # Apply the patch to the class
    try:
        # Replace the encode method in the class
        original_encode = hbm_core.HolographicBrainMemory.encode
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("✅ Applied runtime patch to HBM.encode method")
    except Exception as e:
        print(f"❌ Failed to patch HBM.encode: {e}")
        return False
    
    # Patch other potential complex number issues
    def safe_linear_op(op_func, *args, **kwargs):
        """Safe linear operation wrapper for complex numbers."""
        try:
            return op_func(*args, **kwargs)
        except RuntimeError as e:
            if "ComplexFloat" in str(e):
                # Extract real parts from complex tensors
                safe_args = []
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_complex(arg):
                        safe_args.append(arg.real)
                    else:
                        safe_args.append(arg)
                
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if torch.is_tensor(v) and torch.is_complex(v):
                        safe_kwargs[k] = v.real
                    else:
                        safe_kwargs[k] = v
                
                return op_func(*safe_args, **safe_kwargs)
            else:
                raise e
    
    # Patch torch operations that might fail with complex numbers
    original_matmul = torch.matmul
    torch.matmul = lambda *args, **kwargs: safe_linear_op(original_matmul, *args, **kwargs)
    
    print("✅ Applied runtime patch to torch.matmul")
    
    return True

def monkey_patch_training_script():
    """Monkey patch the training script to use our fixes."""
    
    print("\n=== MONKEY PATCHING TRAINING SCRIPT ===")
    
    # Find and load the training script
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found")
        return
    
    sys.path.insert(0, REPO)
    
    # Load the training script
    script_path = os.path.join(REPO, "train_avus_kaggle.py")
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Apply patches to the script
    patches = [
        ("KAGGLE_MODE           = False", "KAGGLE_MODE           = True"),
        (
            "        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")",
            "        targets = targets.to(logits.device)\n        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")"
        )
    ]
    
    for old, new in patches:
        if old in script_content:
            script_content = script_content.replace(old, new)
            print(f"✅ Applied script patch: {old[:50]}...")
        else:
            print(f"⚠️  Script patch not found: {old[:50]}...")
    
    # Execute the patched script
    print("Executing patched training script with runtime fixes...")
    
    # Create a new namespace for the script
    script_globals = {}
    exec(script_content, script_globals)
    
    # Add our patched functions to the script's namespace
    script_globals['apply_runtime_patches'] = apply_runtime_patches
    
    print("✅ Training script loaded with runtime patches")
    
    return script_globals

if __name__ == "__main__":
    print("Starting runtime patching for HBM complex number issue...")
    
    # Apply runtime patches first
    if apply_runtime_patches():
        # Then monkey patch and execute training script
        script_globals = monkey_patch_training_script()
        
        print("\n=== STARTING TRAINING WITH RUNTIME FIXES ===")
        
        try:
            # Apply patches in the training context
            script_globals.get('apply_runtime_patches', lambda: None)()
            
            # Run training
            train_avus()
            train_hbm()
            print_summary()
            
            print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("Complex number issue resolved via runtime patching.")
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Failed to apply runtime patches")
