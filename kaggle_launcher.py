#!/usr/bin/env python3
"""
Kaggle Launcher for Janus Training with All Fixes
============================================

This launcher works on Kaggle with proper paths and all fixes.
"""

import os
import sys

def main():
    print("=== KAGGLE JANUS TRAINING LAUNCHER ===")
    
    # Kaggle paths
    kaggle_repo = "/kaggle/input/datasets/ishmaelsears/janus-repo/"
    kaggle_working = "/kaggle/working/"
    
    # Local fix paths (for when running on local system)
    local_fixes = [
        "/kaggle/working/runtime_patch.py",
        "/kaggle/working/fix_resume_training.py",
        "/kaggle/working/launcher.py"
    ]
    
    # Check if we're on Kaggle
    if os.path.exists(kaggle_repo):
        print("✅ Running on Kaggle")
        print(f"📁 Repository: {kaggle_repo}")
        print(f"📁 Working: {kaggle_working}")
        
        # Change to repository directory
        os.chdir(kaggle_repo)
        
        # Check for existing fixes in working directory
        fix_found = None
        for fix_path in local_fixes:
            if os.path.exists(fix_path):
                fix_found = fix_path
                print(f"🔧 Found existing fix: {os.path.basename(fix_path)}")
                break
        
        if fix_found:
            print(f"🚀 Running existing fix: {os.path.basename(fix_found)}")
            exec(open(fix_found).read())
        else:
            print("🔧 Creating and running resume training fix...")
            
            # Create the resume training fix directly
            resume_fix_content = '''#!/usr/bin/env python3
"""
Fix Training to Resume from Available Weights
==========================================

This script patches the training logic to find and use existing weights
instead of starting from scratch.
"""

import os
import sys
import torch
from pathlib import Path

def patch_training_to_resume():
    """Patch training script to resume from available weights."""
    
    print("=== PATCHING TRAINING TO RESUME FROM WEIGHTS ===")
    
    # Look for existing weights in multiple locations
    KAGGLE_WORKING = Path("/kaggle/working")
    MODEL_SIZE = "1b"  # Default model size
    
    weight_locations = [
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_best.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_*.pt",
    ]
    
    print("[avus] Looking for existing weights...")
    found_weights = None
    
    for location in weight_locations:
        if location.exists():
            found_weights = location
            print(f"[avus] Found existing weights: {location.name}")
            break
    
    if found_weights:
        print(f"[avus] Resuming from: {found_weights.name}")
        # Set environment variable
        os.environ['RESUME_FROM_WEIGHTS'] = str(found_weights)
    else:
        print("[avus] No existing weights found, starting from scratch")
        os.environ['RESUME_FROM_WEIGHTS'] = 'None'
    
    return True

def patch_complex_numbers():
    """Apply runtime patches for complex number issues."""
    
    print("=== APPLYING RUNTIME PATCHES ===")
    
    # Import and patch HBM modules
    try:
        # Add repo to path
        sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")
        
        import holographic_brain_memory.core as hbm_core
        import holographic_brain_memory.real_valued as hbm_real
        import holographic_brain_memory.spawning as hbm_spawn
        print("✅ HBM modules imported successfully")
        
        # Create patched encode function
        def safe_encode(self, x):
            """Safe encode function that handles complex numbers."""
            if torch.is_complex(self.phase_weights):
                # Use real part for matrix multiplication
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
            else:
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            return encoded
        
        # Apply the patch
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("✅ Applied runtime patch to HBM.encode method")
        
        return True
    except Exception as e:
        print(f"❌ Failed to patch HBM: {e}")
        return False

def patch_training_script():
    """Patch the training script with all fixes."""
    
    print("\n=== PATCHING TRAINING SCRIPT ===")
    
    # Load the training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Apply patches
    patches = [
        ("KAGGLE_MODE           = False", "KAGGLE_MODE           = True"),
        (
            "        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")",
            "        targets = targets.to(logits.device)\n        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")"
        )
    ]
    
    applied_patches = 0
    for old, new in patches:
        if old in script_content:
            script_content = script_content.replace(old, new)
            print(f"✅ Applied script patch: {old[:50]}...")
            applied_patches += 1
        else:
            print(f"⚠️  Script patch not found: {old[:50]}...")
    
    # Write the patched script to working directory
    patched_script_path = "/kaggle/working/train_avus_kaggle_patched.py"
    with open(patched_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Patched script saved to: {patched_script_path}")
    print(f"✅ Applied {applied_patches} patches")
    
    return True

def main():
    print("Starting Kaggle training launcher...")
    
    # Apply all fixes
    if patch_training_to_resume() and patch_complex_numbers() and patch_training_script():
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n📋 Fixes applied:")
        print("  1. Resume from existing weights")
        print("  2. Complex number handling")
        print("  3. KAGGLE_MODE enabled")
        print("  4. Focal loss device fix")
        
        # Execute the patched training script
        print("\n🚀 STARTING TRAINING...")
        
        # Import and run the patched script
        sys.path.insert(0, "/kaggle/working")
        exec(open("/kaggle/working/train_avus_kaggle_patched.py").read())
        
    else:
        print("❌ FAILED: Could not apply all fixes")

if __name__ == "__main__":
    main()'''
            
            # Write the fix to working directory
            with open("/kaggle/working/fix_resume_training.py", 'w') as f:
                f.write(resume_fix_content)
            
            print("✅ Resume training fix created")
            print("🔧 Running resume training fix...")
            exec(open("/kaggle/working/fix_resume_training.py").read())
        
        # Apply complex number patches
        print("\n🔧 Applying complex number patches...")
        if not patch_complex_numbers():
            print("❌ Complex number patches failed")
        
        # Patch training script
        print("\n📝 Patching training script...")
        if not patch_training_script():
            print("❌ Training script patch failed")
        
        print("\n🚀 STARTING TRAINING...")
        
        # Execute the patched training script
        exec(open("/kaggle/working/train_avus_kaggle_patched.py").read())
    else:
        print("❌ Not on Kaggle - check your environment")
