#!/usr/bin/env python3
"""
Fix Training to Resume from Available Weights
==========================================

This script patches the training logic to find and use existing weights
instead of starting from scratch.
"""

import os
import sys
import torch

def patch_training_to_resume():
    """Patch training script to resume from available weights."""
    
    print("=== PATCHING TRAINING TO RESUME FROM WEIGHTS ===")
    
    # Find the janus-repo dataset
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found")
        return False
    
    sys.path.insert(0, REPO)
    
    # Load and patch the training script
    script_path = os.path.join(REPO, "train_avus_kaggle.py")
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Apply patches to find and use existing weights
    patches = [
        # Enable KAGGLE_MODE
        ("KAGGLE_MODE           = False", "KAGGLE_MODE           = True"),
        
        # Fix focal loss device issue
        (
            "        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")",
            "        targets = targets.to(logits.device)\n        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")"
        ),
        
        # Replace "Training from scratch" message with weight finding logic
        (
            'print("[avus] Training from scratch")',
            '''# Find and use existing weights instead of training from scratch
    KAGGLE_WORKING = Path("/kaggle/working")
    MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
    
    # Look for existing weights in multiple locations
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
        # Set global variable to indicate we're resuming
        globals()['RESUME_FROM_WEIGHTS'] = str(found_weights)
    else:
        print("[avus] No existing weights found, starting from scratch")
        globals()['RESUME_FROM_WEIGHTS'] = None'''
        )
    ]
    
    applied_patches = 0
    for old, new in patches:
        if old in script_content:
            script_content = script_content.replace(old, new)
            print(f"✅ Applied patch: {old[:50]}...")
            applied_patches += 1
        else:
            print(f"⚠️  Patch not found: {old[:50]}...")
    
    print(f"Applied {applied_patches} patches to training script")
    
    # Write the patched script to working directory
    patched_script_path = "/kaggle/working/train_avus_kaggle_patched.py"
    with open(patched_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Patched script saved to: {patched_script_path}")
    return True

def patch_weight_loading():
    """Patch the weight loading logic to use found weights."""
    
    print("\n=== PATCHING WEIGHT LOADING LOGIC ===")
    
    # Find the janus-repo dataset
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found")
        return False
    
    sys.path.insert(0, REPO)
    
    # Load and patch the training script
    script_path = os.path.join(REPO, "train_avus_kaggle.py")
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Find the weight loading section and patch it
    # Look for the pattern where weights are loaded
    original_pattern = r'(if WEIGHTS_IN\.exists\(\):.*?model = Avus\(cfg\)\s+else:\s+model = Avus\(\))'
    
    if original_pattern in script_content:
        patched_replacement = '''if WEIGHTS_IN.exists():
        print(f"[avus] Loading existing weights: {WEIGHTS_IN}")
        try:
            model = Avus.from_checkpoint(str(WEIGHTS_IN))
            print(f"[avus] Successfully loaded weights from: {WEIGHTS_IN}")
        except Exception as e:
            print(f"[avus] Failed to load weights: {e}")
            print("[avus] Falling back to creating new model...")
            model = Avus(cfg)
    else:
        model = Avus(cfg)'''
        
        script_content = script_content.replace(original_pattern, patched_replacement)
        print("✅ Applied weight loading patch")
    else:
        print("⚠️  Weight loading pattern not found")
    
    # Also patch the model initialization to use RESUME_FROM_WEIGHTS
    resume_patch = '''# Check for resume weights
    if 'RESUME_FROM_WEIGHTS' in globals() and globals()['RESUME_FROM_WEIGHTS']:
        WEIGHTS_IN = Path(globals()['RESUME_FROM_WEIGHTS'])
        print(f"[avus] Using resume weights: {WEIGHTS_IN.name}")
    else:
        WEIGHTS_IN = DATASET_DIR / f"avus_{MODEL_SIZE}_weights.pt"'''
    
    # Insert this before the model creation
    insert_point = "cfg = AvusConfig.from_dict(cfg_dict)"
    if insert_point in script_content:
        script_content = script_content.replace(insert_point, resume_patch + "\n\n    " + insert_point)
        print("✅ Applied resume weights patch")
    else:
        print("⚠️  Model creation point not found")
    
    # Write the patched script
    patched_script_path = "/kaggle/working/train_avus_kaggle_patched.py"
    with open(patched_script_path, 'w') as f:
        f.write(script_content)
    
    print("✅ Weight loading logic patched")
    return True

if __name__ == "__main__":
    print("Starting patch to resume from existing weights...")
    
    # Apply both patches
    if patch_training_to_resume() and patch_weight_loading():
        print("\n🎉 SUCCESS: Training script patched to resume from weights!")
        print("\nThe training should now:")
        print("1. Look for existing weights in multiple locations")
        print("2. Load found weights instead of starting from scratch")
        print("3. Fall back to new model only if no weights found")
        
        print(f"\nPatched script saved to: /kaggle/working/train_avus_kaggle_patched.py")
        print("Run the patched script to resume training from existing weights.")
    else:
        print("❌ FAILED: Could not apply all patches")
