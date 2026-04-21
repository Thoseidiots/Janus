#!/usr/bin/env python3
"""
FINAL Fix for Resume Training Logic
==================================

This script patches the training script to properly check for and load existing weights
instead of always starting from scratch.
"""

import os
import sys
from pathlib import Path

def fix_resume_training():
    """Patch the training script to resume from existing weights."""
    
    print("=== FIXING RESUME TRAINING LOGIC ===")
    
    # Find the training script
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "janus-repo" in dirs:
            REPO = os.path.join(root, "janus-repo")
            break
    
    if REPO is None:
        print("janus-repo not found")
        return False
    
    script_path = os.path.join(REPO, "train_avus_kaggle.py")
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    # Read the current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Find the problematic line where model is created from scratch
    problematic_line = "    model = Avus(cfg).to(device)"
    
    if problematic_line not in script_content:
        print("Could not find model creation line")
        return False
    
    # Replace with resume logic
    replacement = '''    # Check for existing weights to resume training
    KAGGLE_WORKING = Path("/kaggle/working")
    weight_candidates = [
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_best.pt",
        DATASET_DIR / f"avus_{MODEL_SIZE}_weights.pt"
    ]
    
    existing_weights = None
    for weight_path in weight_candidates:
        if weight_path.exists():
            existing_weights = weight_path
            print(f"[avus] Found existing weights: {weight_path.name}")
            break
    
    if existing_weights:
        print(f"[avus] Resuming from: {existing_weights.name}")
        try:
            model = Avus.from_checkpoint(str(existing_weights))
            print(f"[avus] Successfully loaded weights from checkpoint")
        except Exception as e:
            print(f"[avus] Failed to load checkpoint: {e}")
            print("[avus] Falling back to creating new model...")
            model = Avus(cfg).to(device)
    else:
        print("[avus] No existing weights found, training from scratch")
        model = Avus(cfg).to(device)'''
    
    # Apply the fix
    script_content = script_content.replace(problematic_line, replacement)
    
    # Write the fixed script to working directory
    fixed_script_path = "/kaggle/working/train_avus_kaggle_fixed_resume.py"
    with open(fixed_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Fixed training script saved to: {fixed_script_path}")
    return True

if __name__ == "__main__":
    if fix_resume_training():
        print("\nResume training fix applied!")
        print("Run the fixed script:")
        print("exec(open('/kaggle/working/train_avus_kaggle_fixed_resume.py').read())")
    else:
        print("Failed to apply resume training fix")
