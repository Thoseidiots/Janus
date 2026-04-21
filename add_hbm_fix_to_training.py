#!/usr/bin/env python3
"""
Add HBM Complex Number Fix Directly to Training Script
====================================================

This script adds the HBM complex number fix directly to train_avus_kaggle.py
so it's applied automatically when training starts.
"""

import os
import sys
from pathlib import Path

def add_hbm_fix_to_training():
    """Add HBM complex number fix directly to training script."""
    
    print("=== ADDING HBM FIX TO TRAINING SCRIPT ===")
    
    # Training script path
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    # Read current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # HBM complex number fix code to add at the beginning
    hbm_fix_code = '''# HBM Complex Number Fix - Applied at training start
import sys
import torch

# Add repo to path
sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")

def apply_hbm_complex_fix():
    """Apply HBM complex number fix before training starts."""
    try:
        import holographic_brain_memory.core as hbm_core
        
        def safe_encode(self, x):
            """Safe encode function that handles complex numbers."""
            if torch.is_complex(self.phase_weights):
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
                print("[HBM] Using real part of complex weights")
            else:
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            return encoded
        
        # Apply the patch
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("[HBM] Complex number fix applied successfully")
        return True
    except Exception as e:
        print(f"[HBM] Failed to apply complex fix: {e}")
        return False

# Apply HBM fix immediately
apply_hbm_complex_fix()

'''
    
    # Find the imports section to add the fix after
    imports_end = script_content.find("from pathlib import Path")
    if imports_end == -1:
        imports_end = script_content.find("import os, sys")
    
    if imports_end != -1:
        # Find the end of the imports section
        next_line = script_content.find('\n', imports_end)
        if next_line != -1:
            # Insert HBM fix after imports
            enhanced_script = script_content[:next_line+1] + hbm_fix_code + script_content[next_line+1:]
            print("HBM fix added after imports section")
        else:
            print("Could not find end of imports section")
            return False
    else:
        # Alternative: add at the very beginning
        enhanced_script = hbm_fix_code + script_content
        print("HBM fix added at beginning of script")
    
    # Write the enhanced script
    enhanced_script_path = "/kaggle/working/train_avus_kaggle_with_hbm_fix.py"
    with open(enhanced_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"Enhanced script saved to: {enhanced_script_path}")
    return True

def create_comprehensive_training_script():
    """Create a comprehensive training script with all fixes."""
    
    print("=== CREATING COMPREHENSIVE TRAINING SCRIPT ===")
    
    # Read the original training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Comprehensive fixes to add
    comprehensive_fixes = '''# COMPREHENSIVE TRAINING FIXES
# =========================

# Fix 1: HBM Complex Number Fix
import sys
import torch
sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")

def apply_hbm_complex_fix():
    try:
        import holographic_brain_memory.core as hbm_core
        def safe_encode(self, x):
            if torch.is_complex(self.phase_weights):
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
            else:
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            return encoded
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("[COMPREHENSIVE] HBM complex fix applied")
        return True
    except Exception as e:
        print(f"[COMPREHENSIVE] HBM fix failed: {e}")
        return False

# Fix 2: Recursion-Safe Device-Aware Linear
import torch.nn as nn
_TRUE_PYTORCH_LINEAR_FORWARD = nn.Linear.__dict__['forward']

def _safe_device_aware_linear(self, x):
    if self.weight.device != x.device:
        self.weight.data = self.weight.data.to(x.device)
    if self.bias is not None and self.bias.device != x.device:
        self.bias.data = self.bias.data.to(x.device)
    return _TRUE_PYTORCH_LINEAR_FORWARD(self, x)

if not hasattr(nn.Linear, '_comprehensive_fix_applied'):
    nn.Linear.forward = _safe_device_aware_linear
    nn.Linear._comprehensive_fix_applied = True
    print("[COMPREHENSIVE] Device-aware linear fix applied")

# Apply all fixes
apply_hbm_complex_fix()

'''
    
    # Find where to insert the comprehensive fixes
    insert_point = script_content.find("import os, sys, json")
    if insert_point != -1:
        # Insert before the first import
        enhanced_script = comprehensive_fixes + script_content
        print("Comprehensive fixes added at beginning")
    else:
        # Alternative: add at very beginning
        enhanced_script = comprehensive_fixes + script_content
        print("Comprehensive fixes added at beginning (fallback)")
    
    # Also add auto cleanup after each epoch
    epoch_completion_line = """        print(f"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}")"""
    
    cleanup_code = '''
        # COMPREHENSIVE CLEANUP - Delete old files to prevent disk full
        KAGGLE_WORKING = Path("/kaggle/working")
        MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
        
        print(f"[cleanup] Starting comprehensive cleanup...")
        
        # Delete old epoch files (keep only current)
        epoch_files = list(KAGGLE_WORKING.glob(f"avus_{MODEL_SIZE}_epoch_*.pt"))
        current_epoch_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch+1}.pt"
        
        for epoch_file in epoch_files:
            if epoch_file != current_epoch_file:
                try:
                    size_mb = epoch_file.stat().st_size / (1024*1024)
                    epoch_file.unlink()
                    print(f"[cleanup] DELETED: {epoch_file.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"[cleanup] Failed to delete {epoch_file.name}: {e}")
        
        # Delete database files (they take too much space)
        for db_file in KAGGLE_WORKING.glob("*.db"):
            try:
                size_mb = db_file.stat().st_size / (1024*1024)
                db_file.unlink()
                print(f"[cleanup] DELETED DB: {db_file.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {db_file.name}: {e}")
        
        # Check disk space
        try:
            stat = os.statvfs(str(KAGGLE_WORKING))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_gb = total_gb - free_gb
            print(f"[cleanup] Disk: {used_gb:.1f}/{total_gb:.1f} GB ({free_gb:.2f} GB free)")
        except Exception as e:
            print(f"[cleanup] Could not check disk space: {e}")'''
    
    if epoch_completion_line in enhanced_script:
        enhanced_script = enhanced_script.replace(
            epoch_completion_line,
            epoch_completion_line + cleanup_code
        )
        print("Comprehensive cleanup added after each epoch")
    
    # Write the comprehensive script
    comprehensive_script_path = "/kaggle/working/train_avus_kaggle_comprehensive.py"
    with open(comprehensive_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"Comprehensive script saved to: {comprehensive_script_path}")
    return True

if __name__ == "__main__":
    print("Adding HBM complex number fix to training script...")
    
    # Add just HBM fix
    hbm_fix_added = add_hbm_fix_to_training()
    
    # Create comprehensive script with all fixes
    comprehensive_created = create_comprehensive_training_script()
    
    print(f"\n{'='*60}")
    print("TRAINING SCRIPT FIXES:")
    print(f"{'='*60}")
    print(f"HBM fix added: {'YES' if hbm_fix_added else 'NO'}")
    print(f"Comprehensive script created: {'YES' if comprehensive_created else 'NO'}")
    
    if comprehensive_created:
        print(f"\nTo use training with ALL fixes:")
        print("exec(open('/kaggle/working/train_avus_kaggle_comprehensive.py').read())")
        print(f"\nThis script includes:")
        print("  - HBM complex number fix")
        print("  - Recursion-safe device-aware linear")
        print("  - Automatic cleanup after each epoch")
        print("  - All previous fixes integrated")
    
    print(f"\n{'='*60}")
