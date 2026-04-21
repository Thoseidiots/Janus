#!/usr/bin/env python3
"""
Fix Auto Cleanup and HBM Complex Number Issues
============================================

This script fixes both the auto cleanup not working and the HBM complex number error.
"""

import os
import sys
import shutil
from pathlib import Path

def fix_auto_cleanup():
    """Fix auto cleanup to actually delete files each epoch."""
    
    print("=== FIXING AUTO CLEANUP ===")
    
    # Find the training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    # Read current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Find the exact epoch completion line and add cleanup after it
    epoch_completion_line = """        print(f"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}")"""
    
    # Aggressive cleanup code to insert
    aggressive_cleanup = '''
        # AGGRESSIVE CLEANUP - Delete old files to prevent disk full
        KAGGLE_WORKING = Path("/kaggle/working")
        MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
        
        print(f"[cleanup] Starting aggressive cleanup...")
        
        # Delete ALL old epoch files except current one
        epoch_files = list(KAGGLE_WORKING.glob(f"avus_{MODEL_SIZE}_epoch_*.pt"))
        current_epoch_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch+1}.pt"
        
        for epoch_file in epoch_files:
            if epoch_file != current_epoch_file:  # Don't delete current epoch
                try:
                    size_mb = epoch_file.stat().st_size / (1024*1024)
                    epoch_file.unlink()
                    print(f"[cleanup] DELETED old epoch: {epoch_file.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"[cleanup] Failed to delete {epoch_file.name}: {e}")
        
        # Delete mid-epoch checkpoints
        midepoch_files = list(KAGGLE_WORKING.glob("*midepoch*.pt"))
        for midepoch in midepoch_files:
            try:
                size_mb = midepoch.stat().st_size / (1024*1024)
                midepoch.unlink()
                print(f"[cleanup] DELETED mid-epoch: {midepoch.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {midepoch.name}: {e}")
        
        # Delete database files (they take too much space)
        db_files = list(KAGGLE_WORKING.glob("*.db"))
        for db_file in db_files:
            try:
                size_mb = db_file.stat().st_size / (1024*1024)
                db_file.unlink()
                print(f"[cleanup] DELETED database: {db_file.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {db_file.name}: {e}")
        
        # Delete fallback files
        fallback_files = list(KAGGLE_WORKING.glob("*fallback*.pt"))
        for fallback in fallback_files:
            try:
                size_mb = fallback.stat().st_size / (1024*1024)
                fallback.unlink()
                print(f"[cleanup] DELETED fallback: {fallback.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {fallback.name}: {e}")
        
        # Delete old charts (keep only latest)
        chart_files = list(KAGGLE_WORKING.glob("skill_chart*.png"))
        chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for old_chart in chart_files[1:]:  # Keep only newest
            try:
                old_chart.unlink()
                print(f"[cleanup] DELETED old chart: {old_chart.name}")
            except Exception as e:
                print(f"[cleanup] Failed to delete {old_chart.name}: {e}")
        
        # Check disk space after cleanup
        try:
            stat = os.statvfs(str(KAGGLE_WORKING))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_gb = total_gb - free_gb
            print(f"[cleanup] Disk after cleanup: {used_gb:.1f}/{total_gb:.1f} GB ({free_gb:.2f} GB free)")
            
            if free_gb < 2.0:
                print("[cleanup] WARNING: Still low on space!")
            else:
                print(f"[cleanup] GOOD: {free_gb:.2f} GB free")
        except Exception as e:
            print(f"[cleanup] Could not check disk space: {e}")
        
        # List remaining files
        remaining_files = list(KAGGLE_WORKING.glob("*.pt"))
        total_size_mb = sum(f.stat().st_size for f in remaining_files) / (1024*1024)
        print(f"[cleanup] Remaining {len(remaining_files)} weight files ({total_size_mb:.1f} MB total)")'''
    
    if epoch_completion_line in script_content:
        enhanced_script = script_content.replace(
            epoch_completion_line,
            epoch_completion_line + aggressive_cleanup
        )
        print("Auto cleanup code added successfully!")
    else:
        print("Could not find epoch completion line")
        return False
    
    # Write the enhanced script
    enhanced_script_path = "/kaggle/working/train_avus_kaggle_with_cleanup.py"
    with open(enhanced_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"Enhanced script saved to: {enhanced_script_path}")
    return True

def fix_hbm_complex_numbers():
    """Fix the HBM complex number error with runtime patching."""
    
    print("=== FIXING HBM COMPLEX NUMBERS ===")
    
    try:
        # Add repo to path
        sys.path.insert(0, "/kaggle/input/datasets/ishmaelsears/janus-repo")
        
        import holographic_brain_memory.core as hbm_core
        import torch
        
        # Create a robust patch for the encode function
        def safe_encode(self, x):
            """Safe encode function that handles complex numbers."""
            # Check if phase_weights is complex
            if torch.is_complex(self.phase_weights):
                # Use real part for matrix multiplication
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.real.unsqueeze(0)).squeeze(1)
                print(f"[HBM] Used real part of complex weights")
            else:
                # Normal case - weights are already real
                encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
            
            return encoded
        
        # Apply the patch
        hbm_core.HolographicBrainMemory.encode = safe_encode
        print("HBM complex number fix applied successfully!")
        
        # Test the fix
        try:
            # Create a test instance to verify the fix
            test_hbm = hbm_core.HolographicBrainMemory(
                in_dim=10, 
                memory_dim=20, 
                capacity=100
            )
            
            # Test with complex weights
            import torch
            test_hbm.phase_weights = torch.complex(torch.randn(20, 10), torch.randn(20, 10))
            
            # Test encoding
            test_input = torch.randn(5, 10)
            result = test_hbm.encode(test_input)
            
            print(f"HBM fix test passed! Result shape: {result.shape}")
            return True
            
        except Exception as test_error:
            print(f"HBM fix test failed: {test_error}")
            return False
            
    except Exception as e:
        print(f"Failed to apply HBM fix: {e}")
        return False

def create_quick_cleanup():
    """Create a quick cleanup function for immediate use."""
    
    cleanup_code = '''
def quick_cleanup():
    """Quick cleanup of /kaggle/working."""
    import os, shutil
    from pathlib import Path
    
    KAGGLE_WORKING = Path("/kaggle/working")
    
    print("[quick_cleanup] Starting immediate cleanup...")
    
    # Delete old epoch files (keep only latest)
    epoch_files = list(KAGGLE_WORKING.glob("avus_*_epoch_*.pt"))
    epoch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for old_epoch in epoch_files[1:]:  # Keep only newest
        try:
            size_mb = old_epoch.stat().st_size / (1024*1024)
            old_epoch.unlink()
            print(f"[quick_cleanup] DELETED: {old_epoch.name} ({size_mb:.1f} MB)")
        except:
            pass
    
    # Delete database files
    for db_file in KAGGLE_WORKING.glob("*.db"):
        try:
            size_mb = db_file.stat().st_size / (1024*1024)
            db_file.unlink()
            print(f"[quick_cleanup] DELETED DB: {db_file.name} ({size_mb:.1f} MB)")
        except:
            pass
    
    # Check remaining space
    try:
        stat = os.statvfs(str(KAGGLE_WORKING))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"[quick_cleanup] Free space: {free_gb:.2f} GB")
    except:
        pass'''
    
    with open("/kaggle/working/quick_cleanup.py", 'w') as f:
        f.write(cleanup_code)
    
    print("Quick cleanup function created: /kaggle/working/quick_cleanup.py")
    return True

if __name__ == "__main__":
    print("Fixing auto cleanup and HBM complex number issues...")
    
    # Fix auto cleanup
    cleanup_fixed = fix_auto_cleanup()
    
    # Fix HBM complex numbers
    hbm_fixed = fix_hbm_complex_numbers()
    
    # Create quick cleanup
    quick_cleanup_created = create_quick_cleanup()
    
    print(f"\n{'='*60}")
    print("FIX SUMMARY:")
    print(f"{'='*60}")
    print(f"Auto cleanup fixed: {'YES' if cleanup_fixed else 'NO'}")
    print(f"HBM complex fix: {'YES' if hbm_fixed else 'NO'}")
    print(f"Quick cleanup created: {'YES' if quick_cleanup_created else 'NO'}")
    
    if cleanup_fixed:
        print(f"\nTo use training with auto cleanup:")
        print("exec(open('/kaggle/working/train_avus_kaggle_with_cleanup.py').read())")
    
    if quick_cleanup_created:
        print(f"\nFor immediate cleanup:")
        print("exec(open('/kaggle/working/quick_cleanup.py').read())")
        print("quick_cleanup()")
    
    print(f"\n{'='*60}")
