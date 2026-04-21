#!/usr/bin/env python3
"""
Automatic File Cleanup During Training
===================================

This script patches training to automatically delete old files
to prevent disk from filling up.
"""

import os
import shutil
from pathlib import Path

def add_auto_cleanup():
    """Add automatic file deletion to training script."""
    
    print("=== ADDING AUTOMATIC CLEANUP ===")
    
    # Find training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    # Read current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Auto cleanup code to insert after each epoch
    auto_cleanup_code = '''        # Automatic cleanup to prevent disk full
        KAGGLE_WORKING = Path("/kaggle/working")
        MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
        
        # Delete old epoch files (keep only last 2)
        epoch_files = list(KAGGLE_WORKING.glob(f"avus_{MODEL_SIZE}_epoch_*.pt"))
        epoch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_epoch in epoch_files[2:]:  # Keep last 2 epochs
            try:
                size_mb = old_epoch.stat().st_size / (1024*1024)
                old_epoch.unlink()
                print(f"[cleanup] Deleted old epoch: {old_epoch.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {old_epoch.name}: {e}")
        
        # Delete mid-epoch checkpoints
        midepoch_files = list(KAGGLE_WORKING.glob(f"*midepoch*.pt"))
        for midepoch in midepoch_files:
            try:
                size_mb = midepoch.stat().st_size / (1024*1024)
                midepoch.unlink()
                print(f"[cleanup] Deleted mid-epoch: {midepoch.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {midepoch.name}: {e}")
        
        # Delete database files (they take too much space)
        db_files = list(KAGGLE_WORKING.glob("*.db"))
        for db_file in db_files:
            try:
                size_mb = db_file.stat().st_size / (1024*1024)
                db_file.unlink()
                print(f"[cleanup] Deleted database: {db_file.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {db_file.name}: {e}")
        
        # Delete fallback files
        fallback_files = list(KAGGLE_WORKING.glob("*fallback*.pt"))
        for fallback in fallback_files:
            try:
                size_mb = fallback.stat().st_size / (1024*1024)
                fallback.unlink()
                print(f"[cleanup] Deleted fallback: {fallback.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Failed to delete {fallback.name}: {e}")
        
        # Check disk space and report
        try:
            stat = os.statvfs(str(KAGGLE_WORKING))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_gb = total_gb - free_gb
            print(f"[cleanup] Disk after cleanup: {used_gb:.1f}/{total_gb:.1f} GB ({free_gb:.2f} GB free)")
            
            if free_gb < 1.0:
                print("[cleanup] WARNING: Still low on space!")
        except:
            pass
        
        # List remaining files
        remaining_files = list(KAGGLE_WORKING.glob("*.pt"))
        total_size_mb = sum(f.stat().st_size for f in remaining_files) / (1024*1024)
        print(f"[cleanup] Remaining {len(remaining_files)} weight files ({total_size_mb:.1f} MB total)")'''
    
    # Find insertion point (after epoch completion)
    insertion_point = """        print(f"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}")"""
    
    if insertion_point in script_content:
        enhanced_script = script_content.replace(
            insertion_point,
            insertion_point + "\n\n" + auto_cleanup_code
        )
        print("✅ Auto cleanup added to training script")
    else:
        print("❌ Could not find insertion point for auto cleanup")
        return False
    
    # Write enhanced script
    enhanced_script_path = "/kaggle/working/train_avus_kaggle_auto_cleanup.py"
    with open(enhanced_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"✅ Enhanced script saved to: {enhanced_script_path}")
    return True

def create_cleanup_function():
    """Create a standalone cleanup function."""
    
    cleanup_function = '''
def aggressive_cleanup():
    """Aggressive cleanup of /kaggle/working."""
    import os, shutil
    from pathlib import Path
    
    KAGGLE_WORKING = Path("/kaggle/working")
    
    # Delete old epoch files (keep only latest)
    epoch_files = list(KAGGLE_WORKING.glob("avus_*_epoch_*.pt"))
    epoch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for old_epoch in epoch_files[1:]:  # Keep only newest
        try:
            size_mb = old_epoch.stat().st_size / (1024*1024)
            old_epoch.unlink()
            print(f"[cleanup] Deleted: {old_epoch.name} ({size_mb:.1f} MB)")
        except:
            pass
    
    # Delete database files
    for db_file in KAGGLE_WORKING.glob("*.db"):
        try:
            size_mb = db_file.stat().st_size / (1024*1024)
            db_file.unlink()
            print(f"[cleanup] Deleted DB: {db_file.name} ({size_mb:.1f} MB)")
        except:
            pass
    
    # Check remaining space
    try:
        stat = os.statvfs(str(KAGGLE_WORKING))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"[cleanup] Free space: {free_gb:.2f} GB")
    except:
        pass
'''
    
    with open("/kaggle/working/cleanup_function.py", 'w') as f:
        f.write(cleanup_function)
    
    print("✅ Cleanup function created: /kaggle/working/cleanup_function.py")
    return True

if __name__ == "__main__":
    print("Adding automatic file cleanup to training...")
    
    if add_auto_cleanup():
        print("✅ Auto cleanup added to training script!")
        print("\nTo use training with auto cleanup:")
        print("exec(open('/kaggle/working/train_avus_kaggle_auto_cleanup.py').read())")
        
        if create_cleanup_function():
            print("\n✅ Standalone cleanup function also created!")
            print("Run anytime with: exec(open('/kaggle/working/cleanup_function.py').read())")
    else:
        print("❌ Failed to add auto cleanup")
