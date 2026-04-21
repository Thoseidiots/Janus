#!/usr/bin/env python3
"""
Comprehensive Disk Space Fix for Kaggle Training
==============================================

This script aggressively cleans up /kaggle/working to free up space
and prevents the directory from filling up again.
"""

import os
import shutil
import glob
from pathlib import Path

def aggressive_disk_cleanup():
    """Aggressively clean up /kaggle/working to free maximum space."""
    
    print("=== AGGRESSIVE DISK CLEANUP ===")
    
    KAGGLE_WORKING = Path("/kaggle/working")
    
    if not KAGGLE_WORKING.exists():
        print("Kaggle working directory not found")
        return False
    
    # Get initial disk usage
    try:
        stat = os.statvfs(str(KAGGLE_WORKING))
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb
        print(f"[cleanup] Initial disk usage: {used_gb:.1f}/{total_gb:.1f} GB ({free_gb:.2f} GB free)")
    except Exception as e:
        print(f"[cleanup] Could not check disk usage: {e}")
        total_gb = 20.0  # Kaggle limit
        free_gb = 0.0
        used_gb = total_gb
    
    # Files to keep (essential)
    keep_patterns = [
        "*.pt",           # Model weights
        "*.json",         # Config files
        "*.png",          # Charts and plots
        "tokens.pt",      # Pre-tokenized data
        "config_*.json"  # Model configs
    ]
    
    # Files to remove aggressively
    remove_patterns = [
        "*.db",           # Database files
        "*.log",          # Log files
        "*.tmp",          # Temporary files
        "*.cache",        # Cache files
        "*.pyc",         # Python bytecode
        "__pycache__",    # Python cache dirs
        "upload_staging", # Staging directories
        "kaggle_dl_temp", # Download temp dirs
        "checkpoint*",      # Old checkpoints
        "midepoch.pt",    # Mid-epoch checkpoints
        "epoch_*_fallback.pt", # Fallback files
        "*.bak",         # Backup files
        "*.old"          # Old files
    ]
    
    files_removed = 0
    space_freed = 0
    
    print("[cleanup] Starting aggressive cleanup...")
    
    # Remove files and directories
    for pattern in remove_patterns:
        if pattern.startswith("__"):
            # Handle special directories
            for item in KAGGLE_WORKING.glob(pattern):
                if item.is_dir():
                    try:
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"[cleanup] Removed directory: {item.name}")
                        files_removed += 1
                    except Exception:
                        pass
        else:
            # Handle file patterns
            for item in KAGGLE_WORKING.glob(pattern):
                if item.is_file():
                    try:
                        size_mb = item.stat().st_size / (1024*1024)
                        item.unlink()
                        print(f"[cleanup] Removed file: {item.name} ({size_mb:.1f} MB)")
                        files_removed += 1
                        space_freed += size_mb
                    except Exception:
                        pass
    
    # Special cleanup for large files that might be duplicates
    weight_files = list(KAGGLE_WORKING.glob("*.pt"))
    weight_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only the most recent weights
    for old_weight in weight_files[1:]:  # Keep the newest one
        try:
            size_mb = old_weight.stat().st_size / (1024*1024)
            old_weight.unlink()
            print(f"[cleanup] Removed old weight: {old_weight.name} ({size_mb:.1f} MB)")
            files_removed += 1
            space_freed += size_mb
        except Exception:
            pass
    
    # Clean up any remaining large files
    for item in KAGGLE_WORKING.glob("*"):
        if item.is_file() and item.stat().st_size > 100 * 1024 * 1024:  # > 100MB
            should_remove = True
            
            # Check if it's in keep patterns
            for keep_pattern in keep_patterns:
                if item.match(keep_pattern):
                    should_remove = False
                    break
            
            if should_remove:
                try:
                    size_mb = item.stat().st_size / (1024*1024)
                    item.unlink()
                    print(f"[cleanup] Removed large file: {item.name} ({size_mb:.1f} MB)")
                    files_removed += 1
                    space_freed += size_mb
                except Exception:
                    pass
    
    # Final disk usage check
    try:
        stat_after = os.statvfs(str(KAGGLE_WORKING))
        free_gb_after = (stat_after.f_bavail * stat_after.f_frsize) / (1024**3)
        total_gb_after = (stat_after.f_blocks * stat_after.f_frsize) / (1024**3)
        used_gb_after = total_gb_after - free_gb_after
        print(f"[cleanup] Final disk usage: {used_gb_after:.1f}/{total_gb_after:.1f} GB ({free_gb_after:.2f} GB free)")
    except Exception as e:
        print(f"[cleanup] Could not check final disk usage: {e}")
    
    print(f"\n[cleanup] SUMMARY:")
    print(f"  Files removed: {files_removed}")
    print(f"  Space freed: {space_freed/(1024*1024):.2f} MB")
    print(f"  Free space: {free_gb_after:.2f} GB")
    
    return free_gb_after

def setup_continuous_cleanup():
    """Set up continuous cleanup during training."""
    
    print("=== SETTING UP CONTINUOUS CLEANUP ===")
    
    # Patch the training script to include cleanup calls
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Add cleanup call after each epoch
    epoch_cleanup_code = '''        # Aggressive cleanup after each epoch
        from pathlib import Path
        KAGGLE_WORKING = Path("/kaggle/working")
        free_gb = aggressive_disk_cleanup()
        
        if free_gb < 1.0:  # Less than 1GB free
            print(f"[avus] WARNING: Low disk space ({free_gb:.2f} GB) - consider manual cleanup")
        
        # Save epoch checkpoint as file (database disabled for space)
        epoch_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch+1}.pt"
        try:
            import torch
            torch.save(model.state_dict(), epoch_file)
            size_mb = epoch_file.stat().st_size / (1024*1024)
            print(f"[avus] Epoch checkpoint saved: {epoch_file.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[avus] Failed to save epoch checkpoint: {e}")'''
    
    # Find where to insert the cleanup code
    insert_point = "        # Save epoch checkpoint as file (database disabled for space)"
    
    if insert_point in script_content:
        modified_script = script_content.replace(
            insert_point,
            epoch_cleanup_code
        )
        
        # Write the modified script
        modified_script_path = "/kaggle/working/train_avus_kaggle_with_cleanup.py"
        with open(modified_script_path, 'w') as f:
            f.write(modified_script)
        
        print(f"✅ Modified training script saved to: {modified_script_path}")
        print("✅ Continuous cleanup enabled")
        return True
    else:
        print("❌ Could not find insertion point in training script")
        return False

if __name__ == "__main__":
    print("Starting disk space fix setup...")
    
    # First, run immediate cleanup
    free_gb = aggressive_disk_cleanup()
    
    if free_gb < 2.0:
        print(f"\n⚠️  WARNING: Only {free_gb:.2f} GB free after cleanup")
        print("Consider manual intervention or reducing model size")
    
    # Then set up continuous cleanup
    if setup_continuous_cleanup():
        print("\n🎉 SUCCESS: Disk space fix applied!")
        print("Training should now run with automatic cleanup.")
        print("\nTo use the fixed training script:")
        print("exec(open('/kaggle/working/train_avus_kaggle_with_cleanup.py').read())")
    else:
        print("❌ Failed to set up continuous cleanup")
