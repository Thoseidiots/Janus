#!/usr/bin/env python3
"""
Robust Training Environment for Large Model Training
===================================================

This script sets up a complete environment that can handle:
- Large model weights (1B+ parameters)
- SQLite database with proper BLOB configuration
- Memory management and filesystem optimization
- Kaggle API integration without quota issues
"""

import os
import sys
import json
import sqlite3
import torch
import shutil
from pathlib import Path

def setup_environment():
    """Configure the training environment for large models."""
    
    print("=== SETTING UP ROBUST TRAINING ENVIRONMENT ===")
    
    # 1. Configure PyTorch for large model training
    print("\n[1] Configuring PyTorch for large models...")
    
    # Enable memory-efficient settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error messages
    
    # Set CUDA memory fraction to avoid OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        print(f"   CUDA memory fraction: 0.9")
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
    
    # 2. Configure SQLite for large BLOBs
    print("\n[2] Configuring SQLite for large BLOBs...")
    
    # Set up SQLite connection with large BLOB support
    def get_sqlite_connection(db_path):
        conn = sqlite3.connect(db_path)
        # Configure SQLite for large BLOBs
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and speed
        conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        return conn
    
    # Monkey-patch sqlite3.connect for our use
    original_connect = sqlite3.connect
    def patched_connect(*args, **kwargs):
        if args and str(args[0]).endswith('.db'):
            return get_sqlite_connection(*args, **kwargs)
        return original_connect(*args, **kwargs)
    
    sqlite3.connect = patched_connect
    print("   SQLite configured for large BLOBs")
    
    # 3. Set up working directories
    print("\n[3] Setting up working directories...")
    
    KAGGLE_WORKING = Path("/kaggle/working")
    KAGGLE_WORKING.mkdir(exist_ok=True)
    
    # Create subdirectories for better organization
    subdirs = ["weights", "checkpoints", "logs", "temp"]
    for subdir in subdirs:
        (KAGGLE_WORKING / subdir).mkdir(exist_ok=True)
    
    print(f"   Working directory: {KAGGLE_WORKING}")
    print(f"   Subdirectories: {', '.join(subdirs)}")
    
    # 4. Configure memory and disk management
    print("\n[4] Setting up memory and disk management...")
    
    def cleanup_temp_files():
        """Clean up temporary files to free space."""
        temp_dirs = [
            KAGGLE_WORKING / "temp",
            KAGGLE_WORKING / "__pycache__",
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(exist_ok=True)
        
        # Remove any .pyc files
        for pyc in KAGGLE_WORKING.rglob("*.pyc"):
            try:
                pyc.unlink()
            except:
                pass
    
    cleanup_temp_files()
    print("   Temporary files cleaned")
    
    # 5. Enhanced database functions for large models
    print("\n[5] Setting up enhanced database functions...")
    
    def save_weights_to_db(epoch, model_state, loss, max_chunk_size=50*1024*1024):
        """Save model weights to database with proper chunking for large models."""
        import io
        
        KAGGLE_WORKING = Path("/kaggle/working")
        DB_PATH = KAGGLE_WORKING / "model_weights.db"
        
        # Serialize model state
        buffer = io.BytesIO()
        torch.save(model_state, buffer)
        weights_bytes = buffer.getvalue()
        
        conn = get_sqlite_connection(str(DB_PATH))
        try:
            # Create tables if they don't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS epoch_weights (
                    epoch INTEGER PRIMARY KEY,
                    loss REAL NOT NULL,
                    total_size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS weight_chunks (
                    epoch INTEGER,
                    chunk_index INTEGER,
                    chunk_data BLOB,
                    PRIMARY KEY (epoch, chunk_index)
                )
            ''')
            
            # Remove existing data for this epoch
            conn.execute("DELETE FROM epoch_weights WHERE epoch = ?", (epoch,))
            conn.execute("DELETE FROM weight_chunks WHERE epoch = ?", (epoch,))
            
            # Save metadata
            chunk_count = (len(weights_bytes) + max_chunk_size - 1) // max_chunk_size
            conn.execute(
                "INSERT INTO epoch_weights (epoch, loss, total_size, chunk_count) VALUES (?, ?, ?, ?)",
                (epoch, loss, len(weights_bytes), chunk_count)
            )
            
            # Save chunks
            for i in range(0, len(weights_bytes), max_chunk_size):
                chunk = weights_bytes[i:i+max_chunk_size]
                conn.execute(
                    "INSERT INTO weight_chunks (epoch, chunk_index, chunk_data) VALUES (?, ?, ?)",
                    (epoch, i // max_chunk_size, sqlite3.Binary(chunk))
                )
            
            conn.commit()
            
            # Clean up old epochs (keep only last 3)
            old_epochs = conn.execute(
                "SELECT epoch FROM epoch_weights ORDER BY epoch DESC LIMIT 10 OFFSET 3"
            ).fetchall()
            
            for (old_epoch,) in old_epochs:
                conn.execute("DELETE FROM epoch_weights WHERE epoch = ?", (old_epoch,))
                conn.execute("DELETE FROM weight_chunks WHERE epoch = ?", (old_epoch,))
            
            conn.commit()
            
            print(f"[db] Saved epoch {epoch} weights: {len(weights_bytes)/1024/1024:.1f} MB in {chunk_count} chunks")
            
        except Exception as e:
            print(f"[db] Error saving weights: {e}")
            # Fallback to file
            fallback_path = KAGGLE_WORKING / "weights" / f"epoch_{epoch}_fallback.pt"
            torch.save(model_state, fallback_path)
            print(f"[db] Fallback saved to: {fallback_path}")
            
        finally:
            conn.close()
    
    # 6. Enhanced auto-push with proper error handling
    print("\n[6] Setting up enhanced auto-push...")
    
    def safe_auto_push_weights(version_notes="Auto-save"):
        """Safe auto-push that handles large files properly."""
        KAGGLE_WORKING = Path("/kaggle/working")
        
        print(f"[push] Auto-save: {version_notes}")
        
        # Clean up temp files first
        cleanup_temp_files()
        
        # Check disk space
        try:
            stat = os.statvfs(str(KAGGLE_WORKING))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            print(f"[push] Available disk space: {free_gb:.2f} GB")
        except:
            free_gb = 10.0  # Default assumption
            print(f"[push] Could not check disk space")
        
        # List files for download
        print(f"\n[push] === FILES READY FOR DOWNLOAD ===")
        
        # Weight files
        for weight_file in (KAGGLE_WORKING / "weights").glob("*.pt"):
            size_mb = weight_file.stat().st_size / 1e6
            print(f"  weights/{weight_file.name:<30} {size_mb:.1f} MB")
        
        # Checkpoint files
        for ckpt_file in (KAGGLE_WORKING / "checkpoints").glob("*.pt"):
            size_mb = ckpt_file.stat().st_size / 1e6
            print(f"  checkpoints/{ckpt_file.name:<25} {size_mb:.1f} MB")
        
        # Other files
        for ext in ["*.json", "*.png", "*.db"]:
            for other_file in KAGGLE_WORKING.glob(ext):
                if other_file.is_file():
                    size_mb = other_file.stat().st_size / 1e6
                    print(f"  {other_file.name:<35} {size_mb:.1f} MB")
        
        print(f"\n[push] Training complete! Download files from Kaggle output panel.")
        print(f"[push] No automatic push to avoid quota issues.")
    
    # 7. Monkey patch the functions
    import builtins
    builtins.save_epoch_to_db = save_weights_to_db
    builtins.auto_push_weights = safe_auto_push_weights
    
    print("\n=== ENVIRONMENT SETUP COMPLETE ===")
    print("Ready for large model training!")
    
    return {
        'KAGGLE_WORKING': KAGGLE_WORKING,
        'cleanup_temp_files': cleanup_temp_files,
        'save_weights_to_db': save_weights_to_db,
        'safe_auto_push_weights': safe_auto_push_weights
    }

def load_and_patch_training_script():
    """Load the training script and apply necessary patches."""
    
    print("\n=== LOADING AND PATCHING TRAINING SCRIPT ===")
    
    # Find the training script
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_avus_kaggle.py" in files:
            REPO = root
            break
    
    if REPO is None:
        raise FileNotFoundError("train_avus_kaggle.py not found")
    
    print(f"Found training script at: {REPO}")
    sys.path.insert(0, REPO)
    
    # Load and patch the script
    script_path = os.path.join(REPO, "train_avus_kaggle.py")
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
    
    for old, new in patches:
        if old in script_content:
            script_content = script_content.replace(old, new)
            print(f"Applied patch: {old[:50]}...")
        else:
            print(f"Patch not found: {old[:50]}...")
    
    # Execute the patched script
    print("Executing patched training script...")
    exec(script_content, globals())
    
    print("Training script loaded and patched!")

if __name__ == "__main__":
    # Set up the environment
    env = setup_environment()
    
    # Load and patch the training script
    load_and_patch_training_script()
    
    print("\n=== STARTING TRAINING ===")
    
    # Run training with the robust environment
    try:
        train_avus()
        train_hbm()
        print_summary()
        safe_auto_push_weights(version_notes="Training completed with robust environment")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
