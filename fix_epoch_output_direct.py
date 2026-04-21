#!/usr/bin/env python3
"""
Direct Fix for Epoch Output Formatting
===================================

This script directly patches the training script to add clear epoch output
by finding and replacing the exact output pattern.
"""

import os
import sys
from pathlib import Path

def fix_epoch_output_directly():
    """Directly patch the epoch output in training script."""
    
    print("=== DIRECT EPOCH OUTPUT FIX ===")
    
    # Find the training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"Training script not found: {script_path}")
        return False
    
    # Read the current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Find the exact epoch completion line from the user's output
    old_pattern = """        print(f"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}")"""
    
    # Create enhanced replacement
    new_pattern = """        # Enhanced epoch summary with clear formatting
        print(f"\\n{'='*60}")
        print(f"EPOCH {epoch+1} COMPLETE")
        print(f"{'='*60}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Time: {time.time() - t0:.1f}s")
        print(f"Steps: {len(loader)}")
        print(f"Learning Rate: {lr:.6f}")
        
        # Model statistics
        if hasattr(model, 'count_parameters'):
            total_params = model.count_parameters()
            print(f"Model Parameters: {total_params:,}")
        
        # Memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory Used: {gpu_memory:.2f} GB")
        
        # Weight file information
        KAGGLE_WORKING = Path("/kaggle/working")
        MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
        
        main_weights = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
        if main_weights.exists():
            size_mb = main_weights.stat().st_size / 1024**2
            print(f"Main Weights: {main_weights.name} ({size_mb:.1f} MB)")
        
        epoch_weights = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch+1}.pt"
        if epoch_weights.exists():
            size_mb = epoch_weights.stat().st_size / 1024**2
            print(f"Epoch Weights: {epoch_weights.name} ({size_mb:.1f} MB)")
        
        # Skill tree progress (if available)
        if skill_tree and sampler:
            current_domain = sampler.next_domain()
            print(f"Current Domain: {current_domain}")
            print(f"Skill Progress: {len(skill_tree.skills)} skills learned")
        
        # Disk space check
        try:
            stat = os.statvfs(str(KAGGLE_WORKING))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_gb = total_gb - free_gb
            print(f"Disk Usage: {used_gb:.1f}/{total_gb:.1f} GB ({free_gb:.2f} GB free)")
            
            if free_gb < 2.0:
                print("WARNING: Low disk space!")
        except:
            pass
        
        print(f"{'='*60}")
        print(f"EPOCH {epoch+1} SUMMARY COMPLETE")
        print(f"{'='*60}\\n")"""
    
    # Apply the fix
    if old_pattern in script_content:
        enhanced_script = script_content.replace(old_pattern, new_pattern)
        print("✅ Enhanced epoch output applied successfully!")
    else:
        print("⚠️  Original pattern not found, trying alternative...")
        
        # Try alternative pattern
        alt_pattern = """print(f"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}")"""
        if alt_pattern in script_content:
            enhanced_script = script_content.replace(alt_pattern, new_pattern)
            print("✅ Enhanced epoch output applied via alternative pattern!")
        else:
            print("❌ Could not find epoch completion pattern")
            return False
    
    # Write the enhanced script
    enhanced_script_path = "/kaggle/working/train_avus_kaggle_enhanced_final.py"
    with open(enhanced_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"✅ Enhanced script saved to: {enhanced_script_path}")
    print("\\n📋 Enhanced Features Added:")
    print("  • Clear epoch header with separators")
    print("  • Detailed metrics (loss, time, steps, LR)")
    print("  • Model statistics (parameters, memory)")
    print("  • File information (weights, sizes)")
    print("  • Skill progress tracking")
    print("  • Disk space monitoring")
    print("  • Visual formatting for readability")
    
    return True

if __name__ == "__main__":
    if fix_epoch_output_directly():
        print("\\n🎉 SUCCESS: Enhanced epoch output ready!")
        print("\\nTo use enhanced training script:")
        print("exec(open('/kaggle/working/train_avus_kaggle_enhanced_final.py').read())")
    else:
        print("❌ Failed to enhance epoch output")
