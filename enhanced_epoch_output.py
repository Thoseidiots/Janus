#!/usr/bin/env python3
"""
Enhanced Epoch Output for Training
==================================

This script patches the training to provide clear, detailed output after every epoch.
"""

import os
import sys
from pathlib import Path

def enhance_epoch_output():
    """Add comprehensive epoch output to training script."""
    
    print("=== ENHANCING EPOCH OUTPUT ===")
    
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
    
    # Enhanced epoch output code
    enhanced_epoch_output = '''        # Enhanced epoch summary
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
        from pathlib import Path
        KAGGLE_WORKING = Path("/kaggle/working")
        MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
        
        main_weights = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
        if main_weights.exists():
            size_mb = main_weights.stat().st_size / 1024**2
            print(f"Main Weights: {main_weights.name} ({size_mb:.1f} MB)")
        
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
        print(f"{'='*60}\\n")'''
    
    # Find where to insert the enhanced output (after the epoch loop)
    # Look for the existing epoch completion message
    existing_pattern = "print(f\"\\n[avus] Epoch {epoch+1} complete \u2014 loss={avg_loss:.4f}\")"
    
    if existing_pattern in script_content:
        # Replace with enhanced output
        enhanced_script = script_content.replace(existing_pattern, enhanced_epoch_output)
        print("Enhanced epoch output applied to existing pattern")
    else:
        # Look for alternative pattern
        alt_pattern = "avg_loss = epoch_loss / len(loader)"
        
        if alt_pattern in script_content:
            # Insert after the avg_loss calculation
            enhanced_script = script_content.replace(
                alt_pattern,
                alt_pattern + "\n\n" + enhanced_epoch_output
            )
            print("Enhanced epoch output inserted after avg_loss calculation")
        else:
            print("Could not find insertion point for epoch output")
            return False
    
    # Write the enhanced script
    enhanced_script_path = "/kaggle/working/train_avus_kaggle_enhanced.py"
    with open(enhanced_script_path, 'w') as f:
        f.write(enhanced_script)
    
    print(f"Enhanced training script saved to: {enhanced_script_path}")
    return True

def add_progress_bars():
    """Add progress bars to training loops."""
    
    print("=== ADDING PROGRESS BARS ===")
    
    # This would require tqdm library, but let's create a simple progress indicator
    progress_code = '''
# Simple progress indicator
def print_progress(current, total, prefix="Progress"):
    percent = (current / total) * 100
    filled = int(percent / 5)  # 20 characters for 100%
    bar = '[' + '=' * filled + ' ' * (20 - filled) + ']'
    print(f"\\r{prefix}: {bar} {percent:.1f}% ({current}/{total})", end="", flush=True)
    
    if current == total:
        print()  # New line when complete'''
    
    print("Progress bar code ready for integration")
    return True

if __name__ == "__main__":
    print("Enhancing epoch output for better training visibility...")
    
    if enhance_epoch_output():
        print("Enhanced epoch output applied successfully!")
        print("\\nTo use the enhanced training script:")
        print("exec(open('/kaggle/working/train_avus_kaggle_enhanced.py').read())")
        
        if add_progress_bars():
            print("Progress indicators ready")
    else:
        print("Failed to enhance epoch output")
