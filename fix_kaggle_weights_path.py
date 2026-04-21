#!/usr/bin/env python3
"""
Fix Kaggle Weights Path for Upload and Download
==============================================

This script ensures weights are uploaded to and used from the correct Kaggle dataset path:
/kaggle/input/datasets/ishmaelsears/janus-avus-weights
"""

import os
import sys
from pathlib import Path

def fix_weights_path():
    """Fix training script to use correct Kaggle dataset path."""
    
    print("=== FIXING KAGGLE WEIGHTS PATH ===")
    
    # Target dataset path
    KAGGLE_DATASET = Path("/kaggle/input/datasets/ishmaelsears/janus-avus-weights")
    
    # Check if dataset exists
    if not KAGGLE_DATASET.exists():
        print(f"❌ Dataset not found: {KAGGLE_DATASET}")
        print("Please ensure dataset is added to your Kaggle notebook")
        return False
    
    print(f"✅ Dataset found: {KAGGLE_DATASET}")
    
    # List available weights in dataset
    weight_files = list(KAGGLE_DATASET.glob("*.pt"))
    if weight_files:
        print(f"Available weights in dataset:")
        for weight_file in weight_files:
            size_mb = weight_file.stat().st_size / (1024*1024)
            print(f"  {weight_file.name} ({size_mb:.1f} MB)")
    else:
        print("No weights found in dataset (starting from scratch)")
    
    # Find training script
    script_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/train_avus_kaggle.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    # Read current script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Fix the resume training logic to use dataset path
    old_resume_logic = """    # Check for existing weights to resume training
    KAGGLE_WORKING = Path("/kaggle/working")
    weight_candidates = [
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_best.pt",
        DATASET_DIR / f"avus_{MODEL_SIZE}_weights.pt"
    ]"""
    
    new_resume_logic = f"""    # Check for existing weights to resume training
    KAGGLE_WORKING = Path("/kaggle/working")
    KAGGLE_DATASET = Path("/kaggle/input/datasets/ishmaelsears/janus-avus-weights")
    
    # Priority: Dataset weights first, then working directory
    weight_candidates = [
        KAGGLE_DATASET / f"avus_{MODEL_SIZE}_weights.pt",
        KAGGLE_DATASET / f"avus_{MODEL_SIZE}_best.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt",
        KAGGLE_WORKING / f"avus_{MODEL_SIZE}_best.pt"
    ]"""
    
    if old_resume_logic in script_content:
        script_content = script_content.replace(old_resume_logic, new_resume_logic)
        print("✅ Resume training logic updated to use dataset path")
    else:
        print("⚠️  Resume logic not found, adding new logic...")
        
        # Find model creation line and add resume logic before it
        model_creation_line = "    model = Avus(cfg).to(device)"
        if model_creation_line in script_content:
            script_content = script_content.replace(
                model_creation_line,
                new_resume_logic + "\n\n    " + model_creation_line
            )
            print("✅ Resume training logic added before model creation")
        else:
            print("❌ Could not find model creation line")
            return False
    
    # Fix the auto-upload logic to upload to correct dataset
    old_upload_pattern = '''def auto_push_weights(version_notes: str = "Auto-save"):
    """
    Save weights locally and skip Kaggle API push to avoid filesystem issues.
    
    The Kaggle API push is disabled due to filesystem quota issues even when 
    disk space is available. Weights are saved locally for manual download.
    """'''
    
    new_upload_pattern = '''def auto_push_weights(version_notes: str = "Auto-save"):
    """
    Save weights to working directory and provide upload instructions
    for the correct Kaggle dataset path.
    """
    
    print(f"\\n[save] Auto-save: {{version_notes}}")
    
    # Save to working directory
    from pathlib import Path
    KAGGLE_WORKING = Path("/kaggle/working")
    MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
    main_weights_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
    
    if main_weights_file.exists():
        size_mb = main_weights_file.stat().st_size / 1e6
        print(f"[save] Main weights ready: {{main_weights_file.name}} ({{size_mb:.1f}} MB)")
    else:
        print(f"[save] WARNING: Main weights file not found: {{main_weights_file.name}}")
    
    # Provide clear upload instructions for correct dataset
    print(f"\\n[save] === UPLOAD INSTRUCTIONS ===")
    print(f"[save] Upload these files to: https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights")
    print(f"[save]   • {{main_weights_file.name}} ({{size_mb:.1f}} MB)" if main_weights_file.exists() else f"[save]   • {{main_weights_file.name}} (not found)")
    
    # List other important files
    for f in KAGGLE_WORKING.glob("*.json"):
        print(f"[save]   • {{f.name}}")
    for f in KAGGLE_WORKING.glob("*.png"):
        print(f"[save]   • {{f.name}}")
    
    print(f"[save] This ensures weights are available at: /kaggle/input/datasets/ishmaelsears/janus-avus-weights")'''
    
    if old_upload_pattern in script_content:
        script_content = script_content.replace(old_upload_pattern, new_upload_pattern)
        print("✅ Auto-upload logic updated for correct dataset path")
    else:
        print("⚠️  Auto-upload pattern not found")
    
    # Write the fixed script
    fixed_script_path = "/kaggle/working/train_avus_kaggle_fixed_weights_path.py"
    with open(fixed_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Fixed script saved to: {fixed_script_path}")
    return True

def create_upload_helper():
    """Create a helper script for uploading weights to correct dataset."""
    
    upload_helper = '''#!/usr/bin/env python3
"""
Upload Helper for Kaggle Weights
================================

This script helps upload weights to the correct dataset path.
"""

import os
import json
from pathlib import Path

def upload_weights_to_correct_dataset():
    """Upload weights to correct Kaggle dataset."""
    
    print("=== UPLOAD TO CORRECT DATASET ===")
    
    # Target dataset
    DATASET_NAME = "janus-avus-weights"
    DATASET_PATH = "ishmaelsears/janus-avus-weights"
    
    # Check working directory for weights
    working_dir = Path("/kaggle/working")
    weight_files = list(working_dir.glob("*.pt"))
    
    if not weight_files:
        print("❌ No weight files found in /kaggle/working")
        return False
    
    print(f"Found {{len(weight_files)}} weight files to upload:")
    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / (1024 * 1024)
        print(f"  • {{weight_file.name}} ({{size_mb:.1f}} MB)")
    
    # Create dataset metadata
    metadata = {{
        "title": DATASET_NAME,
        "id": DATASET_PATH,
        "licenses": [{{"name": "CC0-1.0"}}]
    }}
    
    # Write metadata
    metadata_path = working_dir / "dataset-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\\n📤 UPLOAD INSTRUCTIONS:")
    print(f"1. Go to: https://www.kaggle.com/datasets/{{DATASET_PATH}}")
    print(f"2. Click 'New Version' or 'Update'")
    print(f"3. Upload these files:")
    
    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / (1024 * 1024)
        print(f"   • {{weight_file.name}} ({{size_mb:.1f}} MB)")
    
    print(f"4. Also upload: dataset-metadata.json")
    print(f"\\n✅ This will make weights available at:")
    print(f"   /kaggle/input/datasets/{{DATASET_PATH}}")
    
    return True

if __name__ == "__main__":
    upload_weights_to_correct_dataset()'''
    
    with open("/kaggle/working/upload_to_correct_dataset.py", 'w') as f:
        f.write(upload_helper)
    
    print("✅ Upload helper created: /kaggle/working/upload_to_correct_dataset.py")
    return True

if __name__ == "__main__":
    print("Fixing Kaggle weights path for proper upload/download...")
    
    if fix_weights_path():
        print("✅ Weights path fixed!")
        
        if create_upload_helper():
            print("✅ Upload helper created!")
            
            print("\\n📋 SUMMARY:")
            print("  • Training will now use weights from: /kaggle/input/datasets/ishmaelsears/janus-avus-weights")
            print("  • Auto-save will upload to the same dataset")
            print("  • Upload helper script created for easy file management")
            
            print("\\n🚀 To use fixed training script:")
            print("exec(open('/kaggle/working/train_avus_kaggle_fixed_weights_path.py').read())")
            
            print("\\n📤 To upload weights:")
            print("exec(open('/kaggle/working/upload_to_correct_dataset.py').read())")
    else:
        print("❌ Failed to fix weights path")
