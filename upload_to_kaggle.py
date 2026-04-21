#!/usr/bin/env python3
"""
Upload Fixed Weights to Kaggle Dataset
======================================

This script uploads your fixed weights to the Kaggle dataset
so they persist and can be used for future training sessions.
"""

import os
import json
from pathlib import Path

def upload_weights_to_kaggle():
    """Upload fixed weights to Kaggle dataset."""
    
    print("=== UPLOADING WEIGHTS TO KAGGLE DATASET ===")
    
    # Check for weights in working directory
    working_dir = Path("/kaggle/working")
    
    weight_files = list(working_dir.glob("*.pt"))
    if not weight_files:
        print("❌ No weight files found in /kaggle/working")
        return False
    
    print(f"Found {len(weight_files)} weight files:")
    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / (1024 * 1024)
        print(f"  {weight_file.name} ({size_mb:.1f} MB)")
    
    # Create dataset metadata
    metadata = {
        "title": "janus-avus-weights",
        "id": "ishmaelsears/janus-avus-weights",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    # Write metadata file
    metadata_path = working_dir / "dataset-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset metadata created: {metadata_path}")
    
    # Instructions for manual upload
    print(f"\n📤 MANUAL UPLOAD INSTRUCTIONS:")
    print(f"1. Go to: https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights")
    print(f"2. Click 'New Version' or 'Update'")
    print(f"3. Upload these files:")
    
    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / (1024 * 1024)
        print(f"   • {weight_file.name} ({size_mb:.1f} MB)")
    
    print(f"4. Also upload: dataset-metadata.json")
    print(f"\n💡 This will make weights available for future training sessions!")
    
    return True

if __name__ == "__main__":
    upload_weights_to_kaggle()
