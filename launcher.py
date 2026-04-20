#!/usr/bin/env python3
"""
Simple Launcher for Janus Training with Fixes
======================================

This launcher runs the fix scripts from the correct Kaggle path.
"""

import os
import sys

def main():
    print("=== JANUS TRAINING LAUNCHER ===")
    
    # Change to the correct directory
    repo_path = "/kaggle/input/datasets/ishmaelsears/janus-repo/"
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found at: {repo_path}")
        return
    
    print(f"✅ Found repository at: {repo_path}")
    
    # Change to repository directory
    os.chdir(repo_path)
    
    # Run the complex number fix
    fix_script_path = "c:/Users/legac/Downloads/Janus-workspace/runtime_patch.py"
    if os.path.exists(fix_script_path):
        print("🔧 Running complex number fix...")
        exec(open(fix_script_path).read())
    else:
        print("❌ Complex number fix script not found")
    
    # Run the resume training fix
    resume_fix_path = "c:/Users/legac/Downloads/Janus-workspace/fix_resume_training.py"
    if os.path.exists(resume_fix_path):
        print("🔧 Running resume training fix...")
        exec(open(resume_fix_path).read())
    else:
        print("❌ Resume training fix script not found")
    
    print("\n🚀 Starting training with all fixes applied!")

if __name__ == "__main__":
    main()
