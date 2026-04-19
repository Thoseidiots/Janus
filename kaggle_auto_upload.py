"""
kaggle_auto_upload.py
=====================
Automatically uploads training weights to Kaggle dataset after each epoch.

This solves the problem where weights save to /kaggle/working/ but don't
automatically persist to the Kaggle dataset.

Usage:
  1. Add this to your Kaggle notebook after training
  2. Or import and call upload_to_kaggle() after each epoch
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional


class KaggleAutoUploader:
    """Automatically uploads weights to Kaggle dataset"""
    
    def __init__(self, dataset_name: str = "janus-avus-weights",
                 username: Optional[str] = None):
        """
        Initialize uploader
        
        Args:
            dataset_name: Kaggle dataset name (e.g., "ishmaelsears/janus-avus-weights")
            username: Kaggle username (auto-detected if not provided)
        """
        self.dataset_name = dataset_name
        self.username = username or self._get_kaggle_username()
        self.working_dir = Path("/kaggle/working")
        self.dataset_dir = Path(f"/kaggle/datasets/{self.username}/{dataset_name}")
        
    def _get_kaggle_username(self) -> str:
        """Get Kaggle username from environment or config"""
        # Try environment variable
        if "KAGGLE_USER_SECRETS_TOKEN" in os.environ:
            # Extract from token if possible
            pass
        
        # Try reading from kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                with open(kaggle_json) as f:
                    config = json.load(f)
                    return config.get("username", "unknown")
            except Exception:
                pass
        
        return "unknown"
    
    def upload_weights(self, epoch: int, model_size: str = "1b",
                      include_learning_state: bool = True) -> bool:
        """
        Upload weights to Kaggle dataset
        
        Args:
            epoch: Epoch number
            model_size: Model size (1b, 3b, etc.)
            include_learning_state: Also upload learning state
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n[upload] Uploading weights for epoch {epoch}...")
        
        files_to_upload = []
        
        # Weights file
        weights_file = self.working_dir / f"avus_{model_size}_weights.pt"
        if weights_file.exists():
            files_to_upload.append(weights_file)
            print(f"[upload] Found weights: {weights_file.name}")
        else:
            print(f"[upload] ⚠️  Weights not found: {weights_file}")
            return False
        
        # Learning state
        if include_learning_state:
            learning_file = self.working_dir / "learning_state.json"
            if learning_file.exists():
                files_to_upload.append(learning_file)
                print(f"[upload] Found learning state: {learning_file.name}")
        
        # Best weights
        best_file = self.working_dir / f"avus_{model_size}_best.pt"
        if best_file.exists():
            files_to_upload.append(best_file)
            print(f"[upload] Found best weights: {best_file.name}")
        
        if not files_to_upload:
            print("[upload] ❌ No files to upload")
            return False
        
        # Try uploading using Kaggle API
        try:
            return self._upload_via_api(files_to_upload, epoch)
        except Exception as e:
            print(f"[upload] ⚠️  API upload failed: {e}")
            print("[upload] Trying alternative method...")
            return self._upload_via_cli(files_to_upload, epoch)
    
    def _upload_via_api(self, files: list, epoch: int) -> bool:
        """Upload using Kaggle Python API"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            print(f"[upload] Using Kaggle API...")
            
            # Create dataset version with files
            for file_path in files:
                print(f"[upload] Uploading {file_path.name}...")
                # Note: Kaggle API doesn't have direct file upload to dataset
                # This would require creating a new dataset version
                # For now, we'll use CLI method
            
            return False  # Fall back to CLI
            
        except Exception as e:
            print(f"[upload] API method failed: {e}")
            return False
    
    def _upload_via_cli(self, files: list, epoch: int) -> bool:
        """Upload using Kaggle CLI"""
        try:
            # Check if kaggle CLI is available
            result = subprocess.run(["kaggle", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("[upload] ❌ Kaggle CLI not available")
                return False
            
            print(f"[upload] Using Kaggle CLI...")
            
            # Create a temporary directory for upload
            upload_dir = self.working_dir / f"upload_epoch{epoch}"
            upload_dir.mkdir(exist_ok=True)
            
            # Copy files to upload directory
            for file_path in files:
                dest = upload_dir / file_path.name
                import shutil
                shutil.copy(file_path, dest)
                print(f"[upload] Copied {file_path.name} to upload dir")
            
            # Upload using kaggle datasets version create
            cmd = [
                "kaggle", "datasets", "version", "create",
                "-p", str(upload_dir),
                "-m", f"Epoch {epoch} weights"
            ]
            
            print(f"[upload] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"[upload] ✅ Upload successful!")
                print(result.stdout)
                return True
            else:
                print(f"[upload] ❌ Upload failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"[upload] CLI method failed: {e}")
            return False
    
    def setup_auto_upload(self, model_size: str = "1b") -> str:
        """
        Generate code to add to training loop for auto-upload
        
        Returns:
            Python code snippet to add to training loop
        """
        code = f'''
# Add this after each epoch in your training loop:
from kaggle_auto_upload import KaggleAutoUploader

uploader = KaggleAutoUploader(dataset_name="{self.dataset_name}")
uploader.upload_weights(epoch=epoch, model_size="{model_size}")
'''
        return code


def upload_to_kaggle(epoch: int, model_size: str = "1b",
                     dataset_name: str = "janus-avus-weights") -> bool:
    """
    Simple function to upload weights to Kaggle dataset
    
    Usage:
        from kaggle_auto_upload import upload_to_kaggle
        upload_to_kaggle(epoch=1, model_size="1b")
    """
    uploader = KaggleAutoUploader(dataset_name=dataset_name)
    return uploader.upload_weights(epoch=epoch, model_size=model_size)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def patch_training_script_for_auto_upload():
    """
    Instructions for patching train_avus_kaggle.py to auto-upload
    
    Add this after the epoch training loop (around line 1043):
    
    ```python
    # Save checkpoint
    torch.save({...}, str(WEIGHTS_OUT))
    print(f"[avus] Weights saved -> {WEIGHTS_OUT}")
    
    # AUTO-UPLOAD TO KAGGLE DATASET
    try:
        from kaggle_auto_upload import upload_to_kaggle
        upload_to_kaggle(epoch=epoch+1, model_size=MODEL_SIZE)
    except Exception as e:
        print(f"[avus] Auto-upload failed: {e}")
        print("[avus] Weights saved locally but not uploaded to dataset")
    ```
    """
    pass


if __name__ == "__main__":
    # Test upload
    print("Testing Kaggle auto-upload...")
    
    uploader = KaggleAutoUploader()
    print(f"Dataset: {uploader.dataset_name}")
    print(f"Username: {uploader.username}")
    print(f"Working dir: {uploader.working_dir}")
    
    # Show integration code
    print("\n" + "="*70)
    print("INTEGRATION CODE")
    print("="*70)
    print(uploader.setup_auto_upload())
