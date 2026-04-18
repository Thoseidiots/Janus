"""
Janus Weights Fix - Download and Load Real 3B Weights

Downloads real 3B weights from KaggleHub and fixes the weight loading
issue in avus_inference.py to ensure all systems use real weights.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JanusWeightsFix:
    """Fix missing weights issue"""
    
    def __init__(self):
        self.weights_dir = Path("weights")
        self.config_dir = Path(".")
        self.kaggle_dataset = "ishmaelsears/janus-avus-weights"
        
        print("Janus Weights Fix")
        print("=" * 30)
        print("Downloading real 3B weights")
        print("Fixing weight loading issues")
        print()
    
    def fix_weights(self):
        """Main method to fix weights"""
        print("STEP 1: DOWNLOADING REAL WEIGHTS")
        print("-" * 35)
        
        # Download weights using KaggleHub
        weights_path = self._download_weights()
        
        if not weights_path:
            print("Failed to download weights")
            return False
        
        print(f"Weights downloaded to: {weights_path}")
        
        print("\nSTEP 2: VERIFYING WEIGHTS")
        print("-" * 30)
        
        # Verify weights exist
        if not self._verify_weights(weights_path):
            print("Weights verification failed")
            return False
        
        print("Weights verified successfully")
        
        print("\nSTEP 3: FIXING CONFIGURATION")
        print("-" * 30)
        
        # Fix avus_inference.py configuration
        if not self._fix_avus_inference():
            print("Failed to fix avus_inference.py")
            return False
        
        print("Configuration fixed successfully")
        
        print("\nSTEP 4: TESTING WEIGHT LOADING")
        print("-" * 35)
        
        # Test weight loading
        if not self._test_weight_loading():
            print("Weight loading test failed")
            return False
        
        print("Weight loading test passed")
        
        print("\nSTEP 5: VERIFYING ALL SYSTEMS")
        print("-" * 35)
        
        # Verify all systems can use weights
        if not self._verify_all_systems():
            print("System verification failed")
            return False
        
        print("All systems verified successfully")
        
        print("\nWEIGHTS FIX COMPLETE!")
        print("Real 3B weights are now available")
        print("All systems can use real weights")
        
        return True
    
    def _download_weights(self):
        """Download weights using KaggleHub"""
        try:
            print("Attempting to download weights from KaggleHub...")
            
            # Try to import kagglehub
            try:
                import kagglehub
                print("kagglehub imported successfully")
            except ImportError:
                print("kagglehub not available, installing...")
                os.system("pip install kagglehub")
                import kagglehub
            
            # Download weights
            print(f"Downloading from: {self.kaggle_dataset}")
            path = kagglehub.dataset_download(self.kaggle_dataset)
            print(f"Downloaded to: {path}")
            
            # Find weights file
            weights_files = list(Path(path).glob("*.pt"))
            if not weights_files:
                print("No .pt files found in downloaded dataset")
                return None
            
            weights_file = weights_files[0]
            print(f"Found weights file: {weights_file}")
            print(f"Size: {weights_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Copy weights to local directory
            local_weights_path = Path("avus_3b_weights.pt")
            shutil.copy2(weights_file, local_weights_path)
            print(f"Copied weights to: {local_weights_path}")
            
            return local_weights_path
            
        except Exception as e:
            print(f"Download failed: {e}")
            return None
    
    def _verify_weights(self, weights_path):
        """Verify weights file exists and is valid"""
        try:
            if not weights_path.exists():
                print(f"Weights file not found: {weights_path}")
                return False
            
            size = weights_path.stat().st_size
            print(f"Weights file size: {size / (1024*1024):.1f} MB")
            
            if size < 100 * 1024 * 1024:  # Less than 100MB
                print("Weights file seems too small")
                return False
            
            # Try to load weights
            try:
                import torch
                weights = torch.load(weights_path, map_location='cpu')
                print(f"Weights loaded successfully")
                print(f"Keys in weights: {list(weights.keys())[:5]}...")
                return True
            except Exception as e:
                print(f"Failed to load weights: {e}")
                return False
                
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    def _fix_avus_inference(self):
        """Fix avus_inference.py to use correct weights path"""
        try:
            avus_inference_path = Path("avus_inference.py")
            
            if not avus_inference_path.exists():
                print("avus_inference.py not found")
                return False
            
            # Read current file
            with open(avus_inference_path, 'r') as f:
                content = f.read()
            
            # Check if weights loading needs fixing
            if "avus_3b_weights.pt" not in content:
                print("Adding weights path to avus_inference.py")
                
                # Add weights path after REPO_ROOT definition
                repo_root_line = "REPO_ROOT = Path(__file__).parent"
                if repo_root_line in content:
                    new_content = content.replace(
                        repo_root_line,
                        f"{repo_root_line}\n\n# Local weights path\nWEIGHTS_PATH = REPO_ROOT / \"avus_3b_weights.pt\""
                    )
                    
                    # Save updated content
                    with open(avus_inference_path, 'w') as f:
                        f.write(new_content)
                    
                    print("Updated avus_inference.py with weights path")
                else:
                    print("Could not find REPO_ROOT line in avus_inference.py")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Failed to fix avus_inference.py: {e}")
            return False
    
    def _test_weight_loading(self):
        """Test weight loading"""
        try:
            print("Testing weight loading...")
            
            # Import avus_inference
            try:
                import sys
                sys.path.insert(0, '.')
                import avus_inference
                print("avus_inference imported successfully")
            except ImportError as e:
                print(f"Failed to import avus_inference: {e}")
                return False
            
            # Create AvusInference instance
            try:
                inference = avus_inference.AvusInference()
                print("AvusInference instance created")
            except Exception as e:
                print(f"Failed to create AvusInference: {e}")
                return False
            
            # Try to load weights
            try:
                success = inference.load()
                if success:
                    print("Weights loaded successfully")
                    print(f"Model ready with {inference.params:,} parameters")
                    return True
                else:
                    print("Weight loading returned False")
                    return False
            except Exception as e:
                print(f"Weight loading failed: {e}")
                return False
                
        except Exception as e:
            print(f"Weight loading test failed: {e}")
            return False
    
    def _verify_all_systems(self):
        """Verify all systems can use weights"""
        try:
            print("Verifying all systems...")
            
            # Test avus_brain
            try:
                from avus_brain import AvusBrain
                brain = AvusBrain()
                if brain.ensure_loaded():
                    print("AvusBrain: OK")
                    
                    # Test generation
                    response = brain.ask("Test response - say 'WEIGHTS LOADED'", max_tokens=10)
                    if "WEIGHTS LOADED" in response:
                        print("AvusBrain generation: OK")
                    else:
                        print("AvusBrain generation: FAILED")
                        return False
                else:
                    print("AvusBrain: FAILED to load")
                    return False
            except Exception as e:
                print(f"AvusBrain test failed: {e}")
                return False
            
            # Test dual task manager
            try:
                from janus_dual_task_manager import JanusDualTaskManager
                manager = JanusDualTaskManager()
                if manager.avus_brain:
                    print("DualTaskManager: OK")
                else:
                    print("DualTaskManager: FAILED - no AI brain")
                    return False
            except Exception as e:
                print(f"DualTaskManager test failed: {e}")
                return False
            
            # Test unified system
            try:
                from janus_unified_system import JanusUnifiedSystem
                system = JanusUnifiedSystem()
                if system.avus_brain:
                    print("UnifiedSystem: OK")
                else:
                    print("UnifiedSystem: FAILED - no AI brain")
                    return False
            except Exception as e:
                print(f"UnifiedSystem test failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"System verification failed: {e}")
            return False

def main():
    """Main function"""
    fixer = JanusWeightsFix()
    success = fixer.fix_weights()
    
    if success:
        print("\nWEIGHTS FIX SUCCESSFUL!")
        print("All systems now use real 3B weights")
        print("Ready for production operation")
    else:
        print("\nWEIGHTS FIX FAILED!")
        print("Some issues remain to be resolved")
    
    return success

if __name__ == "__main__":
    print("Janus Weights Fix")
    print("Downloading real 3B weights and fixing loading")
    print()
    
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nWeights fix interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nWeights fix error: {e}")
        exit(1)
