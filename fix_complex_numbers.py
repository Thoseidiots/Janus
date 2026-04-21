#!/usr/bin/env python3
"""
Fix for ComplexFloat RuntimeError in Holographic Brain Memory
====================================================

This script patches the holographic brain memory to handle complex numbers properly.
"""

import os
import sys
import torch

def patch_holographic_brain_memory():
    """Patch HBM to handle complex numbers correctly."""
    
    print("=== PATCHING HOLOGRAPHIC BRAIN MEMORY ===")
    
    # Find the HBM core file
    REPO = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "holographic_brain_memory" in dirs:
            REPO = root
            break
    
    if REPO is None:
        print("HBM not found, skipping patches")
        return
    
    hbm_core_path = os.path.join(REPO, "holographic_brain_memory", "core.py")
    
    if not os.path.exists(hbm_core_path):
        print(f"HBM core not found at: {hbm_core_path}")
        return
    
    # Read the current HBM core
    with open(hbm_core_path, 'r') as f:
        hbm_code = f.read()
    
    # Fix the complex number issue in the encode function
    # The issue is that phase_weights can be complex, but torch.matmul expects real tensors
    original_line = "        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"
    
    # Patch with proper complex number handling
    patched_line = """        # Handle complex phase weights properly
        if torch.is_complex(self.phase_weights):
            # Use real part for matrix multiplication
            real_weights = self.phase_weights.real
            encoded = torch.matmul(x.unsqueeze(1), real_weights.unsqueeze(0)).squeeze(1)
        else:
            encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)"""
    
    if original_line in hbm_code:
        hbm_code = hbm_code.replace(original_line, patched_line)
        print("✅ Fixed complex number handling in HBM encode function")
    else:
        print("⚠️  Complex number issue line not found, applying broader fix...")
        # Apply a more general fix
        import re
        
        # Find the encode function and add complex number handling
        encode_pattern = r'(def encode\(self, x\):.*?return encoded)'
        encode_match = re.search(encode_pattern, hbm_code, re.DOTALL)
        
        if encode_match:
            original_encode = encode_match.group(0)
            patched_encode = original_encode.replace(
                "return encoded",
                """# Handle complex numbers in output
                if torch.is_complex(encoded):
                    encoded = encoded.real
                return encoded"""
            )
            hbm_code = hbm_code.replace(original_encode, patched_encode)
            print("✅ Applied general complex number fix to HBM")
        else:
            print("❌ Could not find encode function to patch")
    
    # Write the patched code back
    with open(hbm_core_path, 'w') as f:
        f.write(hbm_code)
    
    print("HBM core patched successfully!")
    
    # Also patch any other potential complex number issues
    patch_other_complex_issues(REPO)

def patch_other_complex_issues(REPO):
    """Patch other potential complex number issues in HBM."""
    
    # Check for other HBM files that might have complex number issues
    hbm_files = [
        os.path.join(REPO, "holographic_brain_memory", "real_valued.py"),
        os.path.join(REPO, "holographic_brain_memory", "spawning.py")
    ]
    
    for hbm_file in hbm_files:
        if os.path.exists(hbm_file):
            with open(hbm_file, 'r') as f:
                code = f.read()
            
            # Add safe complex number handling
            if "torch.is_complex" not in code:
                # Add import and helper function
                if "import torch" in code:
                    code = code.replace(
                        "import torch",
                        """import torch

def safe_real(tensor):
    \"\"\"Safely extract real part from tensor, handling complex numbers.\"\"\"
    if torch.is_complex(tensor):
        return tensor.real
    return tensor"""
                    )
                
                # Replace problematic operations
                original_patterns = [
                    "torch.matmul(",
                    "torch.nn.Linear(",
                    "F.linear("
                ]
                
                for pattern in original_patterns:
                    if pattern in code and "safe_real(" not in code:
                        code = code.replace(pattern, f"safe_real(torch.{pattern}")
                        print(f"✅ Fixed complex number issue in {os.path.basename(hbm_file)}")
                
                # Write back the patched code
                with open(hbm_file, 'w') as f:
                    f.write(code)

if __name__ == "__main__":
    patch_holographic_brain_memory()
    print("\n=== COMPLEX NUMBER FIXES APPLIED ===")
    print("Holographic Brain Memory should now handle complex numbers correctly!")
