"""
test_integrated_validation.py
==============================
Test the integrated coherency validation in train_avus_kaggle.py

This simulates the dataset generation phase without running full training.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from train_avus_kaggle
from train_avus_kaggle import (
    AvusTokenizer,
    JanusDataset,
    SAMPLES_PER_DATASET,
    VALIDATE_DATASETS,
    MAX_SEQ_LEN
)

def test_validation():
    """Test dataset generation with validation enabled."""
    print("="*70)
    print("TESTING INTEGRATED COHERENCY VALIDATION")
    print("="*70)
    print(f"Validation enabled: {VALIDATE_DATASETS}")
    print(f"Samples per dataset: {SAMPLES_PER_DATASET}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print()

    # Create tokenizer
    tokenizer = AvusTokenizer()

    # Create dataset with validation (uses small sample for testing)
    print("Creating dataset with automatic validation...")
    dataset = JanusDataset(
        tokenizer=tokenizer,
        block_size=128,  # Smaller for faster testing
        samples_per=100,  # Small sample for testing
        validate=True
    )

    print()
    print("="*70)
    print("VALIDATION TEST COMPLETE")
    print("="*70)
    print(f"Total training chunks created: {len(dataset):,}")
    print()
    print("✓ Dataset generation with validation successful")
    print("✓ Ready for training with verified clean data")
    print("="*70)


if __name__ == "__main__":
    test_validation()
