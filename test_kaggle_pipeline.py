"""
Test Kaggle Training Pipeline
Validates that the Kaggle training pipeline is ready to use
"""

import sys

def test_imports():
    """Test that all required modules import correctly"""
    print("\n[TEST] Kaggle Pipeline Imports...")
    try:
        from train_avus_kaggle import (
            train_avus,
            train_hbm,
            print_summary,
            generate_3d_pairs,
            generate_screen_action_pairs,
            generate_language_pairs,
            generate_reasoning_pairs,
            AvusTokenizer
        )
        print("  ✅ All imports: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Imports: FAIL - {e}")
        return False


def test_data_generators():
    """Test that data generators work"""
    print("\n[TEST] Data Generators...")
    try:
        from train_avus_kaggle import (
            generate_3d_pairs,
            generate_screen_action_pairs,
            generate_language_pairs,
            generate_reasoning_pairs
        )
        
        # Test each generator
        samples_3d = generate_3d_pairs(10)
        samples_screen = generate_screen_action_pairs(10)
        samples_lang = generate_language_pairs(10)
        samples_reason = generate_reasoning_pairs(10)
        
        assert len(samples_3d) == 10
        assert len(samples_screen) == 10
        assert len(samples_lang) == 10
        assert len(samples_reason) == 10
        
        # Check format
        assert '<|startoftext|>' in samples_3d[0]
        assert '<|endoftext|>' in samples_3d[0]
        
        print("  ✅ Data Generators: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Data Generators: FAIL - {e}")
        return False


def test_tokenizer():
    """Test that tokenizer works"""
    print("\n[TEST] Tokenizer...")
    try:
        from train_avus_kaggle import AvusTokenizer
        
        tokenizer = AvusTokenizer()
        
        # Test encode
        text = "<|startoftext|>Hello world<|endoftext|>"
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
        
        # Test decode
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        
        print("  ✅ Tokenizer: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Tokenizer: FAIL - {e}")
        return False


def test_config():
    """Test that configuration is valid"""
    print("\n[TEST] Configuration...")
    try:
        import train_avus_kaggle as kaggle
        
        # Check required config variables
        assert hasattr(kaggle, 'MODEL_SIZE')
        assert hasattr(kaggle, 'AVUS_EPOCHS')
        assert hasattr(kaggle, 'HBM_EPOCHS')
        assert hasattr(kaggle, 'BATCH_SIZE')
        assert hasattr(kaggle, 'GRAD_ACCUM_STEPS')
        assert hasattr(kaggle, 'MAX_SEQ_LEN')
        
        # Check values are reasonable
        assert kaggle.MODEL_SIZE in ['1b', '3b', '7b', '13b', '34b', '70b', 'growing']
        assert kaggle.AVUS_EPOCHS > 0
        assert kaggle.BATCH_SIZE > 0
        assert kaggle.MAX_SEQ_LEN > 0
        
        print(f"  Model: {kaggle.MODEL_SIZE}")
        print(f"  Epochs: {kaggle.AVUS_EPOCHS}")
        print(f"  Batch Size: {kaggle.BATCH_SIZE}")
        print(f"  Grad Accum: {kaggle.GRAD_ACCUM_STEPS}")
        print(f"  Max Seq Len: {kaggle.MAX_SEQ_LEN}")
        print(f"  Kaggle Mode: {kaggle.KAGGLE_MODE}")
        print("  ✅ Configuration: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Configuration: FAIL - {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available"""
    print("\n[TEST] Dependencies...")
    try:
        import torch
        import tiktoken
        
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("  ✅ Dependencies: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Dependencies: FAIL - {e}")
        return False


def test_skill_curriculum():
    """Test that skill curriculum is available"""
    print("\n[TEST] Skill Curriculum...")
    try:
        import train_avus_kaggle as kaggle
        
        if kaggle.SKILL_CURRICULUM:
            print("  ✅ Skill Curriculum: AVAILABLE")
            return True
        else:
            print("  ⚠️  Skill Curriculum: NOT AVAILABLE (will use fixed curriculum)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"  ❌ Skill Curriculum: FAIL - {e}")
        return False


def test_hbm_modules():
    """Test that HBM modules are available"""
    print("\n[TEST] HBM Modules...")
    try:
        import train_avus_kaggle as kaggle
        
        if kaggle.HBM_AVAILABLE:
            print("  ✅ HBM Modules: AVAILABLE")
            return True
        else:
            print("  ⚠️  HBM Modules: NOT AVAILABLE (HBM training will be skipped)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"  ❌ HBM Modules: FAIL - {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("KAGGLE TRAINING PIPELINE TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_config,
        test_dependencies,
        test_data_generators,
        test_tokenizer,
        test_skill_curriculum,
        test_hbm_modules,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ KAGGLE PIPELINE READY!")
        print("\nYou can now:")
        print("  1. Upload to Kaggle notebook")
        print("  2. Set accelerator to GPU T4 x2")
        print("  3. Run: python train_avus_kaggle.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        print("Fix the issues before running on Kaggle")
        return 1


if __name__ == '__main__':
    sys.exit(main())
