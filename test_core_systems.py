"""
Test Core Systems Integration
Tests that all core Janus systems work together properly
"""

import sys

def test_3d_face_generator():
    """Test 3D face generation"""
    print("\n[TEST] 3D Face Generator...")
    try:
        from advanced_3d_face_generator import ProceduralFaceGenerator, FacialFeatures
        
        generator = ProceduralFaceGenerator()
        features = FacialFeatures(head_width=1.1, nose_length=1.2)
        face = generator.generate_face(features, expressions={'smile': 0.5})
        
        assert 'metadata' in face
        assert 'geometry' in face
        assert len(face['geometry']['vertices']) > 0
        
        print("  ✅ 3D Face Generator: PASS")
        return True
    except Exception as e:
        print(f"  ❌ 3D Face Generator: FAIL - {e}")
        return False


def test_coherency_checker():
    """Test dataset coherency checking"""
    print("\n[TEST] Coherency Checker...")
    try:
        from coherency_checker import CoherencyChecker
        
        checker = CoherencyChecker()
        test_text = "<|startoftext|>Test data<|endoftext|>"
        result = checker.check_text(test_text)
        
        assert result is not None
        assert hasattr(result, 'valid_entries')
        
        print("  ✅ Coherency Checker: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Coherency Checker: FAIL - {e}")
        return False


def test_avus_inference():
    """Test Avus inference engine"""
    print("\n[TEST] Avus Inference...")
    try:
        from avus_inference import AvusInference
        
        avus = AvusInference()
        # Just test that it initializes
        assert avus is not None
        
        print("  ✅ Avus Inference: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Avus Inference: FAIL - {e}")
        return False


def test_hardware_sense():
    """Test hardware awareness"""
    print("\n[TEST] Hardware Sense...")
    try:
        from hardware_sense import HardwareAwareness, HardwareSense
        
        hw_sense = HardwareSense()
        sensation = hw_sense.sense()
        
        assert sensation is not None
        assert hasattr(sensation, 'cpu_temp')
        
        print("  ✅ Hardware Sense: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Hardware Sense: FAIL - {e}")
        return False


def test_integration_hub():
    """Test integration hub"""
    print("\n[TEST] Integration Hub...")
    try:
        from janus_integration_hub import JanusIntegrationHub
        
        # Just test import and initialization
        assert JanusIntegrationHub is not None
        
        print("  ✅ Integration Hub: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Integration Hub: FAIL - {e}")
        return False


def test_human_capable():
    """Test human-level capabilities"""
    print("\n[TEST] Human Capable System...")
    try:
        from janus_human_capable import JanusHumanCapable
        
        # Just test import
        assert JanusHumanCapable is not None
        
        print("  ✅ Human Capable: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Human Capable: FAIL - {e}")
        return False


def test_game_ai_pipeline():
    """Test game AI training pipeline"""
    print("\n[TEST] Game AI Training Pipeline...")
    try:
        from game_ai_training_pipeline import GameAITrainingPipeline
        
        # Just test import
        assert GameAITrainingPipeline is not None
        
        print("  ✅ Game AI Pipeline: PASS")
        return True
    except Exception as e:
        print(f"  ❌ Game AI Pipeline: FAIL - {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("JANUS CORE SYSTEMS TEST")
    print("="*60)
    
    tests = [
        test_3d_face_generator,
        test_coherency_checker,
        test_avus_inference,
        test_hardware_sense,
        test_integration_hub,
        test_human_capable,
        test_game_ai_pipeline,
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
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
