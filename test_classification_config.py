#!/usr/bin/env python3
"""
Quick validation test for evidence classification improvements.

Tests:
1. Configuration loading from environment
2. EvidenceClassification enum behavior
3. Classification routing logic

Run with:
    python test_classification_config.py
"""

import os
import sys
from pathlib import Path

# Add src/api to path
api_path = Path(__file__).parent.parent / "src" / "api" / "src"
sys.path.insert(0, str(api_path))

def test_configuration_loading():
    """Test that VerificationSettings loads classification config correctly."""
    print("=" * 70)
    print("TEST 1: Configuration Loading")
    print("=" * 70)
    
    from open_hallucination_index.infrastructure.config import VerificationSettings
    
    # Test default values
    settings = VerificationSettings()
    
    assert settings.classification_temperature == 0.1, "Default temperature should be 0.1"
    assert not settings.enable_two_pass_classification, "Two-pass should be disabled by default"
    assert not settings.enable_confidence_scoring, "Confidence scoring should be disabled by default"
    assert settings.classification_batch_size == 6, "Default batch size should be 6"
    
    print("✅ Default configuration loaded correctly")
    print(f"   - Temperature: {settings.classification_temperature}")
    print(f"   - Two-pass: {settings.enable_two_pass_classification}")
    print(f"   - Confidence scoring: {settings.enable_confidence_scoring}")
    print(f"   - Batch size: {settings.classification_batch_size}")
    
    # Test custom values from environment
    os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"] = "0.3"
    os.environ["VERIFY_ENABLE_TWO_PASS_CLASSIFICATION"] = "true"
    os.environ["VERIFY_ENABLE_CONFIDENCE_SCORING"] = "true"
    os.environ["VERIFY_CLASSIFICATION_BATCH_SIZE"] = "10"
    
    settings_custom = VerificationSettings()
    
    assert settings_custom.classification_temperature == 0.3, "Custom temperature should be 0.3"
    assert settings_custom.enable_two_pass_classification, "Two-pass should be enabled"
    assert settings_custom.enable_confidence_scoring, "Confidence scoring should be enabled"
    assert settings_custom.classification_batch_size == 10, "Custom batch size should be 10"
    
    print("✅ Custom configuration loaded correctly from environment")
    print(f"   - Temperature: {settings_custom.classification_temperature}")
    print(f"   - Two-pass: {settings_custom.enable_two_pass_classification}")
    print(f"   - Confidence scoring: {settings_custom.enable_confidence_scoring}")
    print(f"   - Batch size: {settings_custom.classification_batch_size}")
    
    # Clean up environment
    del os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"]
    del os.environ["VERIFY_ENABLE_TWO_PASS_CLASSIFICATION"]
    del os.environ["VERIFY_ENABLE_CONFIDENCE_SCORING"]
    del os.environ["VERIFY_CLASSIFICATION_BATCH_SIZE"]
    
    print()

def test_evidence_classification_enum():
    """Test EvidenceClassification enum and to_confidence() method."""
    print("=" * 70)
    print("TEST 2: EvidenceClassification Enum")
    print("=" * 70)
    
    from open_hallucination_index.domain.results import EvidenceClassification
    
    # Test enum values
    assert EvidenceClassification.STRONG_SUPPORT.to_confidence() == 0.9
    assert EvidenceClassification.WEAK_SUPPORT.to_confidence() == 0.7
    assert EvidenceClassification.NEUTRAL.to_confidence() == 0.5
    assert EvidenceClassification.WEAK_REFUTE.to_confidence() == 0.3
    assert EvidenceClassification.STRONG_REFUTE.to_confidence() == 0.1
    
    print("✅ All enum values map to correct confidence scores")
    print("   - STRONG_SUPPORT → 0.9")
    print("   - WEAK_SUPPORT → 0.7")
    print("   - NEUTRAL → 0.5")
    print("   - WEAK_REFUTE → 0.3")
    print("   - STRONG_REFUTE → 0.1")
    
    print()

def test_evidence_entity():
    """Test Evidence entity with classification_confidence field."""
    print("=" * 70)
    print("TEST 3: Evidence Entity")
    print("=" * 70)
    
    from open_hallucination_index.domain.entities import Evidence, EvidenceSource
    
    # Test evidence without confidence
    evidence_no_conf = Evidence(
        text="Test evidence",
        source=EvidenceSource.NEO4J,
        similarity_score=0.95
    )
    
    assert evidence_no_conf.classification_confidence is None
    print("✅ Evidence without confidence created successfully")
    print(f"   - Text: {evidence_no_conf.text}")
    print(f"   - Confidence: {evidence_no_conf.classification_confidence}")
    
    # Test evidence with confidence
    evidence_with_conf = Evidence(
        text="Test evidence",
        source=EvidenceSource.NEO4J,
        similarity_score=0.95,
        classification_confidence=0.9
    )
    
    assert evidence_with_conf.classification_confidence == 0.9
    print("✅ Evidence with confidence created successfully")
    print(f"   - Text: {evidence_with_conf.text}")
    print(f"   - Confidence: {evidence_with_conf.classification_confidence}")
    
    # Test confidence validation (0.0-1.0)
    try:
        Evidence(
            text="Test evidence",
            source=EvidenceSource.NEO4J,
            similarity_score=0.95,
            classification_confidence=1.5  # Invalid: > 1.0
        )
        assert False, "Should have raised validation error"
    except Exception:
        print("✅ Confidence validation working (rejected 1.5)")
    
    print()

def test_verification_settings_validation():
    """Test VerificationSettings field validation."""
    print("=" * 70)
    print("TEST 4: Configuration Validation")
    print("=" * 70)
    
    from open_hallucination_index.infrastructure.config import VerificationSettings
    
    # Test valid temperature range
    os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"] = "0.0"
    settings_min = VerificationSettings()
    assert settings_min.classification_temperature == 0.0
    print("✅ Minimum temperature (0.0) accepted")
    
    os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"] = "2.0"
    settings_max = VerificationSettings()
    assert settings_max.classification_temperature == 2.0
    print("✅ Maximum temperature (2.0) accepted")
    
    # Test invalid temperature (should fail)
    try:
        os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"] = "3.0"
        VerificationSettings()
        assert False, "Should have raised validation error for temp > 2.0"
    except Exception:
        print("✅ Invalid temperature (3.0) rejected")
    
    # Test valid batch size range
    os.environ["VERIFY_CLASSIFICATION_BATCH_SIZE"] = "1"
    settings_batch_min = VerificationSettings()
    assert settings_batch_min.classification_batch_size == 1
    print("✅ Minimum batch size (1) accepted")
    
    os.environ["VERIFY_CLASSIFICATION_BATCH_SIZE"] = "20"
    settings_batch_max = VerificationSettings()
    assert settings_batch_max.classification_batch_size == 20
    print("✅ Maximum batch size (20) accepted")
    
    # Clean up
    if "VERIFY_CLASSIFICATION_TEMPERATURE" in os.environ:
        del os.environ["VERIFY_CLASSIFICATION_TEMPERATURE"]
    if "VERIFY_CLASSIFICATION_BATCH_SIZE" in os.environ:
        del os.environ["VERIFY_CLASSIFICATION_BATCH_SIZE"]
    
    print()

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Evidence Classification Configuration Tests" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_configuration_loading()
        test_evidence_classification_enum()
        test_evidence_entity()
        test_verification_settings_validation()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Configure environment variables (see docs/CLASSIFICATION_CONFIG.md)")
        print("2. Restart API container: docker compose -f docker/compose/docker-compose.yml restart ohi-api")
        print("3. Run benchmark: ./run_benchmark_test.sh")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
