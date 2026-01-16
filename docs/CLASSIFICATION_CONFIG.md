# Evidence Classification Configuration Guide

## Overview

The OHI verification system now supports four advanced classification strategies to improve accuracy and reduce excessive NEUTRAL classifications:

1. **Improved Classification Prompt** - More permissive prompt with examples
2. **Adjustable Temperature** - Configurable LLM temperature for balanced classifications
3. **Two-Pass Classification** - Relaxed first pass, strict verification second pass
4. **Confidence-Weighted Scoring** - 5-level classification with confidence weights

## Environment Variables

All settings use the `VERIFY_` prefix and can be configured in `.env` files or environment:

### Basic Configuration

```bash
# Classification temperature (0.0-2.0)
# Lower = more conservative, Higher = more creative
# Default: 0.1 (very conservative)
# Recommended: 0.3-0.5 for balanced classifications
VERIFY_CLASSIFICATION_TEMPERATURE=0.3

# Two-pass classification (Pass 1: relaxed → Pass 2: strict)
# Reduces false NEUTRAL classifications
# Default: false
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=true

# Confidence-weighted scoring (5-level classification)
# Provides nuanced trust scores instead of binary counts
# Default: false
VERIFY_ENABLE_CONFIDENCE_SCORING=true

# Batch size for evidence classification
# Higher = more efficient, Lower = more granular control
# Default: 6
# Range: 1-20
VERIFY_CLASSIFICATION_BATCH_SIZE=6
```

## Configuration Profiles

### Profile 1: Conservative (Baseline)

**Use when:** Accuracy is critical, false positives must be minimized

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.1
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

**Expected behavior:**
- Strict evidence classification
- Higher NEUTRAL rate (~30-40%)
- Lower false positive rate
- More conservative trust scores

### Profile 2: Balanced (Recommended)

**Use when:** Need good accuracy with reasonable NEUTRAL rate

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

**Expected behavior:**
- Improved prompt with examples
- Moderate temperature for nuanced classification
- Reduced NEUTRAL rate (~15-25%)
- Balanced trust scores

### Profile 3: Two-Pass Verification

**Use when:** Want to minimize false NEUTRAL without sacrificing accuracy

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=true
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

**Expected behavior:**
- Pass 1: Relaxed classification (temp+0.2) to catch borderline evidence
- Pass 2: Strict verification of SUPPORTS/REFUTES
- Lowest NEUTRAL rate (~10-20%)
- Higher confidence in classifications

### Profile 4: Advanced (Research)

**Use when:** Exploring confidence-weighted scoring, research deployments

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=true
```

**Expected behavior:**
- 5-level classification: STRONG_SUPPORT/WEAK_SUPPORT/NEUTRAL/WEAK_REFUTE/STRONG_REFUTE
- Confidence weights: 0.9/0.7/0.5/0.3/0.1
- Weighted trust scores (sum of confidences vs count-based)
- More nuanced verification results

### Profile 5: Maximum Precision

**Use when:** Need highest accuracy, research validation, critical domains

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.4
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=true
VERIFY_ENABLE_CONFIDENCE_SCORING=true
```

**Expected behavior:**
- Combines all advanced features
- Two-pass with confidence-weighted scoring
- Most nuanced classifications
- Highest latency (2x classification + weighted scoring)

## How It Works

### Standard Classification Flow

```
Evidence → LLM Classification → SUPPORTS/REFUTES/NEUTRAL → Count-based status
```

### Two-Pass Classification Flow

```
Evidence → Pass 1 (Relaxed, temp+0.2) → NEUTRAL? 
                                           ↓ No
                                        Return classification
                                           ↓ Yes
          Pass 2 (Strict, original temp) → Final classification
```

### Confidence-Weighted Flow

```
Evidence → LLM Classification → STRONG_SUPPORT(0.9)/WEAK_SUPPORT(0.7)/
                                 NEUTRAL(0.5)/WEAK_REFUTE(0.3)/
                                 STRONG_REFUTE(0.1)
                              → Sum confidences → Weighted status
```

## Deployment Strategy

### Stage 1: Baseline + Improved Prompt

1. Deploy with all features disabled
2. Validate no regressions
3. Monitor NEUTRAL classification rate

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.1
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

### Stage 2: Gradual Temperature Increase

1. Increase temperature to 0.3
2. A/B test 10% → 50% → 100%
3. Monitor accuracy and NEUTRAL rate

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

### Stage 3: Enable Two-Pass

1. Enable two-pass classification
2. A/B test on subset of claims
3. Compare latency vs accuracy tradeoff

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=true
VERIFY_ENABLE_CONFIDENCE_SCORING=false
```

### Stage 4: Evaluate Confidence Scoring

1. Research deployment only
2. Collect metrics on weighted vs count-based trust scores
3. Validate calibration

```bash
VERIFY_CLASSIFICATION_TEMPERATURE=0.3
VERIFY_ENABLE_TWO_PASS_CLASSIFICATION=false
VERIFY_ENABLE_CONFIDENCE_SCORING=true
```

## Performance Considerations

### Latency Impact

- **Standard Classification:** Baseline
- **Improved Prompt:** +5% (longer prompt)
- **Temperature Adjustment:** No impact
- **Two-Pass:** +80-100% (2x LLM calls for NEUTRAL evidence)
- **Confidence Scoring:** +10% (5-level classification + weighted scoring)

### Optimization Tips

1. **Use batch processing:** Set `VERIFY_CLASSIFICATION_BATCH_SIZE=6` (default)
2. **Enable two-pass selectively:** Only for high-stakes verifications
3. **Profile before full rollout:** Test with benchmark suite
4. **Monitor LLM cost:** Two-pass doubles LLM API calls

## Monitoring & Metrics

Track these metrics per configuration:

1. **NEUTRAL Classification Rate:** % of evidence classified as NEUTRAL
2. **Accuracy:** % of correct verifications (vs ground truth)
3. **Latency:** p50/p95/p99 verification time
4. **Trust Score Calibration:** Correlation between scores and accuracy

## Example API Usage

Test classification configuration via API:

```bash
# Test with balanced configuration
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Eiffel Tower is located in Paris, France.",
    "strategy": "adaptive",
    "max_sources": 3
  }'
```

Check current configuration:

```bash
# View classification settings in logs
docker logs ohi-api 2>&1 | grep "Classification"
```

Expected output:
```
[CONFIG] Classification: temp=0.30, two_pass=True, confidence_scoring=False
```

## Troubleshooting

### High NEUTRAL Rate (>30%)

- **Solution:** Increase temperature to 0.3-0.5
- **Or:** Enable two-pass classification

### Low Accuracy

- **Solution:** Decrease temperature to 0.1-0.2
- **Or:** Disable two-pass (may be too permissive)

### High Latency

- **Solution:** Disable two-pass classification
- **Or:** Reduce batch size to 3-4 for faster individual classifications
- **Or:** Use standard classification instead of confidence scoring

### Inconsistent Classifications

- **Solution:** Lower temperature for more deterministic behavior
- **Or:** Enable two-pass for verification of borderline cases

## Testing

Run benchmark suite with different profiles:

```bash
# Baseline
docker exec ohi-benchmark python -m benchmark --config conservative

# Balanced
docker exec ohi-benchmark python -m benchmark --config balanced

# Two-pass
docker exec ohi-benchmark python -m benchmark --config two-pass

# Advanced
docker exec ohi-benchmark python -m benchmark --config advanced
```

Compare results in `benchmark_results/` directory.

## References

- [API Documentation](API.md)
- [Verification Pipeline](../src/api/README.md#verification-pipeline)
- [Testing Guide](TESTING.md)
- [Benchmark Suite](../src/benchmark/README.md)
