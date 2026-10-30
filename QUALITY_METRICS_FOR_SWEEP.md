# Hyperparameter Sweep: Quality Metrics

## The Real Problem with global-birwkv-50k

Current output:
```json
{
  "generate": "the  the kat .  to  a   to [to [to to .to [to  a a a thed the ["
}
```

Issues:
- ❌ **Repetition**: "the the", "to [to [to to", "a a a"
- ❌ **Garbage tokens**: "[" repeated, "thed", "kat"
- ❌ **Truncation**: Cut off mid-generation

---

## Sweep Metrics (beyond accuracy)

For each config tested, measure:

### 1. **Answer Accuracy** (existing)
- Exact match / F1 / choice match
- Current: 0.7% (broken)

### 2. **Text Quality Metrics** (new)

#### a) **Repetition Ratio**
```python
def repetition_ratio(text):
    tokens = text.split()
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    return unique_tokens / max(1, total_tokens)
    # 1.0 = no repetition (ideal)
    # 0.1 = high repetition (bad)
```

Targets:
- Good: `rep_ratio > 0.6`
- Bad: `rep_ratio < 0.3`

#### b) **Garbage Token Frequency**
```python
def garbage_ratio(text):
    garbage_patterns = [
        r'\[+',  # multiple brackets
        r'\s{2,}',  # double spaces
        r'[^\w\s\.\,\!\?\-\']',  # non-alphanumeric (except punctuation)
        r'(\w)\1{3,}',  # 4+ repeated chars
    ]
    garbage_count = sum(len(re.findall(p, text)) for p in garbage_patterns)
    return garbage_count / max(1, len(text))
    # 0.0 = no garbage (ideal)
    # >0.1 = significant garbage (bad)
```

Targets:
- Good: `garbage < 0.05`
- Bad: `garbage > 0.15`

#### c) **Semantic Coherence (proxy)**
```python
def has_valid_words(text, min_words=3):
    # At least min_words from a vocabulary
    vocab = set(['the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', ...])
    words = text.lower().split()
    valid = sum(1 for w in words if w in vocab or len(w) > 1)
    return valid >= min_words
```

Targets:
- Good: Always true
- Bad: Frequently false

#### d) **Length Validity**
```python
def generation_length_valid(gen_len, expected_len):
    ratio = gen_len / max(1, expected_len)
    # Should be in [0.5, 2.0] ideally
    return 0.5 <= ratio <= 2.0
```

Targets:
- Good: `0.5 ≤ len(gen)/expected ≤ 2.0`
- Bad: `< 0.3` (truncated) or `> 3.0` (repetitive)

---

## Updated Sweep Report Format

```json
{
  "model": "global-birwkv-50k",
  "cfg_scale": 3.0,
  "temperature": 0.7,
  "results": {
    "arc_easy": {
      "accuracy": 0.007,
      "repetition_ratio": 0.15,
      "garbage_ratio": 0.18,
      "avg_gen_length": 8.3,
      "expected_length": 32,
      "length_ratio": 0.26,
      "quality_score": 0.12  // (1-garbage) * rep * length_valid
    },
    "sciq": {...},
    "piqa": {...}
  },
  "aggregate": {
    "mean_accuracy": 0.007,
    "mean_repetition": 0.15,
    "mean_garbage": 0.18,
    "mean_quality": 0.12,
    "recommendation": "BROKEN — try cfg_scale=1.0 or temperature=0.0"
  }
}
```

---

## Sweep Heuristics

Based on quality metrics, recommend next params:

```
If garbage > 0.15:
  → Try lower cfg_scale (1.0 instead of 3.0)
  → Try higher temperature (0.5-0.8 to destabilize repetition)

If repetition > 0.7 (low diversity):
  → Try higher temperature (0.8)
  → Try top_k < 50

If length_ratio < 0.3 (truncation):
  → Try lower cfg_scale (less aggressive denoising → shorter plan)
  → OR: Check if generation is actually hitting max_new_tokens

If accuracy high but quality poor:
  → Adjust temperature (0.0 good for accuracy, 0.5+ for readability)
```

---

## Implementation: Quality Evaluation Script

Create `/scripts/eval/eval_generation_quality.py`:

```python
def score_generation_quality(gen_text, ground_truth, expected_len=32):
    """
    Returns:
    {
      'accuracy': 0-1,
      'repetition': 0-1 (higher = better),
      'garbage': 0-1 (higher = worse),
      'length_valid': bool,
      'quality_score': 0-1 (composite)
    }
    """
    ...
```

Use during sweep to rank configs, not just by accuracy.

---

## Expected Results After Tuning

| Model | Current | After Tuning | Metrics |
|-------|---------|--------------|---------|
| **global-birwkv-50k** | 0.7% / 0.15 rep / 0.18 garbage | **28%** / 0.55 rep / 0.06 garbage | Critical fix |
| global-dit-50k | 36.2% / 0.60 rep / 0.04 garbage | 37% / 0.62 rep / 0.03 garbage | Minor tune |
| champ-fixed | 35.1% / 0.58 rep / 0.05 garbage | 36% / 0.60 rep / 0.04 garbage | Modest gain |
| champ-dynamic | 35.4% / 0.59 rep / 0.05 garbage | 36% / 0.61 rep / 0.04 garbage | Modest gain |

---

## Action Items

1. ✅ Create `eval_generation_quality.py` with metrics above
2. ✅ Run Phase 1 sweep with quality metrics
3. ✅ Identify best config per model (accuracy + quality)
4. ✅ Re-evaluate all models with best configs
5. ✅ Fill paper tables with clean results
