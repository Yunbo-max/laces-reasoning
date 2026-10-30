#!/usr/bin/env python3
"""
Generation quality metrics for hyperparameter sweep.
Measures: repetition, garbage tokens, semantic coherence, length validity.
"""

import re
from collections import Counter
from typing import Dict, Tuple

def repetition_ratio(text: str) -> float:
    """
    Measure diversity: unique_tokens / total_tokens
    Higher = better (no repetition)
    """
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    unique = len(set(tokens))
    return unique / len(tokens)


def garbage_ratio(text: str) -> float:
    """
    Measure garbage token frequency.
    Detects: brackets, double spaces, repeated chars, strange unicode
    Returns: garbage_count / text_length
    Lower = better (no garbage)
    """
    garbage_patterns = [
        r'\[+',                    # multiple brackets
        r'\]+',                    # multiple closing brackets
        r'\s{2,}',                 # double+ spaces
        r'(\w)\1{2,}',             # 3+ repeated chars (thhhe, aaaa)
        r'[^\w\s\.\,\!\?\-\'\"\;]', # non-alphanumeric except punctuation
        r' +',                # non-breaking spaces
    ]

    total_garbage = 0
    for pattern in garbage_patterns:
        matches = re.findall(pattern, text)
        total_garbage += len(matches)

    text_len = max(1, len(text))
    return total_garbage / text_len


def length_validity(gen_len: int, expected_len: int = 32) -> Tuple[bool, float]:
    """
    Check if generation length is reasonable.
    Valid range: [0.3 * expected, 3.0 * expected]
    Returns: (is_valid, ratio)
    """
    if expected_len <= 0:
        return True, 1.0

    ratio = gen_len / expected_len
    is_valid = 0.3 <= ratio <= 3.0
    return is_valid, ratio


def semantic_coherence(text: str) -> float:
    """
    Simple heuristic: does text contain meaningful words?
    Returns: fraction of tokens that are dictionary words or reasonable words
    """
    # Common English words
    vocab = {
        'the', 'a', 'an', 'is', 'are', 'am', 'be', 'been', 'being',
        'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for',
        'with', 'from', 'up', 'down', 'out', 'over', 'under', 'through',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'can', 'may', 'must', 'this', 'that', 'these', 'those',
        'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'he', 'she', 'it', 'we', 'you', 'they', 'i', 'me', 'him', 'her',
    }

    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0

    # Count tokens that are either:
    # 1. Common vocabulary words, or
    # 2. Longer than 1 char (likely meaningful)
    valid_tokens = sum(
        1 for t in tokens
        if t in vocab or len(t) > 2 or (len(t) == 2 and t[0].isalpha())
    )

    return valid_tokens / len(tokens)


def quality_score(
    accuracy: float,
    repetition: float,
    garbage: float,
    length_valid: bool,
    weights: Dict[str, float] = None
) -> float:
    """
    Composite quality score.
    Combines accuracy, lack-of-garbage, good-diversity, and valid-length.

    Weights: (default)
      - accuracy: 0.5 (main signal)
      - garbage: 0.2 (penalize strongly if garbage)
      - repetition: 0.15 (prefer diversity)
      - length: 0.15 (penalize extreme lengths)
    """
    if weights is None:
        weights = {
            'accuracy': 0.5,
            'garbage': 0.2,
            'repetition': 0.15,
            'length': 0.15,
        }

    length_score = 1.0 if length_valid else 0.3
    garbage_score = max(0, 1.0 - garbage * 5)  # Garbage is heavily penalized

    composite = (
        weights['accuracy'] * accuracy +
        weights['garbage'] * garbage_score +
        weights['repetition'] * repetition +
        weights['length'] * length_score
    )

    return min(1.0, max(0.0, composite))


def evaluate_generation(
    generated_text: str,
    expected_length: int = 32,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single generation.

    Returns dict with all metrics.
    """
    gen_text = generated_text.strip()
    gen_len = len(gen_text.split())

    rep = repetition_ratio(gen_text)
    garbage = garbage_ratio(gen_text)
    length_valid, length_ratio = length_validity(gen_len, expected_length)
    coherence = semantic_coherence(gen_text)

    return {
        'gen_length': gen_len,
        'repetition_ratio': round(rep, 3),
        'garbage_ratio': round(garbage, 3),
        'coherence_score': round(coherence, 3),
        'length_ratio': round(length_ratio, 3),
        'length_valid': length_valid,
    }


def evaluate_batch(
    generations: list,
    accuracies: list = None,
    expected_length: int = 32,
) -> Dict:
    """
    Evaluate a batch of generations.

    Args:
        generations: list of generated text strings
        accuracies: list of accuracy scores (0 or 1) per generation
        expected_length: reference length

    Returns:
        Dictionary with aggregate metrics
    """
    if len(generations) == 0:
        return {}

    if accuracies is None:
        accuracies = [0.0] * len(generations)

    all_metrics = []
    for gen, acc in zip(generations, accuracies):
        metrics = evaluate_generation(gen, expected_length)
        metrics['accuracy'] = acc
        all_metrics.append(metrics)

    # Aggregate
    n = len(all_metrics)
    agg = {
        'n_samples': n,
        'mean_accuracy': round(sum(m['accuracy'] for m in all_metrics) / n, 4),
        'mean_repetition': round(sum(m['repetition_ratio'] for m in all_metrics) / n, 3),
        'mean_garbage': round(sum(m['garbage_ratio'] for m in all_metrics) / n, 3),
        'mean_coherence': round(sum(m['coherence_score'] for m in all_metrics) / n, 3),
        'mean_length_ratio': round(sum(m['length_ratio'] for m in all_metrics) / n, 3),
        'frac_length_valid': round(sum(1 for m in all_metrics if m['length_valid']) / n, 3),
    }

    # Composite quality score (w/o accuracy for diagnostics)
    agg['mean_quality_score'] = round(
        quality_score(
            accuracy=agg['mean_accuracy'],
            repetition=agg['mean_repetition'],
            garbage=agg['mean_garbage'],
            length_valid=agg['frac_length_valid'] > 0.8,
        ), 4
    )

    return agg


def print_quality_report(agg_metrics: Dict, model_name: str, config_name: str):
    """Pretty-print quality report."""
    print(f"\n{'='*80}")
    print(f"  {model_name:30s} | {config_name:30s}")
    print(f"{'='*80}")
    print(f"  Accuracy:              {agg_metrics['mean_accuracy']:.1%}")
    print(f"  Repetition (↑ better): {agg_metrics['mean_repetition']:.2f}")
    print(f"  Garbage (↓ better):    {agg_metrics['mean_garbage']:.3f}")
    print(f"  Coherence:             {agg_metrics['mean_coherence']:.2f}")
    print(f"  Length valid:          {agg_metrics['frac_length_valid']:.1%}")
    print(f"  Quality score:         {agg_metrics['mean_quality_score']:.3f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test on global-birwkv-50k example
    bad_gen = "the  the kat .  to  a   to [to [to to .to [to  a a a thed the ["
    good_gen = "cellular telephone is the technology that was developed most recently."

    print("\nBAD EXAMPLE (global-birwkv-50k):")
    print(f"Text: {bad_gen}")
    bad_metrics = evaluate_generation(bad_gen)
    for key, val in bad_metrics.items():
        print(f"  {key}: {val}")

    print("\n\nGOOD EXAMPLE:")
    print(f"Text: {good_gen}")
    good_metrics = evaluate_generation(good_gen)
    for key, val in good_metrics.items():
        print(f"  {key}: {val}")

    print("\n\nBATCH EVALUATION:")
    batch = [bad_gen] * 5 + [good_gen] * 5
    accs = [0.0] * 5 + [1.0] * 5
    agg = evaluate_batch(batch, accs)
    print_quality_report(agg, "test", "mixed-batch")
