"""Topology diversity entropy (T3 / F3).

IMPORTANT (frozen wording, Spec v1.0): with a finite candidate budget of k=5,
this is the *estimated diversity entropy under finite candidate budget* — an
empirical descriptor of the observed candidate multiset, NOT an estimate of
the entropy of the model's true generative distribution. Always report it
together with the candidate budget k.
"""
import numpy as np


def topology_entropy(labels):
    """Shannon entropy (nats) of the empirical topology-label distribution.

    labels: list of candidate topology labels (e.g. h1 values), N >= 1.
    Returns (H, H_norm) where H_norm = H / log(N) in [0, 1];
    H_norm is None when N < 2 (undefined normalization).
    """
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    n = int(counts.sum())
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    h_norm = float(h / np.log(n)) if n >= 2 else None
    return h, h_norm


def candidate_entropy_stats(per_sample_labels):
    """Aggregate entropy stats over samples.

    per_sample_labels: list of label-lists, one per input sample.
    Only samples with >= 2 valid candidates are included.
    """
    hs, h_norms = [], []
    for labels in per_sample_labels:
        if len(labels) < 2:
            continue
        h, hn = topology_entropy(labels)
        hs.append(h)
        if hn is not None:
            h_norms.append(hn)
    return {
        'n_samples': len(hs),
        'entropy_mean': float(np.mean(hs)) if hs else None,
        'normalized_entropy_mean': float(np.mean(h_norms)) if h_norms else None,
        'zero_entropy_rate': float(np.mean([h == 0 for h in hs])) if hs else None,
        'candidate_budget_note': ('estimated diversity entropy under finite '
                                  'candidate budget k; not the generative '
                                  'distribution entropy'),
    }
