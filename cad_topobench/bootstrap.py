"""Sample-level bootstrap confidence intervals (Spec v1.0 §2.3).

Statistical unit = the input sample (e.g. 8,052 test inputs), NOT the
candidates within a sample. One bootstrap replicate resamples N inputs with
replacement and recomputes the statistic.
"""
import numpy as np


def bootstrap_ci(stat_fn, n_samples, B=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a statistic over resampled sample indices.

    stat_fn(idx: np.ndarray) -> float
        computes the statistic on the samples at positions `idx`
        (may contain duplicates due to resampling with replacement).
    n_samples: total number of samples available for resampling.
    Returns dict(mean, lo, hi, B) where [lo, hi] is the (1-alpha) CI.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_samples, size=n_samples)
        stats[b] = stat_fn(idx)
    return {
        'mean': float(np.mean(stats)),
        'lo': float(np.quantile(stats, alpha / 2)),
        'hi': float(np.quantile(stats, 1 - alpha / 2)),
        'B': B,
    }


def ci_of_values(stat_fn, n_samples, B=1000, seed=0):
    """Convenience wrapper returning (mean, lo, hi) tuple."""
    r = bootstrap_ci(stat_fn, n_samples, B=B, seed=seed)
    return r['mean'], r['lo'], r['hi']


def format_ci(point, lo, hi, digits=4, pct=False):
    """Format '0.605 [0.594, 0.616]' (or percentage) for tables."""
    if pct:
        f = lambda v: f'{v * 100:.{digits - 2}f}%'
    else:
        f = lambda v: f'{v:.{digits}f}'
    return f'{f(point)} [{f(lo)}, {f(hi)}]'
