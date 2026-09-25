"""Kendall tau and Topological Misranking Rate (T2 / F1).

Core question: can the geometry-error ranking predict the topology-error ranking?

  tau   = Kendall rank correlation between geometry error and topology error.
  R_inv = fraction of sample pairs where geometry ranks A better than B
          but topology ranks A worse than B (and vice versa).

R_inv is the reviewer-friendly companion of tau: it counts concrete
"geometry said better, topology said worse" inversions.
"""
import numpy as np
from scipy import stats as scipy_stats


def kendall_tau(geo_err, topo_err):
    """Kendall's tau between geometry error and topology error rankings."""
    tau, p = scipy_stats.kendalltau(geo_err, topo_err)
    return float(tau), float(p)


def misranking_rate(geo_err, topo_err, n_pairs=1_000_000, seed=0):
    """Topological Misranking Rate R_inv.

    Uniformly sample pairs (i, j). A pair is *comparable* when both metrics
    strictly order it (no tie in either). It is an *inversion* when the two
    orderings disagree:
        (g_i < g_j and t_i > t_j) or (g_i > g_j and t_i < t_j)

    R_inv = #inversions / #comparable pairs.
    """
    g = np.asarray(geo_err, dtype=np.float64)
    t = np.asarray(topo_err, dtype=np.float64)
    n = len(g)
    if n < 2:
        return {'r_inv': None, 'n_pairs': 0, 'n_comparable': 0}
    rng = np.random.default_rng(seed)
    m = min(n_pairs, n * (n - 1))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    keep = i != j
    i, j = i[keep], j[keep]

    dg = np.sign(g[i] - g[j])
    dt = np.sign(t[i] - t[j])
    comparable = (dg != 0) & (dt != 0)
    n_comp = int(comparable.sum())
    if n_comp == 0:
        return {'r_inv': None, 'n_pairs': int(keep.sum()), 'n_comparable': 0}
    inversions = int((dg[comparable] != dt[comparable]).sum())
    return {
        'r_inv': inversions / n_comp,
        'n_pairs': int(keep.sum()),
        'n_comparable': n_comp,
        'n_inversions': inversions,
        'note': 'fraction of sample pairs where geometry and topology rankings disagree',
    }
