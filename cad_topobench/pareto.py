"""Geometry-topology trade-off landscape (T2 / F1).

Terminology (frozen in Spec v1.0): we describe the (geometry error, topology
error) scatter as the *geometry-topology trade-off landscape* and report its
lower Pareto envelope. We deliberately avoid claims of "Pareto superiority".
"""
import numpy as np


def pareto_envelope(geo_err, topo_err):
    """Lower Pareto envelope of the (geometry error, topology error) landscape.

    Returns (frontier_geo, frontier_topo) sorted by ascending geometry error.
    A point is on the envelope iff no other point has both a smaller-or-equal
    geometry error AND a strictly smaller topology error. Ties in geometry
    error are therefore collapsed first: within each unique geometry-error
    group only the minimum topology error can be on the envelope (any other
    member of the group is dominated by it). Without this pre-aggregation a
    tied point encountered earlier in the sorted sweep would survive even
    when a same-geometry point with lower topology error appears later,
    producing spurious vertical segments in the envelope.
    """
    g = np.asarray(geo_err, dtype=np.float64)
    t = np.asarray(topo_err, dtype=np.float64)
    order = np.argsort(g, kind='stable')
    g_s, t_s = g[order], t[order]
    # collapse exact geometry ties: per unique g, keep min t
    ug, inv = np.unique(g_s, return_inverse=True)
    t_min = np.full(len(ug), np.inf)
    np.minimum.at(t_min, inv, t_s)
    # running minimum over the de-duplicated points
    keep = np.zeros(len(ug), dtype=bool)
    best = np.inf
    for k in range(len(ug)):
        if t_min[k] < best:
            keep[k] = True
            best = t_min[k]
    return ug[keep], t_min[keep]


def layered_envelope(geo_err, topo_err):
    """Per-topology-error-layer geometry statistics.

    For each observed topology error level e: count, and the CD quantiles of
    samples at that level. Shows how wide the geometry spread is inside each
    topology-error band (the 'trade-off landscape' cross-sections).
    """
    g = np.asarray(geo_err, dtype=np.float64)
    t = np.asarray(topo_err, dtype=np.float64)
    layers = {}
    for e in sorted(set(t.astype(int).tolist())):
        mask = t == e
        ge = g[mask]
        layers[int(e)] = {
            'n': int(mask.sum()),
            'cd_median': float(np.median(ge)),
            'cd_q10': float(np.quantile(ge, 0.10)),
            'cd_q90': float(np.quantile(ge, 0.90)),
            'cd_min': float(ge.min()),
        }
    return layers


def selection_analysis(geo_err, topo_err, frac=0.10):
    """What does geometry-only selection buy you?

    Select the `frac` best samples by geometry error and report topology
    stats inside the selection; contrast with the topology-best `frac`.
    """
    g = np.asarray(geo_err, dtype=np.float64)
    t = np.asarray(topo_err, dtype=np.float64)
    thr_g = np.quantile(g, frac)
    thr_t = np.quantile(t, frac)
    geo_sel = t[g <= thr_g]
    topo_sel = g[t <= thr_t]
    return {
        'frac': frac,
        'geometry_selected': {
            'n': int(len(geo_sel)),
            'topology_error_mean': float(geo_sel.mean()),
            'topology_mismatch_rate': float((geo_sel > 0).mean()),
        },
        'topology_selected': {
            'n': int(len(topo_sel)),
            'geometry_error_mean': float(topo_sel.mean()),
            'geometry_error_median': float(np.median(topo_sel)),
        },
        'note': 'geometry-only selection does not yield topology-reliable samples',
    }
