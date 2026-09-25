"""Topology utilities for CAD-TopoBench.

Topology signature = (beta0, beta1, beta2) Betti numbers of the B-Rep solid.
Failure taxonomy (F1-F4) classification helpers.
"""

FAILURE_TYPES = [
    'exact_match',          # identical (b0, b1, b2)
    'lost_holes',           # gt_b1 > 0 and gen_b1 == 0
    'hallucinated_holes',   # gt_b1 == 0 and gen_b1 > 0
    'hole_count_mismatch',  # both > 0 (or one > 0) but b1 differs
    'component_mismatch',   # b0 differs (multi-body errors)
    'void_mismatch',        # b2 differs
]


def betti_to_sig(betti):
    """Normalize a betti list [b0, b1, b2] to a tuple."""
    return tuple(int(x) for x in betti[:3])


def euler_characteristic(betti):
    b0, b1, b2 = betti_to_sig(betti)
    return b0 - b1 + b2


def classify_topology_failure(gt_betti, gen_betti):
    """Classify the topology failure type of a (gt, gen) pair.

    Returns ['exact_match'] when signatures are identical; otherwise a list
    of applicable failure tags. Callers computing failure statistics MUST
    exclude 'exact_match' (see metrics.task1, which pops it explicitly).
    """
    g0, g1, g2 = betti_to_sig(gt_betti)
    p0, p1, p2 = betti_to_sig(gen_betti)
    if (g0, g1, g2) == (p0, p1, p2):
        return ['exact_match']
    tags = []
    if g1 > 0 and p1 == 0:
        tags.append('lost_holes')
    if g1 == 0 and p1 > 0:
        tags.append('hallucinated_holes')
    if g1 != p1 and g1 > 0 and p1 > 0:
        tags.append('hole_count_mismatch')
    if g0 != p0:
        tags.append('component_mismatch')
    if g2 != p2:
        tags.append('void_mismatch')
    if not tags:
        tags.append('hole_count_mismatch')
    return tags


def h1_error(gt_h1, gen_h1):
    """Absolute H1 (hole count) error."""
    return abs(int(gt_h1) - int(gen_h1))
