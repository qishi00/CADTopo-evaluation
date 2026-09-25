"""Diagnostic metrics for CAD-TopoBench (Tasks T1-T4 / Failures F1-F4).

All functions take the unified benchmark records (dataset.py schema) plus,
where needed, geometry side-files produced by compute_geometry.py.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as scipy_stats

from .topology import betti_to_sig, classify_topology_failure, h1_error
from .kendall import kendall_tau, misranking_rate
from .pareto import pareto_envelope, layered_envelope, selection_analysis
from .entropy import candidate_entropy_stats
from .bootstrap import bootstrap_ci

from .paths import EXTERNAL
PH_DIR = str(EXTERNAL / 'ph_results')


# ------------------------------------------------------------------ T1 / F2

def task1_topology_fidelity(records):
    """Topology Fidelity Evaluation.

    Returns signature match rate, H1 accuracy, confusion matrix, and the
    hole-error decomposition (lost / hallucinated / count mismatch),
    plus beta0 (multi-body) analysis and invalid rates.
    """
    both = [r for r in records if r['gt']['success'] and r['gen']['success']]
    n = len(both)

    sig_match = sum(1 for r in both
                    if betti_to_sig(r['gt']['betti']) == betti_to_sig(r['gen']['betti']))
    h1_match = sum(1 for r in both if r['gt']['h1'] == r['gen']['h1'])

    # confusion matrix over h1 (rows = gt, cols = gen)
    max_h1 = max([r['gt']['h1'] for r in both] + [r['gen']['h1'] for r in both])
    cap = min(max_h1, 12)  # aggregate the long tail into a "cap+" bin
    cm = np.zeros((cap + 1, cap + 1), dtype=int)
    for r in both:
        i = min(r['gt']['h1'], cap)
        j = min(r['gen']['h1'], cap)
        cm[i, j] += 1

    fail_counter = Counter()
    for r in both:
        for t in classify_topology_failure(r['gt']['betti'], r['gen']['betti']):
            fail_counter[t] += 1
    fail_counter.pop('exact_match', None)

    # beta0 analysis
    b0_diff = [r['gen']['betti'][0] - r['gt']['betti'][0] for r in both]
    # conditional accuracy: how bad is topology on holed GT vs unholed GT
    holed = [r for r in both if r['gt']['h1'] > 0]
    unholed = [r for r in both if r['gt']['h1'] == 0]
    h1_acc = lambda rs: (sum(1 for r in rs if r['gt']['h1'] == r['gen']['h1'])
                         / max(len(rs), 1))

    # h1 error magnitude distribution
    h1_err = [h1_error(r['gt']['h1'], r['gen']['h1']) for r in both]

    return {
        'n_comparable': n,
        'n_gt_parsed': sum(1 for r in records if r['gt']['success']),
        'n_gen_parsed': sum(1 for r in records if r['gen']['success']),
        'gen_invalid_rate': 1 - sum(1 for r in records if r['gen']['success']) / len(records),
        'signature_exact_match': sig_match / n,
        'h1_exact_match': h1_match / n,
        'h1_exact_match_holed_gt': h1_acc(holed),
        'h1_exact_match_unholed_gt': h1_acc(unholed),
        'n_holed_gt': len(holed),
        'confusion_matrix_h1': cm.tolist(),
        'confusion_matrix_cap': cap,
        'failure_rates': {k: v / n for k, v in fail_counter.items()},
        'failure_counts': dict(fail_counter),
        'beta0_mean_diff': float(np.mean(b0_diff)),
        'beta0_mismatch_rate': sum(1 for d in b0_diff if d != 0) / n,
        'h1_error_mean': float(np.mean(h1_err)),
        'h1_error_dist': dict(sorted(Counter(h1_err).items())),
        'gt_hole_ratio': len(holed) / n,
        'gen_hole_ratio': sum(1 for r in both if r['gen']['h1'] > 0) / n,
    }


# ------------------------------------------------------------------ T2 / F1

def task2_geometry_topology_consistency(records, geometry):
    """Geometry-Topology Consistency: is CD a reliable proxy for topology?

    geometry: {id: {cd, f_score_0.01, gt_built, gen_built}}
    """
    geo_by_id = {g['id']: g for g in geometry if g.get('cd') is not None}
    rows = []
    for r in records:
        g = geo_by_id.get(r['id'])
        if g is None or not (r['gt']['success'] and r['gen']['success']):
            continue
        rows.append({
            'id': r['id'],
            'cd': g['cd'],
            'f_score': g['f_score_0.01'],
            'h1_err': h1_error(r['gt']['h1'], r['gen']['h1']),
            'sig_mismatch': int(betti_to_sig(r['gt']['betti'])
                                != betti_to_sig(r['gen']['betti'])),
            'gt_h1': r['gt']['h1'],
        })
    cd = np.array([x['cd'] for x in rows])
    fs = np.array([x['f_score'] for x in rows])
    h1e = np.array([x['h1_err'] for x in rows])
    mis = np.array([x['sig_mismatch'] for x in rows])

    tau, tau_p = kendall_tau(cd, h1e)
    rinv = misranking_rate(cd, h1e, n_pairs=1_000_000, seed=0)
    out = {
        'n': len(rows),
        'cd_mean': float(cd.mean()),
        'cd_median': float(np.median(cd)),
        'f_score_mean': float(fs.mean()),
        'pearson_cd_h1err': float(scipy_stats.pearsonr(cd, h1e)[0]),
        'spearman_cd_h1err': float(scipy_stats.spearmanr(cd, h1e)[0]),
        'pearson_cd_sigmismatch': float(scipy_stats.pearsonr(cd, mis)[0]),
        'spearman_cd_sigmismatch': float(scipy_stats.spearmanr(cd, mis)[0]),
        'pearson_fscore_h1err': float(scipy_stats.pearsonr(fs, h1e)[0]),
        'point_biserial_note': 'sig_mismatch is binary; pearson == point-biserial',
        'kendall_tau_cd_h1err': tau,
        'kendall_tau_p': tau_p,
        'topological_misranking_rate': rinv,
        'tradeoff_landscape': {
            'layered_envelope': layered_envelope(cd, h1e),
            'selection_analysis': selection_analysis(cd, h1e),
        },
    }
    # pareto envelope polyline for the figure (downsampled)
    fg, ft = pareto_envelope(cd, h1e)
    out['_pareto_frontier'] = [fg.tolist(), ft.tolist()]

    # the killer cut: among geometrically near-perfect reconstructions,
    # how often is topology still wrong?
    for q, name in [(0.10, 'top10pct_cd'), (0.25, 'top25pct_cd')]:
        thr = np.quantile(cd, q)
        best = [x for x in rows if x['cd'] <= thr]
        out[f'mismatch_rate_in_{name}'] = (
            sum(x['sig_mismatch'] for x in best) / max(len(best), 1))
        out[f'h1err_mean_in_{name}'] = float(np.mean([x['h1_err'] for x in best]))

    # and the inverse: geometrically terrible but topologically right
    thr = np.quantile(cd, 0.90)
    worst = [x for x in rows if x['cd'] >= thr]
    out['match_rate_in_bottom10pct_cd'] = (
        1 - sum(x['sig_mismatch'] for x in worst) / max(len(worst), 1))

    # scatter data for the figure (subsampled)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(3000, len(rows)), replace=False)
    out['_scatter'] = [[rows[i]['cd'], rows[i]['h1_err'], rows[i]['sig_mismatch']]
                       for i in idx]
    return out


# ------------------------------------------------------------------ T3 / F3

def task3_sampling_reliability(records, cand_geometry=None):
    """Sampling Reliability: does sampling explore topology space?

    Per sample: N=5 candidates at temperatures {1e-4, 0.1, 0.3, 0.5, 1.0}.
    Topology diversity D = |unique(h1)| / N_valid.
    """
    per_sample = []
    per_sample_labels = []
    temp_stats = defaultdict(lambda: {'valid': 0, 'h1_match': 0, 'h1_err': []})
    for r in records:
        cands = [c for c in r['candidates'] if c.get('success') and c.get('h1') is not None]
        if len(cands) < 2 or not r['gt']['success']:
            continue
        h1s = [c['h1'] for c in cands]
        per_sample_labels.append(h1s)
        gt_h1 = r['gt']['h1']
        greedy_c = next((c for c in cands if c.get('cand_idx') == 0), None)
        per_sample.append({
            'id': r['id'],
            'diversity': len(set(h1s)) / len(h1s),
            'monomorphic': int(len(set(h1s)) == 1),
            'n_valid': len(h1s),
            'any_match': int(gt_h1 in h1s),
            'best_match': int(min(abs(h - gt_h1) for h in h1s) == 0),
            'greedy_hit': int(greedy_c is not None and greedy_c['h1'] == gt_h1),
            'gt_h1': gt_h1,
            'mean_h1_err': float(np.mean([abs(h - gt_h1) for h in h1s])),
        })
        for c in r['candidates']:
            t = c['temperature']
            if c.get('success') and c.get('h1') is not None:
                temp_stats[t]['valid'] += 1
                temp_stats[t]['h1_match'] += int(c['h1'] == gt_h1)
                temp_stats[t]['h1_err'].append(abs(c['h1'] - gt_h1))

    div = np.array([p['diversity'] for p in per_sample])
    mono = np.array([p['monomorphic'] for p in per_sample], dtype=float)
    all5 = [p for p in per_sample if p['n_valid'] == 5]
    greedy_acc = float(np.mean([p['greedy_hit'] for p in per_sample]))
    oracle_acc = float(np.mean([p['best_match'] for p in per_sample]))
    out = {
        'n_samples': len(per_sample),
        'topology_diversity_mean': float(div.mean()),
        # Headline (pool-size-invariant): fraction of pools whose valid
        # candidates all carry ONE topology label. The legacy D<=1/5 cut is
        # kept for continuity but is only exact for all-5-valid pools (for a
        # partial pool of n_valid<5 a fully monomorphic pool has D=1/n_valid>0.2
        # and is missed — pool-validity distribution then leaks into the rate).
        'monomorphic_rate (unique==1)': float(mono.mean()),
        'monomorphic_rate_all5valid': (
            float(np.mean([p['monomorphic'] for p in all5])) if all5 else None),
        'n_all5valid': len(all5),
        'full_collapse_rate (D<=1/5, legacy)': float(np.mean(div <= 0.2 + 1e-9)),
        'any_candidate_matches_gt': float(np.mean([p['any_match'] for p in per_sample])),
        'all_candidates_wrong': float(np.mean([1 - p['any_match'] for p in per_sample])),
        'greedy_h1_accuracy': greedy_acc,
        'oracle_h1_accuracy (best of 5)': oracle_acc,
        'oracle_gap': oracle_acc - greedy_acc,
        'mean_h1_err_across_candidates': float(np.mean([p['mean_h1_err'] for p in per_sample])),
        'topology_entropy': candidate_entropy_stats(per_sample_labels),
        'per_temperature': {
            str(t): {
                'h1_accuracy': temp_stats[t]['h1_match'] / max(temp_stats[t]['valid'], 1),
                'mean_h1_err': float(np.mean(temp_stats[t]['h1_err']))
                if temp_stats[t]['h1_err'] else None,
                'n_valid': temp_stats[t]['valid'],
            } for t in sorted(temp_stats)
        },
    }

    # geometry diversity vs topology diversity (candidate track subset)
    if cand_geometry:
        geo_by_id = {g['id']: g for g in cand_geometry
                     if g.get('pairwise_cd_mean') is not None}
        xs, ys = [], []
        for p in per_sample:
            g = geo_by_id.get(p['id'])
            if g is not None:
                xs.append(g['pairwise_cd_mean'])
                ys.append(p['diversity'])
        if len(xs) > 10:
            out['geometry_vs_topology_diversity'] = {
                'n': len(xs),
                'pearson': float(scipy_stats.pearsonr(xs, ys)[0]),
                'spearman': float(scipy_stats.spearmanr(xs, ys)[0]),
                'note': 'candidate pairwise CD mean vs topology diversity D',
            }
    out['_diversity_dist'] = dict(sorted(Counter(div.round(3).tolist()).items()))
    return out


# ------------------------------------------------------------------ T4 / F4

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')


def task4_controllability_audit(ph_dir=PH_DIR, results_dir=RESULTS_DIR):
    """Controllability Audit (DeepCAD-only).

    Primary source: v1.1 reference-verified latent experiments under
    results/ (every decoded outcome scored by the reference implementation).
    Fallback: legacy ph_results/ aggregation (pre-reference counters —
    conclusions NOT usable for paper claims).
    """
    v11 = _task4_v11(results_dir)
    if v11 is not None:
        return v11
    return _task4_legacy(ph_dir)


def _task4_v11(results_dir):
    """Aggregate the reference-verified v1.1 latent experiments (§5.3 source)."""
    def _load(name):
        p = os.path.join(results_dir, name)
        if not os.path.exists(p):
            return None
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)

    poc = _load('latent_intervention_poc_v1.1.json')
    clf = _load('latent_classifier_v1.json')
    lin = _load('latent_intervention_v1.1.json')
    if poc is None and clf is None:
        return None

    out = {'provenance': ('v1.1 reference-verified: every decoded outcome was '
                          'scored by the reference implementation (labels v1.0)')}

    if clf:
        out['latent_probe'] = {
            'n_train': clf['n_train'],
            'n_val': clf['n_val'],
            'val_accuracy': clf['best_val_acc'],
            'majority_baseline': clf['majority_baseline'],
            'within_pm1': clf.get('within_pm1'),
            'supervision': clf.get('supervision'),
        }

    if poc:
        out['controlled_intervention'] = {
            'n_per_cell': poc['groups']['remove']['0.0']['n'],
            'alphas': poc['alphas'],
            'groups': poc['groups'],
            'headline': {
                'remove_alpha2_clf_hit': poc['groups']['remove']['2.0']['clf_hit'],
                'remove_alpha2_target_hit': poc['groups']['remove']['2.0']['target_hit'],
                'add_alpha3_target_hit': poc['groups']['add']['3.0']['target_hit'],
                'remove_valid_alpha1': poc['groups']['remove']['1.0']['valid'],
                'remove_valid_alpha3': poc['groups']['remove']['3.0']['valid'],
                'remove_cd_drift_alpha1': poc['groups']['remove']['1.0']['cd_drift'],
                'remove_cd_drift_alpha3': poc['groups']['remove']['3.0']['cd_drift'],
            },
        }

    if lin:
        out['linear_editing_reverify'] = {
            'part': lin.get('part'),
            'n_cases': lin.get('n_cases'),
            'star_valid_rate': lin.get('star_valid_rate'),
            'real_fix_rate': lin.get('real_fix_rate'),
            'note': lin.get('note'),
        }
    return out


def _task4_legacy(ph_dir):
    """LEGACY: aggregate pre-reference latent experiments (do not cite).

    Sources (all under ph_results/, pre-reference counters):
      latent_opt_classifier_results.json - gradient latent optimization to fix H1
      end_to_end_validation.json         - real B-Rep verification of z_opt
      corridor_test/corridor_eval_results.json - latent interpolation topology path
      nearest_manifold_test.json         - latent distance between H1 classes
    """
    out = {}

    def _load(rel):
        p = os.path.join(ph_dir, rel)
        if not os.path.exists(p):
            return None
        with open(p, 'r') as f:
            return json.load(f)

    # --- linear editing / latent optimization ---
    lat = _load('latent_opt_classifier_results.json')
    if lat:
        fixed = sum(1 for x in lat if x.get('fixed'))
        out['latent_optimization'] = {
            'n_cases': len(lat),
            'classifier_fixed': fixed,
            'classifier_fix_rate': fixed / len(lat),
            'mean_z_delta': float(np.mean([x['z_delta'] for x in lat])),
            'note': 'classifier-predicted H1 fixed; needs real B-Rep verification',
        }
    e2e = _load('end_to_end_validation.json')
    if e2e:
        verified = [x for x in e2e if x.get('h1_real_after') is not None]
        fixed_real = sum(1 for x in e2e if x.get('fixed_real'))
        out['latent_optimization_real_brep'] = {
            'n_cases': len(e2e),
            'n_verifiable': len(verified),
            'fixed_real': fixed_real,
            'real_fix_rate': fixed_real / len(e2e),
            'conclusion': 'classifier-level fix does NOT transfer to real B-Rep',
        }

    # --- corridor (interpolation) test ---
    cor = _load('corridor_test/corridor_eval_results.json')
    if cor:
        by_pair = defaultdict(list)
        for x in cor:
            by_pair[x['pair_id']].append(x)
        n_pairs = len(by_pair)
        n_invalid_pts = sum(1 for x in cor if not x.get('success'))
        # monotonicity: does h1 move gradually src->tgt or jump?
        abrupt = 0
        for pid, pts in by_pair.items():
            pts = sorted([p for p in pts if p.get('success') and p.get('h1') is not None],
                         key=lambda p: p['alpha'])
            src_h1 = pts[0]['src_h1'] if pts else None
            tgt_h1 = pts[0]['tgt_h1'] if pts else None
            if src_h1 is None:
                continue
            hs = [p['h1'] for p in pts]
            # an abrupt transition: max single-step jump > 1
            jumps = [abs(hs[i + 1] - hs[i]) for i in range(len(hs) - 1)]
            if jumps and max(jumps) > 1:
                abrupt += 1
        out['corridor_interpolation'] = {
            'n_pairs': n_pairs,
            'n_points': len(cor),
            'invalid_point_rate': n_invalid_pts / len(cor),
            'pairs_with_abrupt_h1_jump': abrupt,
            'abrupt_jump_rate': abrupt / max(n_pairs, 1),
        }

    # --- nearest manifold distance ---
    nm = _load('nearest_manifold_test.json')
    if nm:
        out['nearest_manifold_distance'] = {
            k: {'mean_dist': v['mean_dist'], 'pct_under_2': v.get('pct_under_2')}
            for k, v in nm.items() if isinstance(v, dict) and 'mean_dist' in v
        }
    return out
