"""Relate frozen projection ratios to CORE correctness and refusal-coded errors."""
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


KINDS = ('paraphrase', 'distractor_insertion', 'numeric_perturbation', 'negation_uncertainty')
SEED = 20261127
REPLICATES = 2000


def logistic(design, response):
    coefficient = np.zeros(design.shape[1])
    event_rate = np.mean(response)
    if not 0 < event_rate < 1:
        raise ValueError('Both outcomes are required')
    coefficient[0] = np.log(event_rate / (1 - event_rate))

    def predict(beta):
        return np.einsum('ij,j->i', design, beta)

    def loss(margin):
        return float(np.sum(np.logaddexp(0, margin) - response * margin))

    for _ in range(80):
        margin = predict(coefficient)
        probability = np.exp(-np.logaddexp(0, -margin))
        weights = probability * (1 - probability)
        information = np.einsum('ni,n,nj->ij', design, weights, design)
        gradient = np.einsum('ni,n->i', design, response - probability)
        step = np.linalg.solve(information, gradient)
        if np.max(np.abs(step)) < 1e-8:
            return coefficient, loss(margin)
        candidate = coefficient + step
        while loss(predict(candidate)) > loss(margin) + 1e-12:
            step /= 2
            candidate = coefficient + step
        coefficient = candidate
        if np.max(np.abs(coefficient)) > 80:
            break
    raise ValueError('Logistic fit did not converge')


def cluster_indices(clusters):
    return [np.flatnonzero(clusters == c) for c in np.unique(clusters)]


def bootstrap(designs, outcomes, clusters):
    groups = cluster_indices(clusters)
    rng = np.random.default_rng(SEED)
    results = [[] for _ in designs]
    for _ in range(REPLICATES):
        select = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for j, (design, response) in enumerate(zip(designs, outcomes)):
            try:
                beta, _ = logistic(design[select], response[select])
            except (ValueError, np.linalg.LinAlgError):
                continue
            results[j].append(beta)
    return [np.asarray(item) for item in results]


def interval(samples):
    return np.quantile(samples, [.025, .975], axis=0).T.tolist()


def describe(values):
    return {'n': len(values), 'median': float(np.median(values)),
            'iqr': np.quantile(values, [.25, .75]).tolist()}


def select(rows, kind):
    return [r for r in rows if r['variant'] == kind]


def main(root):
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').open()]
    groups = {int(k): v for k,v in json.loads((root/'image_groups.private.json').read_text()).items()}
    assert len(rows) == 1375 and len(groups) == 150 and len(set(groups.values())) == 129
    fit = defaultdict(list)
    selected_to_owner = {}
    for row in rows:
        if row['role'] == 'NATIVE':
            assert row['routed_to_owner']
            selected_to_owner[row['selected']] = row['owner']
        if row['role'] == 'FIT':
            assert row['routed_to_owner']
            fit[row['selected']].append(row)
    assert len(fit) == 150 and all(len(f) == 4 for f in fit.values())
    for row in rows:
        row['cluster'] = groups[row['owner']]
        row['ratio'] = {}
        for arm in ('MED', 'MDU'):
            r = row['arms'][arm]['projection'] / np.median([v['arms'][arm]['projection'] for v in fit[row['selected']]])
            assert np.isfinite(r) and r > 0
            row['ratio'][arm] = r
    locality = select(rows, 'T2L')
    indirect = [r for r in select(rows, 'negation_uncertainty') if r['audit'] == 'V' and r['routed_to_owner']]
    assert len(locality) == 54 and len(indirect) == 135

    def compare(a, b):
        image_a = defaultdict(list)
        image_b = defaultdict(list)
        for r in a:
            image_a[r['cluster']].append(r['ratio']['MDU'])
        for r in b:
            image_b[r['cluster']].append(r['ratio']['MDU'])
        rng = np.random.default_rng(SEED)
        all_groups = sorted(set(image_a) | set(image_b))
        samples = []
        for _ in range(REPLICATES):
            chosen = [all_groups[i] for i in rng.integers(0, len(all_groups), len(all_groups))]
            a_vals = [v for i in chosen for v in image_a[i]]
            b_vals = [v for i in chosen for v in image_b[i]]
            samples.append(np.median(a_vals) - np.median(b_vals))
        return {
            'comparison_groups': len(all_groups),
            'median_difference': float(np.median([r['ratio']['MDU'] for r in a]) -
                                       np.median([r['ratio']['MDU'] for r in b])),
            'ci95': interval(samples),
        }

    distribution = {}
    for name, panel in (('locality', locality), ('locality_same_route',[r for r in locality if r['routed_to_owner']]),
                        ('indirect_valid',indirect)):
        distribution[name] = {arm: describe([r['ratio'][arm] for r in panel]) for arm in ('MED','MDU')}
        distribution[name]['mdu_at_or_above_0_5'] = int(sum(r['ratio']['MDU'] >= .5 for r in panel))
        distribution[name]['edits'] = len({r['owner'] for r in panel})
        distribution[name]['images'] = len({r['cluster'] for r in panel})
    distribution['locality_minus_indirect'] = compare(locality, indirect)
    by_owner = defaultdict(dict)
    for r in indirect:
        by_owner[r['owner']]['indirect'] = r['ratio']['MDU']
    for r in locality:
        if r['routed_to_owner']:
            by_owner[r['owner']].setdefault('locality', []).append(r['ratio']['MDU'])
    paired = [np.median(d['locality'])-d['indirect'] for d in by_owner.values()
              if 'locality' in d and 'indirect' in d]
    distribution['within_owner_locality_minus_indirect'] = describe(paired)

    def regression(panel, refusal=False, adjust=False, route=False):
        target = np.asarray([int(r['status']['MDU150']['behavior']=='A' if refusal
                                 else r['status']['MDU150']['core']) for r in panel], dtype=float)
        if refusal:
            assert all(r['variant']=='negation_uncertainty' and
                       (r['status']['MDU150']['core'] or r['status']['MDU150']['behavior'] in ('A','O','P')) for r in panel)
        ratios = np.asarray([r['ratio']['MDU'] for r in panel])
        groups_matrix = np.column_stack([[r['variant']==k for r in panel] for k in KINDS[1:]])
        x_base = np.column_stack((np.ones(len(panel)), np.log2(ratios)))
        route_column = [[int(not r['routed_to_owner']) for r in panel]] if route else []
        covariates = [x_base]
        if adjust:
            covariates.extend([np.column_stack((x_base[:,0],groups_matrix)), np.column_stack((x_base,groups_matrix))])
        if route:
            covariates = [np.column_stack((x, *route_column)) for x in covariates]
        clusters = np.asarray([r['cluster'] for r in panel])
        samples = bootstrap(covariates,[target]*len(covariates),clusters)
        fits = [logistic(x,target) for x in covariates]
        result = {
            'n':len(panel),'images':len(set(clusters)), 'positives':int(target.sum()),
            'ranges':{k:describe([r['ratio']['MDU'] for r in panel if r['variant']==k])
                      for k in set(r['variant'] for r in panel)},
            'fits':[{'coef':v.tolist(),'coef_ci95':interval(b),'valid_bootstrap':len(b),
                     'neg_log_likelihood':nll} for (v,nll),b in zip(fits,samples)],
        }
        if not refusal:
            chosen = 2 if adjust else 0
            coefficients = fits[chosen][0]
            b = samples[chosen]
            intercept = coefficients[0] + (coefficients[4] if adjust else 0)
            boot_intercept = b[:, 0] + (b[:, 4] if adjust else 0)
            cutoff = 2 ** ((math.log(4) - intercept) / coefficients[1])
            valid = b[:, 1] > 0
            log_cutoff = (math.log(4) - boot_intercept[valid]) / b[valid, 1]
            cut_interval = [2**x for x in interval(log_cutoff)]
            subset = np.array([r['variant']=='negation_uncertainty' for r in panel])
            over = subset & (ratios >= cutoff)
            result['indirect_80'] = {
                'ratio':cutoff, 'ci95':cut_interval,'bootstrap_increasing':int(valid.sum()),
                'above_n':int(over.sum()), 'above_core':int(target[over].sum()),
                'range':[float(ratios[subset].min()),float(ratios[subset].max())],
            }
        if adjust:
            result['groups'] = {k:{'n':sum(r['variant']==k for r in panel),
                                    'correct':sum(r['variant']==k and r['status']['MDU150']['core'] for r in panel)}
                                for k in KINDS}
        return result

    four_types = [r for r in rows if r['variant'] in KINDS
                  and (r['variant'] != 'negation_uncertainty' or r['audit'] == 'V')]
    assert len(four_types) == 555
    regressions = {
        'correctness_group_adjusted':regression(four_types, adjust=True),
        'correctness_group_and_route_adjusted':regression(four_types, adjust=True, route=True),
        'indirect_correctness':regression(indirect),
        'indirect_refusal':regression(indirect, refusal=True),
    }
    print(json.dumps({'distribution':distribution,'regression':regressions},allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
