"""CPU-only readback of anonymous fixed probes; no new outcome test."""
import json
import statistics as s
from pathlib import Path


def audit(result):
    probes = result['probes']
    assert len(probes) == 32 and result['endpoint_exact_all']
    assert len({(p['expert_order'], p['step']) for p in probes}) == 32
    modes = ('RAW', 'MOMENT', 'PRECOND', 'ADAM')
    rows = []
    for mode in modes:
        selected = [next(c for c in p['candidates'] if c['mode'] == mode) for p in probes]
        adam = [next(c for c in p['candidates'] if c['mode'] == 'ADAM') for p in probes]
        kl = lambda c: s.mean(z['KL_from_prewrite'] for z in c['native_FIT'])
        for metric in ('map_delta_frobenius', 'predictor_residual_delta_norm', 'KL_from_prewrite'):
            value = kl if metric == 'KL_from_prewrite' else lambda c: c[metric]
            ratios = [value(c) / value(a) for c, a in zip(selected, adam)]
            original = next(x for x in result['aggregate'] if x['mode'] == mode and x['metric'] == metric)
            assert s.median(ratios) == original['ratio_to_Adam']['median']
        absolute = [kl(c) for c in selected]
        stats = [z for c in selected for z in c['native_FIT']]
        by_step = []
        for step in (2, 80, 160, 320):
            indices = [i for i, p in enumerate(probes) if p['step'] == step]
            assert len(indices) == 8
            by_step.append(dict(step=step,KL_ratio_median=s.median(kl(selected[i])/kl(adam[i]) for i in indices)))
        rows.append(dict(mode=mode,absolute_KL=dict(minimum=min(absolute),median=s.median(absolute),maximum=max(absolute)),
            argmax_changes=sum(z['argmax_changes'] for z in stats),argmax_correct=sum(z['argmax_correct'] for z in stats),
            token_observations=sum(z['tokens'] for z in stats),by_step=by_step,
            states_KL_above_Adam=sum(kl(c)>kl(a) for c,a in zip(selected,adam))))
    return dict(status='PASS',kind='POSTHOC_DESCRIPTIVE_READBACK',original_aggregate_reproduced=True,rows=rows,
        scope='Repeated native/FIT teacher-forced tokens, not independent samples, free generation, or protection outcomes.')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'reports/medtrace_tt_mechanism_v1_20261009'
    result = audit(json.loads((root / 'RESULTS.json').read_text()))
    (root / 'REVIEW_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
