"""Aggregate the frozen gradient states without reading heldout answers."""
import json
from pathlib import Path
import statistics


def summarize(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    states = read('GRADIENT_AUDIT_FROZEN.json')['states']
    assert len(states) == 56 and all(r['weights_unchanged_exact'] for r in states)
    assert read('WORKER_EXIT.json')['code'] == read('GUARD_EXIT.json')['code'] == 0
    metrics = {}
    for arm in ('REPLAY_DOUBLE', 'PATH_REPLAY'):
        for endpoint in ('INITIAL', 'TERMINAL'):
            selected = [r for r in states if r['arm'] == arm and r['endpoint'] == endpoint]
            assert len(selected) == 14
            summary = {}
            for name in ('source_vs_replay', 'source_vs_raw_fisher', 'source_vs_raw_combined_aux',
                         'source_vs_original_unclipped_total', 'source_vs_priority_unclipped_total'):
                values = [b[name] for r in selected for b in r['blocks'] if b[name]['eligible_nonzero']]
                if values:
                    summary[name] = dict(eligible_blocks=len(values),
                        negative_blocks=sum(v['cosine'] < -1e-6 for v in values),
                        minimum_cosine=min(v['cosine'] for v in values),
                        median_cosine=statistics.median(v['cosine'] for v in values),
                        median_norm_ratio=statistics.median(v['norm_ratio'] for v in values))
            metrics[arm + '/' + endpoint] = summary
    complete = read('COMPLETE.json')
    return dict(status=complete['status'], source_priority_test_admissible=complete['source_priority_test_admissible'],
        eligible_terminal_PATH_blocks=complete['eligible_terminal_PATH_blocks'],
        conflicting_terminal_PATH_blocks=len(complete['conflicting_terminal_PATH_blocks']),
        metrics=metrics, counts=complete['counts'], cost=read('GUARD_EXIT.json'),
        interpretation='Fixed56 endpoint expected gradients, no optimizer step/generation; not a causal trajectory reconstruction')
