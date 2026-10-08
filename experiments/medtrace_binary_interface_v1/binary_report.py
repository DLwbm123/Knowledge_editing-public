"""All main panels, all changed queries and balanced fixed-expert polarity checks."""
import sqlite3
from collections import Counter
import astra_report as ar
import binary as exp
import binary_queue as queue
c, RUN, r = exp.c, exp.RUN, ar.r
LABELS = ('MARGIN_002', 'TYPE_GUARD_002')+exp.ARMS


def main():
    root = queue.q.ROOT; assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db = sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro', uri=True); db.row_factory = sqlite3.Row
    scores = {x['key']: x['correct'] for x in db.execute('SELECT * FROM payload')}
    records, mapping = [], {}
    for x in db.execute('SELECT * FROM consumer'):
        d = c.read(x['path']); assert c.digest(d) == x['output_binding']
        d = dict(d, binding=dict(d['binding'], arm=x['method'], mode=x['mode'],
            phase=dict(d['binding']['phase'], arm=x['method'])))
        records.append((d, x['payload_key'])); mapping[x['method'], x['query_id']] = d, x['payload_key']
    assert len(records) == 6232
    tasks = exp.p.tasks(); panels, coefs, groups = [], {}, {}
    for label in LABELS:
        for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            m, coef, group = r.panel(records, scores, 'RETRO146', label, 0, 'bank_R0', task, tasks, 146)
            m.update(ar.bootstrap(coef, scores, group)); panels.append(m)
            coefs[label, task] = coef; groups.update(group)
    prior = c.read(exp.TYPE/'public/RESULTS.json')
    for panel in panels[:10]:
        old = next(x for x in prior['panels'] if (x['arm'], x['task']) == (panel['arm'], panel['task']))
        for key in ('macro_bounds', 'observations', 'missing', 'edit_units', 'known_correct'):
            assert panel[key] == old[key]
    contrasts = []
    for a, b in [(exp.ARMS[1], 'MARGIN_002'), (exp.ARMS[1], 'TYPE_GUARD_002'), (exp.ARMS[1], exp.ARMS[0])]:
        for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            aa, bb = coefs[a, task], coefs[b, task]; assert set(aa) == set(bb)
            coef = {e: r.combine([aa[e], {k: -v for k, v in bb[e].items()}]) for e in aa}
            bounds = r.score_bounds(r.combine([{k: v/len(coef) for k, v in x.items()} for x in coef.values()]), scores) if coef else [None, None]
            contrasts.append(dict(a=a, b=b, task=task, paired_edits=len(coef),
                delta_bounds_pp=[v*100 if v is not None else None for v in bounds], **ar.bootstrap(coef, scores, groups)))
    inputs = c.read(RUN/'private/INPUTS.json'); changed = [x for x in inputs if not x['negative']]
    all_changed, negative = {}, []
    for arm in exp.ARMS:
        count = Counter()
        for item in changed:
            qid = item['row']['query_id']; before = scores[mapping['MARGIN_002', qid][1]]; after = scores[mapping[arm, qid][1]]
            count['missing' if before is None or after is None else 'gain' if after > before else 'loss' if after < before else 'unchanged'] += 1
        all_changed[arm] = count
        pairs = []
        for item in inputs:
            if item['negative']:
                pos, pk = mapping[arm, item['original_qid']]; neg, nk = mapping[arm, item['row']['query_id']]
                pairs.append((pos, neg, scores[pk], scores[nk]))
        negative.append(dict(arm=arm, pairs=len(pairs), positives_known_correct=sum(a == 1 for _, _, a, b in pairs),
            negatives_known_correct=sum(b == 1 for _, _, a, b in pairs),
            both_known_correct=sum(a == b == 1 for _, _, a, b in pairs),
            missing_pairs=sum(a is None or b is None for _, _, a, b in pairs),
            positive_Yes=sum(a['R0']['raw_answer'] == 'Yes' for a, b, _, _ in pairs),
            negative_Yes=sum(b['R0']['raw_answer'] == 'Yes' for a, b, _, _ in pairs),
            route_on_new_query_evaluated=False, scope='paired decoder diagnostic with original expert fixed'))
    checks = []
    for label in LABELS:
        ds = [d for d, _ in records if d['binding']['arm'] == label and d['binding']['input'].get('role') == 'CHECK']
        assert len(ds) == 4
        checks.append(dict(arm=label, observations=4,
            Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,
            mean_KL=sum(d['U_KL'] for d in ds)/4 if all(d['U_KL'] is not None for d in ds) else None,
            KL_status='NOT_COMPARABLE_DECODER_SUPPORT_CHANGED' if label in exp.ARMS else 'ORIGINAL_TEACHER_KL',
            route_ON=sum(d['effective_expert'] is not None for d in ds), medical_accuracy=False))
    resource = c.read(RUN/'RESOURCE_LEDGER.json'); inherited = c.read(RUN/'private/INHERITED_COST.json')
    result = dict(status='TERMINAL_WITH_MISSING' if any(v is None for v in scores.values()) else 'COMPLETE',
        panels=panels, contrasts=contrasts, all_changed_query_transitions=all_changed,
        polarity_diagnostic=negative, CHECK=checks, baseline_reproduction='PASS',
        scoring=c.read(root/'READY.json'), missing_payloads=sum(v is None for v in scores.values()),
        Judge_attempts=resource['Judge_attempts'], new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],
        GPU_process_hours=c.used()/3600, new_GPU_process_hours=(c.used()-inherited['gpu_seconds_used'])/3600,
        new_training=0, new_outputs=366, CP_enabled=False, independent_confirmation=False,
        scope_qualification=False, decoder_only_negative_challenge=True)
    c.write(RUN/'public/RESULTS.json', result); exp.p.done('REPORT_COMPLETE'); exp.p.progress('RESULTS_COMPLETE_PUBLICATION_PENDING')


if __name__ == '__main__':
    main()
