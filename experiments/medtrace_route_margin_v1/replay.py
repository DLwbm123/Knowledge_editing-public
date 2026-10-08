"""Exact two-output replay: no generation, new judging, training, or score retries."""
import copy
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from margin import PRIMARY_MARGIN, SENSITIVITY_MARGINS, select, selfcheck

RUN = Path(os.environ['RUN_ROOT'])
PARENT = Path(os.environ['MARGIN_PARENT'])
sys.path.insert(0, str(PARENT/'private/tools'))
import astra_report as ar
c, r = ar.common, ar.r


def main():
    started = time.time()
    check = selfcheck()
    lock = c.read(RUN/'private/PROTOCOL.json')
    assert lock['primary_margin'] == PRIMARY_MARGIN
    assert lock['sensitivity_margins'] == list(SENSITIVITY_MARGINS)
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    root = PARENT/'private/judge_gate_astra_medium'
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db = sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    payloads = {x['key']: dict(x) for x in db.execute('SELECT * FROM payload')}
    assert all(x['status'] in ('FORMAT_VALID', 'MISSING') for x in payloads.values())
    scores = {key: x['correct'] for key, x in payloads.items()}
    rows = {}
    for row in db.execute("SELECT * FROM consumer WHERE method IN ('G_R0','G_MODAL_R0_CANDIDATES')"):
        d = c.read(row['path'])
        assert c.digest(d) == row['output_binding']
        rows.setdefault(row['query_id'], {})[row['method']] = (d, dict(row))
    assert len(rows) == 1513 and all(len(v) == 2 for v in rows.values())
    tasks = c.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    indices = {t['edit_id']: i for i, t in enumerate(tasks)}
    labels = {'MARGIN_002': PRIMARY_MARGIN, 'MARGIN_001': SENSITIVITY_MARGINS[0],
              'MARGIN_003': SENSITIVITY_MARGINS[1]}
    records, selections = [], []
    counts = {label: Counter() for label in labels}
    for qid, pair in rows.items():
        base, bc = pair['G_R0']; modal, mc = pair['G_MODAL_R0_CANDIDATES']
        assert base['binding']['judge_input'] == modal['binding']['judge_input']
        assert base['binding']['input'] == modal['binding']['input']
        for field in ('weights', 'execution', 'gate_lock', 'prefix', 'node', 'slot'):
            assert base['binding']['phase'][field] == modal['binding']['phase'][field]
        assert base['route']['activated'] == modal['route']['activated']
        assert base['route']['original_R0'] == modal['route']['original_R0']
        for d, consumer in (pair['G_R0'], pair['G_MODAL_R0_CANDIDATES']):
            records.append((d, consumer['payload_key']))
        original = indices.get(base['effective_expert'])
        candidate = indices.get(modal['effective_expert'])
        for label, threshold in labels.items():
            chosen = select(original, candidate, modal['route']['scores'], threshold)
            use_modal = chosen != original
            source, consumer = pair['G_MODAL_R0_CANDIDATES' if use_modal else 'G_R0']
            assert source['effective_expert'] == (tasks[chosen]['edit_id'] if chosen is not None else None)
            view = dict(source, binding=dict(source['binding'], arm=label,
                        phase=dict(source['binding']['phase'], arm=label)))
            records.append((view, consumer['payload_key']))
            counts[label]['switches'] += use_modal
            counts[label]['reverted_modal_switches'] += candidate != original and not use_modal
            counts[label]['queries'] += 1
            counts[label]['missing_consumers'] += scores[consumer['payload_key']] is None
            selections.append(dict(arm=label, query_id=qid, threshold=threshold,
                advantage=None if original is None else modal['route']['scores'][candidate]-modal['route']['scores'][original],
                selected_expert=source['effective_expert'], source_path=consumer['path'],
                source_output_binding=consumer['output_binding'], payload_key=consumer['payload_key'],
                score_status=payloads[consumer['payload_key']]['status']))
    c.write(RUN/'private/SELECTIONS.json', selections)
    panels, coefs, groups = [], {}, {}
    for label in ('G_R0', 'G_MODAL_R0_CANDIDATES', *labels):
        for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            m, coef, group = r.panel(records, scores, 'RETRO146', label, 0, 'bank_R0', task, tasks, 146)
            m.update(ar.bootstrap(coef, scores, group)); panels.append(m)
            coefs[label, task] = coef; groups.update(group)
        print('PANELS', label, flush=True)
    prior = c.read(PARENT/'public/RESULTS.json')
    for panel in panels[:10]:
        old = next(x for x in prior['panels'] if (x['arm'], x['task']) == (panel['arm'], panel['task']))
        for key in ('macro_bounds', 'observations', 'missing', 'edit_units', 'known_correct'):
            assert panel[key] == old[key], (panel['arm'], panel['task'], key)
    contrasts = []
    for label in labels:
        for control in ('G_R0', 'G_MODAL_R0_CANDIDATES'):
            for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
                a, b = coefs[label, task], coefs[control, task]; assert set(a) == set(b)
                coef = {e: r.combine([a[e], {k: -v for k, v in b[e].items()}]) for e in a}
                bounds = r.score_bounds(r.combine([{k: v/len(coef) for k, v in x.items()} for x in coef.values()]), scores) if coef else [None, None]
                contrasts.append(dict(a=label, b=control, task=task, paired_edits=len(coef),
                    delta_bounds_pp=[v*100 if v is not None else None for v in bounds],
                    **ar.bootstrap(coef, scores, groups)))
            print('CONTRAST', label, control, flush=True)
    checks = []
    for label in ('G_R0', 'G_MODAL_R0_CANDIDATES', *labels):
        ds = [d for d, _ in records if d['binding']['arm'] == label and d['binding']['input'].get('role') == 'CHECK']
        assert len(ds) == 4
        checks.append(dict(arm=label, observations=4, mean_KL=sum(d['U_KL'] for d in ds)/4,
            Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,
            route_ON=sum(d['route']['activated'] for d in ds), medical_accuracy=False))
    keys = {key for _, key in records}
    result = dict(status='TERMINAL_WITH_INHERITED_MISSING' if any(scores[k] is None for k in keys) else 'COMPLETE',
        primary='MARGIN_002', sensitivity_only=['MARGIN_001', 'MARGIN_003'], panels=panels,
        contrasts=contrasts, CHECK=checks, routing=counts, query_inputs=1513, new_route_consumers=len(selections),
        exact_replay_consumers=len(records), inherited_unique_payloads=len(keys),
        inherited_missing_payloads=sum(scores[k] is None for k in keys), new_training=0,
        new_generation=0, new_Judge_attempts=0, new_GPU_process_hours=0,
        Judge_attempts=prior['Judge_attempts'], GPU_process_hours=prior['GPU_process_hours'],
        CP_enabled=False, independent_confirmation=False, threshold_selected_on_exposed_DEV=True,
        medical_protection='NA_SCOPE_NOT_QUALIFIED', source=c.read(RUN/'private/SOURCE_VERSION.json'),
        checks=check, baseline_reproduction='PASS', seconds=time.time()-started)
    c.write(RUN/'public/RESULTS.json', result)
    c.write(RUN/'private/REPORT_COMPLETE.json', dict(status='COMPLETE', epoch=time.time(),
        exact_replay=True, protocol_binding=c.digest(lock), new_checkpoints=0))
    c.write(RUN/'public/PROGRESS.json', dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING', complete=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        c.write(RUN/'private/FAILURE.json', dict(error=repr(error), traceback=traceback.format_exc(), retry=False))
        raise
