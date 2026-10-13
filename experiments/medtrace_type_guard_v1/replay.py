"""Frozen margin routing plus a question-type veto; exact historical OFF reuse."""
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from question_type import incompatible, question_type, selfcheck

RUN = Path(os.environ['RUN_ROOT'])
PARENT = Path(os.environ['TYPE_PARENT'])
MARGIN_RUN = Path(os.environ['MARGIN_PARENT'])
sys.path.insert(0, str(RUN/'private/tools'))
import astra_report as ar
from margin import select
c, r = ar.common, ar.r
LABELS = ('G_R0', 'G_MODAL_R0_CANDIDATES', 'MARGIN_002', 'TYPE_GUARD_002')


def main():
    started = time.time(); tests = selfcheck()
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    assert (MARGIN_RUN/'private/REPORT_COMPLETE.json').exists()
    db = sqlite3.connect('file:'+str(PARENT/'private/judge_gate_astra_medium/queue.sqlite')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    payloads = {x['key']: dict(x) for x in db.execute('SELECT * FROM payload')}
    assert all(x['status'] in ('FORMAT_VALID', 'MISSING') for x in payloads.values())
    scores = {key: x['correct'] for key, x in payloads.items()}
    index = {}
    for row in db.execute('SELECT * FROM consumer'):
        index.setdefault(row['query_id'], {})[row['method']] = dict(row)
    assert len(index) == 1513
    tasks = c.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    task_by_id = {t['edit_id']: t for t in tasks}
    indices = {t['edit_id']: i for i, t in enumerate(tasks)}
    old_selections = {x['query_id']: x for x in c.read(MARGIN_RUN/'private/SELECTIONS.json') if x['arm'] == 'MARGIN_002'}
    records, decisions = [], []
    def load(consumer):
        d = c.read(consumer['path']); assert c.digest(d) == consumer['output_binding']
        return d, consumer
    for qid, consumers in index.items():
        base, bc = load(consumers['G_R0']); modal, mc = load(consumers['G_MODAL_R0_CANDIDATES'])
        original, candidate = indices.get(base['effective_expert']), indices.get(modal['effective_expert'])
        selected = select(original, candidate, modal['route']['scores'], .02)
        chosen, cc = (modal, mc) if selected != original else (base, bc)
        assert chosen['effective_expert'] == old_selections[qid]['selected_expert']
        assert cc['payload_key'] == old_selections[qid]['payload_key']
        query = chosen['binding']['input']['question']
        edit = task_by_id[chosen['effective_expert']]['native']['question'] if selected is not None else ''
        blocked = selected is not None and incompatible(query, edit)
        guarded, gc = chosen, cc
        if blocked:
            # Lookup depends only on a historical OFF route, never on its answer or score.
            for arm in sorted(consumers, key=lambda x: (x != 'G_OLD_MODAL', x)):
                off, oc = load(consumers[arm])
                if off['effective_expert'] is None:
                    guarded, gc = off, oc; break
            else:
                raise RuntimeError('No exact historical OFF output; generation is required')
            assert not guarded['route']['activated'] and guarded['selected_experts'] == []
            assert guarded['binding']['input'] == chosen['binding']['input']
            assert guarded['binding']['judge_input'] == chosen['binding']['judge_input']
            assert guarded['binding']['phase']['weights'] == chosen['binding']['phase']['weights']
            b = guarded['binding']['judge_input']
            if not guarded['binding']['input'].get('role'):
                assert guarded['R0'] == dict(raw_answer=b['output']['model_answer_raw'], raw_token_ids=b['output']['raw_generated_token_ids'])
            else:
                assert guarded['Base_token_consistency'] is True and abs(guarded['U_KL']) < 1e-6
        for label, (source, consumer) in zip(LABELS, [(base, bc), (modal, mc), (chosen, cc), (guarded, gc)]):
            view = dict(source, binding=dict(source['binding'], arm=label,
                        phase=dict(source['binding']['phase'], arm=label)))
            records.append((view, consumer['payload_key']))
        decisions.append(dict(query_id=qid, query_type=question_type(query), edit_type=question_type(edit),
            blocked=blocked, previous_expert=chosen['effective_expert'], selected_expert=guarded['effective_expert'],
            source_path=gc['path'], source_output_binding=gc['output_binding'], payload_key=gc['payload_key']))
    c.write(RUN/'private/DECISIONS.json', decisions)
    ledger = c.read(RUN/'private/EVAL_LEDGER.json')
    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks}
    byqid = {d['query_id']: d for d in decisions}
    composition = {}
    for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
        qids = {q for t in tasks for e in t['events'] if e['task'] == task for q in e['all_probe_query_ids']}
        if task != 'T0':
            qids = {q for q in qids if ledger['Base_correctness'][q] == task.endswith('L')}
        if task.endswith('L'):
            qids = {q for q in qids if (ledger['queries'][q]['image_sha256'], ledger['queries'][q]['question']) not in active}
        composition[task] = dict(unique_queries=len(qids), blocked=sum(byqid[q]['blocked'] for q in qids),
            question_types=dict(Counter(byqid[q]['query_type'] for q in qids)))
    panels, coefs, groups = [], {}, {}
    for label in LABELS:
        for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            m, coef, group = r.panel(records, scores, 'RETRO146', label, 0, 'bank_R0', task, tasks, 146)
            m.update(ar.bootstrap(coef, scores, group)); panels.append(m)
            coefs[label, task] = coef; groups.update(group)
        print('PANELS', label, flush=True)
    prior = c.read(MARGIN_RUN/'public/RESULTS.json')
    for panel in panels[:15]:
        old = next(x for x in prior['panels'] if (x['arm'], x['task']) == (panel['arm'], panel['task']))
        for key in ('macro_bounds', 'observations', 'missing', 'edit_units', 'known_correct'):
            assert panel[key] == old[key], (panel['arm'], panel['task'], key)
    contrasts = []
    for control in LABELS[:3]:
        for task in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
            a, b = coefs['TYPE_GUARD_002', task], coefs[control, task]; assert set(a) == set(b)
            coef = {e: r.combine([a[e], {k: -v for k, v in b[e].items()}]) for e in a}
            bounds = r.score_bounds(r.combine([{k: v/len(coef) for k, v in x.items()} for x in coef.values()]), scores) if coef else [None, None]
            contrasts.append(dict(a='TYPE_GUARD_002', b=control, task=task, paired_edits=len(coef),
                delta_bounds_pp=[v*100 if v is not None else None for v in bounds], **ar.bootstrap(coef, scores, groups)))
    checks = []
    for label in LABELS:
        ds = [d for d, _ in records if d['binding']['arm'] == label and d['binding']['input'].get('role') == 'CHECK']
        assert len(ds) == 4
        checks.append(dict(arm=label, observations=4, mean_KL=sum(d['U_KL'] for d in ds)/4,
            Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,
            route_ON=sum(d['route']['activated'] for d in ds), medical_accuracy=False))
    keys = {key for _, key in records}
    result = dict(status='TERMINAL_WITH_INHERITED_MISSING' if any(scores[k] is None for k in keys) else 'COMPLETE',
        panels=panels, contrasts=contrasts, CHECK=checks, query_inputs=1513,
        guarded_OFF=sum(d['blocked'] for d in decisions), query_types=dict(Counter(d['query_type'] for d in decisions)),
        task_type_composition=composition,
        new_route_consumers=1513, exact_replay_consumers=len(records), inherited_unique_payloads=len(keys),
        inherited_missing_payloads=sum(scores[k] is None for k in keys), new_training=0, new_generation=0,
        new_Judge_attempts=0, new_GPU_process_hours=0, Judge_attempts=prior['Judge_attempts'],
        GPU_process_hours=prior['GPU_process_hours'], CP_enabled=False, independent_confirmation=False,
        rule_designed_after_DEV_inspection=True, semantic_scope_qualification=False,
        diagnostic_bias='All 52 unique evaluated T2L queries are explicit polar; no evaluated T0/T1G/T2G query is explicit polar under this parser.',
        cross_format_generalization='NOT_VALIDATED; a genuine cross-format edit may be incorrectly vetoed',
        source=c.read(RUN/'private/SOURCE_VERSION.json'), checks=tests, baseline_reproduction='PASS', seconds=time.time()-started)
    c.write(RUN/'public/RESULTS.json', result)
    c.write(RUN/'private/REPORT_COMPLETE.json', dict(status='COMPLETE', epoch=time.time(), exact_replay=True,
        protocol_binding=c.digest(c.read(RUN/'private/PROTOCOL.json')), new_checkpoints=0))
    c.write(RUN/'public/PROGRESS.json', dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING', complete=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        c.write(RUN/'private/FAILURE.json', dict(error=repr(error), traceback=traceback.format_exc(), retry=False))
        raise
