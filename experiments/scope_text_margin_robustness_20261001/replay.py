"""Three frozen margins on cached H/Base outputs; no new model or Judge."""
import copy
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/tmp'))
SOURCE = Path(os.environ.get('SOURCE_ROOT', '/tmp'))


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def retain(winner, q, protected, tau):
    return winner if winner is not None and (q >= tau or protected) else None


def consumer_id(row):
    return tuple(row[k] for k in ['mode', 'prefix', 'edit', 'task', 'input_id'])


def selfcheck():
    assert retain(None, None, True, -0.1) is None
    assert retain('e', -0.1, False, -0.1) == 'e'
    assert retain('e', -0.10001, False, -0.1) is None
    assert retain('e', -0.5, True, 0.1) == 'e'
    assert retain('e', 0, False, 0.1) is None
    a = dict(mode='sequential', prefix=4, edit='e', task='T2G', input_id='x')
    assert consumer_id(a) != consumer_id(dict(a, prefix=8))


def main():
    started, cpu = time.time(), time.process_time()
    selfcheck()
    config = read(ROOT/'public/REPLAY_CONFIG.json')
    manifest = read(ROOT/'RUN_MANIFEST.json')
    prior = read(ROOT/'INITIALIZATION_RECOVERY.json') if (ROOT/'INITIALIZATION_RECOVERY.json').exists() else {}
    failed_seconds = prior.get('failed_CPU_wall_upper_bound', 0)
    assert config['taus'] == [-0.1, 0.0, 0.1]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    argv = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert not any(k in argv.lower() for k in ['wangbomin', 'knowledge_editing', 'scope'])
    assert read(SOURCE/'RUN_STATUS.json')['status'] == 'COMPLETE'
    assert read(SOURCE/'public/FINAL_EXECUTION_AUDIT.json')['unresolved'] == 0
    # Bind existing read-only helpers to their original run, never the new ROOT.
    sys.path.insert(0, str(SOURCE))
    previous = os.environ['RUN_ROOT']
    os.environ['RUN_ROOT'] = str(SOURCE)
    try:
        from decision import joint, transitions
        from judge_protocol import read_scores, NAMESPACE
    finally:
        os.environ['RUN_ROOT'] = previous
    scores = read_scores(SOURCE)
    missing = set(read(SOURCE/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'])
    masks = read(SOURCE/'private/BASE_MASKS.json')
    queue = read(SOURCE/'QUEUE.json')
    assert len(queue) == 25 and all(j['status'] == 'COMPLETE' for j in queue)
    outputs, private = {}, []
    write('RUN_STATUS.json', dict(status='RUNNING',phase='FIXED_REPLAY',epoch=started))
    for phase, expected in config['panels'].items():
        refs = read(SOURCE/f'private/references/{phase}.json')
        rows = [r for j in queue if j['phase'] == phase
                for path in (SOURCE/'jobs'/j['id']).rglob('CONSUMERS.json') for r in read(path)]
        original = {consumer_id(r): r for r in rows if r['arm'] == 'R0'}
        txt = {consumer_id(r): r for r in rows if r['arm'] == 'TXT'}
        assert len(original) == len(txt) == len(refs) == expected
        assert original.keys() == txt.keys() == {consumer_id(r) for r in refs}
        old = (read(SOURCE/'public/JOINT_ROUTER_DECISION.json')['candidates']['TXT']
               if phase == 'DEV' else read(SOURCE/'public/REG_JOINT_REVIEW.json'))
        support = old['gates']['negative_prototype_support']
        assert support['status'] == 'PASS'
        frozen = {r['input_id']: r['frozen_base_correct'] for r in refs if r['mode'] == 'EXPOSED_REGRESSION'}
        outputs[phase] = {}
        for tau in config['taus']:
            proposed = []
            for key, a in original.items():
                assert time.time() < manifest['deadline_epoch']
                assert time.time()-started+failed_seconds < config['CPU_wall_seconds_limit'] and not (ROOT/'STOP').exists()
                b = txt[key]
                winner, d = a['route']['logical_edit_id'], b['route_diagnostics']
                if winner is not None:
                    assert math.isfinite(d['q']) and -1.000001 <= d['q'] <= 1.000001
                    assert d['R0_activated'] and a['route']['nearest_distance'] <= a['route']['radius']
                target = retain(winner, d.get('q'), bool(d.get('text_guard')), tau)
                c = copy.deepcopy(b)
                c['route'].update(logical_edit_id=target, activated=target is not None)
                c['judge_key'] = a['judge_key'] if target is not None or winner is None else a['base_judge_key']
                assert c['judge_key'] in scores or c['judge_key'] in missing
                if c['mode'] == 'EXPOSED_REGRESSION':
                    c['frozen_base_correct'] = frozen[c['input_id']]
                if tau == 0:
                    assert c['route']['logical_edit_id'] == b['route']['logical_edit_id']
                    assert c['judge_key'] == b['judge_key']
                proposed.append(c)
                private.append(dict(phase=phase,tau=tau,identity=list(key),judge_key=c['judge_key'],winner=target))
            # Support qualification is inherited in aggregate from the closed run;
            # no prototype/feature construction or new medical labels occur here.
            result = joint(proposed, refs, scores, masks, [dict(negative_status=support['status'])])['candidates']['TXT']
            if tau == 0:
                assert all(result[k] == old[k] for k in result), 'tau0 joint must reproduce executed TXT exactly'
            changed = [r for r in proposed if r['route']['logical_edit_id'] != txt[consumer_id(r)]['route']['logical_edit_id']]
            tr = {}
            # Existing matched/transition helpers operate within one panel only.
            for mode, prefix in sorted({(r['mode'], r['prefix']) for r in proposed}):
                a = [r for r in proposed if (r['mode'], r['prefix']) == (mode, prefix)]
                b = [r for r in txt.values() if (r['mode'], r['prefix']) == (mode, prefix)]
                summary = transitions(a, b, scores)
                summary.pop('expert_switch_matrix')  # Raw identities stay private.
                tr[mode+'/'+str(prefix)] = summary
            outputs[phase][str(tau)] = dict(joint=result,route_changes_vs_executed_tau0=len(changed),
                changed_tasks=dict(Counter(r['task'] for r in changed)),transitions_vs_tau0=tr,
                full_denominator=len(proposed),missing_occurrences=sum(r['judge_key'] not in scores for r in proposed))
    write('private/COUNTERFACTUAL_ROUTES.json', private)
    write('public/MARGIN_AGGREGATES.json',dict(status='COMPLETE_EXPOSED_SENSITIVITY',taus=config['taus'],
        by_phase=outputs,tau0_route_and_joint_reproduction='PASS',original_R0_winner_and_radius_fixed=True,
        support_qualification='INHERITED_FROM_CLOSED_TXT',best_tau_selected=False,independent_CONFIRM=False,new_verified_CAL=0))
    seconds = time.time()-started
    assert seconds+failed_seconds < config['CPU_wall_seconds_limit']
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < config['storage_bytes_limit']
    ledger = dict(CPU_wall_seconds=seconds,CPU_seconds=time.process_time()-cpu,GPU_hours=0,new_Judge=0,
        failed_CPU_wall_upper_bound=failed_seconds,cumulative_CPU_wall_upper_bound=seconds+failed_seconds,
        failed_input_bytes_upper_bound=prior.get('failed_input_bytes_upper_bound', 0),
        cumulative_Judge=read(SOURCE/'RESOURCE_LEDGER.json')['judge_submission_attempt_items'],
        permanent_missing=len(missing),training_steps=0,new_generation=0)
    write('RESOURCE_LEDGER.json',ledger)
    write('PROCESS_RECEIPT.json',dict(pid=os.getpid(),argv=argv,started_epoch=started,CUDA_VISIBLE_DEVICES=''))
    write('public/FINAL_EXECUTION_AUDIT.json',dict(status='COMPLETE',consumers_per_tau=2047,
        taus=3,tau0_reproduction='PASS',all_score_keys_bound_or_permanent_missing=True,**ledger))
    write('RUN_STATUS.json',dict(status='COMPLETE',phase='CLOSED_EXPOSED_SENSITIVITY',epoch=time.time()))
    print(json.dumps({p:{t:dict(status=v['joint']['status'],changes=v['route_changes_vs_executed_tau0']) for t,v in ts.items()} for p,ts in outputs.items()}))


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
        print('PASS: inclusive margin, fixed winner, lexical protection and R0 OFF')
    else:
        try:
            main()
        except Exception as error:
            write('FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
            write('RUN_STATUS.json',dict(status='FAILED_PRESERVED'))
            raise
