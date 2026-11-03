"""Wait for both arms, then reuse prior scores for the frozen 392 events."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

root = Path(os.environ['LOCAL_SCORE_ROOT'])
worktree = Path(os.environ['WORKTREE'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, value):
    path = root / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.chmod(0o600)
    tmp.replace(path)


def ssh(script, timeout=60):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'pro5000', 'python3', '-'],
                       input=script, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError('remote scoring dependency failed: ' + p.stderr[-600:])
    if len(p.stdout.encode()) > 20_000_000:
        raise ValueError('selected packet exceeds size limit')
    return json.loads(p.stdout)


def main():
    sys.path[:0] = [str(worktree / 'experiments' / name) for name in
                    ('med_common25_loss_ablation_v1', 'med_fixed_depth_v1', 'med150_source_review_v1',
                     'med_proposition_generalization_v1')]
    import identities, ledger, score_reuse, summary
    d = identities.consumers(Path('/private/tmp/.r.20261102.06/PACKET_PRIVATE.json'))
    du = identities.consumers(Path('/private/tmp/.r.20261102.05/PACKET_PRIVATE.json'))
    chosen = identities.common(d, du)
    history = json.loads(Path('/private/tmp/.r.20261102.03/HISTORY_INDEX_PRIVATE.json').read_text())
    if history.get('status') != 'AVAILABLE_HISTORY_INDEX_FROZEN' or history.get('history_coverage_complete') is not False:
        raise ValueError('history index missing or incompatible')
    write('HISTORY_INDEX_FROZEN.json', {'status': history['status'], 'history_coverage_complete': False,
                                        'selected_events': len(chosen), 'new_judge_authorized': False})
    remote = os.environ['EXPERIMENT_ROOT']
    old = os.environ['ADDITIONS_ROOT']
    outputs = {}
    for arm in ('CE', 'UNIFORM'):
        remote_arm = str(Path(remote) / arm)
        probe = """import json\nfrom pathlib import Path\nr=Path(%r)\np=r/'EVAL_GUARD_EXIT.json'\nprint(json.dumps({'failed':any((r/n).exists() for n in ('TRAIN_STOP.json','EVAL_STOP.json','TRAIN_GUARD_FAILURE.json','EVAL_GUARD_FAILURE.json')),'done':p.is_file(),'code':json.loads(p.read_text())['code'] if p.is_file() else None}))\n""" % remote_arm
        while True:
            if time.time() >= deadline - 180:
                raise TimeoutError('original deadline')
            state = ssh(probe)
            if state['failed'] or (state['done'] and state['code'] != 0):
                raise RuntimeError('GPU dependency failed')
            if state['done']:
                break
            write('DEPENDENCY_WAIT.json', {'arm': arm, 'time': time.time()})
            time.sleep(20)
        script = """import json,sys\nfrom pathlib import Path\nr=Path(%r);old=Path(%r);sys.path[:0]=[str(r/'tools'),str(Path(%r)/'tools')]\nimport packet\npacket.build(r,old,%r)\nprint((r/'SCORE_PACKET_PRIVATE.json').read_text())\n""" % (remote_arm, old, os.environ['FIXED_ROOT'], chosen)
        outputs[arm] = ssh(script, timeout=180)
        write('PACKET_' + arm + '.private.json', outputs[arm])
    legacy = {key for p in outputs.values() for v in p['payloads'].values() for key in v['legacy_keys']}
    semantic = {key for p in outputs.values() for key in p['payloads']}
    paths = [worktree / p for p in ('.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',
             '.private/med150_exact_switch_v1_20261028/DECISIONS.json',
             '.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',
             '.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json',
             '.private/med150_grace_v1_20261028/scoring/DECISIONS.json')]
    conflicts = set(history['conflicts'])
    for path in paths:
        if not path.is_file():
            continue
        for key, value in ledger.selected_decisions(path, legacy).items():
            if key in history['valid'] and any(history['valid'][key][f] != value[f] for f in score_reuse.FIELDS):
                conflicts.add(key)
            else:
                history['valid'][key] = value
    history['conflicts'] = list(conflicts)
    for suffix in ('01', '02', '03', '05', '06', '08'):
        path = Path('/private/tmp/.r.20261102.' + suffix + '/DECISIONS.json')
        if path.is_file():
            for key, value in ledger.selected_decisions(path, semantic).items():
                history['semantic_valid'].setdefault(key, []).append(value)
    decisions, consumers, scoring, blocked = {}, [], {}, set()
    med_keys = {}
    for arm, packet in outputs.items():
        part = score_reuse.resolve(packet, history)
        blocked.update(part['conflict_unknown'])
        decisions.update(part['known_valid'])
        scoring[arm] = {name: len(value) for name, value in part.items()}
        for row in packet['consumers']:
            row = dict(row)
            prior = d[(row['consumer_id'], 'MED')]
            if any(row[k] != prior[k] for k in ('consumer_id', 'query_id', 'owner', 'role', 'route')):
                raise ValueError('new and historical consumer identities differ')
            if row['arm'] == 'CANDIDATE':
                row['arm'] = arm
                consumers.append(row)
            elif row['arm'] == 'MED':
                if row['key'] != prior['key']:
                    raise ValueError('new and historical MED payload differs')
                if arm == 'CE':
                    med_keys[row['consumer_id']] = row['key']
                    consumers.append(row)
                elif med_keys.get(row['consumer_id']) != row['key']:
                    raise ValueError('candidate arms use different MED payloads')
            else:
                raise ValueError('unexpected score arm')
    keys = {d[(cid, 'MED_D')]['key'] for cid in chosen} | {du[(cid, 'MED_D')]['key'] for cid in chosen}
    keys |= {d[(cid, 'MED')]['key'] for cid in chosen}
    for suffix in ('06', '05', '08'):
        path = Path('/private/tmp/.r.20261102.' + suffix + '/DECISIONS.json')
        for key, value in ledger.selected_decisions(path, keys).items():
            if key in decisions and any(decisions[key][f] != value[f] for f in score_reuse.FIELDS):
                blocked.add(key)
            elif key not in blocked:
                decisions[key] = value
    for key in keys:
        rows = list(history['semantic_valid'].get(key, []))
        if key in decisions:
            rows.append(decisions[key])
        values = {tuple(row.get(f) for f in score_reuse.FIELDS) for row in rows}
        if len(values) > 1 or any(any(type(v) is not bool for v in verdict) for verdict in values):
            blocked.add(key)
    for key in blocked:
        decisions.pop(key, None)
    for cid in chosen:
        for arm, source in (('D', d), ('DU', du)):
            row = dict(source[(cid, 'MED_D')]); row['arm'] = arm; consumers.append(row)
    roles = json.loads((worktree / '.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
    result = summary.compare(consumers, decisions, roles)
    result['scoring'] = scoring
    result['new_judge_attempts'] = 0
    write('RESULTS.json', result)
    write('COMPLETE.json', {'status': result['status'], 'events': 392, 'new_judge_attempts': 0})


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('DRIVER_FAILURE.private.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        write('DRIVER_EXIT.json', {'code': code, 'time': time.time()})
