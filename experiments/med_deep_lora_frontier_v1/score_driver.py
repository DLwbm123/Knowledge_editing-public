"""Freeze the available historical index, wait for predictions, and score once."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(os.environ['FIXED_SCORE_ROOT'])
WORKTREE = Path(os.environ['FIXED_WORKTREE'])
DEADLINE = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def main():
    sys.path[:0] = [str(WORKTREE/'experiments/med_deep_lora_frontier_v1'),
                    str(WORKTREE/'experiments/med_fixed_depth_v1'),
                    str(WORKTREE/'experiments/med_source_teacher_v1'),
                    str(WORKTREE/'experiments/med150_source_review_v1')]
    import ledger
    import score
    prior = WORKTREE/'.private/med146_single_sequential_v133_20261027/scoring_recovery1'
    fixed_scores = Path('/private/tmp/.q.fxdscore')
    extra = [WORKTREE/'.private/med150_alphaedit_v1_20261028/initial_scoring',
             WORKTREE/'.private/med150_alphaedit_v1_20261028/recovery_scoring',
             WORKTREE/'.private/med150_grace_v1_20261028/scoring',
             WORKTREE/'.private/med150_balancedit_v1_20261028/scoring',
             WORKTREE/'.private/med150_memit_extension_v1_20261028/scoring']
    manifests = [prior/'BATCH_MANIFEST.json', fixed_scores/'BATCH_MANIFEST.json'] + [p/'BATCH_MANIFEST.json' for p in extra]
    if not all(x.is_file() for x in manifests):
        raise ValueError('Previously located scoring evidence unavailable')
    result = ledger.build(ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json',
        WORKTREE/'.private/med_fixed_depth_v1_20261030/BASE_M0_SCORE_BINDINGS.json',
        prior/'MERGED_DECISIONS.json', fixed_scores/'PACKET_PRIVATE.json', fixed_scores/'DECISIONS.json', manifests)
    write('HISTORY_INDEX_FROZEN.json', result)
    remote_root = os.environ['REMOTE_ROOT']
    fixed_root = os.environ['FIXED_DEPTH_ROOT']
    source = os.environ['RAW_MED_ROOT']
    probe = '''from pathlib import Path
import json
r=Path(%r)
names=['WORKER_FAILURE.json','GUARD_FAILURE.json','STOP.json']
print(json.dumps(dict(failed=any((r/n).exists() for n in names),ready=(r/'WORKER_EXIT.json').is_file() and (r/'GUARD_EXIT.json').is_file())))
''' % remote_root
    while True:
        if time.time() >= DEADLINE-620 or (ROOT/'STOP.json').exists():
            raise TimeoutError('Original deadline or guard stop')
        state = score.remote(probe)
        if state['failed']:
            raise RuntimeError('GPU dependency failed; no scoring request sent')
        if state['ready']:
            break
        write('DEPENDENCY_WAIT.json', dict(status='WAIT_GPU', time=time.time()))
        time.sleep(30)
    build = '''from pathlib import Path
import json,sys
r=Path(%r);f=Path(%r);s=Path(%r)
sys.path[:0]=[str(r/'tools'),str(f/'tools')]
import packet,extract_reference
packet.build(r,f,s/'INPUTS_FROZEN.json',extract_reference)
print((r/'SCORE_PACKET_PRIVATE.json').read_text())
''' % (remote_root, fixed_root, source)
    packet = score.remote(build, timeout=180)
    if packet.get('status') != 'SCORE_PACKET_FROZEN' or len(packet['consumers']) != 2032:
        raise ValueError('Frozen packet incomplete')
    write('PACKET_PRIVATE.json', packet)
    (ROOT/'PACKET_PRIVATE.json').chmod(0o600)
    # Only now select historical decisions for the already frozen candidate keys.
    history = json.loads((ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json').read_text())
    allowed = {k for item in packet['payloads'].values() for k in item['legacy_keys']}
    conflicts = set(history['conflicts'])
    for folder in extra:
        for key, row in ledger.selected_decisions(folder/'DECISIONS.json', allowed).items():
            if key in history['valid'] and any(history['valid'][key][f] != row[f] for f in ledger.partition.FIELDS):
                conflicts.add(key)
            else:
                history['valid'][key] = row
    history['conflicts'] = sorted(conflicts)
    with (ROOT/'HISTORY_INDEX_PRIVATE.json').open('x') as f:
        json.dump(history, f, ensure_ascii=False)
    (ROOT/'HISTORY_INDEX_PRIVATE.json').chmod(0o600)
    score.main()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('DRIVER_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
    finally:
        write('DRIVER_EXIT.json', dict(time=time.time(), code=code))
