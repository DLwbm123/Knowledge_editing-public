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
    sys.path[:0] = [str(WORKTREE/'experiments/med_full_d_v1'),
                    str(WORKTREE/'experiments/med_proposition_generalization_v1'),
                    str(WORKTREE/'experiments/med_fixed_depth_v1'),
                    str(WORKTREE/'experiments/med_source_teacher_v1'),
                    str(WORKTREE/'experiments/med150_source_review_v1')]
    import ledger
    import nine_score as score
    prior = WORKTREE/'.private/med146_single_sequential_v133_20261027/scoring_recovery1'
    fixed_scores = Path(os.environ['PRIOR_FIXED_SCORE_ROOT'])
    extra = [WORKTREE/'.private/med150_alphaedit_v1_20261028/initial_scoring',
             WORKTREE/'.private/med150_alphaedit_v1_20261028/recovery_scoring',
             WORKTREE/'.private/med150_grace_v1_20261028/scoring',
             WORKTREE/'.private/med150_balancedit_v1_20261028/scoring',
             WORKTREE/'.private/med150_memit_extension_v1_20261028/scoring']
    manifests = [prior/'BATCH_MANIFEST.json', fixed_scores/'BATCH_MANIFEST.json'] + [p/'BATCH_MANIFEST.json' for p in extra]
    if not all(x.is_file() for x in manifests):
        raise ValueError('Previously located scoring evidence unavailable')
    # Reuse an existing identity-only index; it deliberately remains incomplete.
    import shutil
    previous = Path(os.environ['PRIOR_LORA_SCORE_ROOT'])
    initial = previous/'HISTORY_INDEX_PRIVATE.json'
    if not initial.is_file():raise ValueError('Prior identity index unavailable')
    shutil.copyfile(initial,ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json')
    (ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json').chmod(0o600)
    index=json.loads((ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json').read_text())
    if index.get('status')!='AVAILABLE_HISTORY_INDEX_FROZEN':raise ValueError('Invalid history index')
    write('HISTORY_INDEX_FROZEN.json',dict(status=index['status'],history_coverage_complete=False,manifests_indexed=index['manifests_indexed']))
    remote_root = os.environ['REMOTE_ROOT']
    fixed_root = os.environ['FIXED_DEPTH_ROOT']
    source = os.environ['RAW_MED_ROOT']
    span_lock = os.environ['SPAN_LOCK']
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
r=Path(%r);f=Path(%r);s=Path(%r);l=Path(%r)
sys.path[:0]=[str(r/'tools'),str(f/'tools')]
import packet
packet.build(r,f,s/'INPUTS_FROZEN.json',l)
print((r/'SCORE_PACKET_PRIVATE.json').read_text())
''' % (remote_root, fixed_root, source, span_lock)
    packet = score.remote(build, timeout=180)
    if packet.get('status') != 'SCORE_PACKET_FROZEN' or len(packet['consumers']) != 371:
        raise ValueError('Frozen packet incomplete')
    write('PACKET_PRIVATE.json', packet)
    (ROOT/'PACKET_PRIVATE.json').chmod(0o600)
    # Only now select historical decisions for the already frozen candidate keys.
    history = json.loads((ROOT/'HISTORY_INDEX_INITIAL_PRIVATE.json').read_text())
    allowed = {k for item in packet['payloads'].values() for k in item['legacy_keys']}
    conflicts = set(history['conflicts'])
    # The previous LoRA run occurred after the reused index was built. Select only
    # scorer identities needed by these already-frozen candidate predictions.
    candidate_keys=set(packet['payloads'])
    for key,row in ledger.selected_decisions(previous/'DECISIONS.json',candidate_keys).items():
        history['semantic_valid'].setdefault(key,[]).append(row)
    import history_index
    batches=history_index.manifest(previous/'BATCH_MANIFEST.json')
    history['semantic_attempts']=sorted(set(history['semantic_attempts']) | set(history_index.attempted(previous,batches)))
    for folder in extra:
        for key, row in ledger.selected_decisions(folder/'DECISIONS.json', allowed).items():
            if key in history['valid'] and any(history['valid'][key][f] != row[f] for f in ledger.partition.FIELDS):
                conflicts.add(key)
            else:
                history['valid'][key] = row
    nine = Path(os.environ['PRIOR_NINE_SCORE_ROOT'])
    if not (nine/'DECISIONS.json').is_file() or not (nine/'BATCH_MANIFEST.json').is_file():
        raise ValueError('Nine-arm historical scoring evidence unavailable')
    for key,row in ledger.selected_decisions(nine/'DECISIONS.json',candidate_keys).items():
        history['semantic_valid'].setdefault(key,[]).append(row)
    nine_batches=history_index.manifest(nine/'BATCH_MANIFEST.json')
    history['semantic_attempts']=sorted(set(history['semantic_attempts']) | set(history_index.attempted(nine,nine_batches)))
    history['conflicts'] = sorted(conflicts)
    with (ROOT/'HISTORY_INDEX_PRIVATE.json').open('x') as f:
        json.dump(history, f, ensure_ascii=False)
    (ROOT/'HISTORY_INDEX_PRIVATE.json').chmod(0o600)
    import partition
    parts=partition.partition(packet,history['valid'],history['attempts'],history['semantic_valid'],history['semantic_attempts'],history['conflicts'])
    unknown=parts['known_unknown']+parts['conflict_unknown']+parts['history_unverified']
    write('DECISIONS.json',dict(decisions=parts['known_valid'],unknown=unknown))
    write('COMPLETE.json',dict(status='COMPLETE_WITH_POSSIBLE_UNKNOWN',unique_payloads=len(packet['payloads']),reused=len(parts['known_valid']),unknown=len(unknown),new_attempts=0,history_coverage_complete=False))


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
