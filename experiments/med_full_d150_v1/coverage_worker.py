"""Produce a bounded, identity-first training-source and token-coverage lock."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
DEADLINE = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, value):
    path = ROOT / name
    with path.open('x') as f:
        os.chmod(path, 0o600)
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))


def main():
    if time.time() >= DEADLINE - 120:
        raise TimeoutError('Original experiment deadline')
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Preparation deadline')))
    signal.alarm(min(600, max(1, int(DEADLINE - time.time() - 120))))
    cfg = json.loads(Path(os.environ['RUNTIME_CONFIG']).read_text())
    fixed = Path(os.environ['FIXED_ROOT'])
    original = Path(os.environ['ORIGINAL_ROOT'])
    additions = Path(os.environ['ADDITIONS_ROOT'])
    replay = Path(cfg['REPLAY_PARENT']) / 'private'
    sys.path[:0] = [str(ROOT / 'tools'), str(fixed / 'tools')]
    import identity_lock
    import select_training
    fixed_admission = json.loads((fixed / 'ROLE_ADMISSION.json').read_text())
    roles = identity_lock.build(original / 'INPUTS_FROZEN.json',
        json.loads((original / 'DATA_ADMISSION.json').read_text()), fixed_admission,
        json.loads((additions / 'DATA_ADMISSION.json').read_text()),
        json.loads(Path(os.environ['DENY_GROUPS']).read_text()),
        json.loads((ROOT / 'V106_IDENTITIES.private.json').read_text()), replay)
    write('SOURCE_IDENTITY_LOCK.private.json', roles)
    entries = select_training.original_rows(original / 'INPUTS_FROZEN.json', roles)
    entries.extend(select_training.extra_rows(additions / 'ADDITIONAL_INPUTS.json'))
    owners = [entry['task']['order'] for entry in entries]
    if len(owners) != len(set(owners)) or owners != sorted(owners):
        raise ValueError('source identity ordering mismatch')
    if owners[-4:] != [147, 148, 149, 150]:
        raise ValueError('additional source identities missing')
    write('TRAIN_SOURCE_BOUND_PRIVATE.json', entries)
    write('ROLE_ADMISSION.json', {'status':'TRAINING_SOURCE_ROLE_ADMITTED','owners':owners,
        'identity_scope':'existing original training admission, protected-group exclusion and previously admitted VQA-RAD panel; unknown excluded',
        'unresolved_original_owners':len(roles['unknown_owners']), 'additions_receipt':'mechanical_source_and_role_pass_with_semantic_limit'})
    env = os.environ.copy()
    env.update(TASK_INPUT_ROOT=str(ROOT), TASK_PRIVATE_OUT=str(ROOT / 'SPAN_LOCK.private.json'),
        TASK_TOKENIZER=cfg['M3BENCH_MODEL_PATH'], TASK_OFFICIAL_SOURCE=cfg['M3BENCH_LLAVA_SOURCE'],
        TASK_DEADLINE=str(min(DEADLINE - 120, time.time() + 300)), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(ROOT / 'tools' / 'token_coverage.py')],
        env=env, check=True, timeout=min(310, max(1, int(DEADLINE - time.time() - 110))))
    lock = json.loads((ROOT / 'SPAN_LOCK.private.json').read_text())
    if lock['status'] != 'WEAK_TOKEN_PROXY_ONLY' or [x['order'] for x in lock['records']] != owners:
        raise ValueError('coverage owner mismatch')
    by_group = {}
    for name, test in (('original', lambda n: n <= 146), ('additional', lambda n: n >= 147)):
        values = [x for x in lock['records'] if test(x['order'])]
        by_group[name] = {'examined':len(values),'matched':sum(x['status']=='WEAK_TOKEN_MATCH' for x in values),
                          'unmatched':sum(x['status']!='WEAK_TOKEN_MATCH' for x in values)}
    write('COVERAGE_COMPLETE.json', {'status':'COVERAGE_COMPLETE_NOT_TRAINING_ADMISSION',
        'total_owners':150,'source_examined':len(entries),'source_unresolved':len(roles['unknown_owners']),
        'matched':sum(x['status']=='WEAK_TOKEN_MATCH' for x in lock['records']),
        'unmatched':sum(x['status']!='WEAK_TOKEN_MATCH' for x in lock['records']),
        'groups':by_group,'model_calls':0,'judge_calls':0,'eval_qa_decoded':0})


if __name__ == '__main__':
    code = 0
    try:
        write('COVERAGE_WORKER_ONCE', {'started':time.time()})
        main()
    except BaseException:
        code = 1
        write('COVERAGE_FAILURE.private.json', {'traceback':traceback.format_exc()})
        raise
    finally:
        write('COVERAGE_EXIT.json', {'code':code,'time':time.time()})
