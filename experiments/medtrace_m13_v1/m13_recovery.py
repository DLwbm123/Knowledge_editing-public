"""One takeover of three import failures with zero model/optimizer work."""
import importlib.util
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import m13 as run
from module_recovery import live


def zero_import_failure(value):
    return (value['error'] == 'ModuleNotFoundError("No module named \'llava\'")'
            and bool(value['counts']) and all(v == 0 for v in value['counts'].values()))


def check():
    r, c, root = run.r, run.c, run.RUN
    value = dict(error='ModuleNotFoundError("No module named \'llava\'")', counts=dict(updates=0))
    assert zero_import_failure(value)
    assert not zero_import_failure(dict(value, counts=dict(updates=1)))
    assert not zero_import_failure(dict(value, error='numerical failure'))
    assert not zero_import_failure(dict(value, counts={}))
    from m3bench_repro.editors import llava_runtime
    source = root / 'private/official_llava'
    assert llava_runtime.LLAVA_SOURCE.resolve() == source.resolve()
    sys.path.insert(0, str(source))
    assert Path(importlib.util.find_spec('llava').origin).resolve().is_relative_to(source.resolve())
    assert c.read(root / 'EXIT.json')['exit_code'] == 1 and not live(c.read(root / 'START.json'))
    assert c.read(root / 'public/FISHER_ADMISSION.json')['status'] == 'PASS'
    assert (root / 'private/OPERATOR_FISHER.pt').is_file()
    ledger = c.read(root / 'RESOURCE_LEDGER.json')
    assert len(ledger['gpu_sessions']) == 4 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    for part in range(3):
        assert not live(c.read(root / 'private' / f'START_CHAIN_module_train_{part}.json'))
        assert zero_import_failure(c.read(root / 'private' / f'FAILURE_m13_module_train_{part}.json'))
    parent = c.read(root / 'private/FAILURE_m13_m13_controller_0.json')
    assert parent['error'] == 'AssertionError()' and all(v == 0 for v in parent['counts'].values())
    for folder in ('candidates/M13', 'results/M13', 'progress/M13', 'active', 'calibration', 'outputs'):
        assert not any(p.is_file() for p in (root / 'private' / folder).rglob('*'))
    assert not (root / 'private/CANDIDATES_FROZEN.json').exists()
    for path in (root / 'private').glob('COUNTS_module_train_*.json'):
        assert all(v == 0 for v in c.read(path).values())
    assert ledger['Judge_attempts'] == c.read(root / 'private/INHERITED_COST.json')['Judge_attempts']
    with sqlite3.connect('file:' + str(root / 'private/judge_m13_astra_medium/queue.sqlite') + '?mode=ro', uri=True) as db:
        assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM consumer').fetchone()[0] == 0
    return dict(status='PASS', zero_model_work=True, interrupted_trajectories=0,
                official_runtime_import=True, completed_Fisher_reused=True, new_Judge=0,
                extra_failed_GPU_leases=3, epoch=time.time())


def main():
    evidence = check()
    c, root = run.c, run.RUN
    archive = root / 'private/import_recovery1'
    archive.mkdir(exist_ok=False)
    c.write(archive / 'ADMISSION.json', evidence)
    paths = [root / 'START.json', root / 'EXIT.json']
    paths += list((root / 'private').glob('FAILURE_m13_*.json'))
    paths += list((root / 'private').glob('START_CHAIN_module_train_*.json'))
    paths += list((root / 'private').glob('COUNTS_module_train_*.json'))
    for path in paths:
        path.rename(archive / path.name)
    c.write(root / 'START.json', c.read(root / 'private/ENGINEERING_IMPORT_RECOVERY_START.json'))
    for path in (root / 'logs').glob('*.log'):
        shutil.copy2(path, archive / path.name)
    run.admission()
    c.write(root / 'public/PROGRESS.json', dict(status='M13_IMPORT_RECOVERY_TRAINING', complete=False))
    run.admitted_stage('m13.py', 'module_train', range(3))
    run.finish()


if __name__ == '__main__':
    if os.environ.get('M13_RECOVERY_PREFLIGHT') == '1':
        print(check())
    else:
        main()
