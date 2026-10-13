"""One admission-only takeover of the preserved zero-work assembly failure."""
import os
from pathlib import Path
import shutil
import time
import traceback
import guard_probe as g
from guard_admission import require_reference
from module_recovery import live


def zero_counts(counts):
    return set(counts) == set(g.r.COUNTS) and all(v == 0 for v in counts.values())


def selfcheck():
    assert zero_counts(dict(g.r.COUNTS))
    assert not zero_counts(dict(g.r.COUNTS, updates=1))
    assert not zero_counts({})
    return dict(status='PASS', rejects_partial_work=True, rejects_missing_counts=True)


def main():
    c, run = g.c, g.RUN
    require_reference()
    assert not live(c.read(run/'START.json'))
    assert not live(c.read(run/'private/START_CHAIN_module_fisher_0.json'))
    for path in Path('/proc').glob('[0-9]*/environ'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            environment = path.read_bytes().split(b'\0')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        assert ('RUN_ROOT='+str(run)).encode() not in environment, 'Another active chain owns this run'
    assert c.read(run/'EXIT.json')['exit_code'] == 1
    assert (run/'private/GUARD_START.json').exists()
    failures = list((run/'private').glob('FAILURE_guard_*.json'))
    assert {p.name for p in failures} == {'FAILURE_guard_module_fisher_0.json', 'FAILURE_guard_guard_controller_0.json'}
    for path in failures:
        failure = c.read(path)
        assert zero_counts(failure['counts'])
        assert failure['error'] == 'AssertionError()'
        if 'module_fisher' in path.name:
            assert "assert (RUN / 'private/REFERENCE_EXACT.json').exists()" in failure['traceback']
    assert all(zero_counts(c.read(p)) for p in (run/'private').glob('COUNTS_*.json'))
    ledger = c.read(run/'RESOURCE_LEDGER.json')
    inherited = c.read(run/'private/INHERITED_COST.json')
    assert ledger['gpu_sessions'] == [] and ledger['weights_observed_peak_bytes'] == 0
    assert ledger['gpu_seconds_used'] == inherited['gpu_seconds_used']
    assert ledger['Judge_attempts'] == inherited['Judge_attempts']
    assert not list((run/'private').glob('**/*.pt'))
    # C symlinks are inherited assets, not this run's generated model work.
    assert not list((run/'private/nodes').glob('*')) and not list((run/'private/diagnostics').glob('*'))
    assert not list((run/'private/PREFIX_COMPLETE').glob('*'))
    assert not list((run/'private/progress').glob('**/*.json'))
    archive = run/'private/zero_work_takeover1'
    archive.mkdir(exist_ok=False)
    paths = [run/'START.json', run/'EXIT.json', run/'private/GUARD_START.json'] + failures
    paths += list((run/'private').glob('START_CHAIN_*.json')) + list((run/'private').glob('COUNTS_*.json'))
    paths += list((run/'logs').glob('*.log'))
    for i, path in enumerate(paths):
        shutil.copy2(path, archive/f'{i}_{path.name}')
    c.write(archive/'ADMISSION.json', dict(status='STRICT_ZERO_WORK_VERIFIED', selfcheck=selfcheck(),
        epoch=time.time(), original_lock_preserved=True, historical_cost_preserved=True, C_retrained=False))
    marker = run/'private/GUARD_TAKEOVER_ONCE.json'
    with marker.open('x') as handle:
        import json
        json.dump(dict(epoch=time.time(), pid=os.getpid()), handle)
    g.run_stages()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        g.c.write(g.RUN/'private/FAILURE_guard_takeover1.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        g.c.write(g.RUN/'public/PROGRESS.json',dict(status='TAKEOVER_STOPPED_RETAINED',complete=False))
        raise
