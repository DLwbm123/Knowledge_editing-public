"""Finish the unstarted C/M13 suffix while the original Fisher process stays stopped."""
import json
import os
import sys
import inspect
from pathlib import Path
import time
import traceback


def read(path):
    return json.loads(path.read_text())


def complete(run, arm, owner):
    result = read(run / f'private/results/{arm}/{owner}.json')
    assert result['status'] == 'COMPLETE' and result['save_load_exact']
    assert [row['step'] for row in result['curve']] == list(range(1, 161))
    assert (run / f'private/candidates/{arm}/{owner}.pt').is_file()


def plan(run, start=43):
    assert 43 <= start <= 146
    for arm in ('C', 'M13'):
        assert {int(p.stem) for p in (run / f'private/results/{arm}').glob('*.json')} == set(range(1, start))
        for owner in range(1, start):
            complete(run, arm, owner)
    todo = []
    for owner in range(start, 147):
        for arm, marker in [('C', 'CALIBRATION'), ('M13', 'NATIVE_M13')]:
            for path in [f'private/results/{arm}/{owner}.json',
                         f'private/candidates/{arm}/{owner}.pt',
                         f'private/resume146/{arm}_{owner}.pt',
                         f'private/{marker}_ONCE_{owner}']:
                assert not (run / path).exists(), f'Existing work must not be retrained: {path}'
            todo.append((owner, arm))
    return todo


def stopped(pid, ticks, argv):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().split()
    assert fields[2] == 'T' and int(fields[21]) == ticks
    assert (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip() == argv


def main():
    import runtime146 as worker
    import selected_training146 as training
    # Legacy imports prepend their own directories; restore the frozen lane first.
    source = Path(training.__file__).resolve().parent
    sys.path.insert(0, str(source))
    import native_calibration
    import native_followup
    for module, fn in ((native_calibration, native_calibration.main), (native_followup, native_followup.train)):
        assert Path(module.__file__).resolve().parent == source
        inspect.signature(fn).bind(full=True, loaded=(None, None))
    r, c, run = worker.r, worker.c, worker.RUN
    gate = c.read(run / 'private/M13_ONLY_ADMISSION3.json')
    start = gate['owners'][0]
    remaining = 147 - start
    assert gate['status'] == 'PASS' and gate['owners'] == list(range(start, 147))
    validation = read(Path(gate['prefix_validation']) / 'RESULTS.json')
    assert read(Path(gate['prefix_validation']) / 'EXIT.json')['exit_code'] == 0
    assert validation['status'] == 'PASS' and validation['deployment_allowed']
    assert len(validation['exact']) == 8 and all(row['all_exact'] for row in validation['exact'])
    previous = Path('/proc') / str(gate['previous_worker_pid'])
    assert not previous.exists(), 'Previous C/M13 worker must exit before handoff'
    assert read(run / 'private/M13_ONLY_EXIT2.json')['exit_code'] == 1
    barrier = run / f'private/CALIBRATION_ONCE_{start}'
    assert not barrier.exists() and read(run / 'private/PREFIX_HANDOFF3.json')['first_unstarted'] == start
    assert gate['GPU'] == 1 and os.environ['GPU'] == '1'
    assert os.environ['ACTION'] == 'm13_only_continue3' and os.environ['PARTITION'] == '1'
    assert gate['scientific_source'] == c.read(run / 'private/SELECTED_PREFILL_RECOVERY_SOURCE1.json')['commit']
    assert gate['scheduler_source'] == c.read(run / 'private/M13_ONLY_SOURCE3.json')['commit']
    training.require_admission()
    todo = plan(run, start)
    assert len(gate['paused_processes']) == 2
    for process in gate['paused_processes']:
        stopped(**process)
    old = Path('/proc') / str(gate['finished_worker_pid'])
    assert not old.exists() or (old / 'stat').read_text().split()[2] == 'Z'
    assert all(value == 0 for value in r.COUNTS.values()), 'New namespace must not inherit counters'
    with (run / 'private/M13_ONLY_ONCE3').open('x'):
        pass
    original_budget = c.budget

    def budget():
        original_budget()
        ledger = c.read(run / 'RESOURCE_LEDGER.json')
        used = ledger['gpu_seconds_used'] + sum(time.time() - s['started_epoch']
            for s in ledger['gpu_sessions'] if not s.get('ended_epoch'))
        assert used + 1488.5244522094727 < c.read(run / 'RUN_MANIFEST.json')['GPU_seconds_limit']
        assert r.COUNTS['updates'] <= remaining * 320 and r.COUNTS['backwards'] <= remaining * 648
        assert r.COUNTS['operator_backwards'] <= remaining * 160
        assert r.COUNTS['generation_attempts'] == r.COUNTS.get('local_solves', 0) == 0

    c.budget = budget
    from prefix_cache import PrefixCache
    original_trajectory = training.trajectory.run
    def cached_trajectory(*args, **kwargs):
        with PrefixCache(args[1], args[4]) as cache:
            result = original_trajectory(*args, **kwargs)
            assert cache.misses == 5 and cache.hits == 315
            c.write(run / 'private/prefix_cache_receipts' / f'{args[5]}_{os.environ["OWNER"]}.json',
                dict(hits=cache.hits, misses=cache.misses, scheduler=gate['scheduler_source'],
                     validation=gate['prefix_validation'], epoch=time.time()))
            return result
    training.trajectory.run = cached_trajectory
    worker.configure()
    r.NATIVE_FORWARD_CAP = training.caps()['non_generation_forwards']
    with r.p.lease(1):
        runtime, bindings, handle = r.load(1)
        handle.remove()
        handle = runtime.model.register_forward_pre_hook(lambda *args: r.count(*args))
        try:
            # Every suffix owner originally inherited math attention from the previous trajectory.
            r.old.fp32(runtime)
            for owner, arm in todo:
                for process in gate['paused_processes']:
                    stopped(**process)
                budget(); training.check_counts()
                os.environ['OWNER'] = str(owner)
                c.write(run / 'private/M13_ONLY_PROGRESS3.json', dict(owner=owner, arm=arm, epoch=time.time()))
                fn = native_calibration.main if arm == 'C' else native_followup.train
                fn(full=True, loaded=(runtime, bindings))
                complete(run, arm, owner)
                r.save_counts(); training.check_counts()
        finally:
            handle.remove(); r.save_counts()
    assert r.COUNTS['updates'] == remaining * 320 and r.COUNTS['backwards'] == remaining * 648
    assert r.COUNTS['operator_backwards'] == remaining * 160
    for owner in range(1, 147):
        for arm in ('C', 'M13'):
            complete(run, arm, owner)
    c.write(run / 'private/M13_ONLY_COMPLETE3.json', dict(status='TRAINING_COMPLETE',
        C=146, M13=146, Fisher_paused=True, full146_complete=False,
        generation=0, Judge=0, counts=r.COUNTS, epoch=time.time()))


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)
        for arm in ('C', 'M13'):
            for owner in range(1, 43):
                p = run / f'private/results/{arm}/{owner}.json'; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(dict(status='COMPLETE', save_load_exact=True,
                    curve=[dict(step=i) for i in range(1, 161)])))
                p = run / f'private/candidates/{arm}/{owner}.pt'; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
        assert plan(run) == [(owner, arm) for owner in range(43, 147) for arm in ('C', 'M13')]
        for path in ['private/resume146/M13_43.pt', 'private/CALIBRATION_ONCE_43']:
            p = run / path; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
            try:
                plan(run)
            except AssertionError:
                pass
            else:
                raise AssertionError('Partial work incorrectly admitted')
            p.unlink()
        (run / 'private/candidates/M13/42.pt').unlink()
        try:
            plan(run)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing retained candidate incorrectly admitted')
    print('PASS: 42 reused, 104 C/M13 suffix only; partial work and missing candidate rejected; no GPU')


if __name__ == '__main__':
    import sys
    if '--selfcheck' in sys.argv:
        selfcheck()
    else:
        code = 0
        try:
            main()
        except BaseException as error:
            code = 1
            import runtime146 as worker
            worker.r.save_counts()
            worker.c.write(worker.RUN / 'private/M13_ONLY_FAILURE3.json', dict(
                error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, epoch=time.time()))
            raise
        finally:
            import runtime146 as worker
            worker.c.write(worker.RUN / 'private/M13_ONLY_EXIT3.json', dict(exit_code=code, epoch=time.time()))
