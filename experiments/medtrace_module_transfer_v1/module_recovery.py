"""Adopt the live partition; admit only partitions that failed before any model work."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def live(start):
    path = Path('/proc') / str(start['pid']) / 'stat'
    if not path.exists():
        return False
    fields = path.read_text().split()
    return fields[21] == start['start_ticks'] and fields[2] != 'Z'


def zero_work(failure):
    return (failure['error'] == "AssertionError('45 GiB admission reserve unavailable')"
            and bool(failure['counts']) and all(value == 0 for value in failure['counts'].values()))


def expected_parent_failure(failure):
    return (failure['error'] == "RuntimeError('Own finite stage failed; do not automatically retry')"
            and bool(failure['counts']) and all(value == 0 for value in failure['counts'].values()))


def selfcheck():
    failure = dict(error="AssertionError('45 GiB admission reserve unavailable')", counts=dict(updates=0, forwards=0))
    assert zero_work(failure)
    assert not zero_work(dict(failure, counts=dict(updates=1, forwards=0)))
    assert not zero_work(dict(failure, error='numerical failure'))
    assert not zero_work(dict(failure, counts={}))
    assert not live(dict(pid=999999999, start_ticks='0'))
    return dict(status='PASS', rejects_partial_work=True, rejects_other_failures=True)


def free_gpus(busy=()):
    import module_run as r
    allowed = r.c.read(r.RUN / 'RUN_MANIFEST.json')['physical_GPUs']
    expected = r.c.read(r.RUN / 'PLAN_CONFIG.json')['hardware']['UUIDs']
    active = {row['gpu_uuid'] for row in r.c.read(r.RUN / 'RESOURCE_LEDGER.json')['gpu_sessions'] if not row.get('ended_epoch')}
    available = []
    for gpu in [7, 2, 4, 3]:
        if gpu not in allowed or gpu in busy or expected[str(gpu)] in active:
            continue
        row = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert row[0] == expected[str(gpu)]
        if int(row[1]) >= 45 * 1024:
            available.append(gpu)
    return available


def admitted_stage(file, action, parts):
    import module_run as r
    import pipeline
    waiting, jobs = set(parts), {}
    while waiting or any(job.poll() is None for job, _ in jobs.values()):
        r.c.budget()
        assert all(job.poll() is None or job.returncode == 0 for job, _ in jobs.values())
        busy = {gpu for job, gpu in jobs.values() if job.poll() is None}
        for part, gpu in zip(sorted(waiting), free_gpus(busy)):
            jobs[part] = (pipeline.launch(file, action, gpu, part), gpu)
            waiting.remove(part)
        if waiting:
            r.c.write(r.RUN / 'public/RESOURCE_WAIT.json', dict(action=action, waiting_parts=sorted(waiting),
                required_free_MiB=45 * 1024, epoch=time.time(), GPU_lease_started_for_wait=False))
        time.sleep(30)
    assert all(job.returncode == 0 for job, _ in jobs.values())


def main():
    import module_run as r
    import pipeline
    run, c = r.RUN, r.c
    archive = run / 'private/engineering_admission_recovery1'
    archive.mkdir(exist_ok=False)
    assert c.read(run / 'private/REFERENCE_EXACT.json')['status'] == 'PASS'
    assert c.read(run / 'public/FISHER_ADMISSION.json')['status'] == 'PASS'
    controller = c.read(run / 'START.json')
    adopted = c.read(run / 'private/START_CHAIN_module_train_1.json')
    assert live(controller) and live(adopted)
    waiting, jobs = {0, 2}, {}
    for part in waiting:
        failure = run / 'private' / f'FAILURE_module_train_{part}.json'
        assert zero_work(c.read(failure)), 'Only zero-work admission failures are recoverable here'
        assert not live(c.read(run / 'private' / f'START_CHAIN_module_train_{part}.json'))
        for task in r.s.tasks()[part::len(r.GPUS)]:
            for arm in r.mm.ARMS[1:]:
                assert not r.s.point(task, arm).exists()
                assert not (run / 'private/active' / f"{arm}_{task['order']}.pt").exists()
                assert not (run / 'private/progress' / arm / f"{task['order']}.json").exists()
        for name in [failure.name, f'START_CHAIN_module_train_{part}.json']:
            source = run / 'private' / name
            source.rename(archive / name)
        counts = run / 'private' / f'COUNTS_module_train_{part}.json'
        if counts.exists():
            assert all(value == 0 for value in c.read(counts).values())
            counts.rename(archive / counts.name)
    c.write(archive / 'ADMISSION.json', dict(status='ZERO_MODEL_WORK_VERIFIED', parts=[0, 2],
        adopted_partition=1, adopted_start=adopted, original_controller=controller,
        selfcheck=selfcheck(), epoch=time.time(), scientific_protocol_changed=False))
    while waiting or any(job.poll() is None for job, _ in jobs.values()) or live(adopted):
        c.budget()
        assert not list((run / 'private').glob('FAILURE_module_train*.json'))
        assert all(job.poll() is None or job.returncode == 0 for job, _ in jobs.values())
        if not live(adopted):
            assert all(r.s.point(task, arm).exists() for task in r.s.tasks()[1::3] for arm in r.mm.ARMS[1:]), 'Adopted training stopped unfinished'
        busy = {gpu for job, gpu in jobs.values() if job.poll() is None}
        for part, gpu in zip(sorted(waiting), free_gpus(busy)):
            jobs[part] = (pipeline.launch('module_run.py', 'module_train', gpu, part), gpu)
            waiting.remove(part)
        c.write(run / 'public/PROGRESS.json', dict(status='RESOURCE_RECOVERY_TRAINING', complete=False,
            waiting_parts=sorted(waiting), admitted_GPUs={str(part): gpu for part, (_, gpu) in jobs.items()},
            adopted_partition=1, epoch=time.time()))
        time.sleep(30)
    assert all(job.returncode == 0 for job, _ in jobs.values())
    assert all(r.s.point(task, arm).exists() for task in r.s.tasks() for arm in r.mm.ARMS)
    # The original parent is waiting for its healthy child, then reports the preserved admission failures.
    while live(controller):
        c.budget()
        time.sleep(5)
    failure = run / 'private/FAILURE_module_controller_None.json'
    value = c.read(failure)
    assert expected_parent_failure(value)
    failure.rename(archive / failure.name)
    exit_path = run / 'EXIT.json'
    if exit_path.exists():
        exit_path.rename(archive / 'ORIGINAL_EXIT.json')
    c.write(archive / 'TRAINING_COMPLETE.json', dict(epoch=time.time(), candidates=48,
        restarted_training_trajectories=0, strict_state_resumes=0))
    r.finish()


if __name__ == '__main__':
    if os.environ.get('RECOVERY_SELFCHECK') == '1':
        print(json.dumps(selfcheck()))
    else:
        try:
            main()
        except BaseException as error:
            import module_run as r
            r.c.write(r.RUN / 'private/FAILURE_module_recovery.json', dict(error=repr(error),
                traceback=traceback.format_exc(), epoch=time.time(), retry=False))
            r.c.write(r.RUN / 'public/PROGRESS.json', dict(status='RECOVERY_STOPPED_RETAINED', complete=False))
            raise
