"""Select the locked KL coefficient and launch the four-arm pilot on two GPUs."""
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import time


def receive(path, payload):
    with path.open('x') as out:
        json.dump(payload, out, allow_nan=False)


def choose(calibration, deadline):
    while not all((calibration / f'done{i}.json').is_file() for i in range(2)):
        if time.time() >= deadline:
            raise TimeoutError('KL calibration did not finish before its deadline')
        time.sleep(15)
    plan = json.loads((calibration / 'protocol.json').read_text())
    assert plan['phase'] in ('STAGE0_KL_CALIBRATION', 'STAGE0_KL_CALIBRATION_EXTENDED')
    assert plan['calibration_steps'] == 8
    rows = []
    for i in range(2):
        d = json.loads((calibration / f'done{i}.json').read_text())
        assert d['owners'] == 4 and d['disposable_updates'] == 384 and d['generation'] == d['judge'] == 0
        rows.extend(d['gradients'])
    ceiling = statistics.median(v for d in rows for v in d['mdu_kl'])
    assert math.isfinite(ceiling) and ceiling > 0
    candidates = {}
    chosen = None
    for multiplier in plan.get('factors', (0.3, 1., 3.)):
        beta = plan['beta0'] * multiplier
        values = {}
        for arm in ('P1', 'P2', 'P3', 'P4'):
            outcomes = [v for r in rows for v in r['candidates'][str(multiplier)][arm]]
            assert len(outcomes) == 8 and all(math.isfinite(x) and x >= -2e-6 for x in outcomes)
            values[arm] = statistics.median(outcomes)
        candidates[str(multiplier)] = dict(beta=beta, validation_median=values,
                                           passes=max(values.values()) <= ceiling)
        if chosen is None and candidates[str(multiplier)]['passes']:
            chosen = beta
    return dict(reference_median=ceiling, candidates=candidates, beta=chosen)


def limit():
    cap = 64 * 1024**2
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main(root):
    scheduled = json.loads((root / 'queue.json').read_text())
    calibration = root.parent / scheduled['calibration_run']
    result = choose(calibration, scheduled['calibration_deadline'])
    receive(root / 'beta_selection.json', result)
    if result['beta'] is None:
        print('CALIBRATION_NO_COMMON_VALUE', flush=True)
        return
    assert shutil.disk_usage(root).free > 9 * 1024**3
    capacity = subprocess.check_output(
        ['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    gpu = [(int(i), int(f)) for i, f in (line.split(',') for line in capacity.splitlines())]
    assert [i for i, _ in gpu] == [6, 7] and min(f for _, f in gpu) >= 43500
    started = time.time()
    deadline = started + 24 * 3600
    receive(root / 'protocol.json', dict(phase='STAGE1_MATCHED_PILOT', formal_pilot=True,
        physical_gpus=[6, 7], limit_bytes=[(f - 3500) * 1024**2 for _, f in gpu],
        start=started, deadline=deadline, approved_storage_bytes=1_000_000_000,
        teacher_cap_per_lane=2500, generation_cap_per_lane=1000, beta=result['beta'], updates=19200))
    pids = []
    workers = []
    for lane, (device, _) in enumerate(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), Q_LANE=str(lane))
        with (root / 'w').open() as source, (root / f'log{lane}').open('x') as output:
            worker = subprocess.Popen(
                ['timeout', '--signal=TERM', '--kill-after=30s', '84600s', 'python3', '-'],
                stdin=source, stdout=output, stderr=subprocess.STDOUT,
                cwd=root, env=env, start_new_session=True, preexec_fn=limit)
        pids.append(worker.pid)
        workers.append(worker)
    receive(root / 'start.json', dict(pids=pids, physical_gpus=[6, 7], start=started, deadline=deadline))
    joined = ','.join(map(str, pids))
    subprocess.run(['ps', '-ww', '-o', 'pid=,ppid=,comm=,args=', '-p', joined, '--ppid', joined], check=False)
    codes = [worker.wait() for worker in workers]
    receive(root / 'worker_exit.json', dict(codes=codes, end=time.time()))
    if any(codes):
        raise RuntimeError(f'Pilot GPU worker exit codes: {codes}')


if __name__ == '__main__':
    main(Path.cwd())
