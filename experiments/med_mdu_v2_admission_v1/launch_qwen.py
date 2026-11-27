"""Queue the second-backbone mechanism experiment after the first pilot releases its GPUs."""
import json
import os
from pathlib import Path
import subprocess
import time

from launch_pilot import limit, receive


def wait_for_pilot(root, queue):
    pilot = root.parent / queue['pilot']
    while time.time() < queue['queue_deadline']:
        selection = pilot / 'beta_selection.json'
        if (pilot / 'worker_exit.json').is_file():
            return
        if selection.is_file() and json.loads(selection.read_text())['beta'] is None:
            return
        time.sleep(30)
    raise TimeoutError('The first pilot did not release the devices within its recorded window')


def main(root):
    queue = json.loads((root / 'queue.json').read_text())
    wait_for_pilot(root, queue)
    n = len(json.loads((root / 'panel.json').read_text()))
    assert n == 27
    raw = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = [(int(k), int(v)) for k, v in (row.split(',') for row in raw.splitlines())]
    assert [i for i, _ in devices] == [6, 7]
    assert all(f >= queue['minimum_free_mib'] for _, f in devices)
    start = time.time()
    deadline = start + 5 * 3600
    receive(root / 'protocol.json', dict(preflight=False, start=start, deadline=deadline,
        limit_bytes=[min(f - 6000, 32000) * 1024**2 for _, f in devices],
        experts=n, planned_updates=n * 160))
    workers = []
    for lane, (gpu, _) in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), Q_LANE=str(lane))
        with (root / 'w').open() as code, (root / f'log{lane}').open('x') as log:
            workers.append(subprocess.Popen(
                ['timeout', '--signal=TERM', '--kill-after=30s', '17200s', 'python3', '-'],
                stdin=code, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=root,
                start_new_session=True, preexec_fn=limit))
    pids = [w.pid for w in workers]
    receive(root / 'start.json', dict(pids=pids, gpu=[6, 7], start=start, deadline=deadline))
    ids = ','.join(map(str, pids))
    subprocess.run(['ps', '-ww', '-o', 'pid=,ppid=,comm=,args=', '-p', ids, '--ppid', ids], check=False)
    codes = [w.wait() for w in workers]
    receive(root / 'worker_exit.json', dict(codes=codes, end=time.time()))
    if any(codes):
        raise RuntimeError(f'Second backbone exit codes: {codes}')


if __name__ == '__main__':
    main(Path.cwd())
