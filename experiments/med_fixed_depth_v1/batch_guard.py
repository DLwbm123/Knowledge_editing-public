"""Launch four fixed shards only after first-pair engineering evaluation passes."""
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
DEADLINE = 1793447820


def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def allocated(root):
    total = 0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_blocks * 512
        except FileNotFoundError:
            pass
    return total


def main():
    with (ROOT / 'BATCH_GUARD_ONCE').open('x'):
        pass
    while True:
        if time.time() >= DEADLINE - 660:
            raise TimeoutError('Original deadline reached while waiting')
        if any((ROOT / n).exists() for n in ('WORKER_FAILURE.json', 'GUARD_FAILURE.json',
                                              'FIRST_EVAL_FAILURE.json', 'FIRST_EVAL_GUARD_FAILURE.json')):
            raise RuntimeError('First-pair dependency failed')
        required = ('FIRST_PAIR_EVAL_COMPLETE.json', 'FIRST_EVAL_EXIT.json', 'FIRST_EVAL_GUARD_EXIT.json')
        if all((ROOT / n).is_file() for n in required):
            if any(json.loads((ROOT / n).read_text())['code'] != 0 for n in ('FIRST_EVAL_EXIT.json', 'FIRST_EVAL_GUARD_EXIT.json')):
                raise RuntimeError('First evaluation did not exit successfully')
            if json.loads((ROOT / 'FIRST_PAIR_EVAL_COMPLETE.json').read_text())['engineering_status'] != 'PASS':
                raise RuntimeError('First evaluation engineering checks failed')
            break
        time.sleep(10)
    neutral = Path(os.environ['NEUTRAL_ENTRY'])
    gpus = []
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        if int(free) >= 45056:
            gpus.append((index, uuid))
    if len(gpus) < 4:
        raise RuntimeError('Fewer than four GPUs have the required free memory')
    gpus = gpus[:4]
    write('BATCH_GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=allocated(ROOT)))
    processes = []
    def file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    try:
        for part, (gpu, uuid) in enumerate(gpus):
            env = dict(os.environ, ENTRY_MODE='batch_worker', PARTITION=str(part), PARTITIONS='4',
                       CUDA_VISIBLE_DEVICES=gpu, M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
                       M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu,
                       M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu)
            with (ROOT / f'batch_{part}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, str(neutral)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, preexec_fn=file_limit)
            processes.append(child)
        write('BATCH_IDENTITIES.json', dict(time=time.time(), pids=[p.pid for p in processes], gpus=[g[0] for g in gpus]))
        while any(p.poll() is None for p in processes):
            used = allocated(ROOT) + allocated(neutral.parent)
            st = os.statvfs(ROOT)
            free = st.f_bavail * st.f_frsize
            write('BATCH_GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=used, free=free))
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A batch shard failed; preserve partial work')
            if used >= 30_000_000_000 or free <= 8 * 1024**3 or time.time() >= DEADLINE - 45:
                raise RuntimeError('Storage reserve or original deadline reached')
            time.sleep(5)
    except BaseException:
        write('BATCH_STOP.json', dict(time=time.time(), error=traceback.format_exc()))
        for child in processes:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in processes:
            if child.poll() is None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    codes = [p.wait() for p in processes]
    write('BATCH_GUARD_EXIT.json', dict(time=time.time(), codes=codes, hard_end=DEADLINE))
    raise SystemExit(0 if codes == [0, 0, 0, 0] else 1)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('BATCH_GUARD_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
