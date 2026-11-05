"""Independently stop the one-GPU diagnostic at its original resource limits."""
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

root = Path(os.environ['DIAG_ROOT'])
neutral = Path(os.environ['NEUTRAL_ENTRY'])
end = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, obj):
    p = root/name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(obj))
    t.replace(p)


def allocated(path):
    total = 0
    for f in path.rglob('*'):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_blocks * 512
        except FileNotFoundError:
            pass
    return total


def limit():
    resource.setrlimit(resource.RLIMIT_FSIZE, (100_000_000, 100_000_000))


def main():
    if time.time() >= end-600:
        raise TimeoutError('original deadline near')
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    if not gpu.isdigit():
        raise ValueError('explicit GPU index required')
    info = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if info[0].strip() != os.environ['GPU_UUID'] or int(info[1]) < 46080:
        raise RuntimeError('GPU identity or free memory changed')
    with (root/'GUARD_ONCE').open('x'):
        pass
    write('GUARD_HEALTH.json', {'time': time.time(), 'bytes': allocated(root)+allocated(neutral.parent), 'hard_end': end})
    env = dict(os.environ, ENTRY_MODE='worker', M3BENCH_FORMAL_EXPECTED_GPU_UUID=info[0].strip(),
               M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu, M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu)
    with (root/'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(neutral)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 cwd='/data/bmw', start_new_session=True, preexec_fn=limit)
        write('WORKER_IDENTITY.private.json', {'pid': child.pid, 'time': time.time()})
        stopped = False
        try:
            while child.poll() is None:
                size = allocated(root)+allocated(neutral.parent)
                fs = os.statvfs(root)
                free = fs.f_bavail*fs.f_frsize
                write('GUARD_HEALTH.json', {'time': time.time(), 'bytes': size, 'free': free, 'hard_end': end})
                if size >= 900_000_000 or free <= 8*1024**3 or time.time() >= end-45:
                    raise RuntimeError('storage reserve or original deadline')
                time.sleep(5)
        except BaseException:
            stopped = True
            write('STOP.json', {'time': time.time()})
            write('GUARD_FAILURE.private.json', {'error': traceback.format_exc()})
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        code = child.wait()
        if stopped and code == 0:
            code = 1
        write('GUARD_EXIT.json', {'code': code, 'hard_end': end, 'time': time.time()})
        return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        write('GUARD_FAILURE.private.json', {'error': traceback.format_exc()})
        write('GUARD_EXIT.json', {'code': 1, 'hard_end': end})
        raise
