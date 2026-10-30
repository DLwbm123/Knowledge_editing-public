"""Wait for the first training pair, then guard its real generation stage."""
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
    with (ROOT / 'FIRST_EVAL_GUARD_ONCE').open('x'):
        pass
    while True:
        if time.time() >= DEADLINE - 660:
            raise TimeoutError('Original deadline reached while waiting for training')
        if (ROOT / 'WORKER_FAILURE.json').exists() or (ROOT / 'GUARD_FAILURE.json').exists():
            raise RuntimeError('Training dependency failed')
        required = ('FIRST_PAIR_TRAIN_COMPLETE.json', 'WORKER_EXIT.json', 'GUARD_EXIT.json')
        if all((ROOT / n).is_file() for n in required):
            if any(json.loads((ROOT / n).read_text())['code'] != 0 for n in ('WORKER_EXIT.json', 'GUARD_EXIT.json')):
                raise RuntimeError('Training dependency did not exit successfully')
            break
        time.sleep(10)
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    row = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if row[0].strip() != os.environ['GPU_UUID'] or int(row[1]) < 45056:
        raise RuntimeError('GPU identity or available memory insufficient')
    neutral = Path(os.environ['NEUTRAL_ENTRY'])
    write('FIRST_EVAL_GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=allocated(ROOT)))
    def file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    with (ROOT / 'first_eval.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(neutral)], env=dict(os.environ, ENTRY_MODE='first_eval'),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=file_limit)
    write('FIRST_EVAL_WORKER_IDENTITY.json', dict(pid=child.pid, time=time.time(), gpu=gpu))
    try:
        while child.poll() is None:
            used = allocated(ROOT) + allocated(neutral.parent)
            st = os.statvfs(ROOT)
            free = st.f_bavail * st.f_frsize
            write('FIRST_EVAL_GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=used, free=free))
            if used >= 30_000_000_000 or free <= 8 * 1024**3 or time.time() >= DEADLINE - 45:
                raise RuntimeError('Storage reserve or original deadline reached')
            time.sleep(5)
    except BaseException:
        write('FIRST_EVAL_STOP.json', dict(time=time.time(), error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    code = child.wait()
    write('FIRST_EVAL_GUARD_EXIT.json', dict(time=time.time(), code=code, hard_end=DEADLINE))
    raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('FIRST_EVAL_GUARD_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
