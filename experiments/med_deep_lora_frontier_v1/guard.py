"""Independent guard for one fixed first pair; no experiment retries."""
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
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    if ',' in gpu or not gpu.isdigit():
        raise ValueError('One physical device required for first pair')
    row = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if row[0].strip() != os.environ['GPU_UUID'] or int(row[1]) < 45056:
        raise RuntimeError('GPU identity or available memory insufficient')
    if time.time() >= DEADLINE - 660:
        raise TimeoutError('Not enough time before original deadline')
    if json.loads((Path(os.environ['FIXED_DEPTH_ROOT']) / 'ROLE_ADMISSION.json').read_text())['status'] != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED':
        raise ValueError('Role admission missing')
    scoring = json.loads((ROOT / 'SCORING_CHAIN_ADMISSION.json').read_text())
    if scoring.get('status') != 'SCORING_CHAIN_VERIFIED' or scoring.get('automatic_retry') is not False:
        raise ValueError('Scoring dependency unavailable')
    with (ROOT / 'GUARD_ONCE').open('x'):
        pass
    neutral = Path(os.environ['NEUTRAL_ENTRY'])
    write('GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=allocated(ROOT)))
    def file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    with (ROOT / 'worker.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(neutral)], env=dict(os.environ, ENTRY_MODE='worker'),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=file_limit)
    write('WORKER_IDENTITY.json', dict(pid=child.pid, time=time.time(), gpu=gpu))
    try:
        while child.poll() is None:
            used = allocated(ROOT) + allocated(neutral.parent)
            st = os.statvfs(ROOT)
            free = st.f_bavail * st.f_frsize
            write('GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=used, free=free))
            if used >= 30_000_000_000 or free <= 8 * 1024**3 or time.time() >= DEADLINE - 45:
                raise RuntimeError('Storage reserve or original deadline reached')
            time.sleep(5)
    except BaseException:
        write('STOP.json', dict(time=time.time(), error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    code = child.wait()
    write('GUARD_EXIT.json', dict(time=time.time(), code=code, hard_end=DEADLINE))
    raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('GUARD_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
