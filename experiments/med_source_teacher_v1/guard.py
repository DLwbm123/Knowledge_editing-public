"""One independent clock and storage guard for the sequential GPU stages."""
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
    path = ROOT / name; tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def allocated(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink(): total += path.stat().st_blocks * 512
        except FileNotFoundError: pass
    return total


def main():
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    if not gpu.isdigit(): raise ValueError('One physical GPU required')
    row = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if row[0].strip() != os.environ['GPU_UUID'] or int(row[1]) < 45056:
        raise RuntimeError('GPU identity or memory insufficient')
    if time.time() >= DEADLINE - 660 or DEADLINE - time.time() > 86400:
        raise TimeoutError('Invalid original experiment deadline')
    admission = json.loads((ROOT / 'SCORING_CHAIN_ADMISSION.json').read_text())
    if admission.get('status') != 'SCORING_CHAIN_VERIFIED' or admission.get('automatic_retry') is not False:
        raise ValueError('Independent scoring chain not admitted')
    with (ROOT / 'GUARD_ONCE').open('x'): pass
    neutral = Path(os.environ['NEUTRAL_ENTRY'])
    def limit(): resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    for phase in ('train', 'eval', 'packet'):
        write('GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=allocated(ROOT)))
        with (ROOT / f'{phase}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(neutral)], env=dict(os.environ, ENTRY_MODE=phase),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=limit)
        write(f'{phase.upper()}_IDENTITY.json', dict(pid=child.pid, time=time.time()))
        try:
            while child.poll() is None:
                used = allocated(ROOT) + allocated(neutral.parent)
                stat = os.statvfs(ROOT); free = stat.f_bavail * stat.f_frsize
                write('GUARD_HEALTH.json', dict(time=time.time(), hard_end=DEADLINE, bytes=used, free=free))
                if used >= 30_000_000_000 or free <= 8 * 1024**3 or time.time() >= DEADLINE - 45:
                    raise RuntimeError('Storage reserve or deadline reached')
                time.sleep(5)
        except BaseException:
            write('STOP.json', dict(time=time.time(), error=traceback.format_exc()))
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        code = child.wait()
        write(f'{phase.upper()}_GUARD_EXIT.json', dict(code=code, hard_end=DEADLINE))
        if code != 0: raise SystemExit(code)


if __name__ == '__main__':
    try: main()
    except Exception:
        write('GUARD_FAILURE.json', dict(error=traceback.format_exc()))
        raise
