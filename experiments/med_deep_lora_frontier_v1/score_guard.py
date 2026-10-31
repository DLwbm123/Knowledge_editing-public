"""Independent local score guard; no automatic rerun of attempted payloads."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['FIXED_SCORE_ROOT'])
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
    if time.time() >= DEADLINE - 660:
        raise TimeoutError('Original deadline too close')
    with (ROOT / 'GUARD_ONCE').open('x'):
        pass
    neutral = Path(os.environ['NEUTRAL_ENTRY'])
    def file_limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    with (ROOT / 'worker.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(neutral)], env=dict(os.environ, ENTRY_MODE='score_driver'),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=file_limit)
    write('WORKER_IDENTITY.json', dict(pid=child.pid, time=time.time()))
    try:
        while child.poll() is None:
            used = allocated(ROOT) + allocated(neutral.parent) + allocated(Path(os.environ['FIXED_PRIVATE_ROOT']))
            st = os.statvfs(ROOT)
            free = st.f_bavail * st.f_frsize
            write('GUARD_HEALTH.json', dict(time=time.time(), bytes=used, free=free, hard_end=DEADLINE))
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
    if code == 0:
        sys.path.insert(0, str(Path(os.environ['FIXED_WORKTREE']) / 'experiments/med_deep_lora_frontier_v1'))
        import report_driver
        report_driver.main()
    raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('GUARD_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
