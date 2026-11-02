"""Independent bounded preparation guard; shares the experiment's original clock."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(os.environ['EXPERIMENT_ROOT'])
hard_end = float(os.environ['EXPERIMENT_DEADLINE'])
started = time.time()
child = None
code = 1
reason = None
try:
    if (root / 'COVERAGE_GUARD_ONCE').exists():
        raise RuntimeError('guard already used')
    (root / 'COVERAGE_GUARD_ONCE').write_text('started')
    child = subprocess.Popen([os.environ['TRAIN_PYTHON'], os.environ['NEUTRAL_WORKER']],
        env=os.environ.copy(), start_new_session=True)
    while child.poll() is None:
        total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        fs = os.statvfs(root)
        if time.time() >= min(hard_end - 120, started + 600):
            reason = 'TIME_LIMIT'
        elif total >= 50_000_000:
            reason = 'PREPARATION_STORAGE_LIMIT'
        elif fs.f_bavail * fs.f_frsize < 8 * 1024**3:
            reason = 'FREE_SPACE_LIMIT'
        if reason:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        time.sleep(1)
    code = child.returncode if reason is None else 1
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    with (root / 'COVERAGE_GUARD_EXIT.json').open('x') as f:
        json.dump({'code':code, 'reason':reason, 'finished':time.time()}, f)
