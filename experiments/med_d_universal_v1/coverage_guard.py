"""Independent one-shot CPU source-coverage guard, sharing the experiment clock."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

root = Path(os.environ['OUTPUT_ROOT'])
deadline = float(os.environ['DEADLINE'])
started = time.time()
child = None
code = 1
reason = 'incomplete'
try:
    with (root / 'COVERAGE_GUARD_ONCE').open('x'):
        pass
    if deadline - started <= 120:
        raise TimeoutError('experiment deadline')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
    with (root / 'coverage.log').open('x') as log:
        child = subprocess.Popen([os.environ['PYTHON_BIN'], os.environ['NEUTRAL_WORKER']],
            env=env, cwd='/data/bmw', stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=limit)
        while child.poll() is None:
            used = sum(p.stat().st_blocks * 512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
            fs = os.statvfs(root)
            if used >= 50_000_000 or fs.f_bavail * fs.f_frsize <= 8 * 1024**3:
                raise RuntimeError('storage guard')
            if time.time() >= min(started + 600, deadline - 120):
                raise TimeoutError('coverage deadline')
            time.sleep(1)
        code = child.wait()
        reason = 'completed' if code == 0 else 'worker_failure'
except BaseException as e:
    reason = type(e).__name__
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    code = 1
finally:
    with (root / 'COVERAGE_GUARD_EXIT.json').open('x') as f:
        json.dump({'code': code, 'reason': reason, 'elapsed_seconds': round(time.time()-started, 3)}, f)
