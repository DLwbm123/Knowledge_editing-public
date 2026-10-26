"""An independent finite resource guard, not a recurring chat monitor."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def size(root):
    return sum(p.stat().st_blocks * 512 for p in root.rglob('*')
               if p.is_file() and not p.is_symlink())


def main():
    with (ROOT/'GUARD_ONCE').open('x'):
        pass
    lock = json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
    start = time.time()
    end = start + lock['budgets']['max_elapsed_seconds']
    write('CLOCK.json', dict(original_start=start, hard_end=end, retries_share_clock=True))
    env = dict(os.environ, ENTRY_MODE='worker')
    with (ROOT/'worker.log').open('a') as log:
        child = subprocess.Popen([sys.executable, os.environ['NEUTRAL_ENTRY']], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path('/proc', str(child.pid), 'stat').read_text().split(') ',1)[1].split()[19]
    write('WORKER_IDENTITY.json', dict(pid=child.pid, start_ticks=ticks, start=start,
                                      argv=[sys.executable, os.environ['NEUTRAL_ENTRY']]))
    budget = lock['budgets']
    try:
        while child.poll() is None:
            used = size(ROOT)
            other_path = Path(os.environ['OTHER_RESOURCE_HEALTH'])
            other = json.loads(other_path.read_text())
            assert time.time()-other['time'] < 90, 'Other managed storage ledger unavailable'
            global_used = used + other['new_bytes']
            free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            health = dict(time=time.time(), new_bytes=used, other_managed_new_bytes=other['new_bytes'],
                          global_managed_new_bytes=global_used, free_bytes=free,
                          seconds=time.time()-start, hard_end=end)
            write('GUARD_HEALTH.json', health)
            assert used < budget['new_bytes_limit'] - 100_000_000, 'Owned storage pre-stop'
            assert global_used < 47_000_000_000, 'Global managed storage pre-stop'
            assert free > budget['minimum_free_bytes'], 'Disk reserve'
            assert time.time() < end - 45, 'Frozen wall-clock pre-stop'
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
    write('GUARD_EXIT.json', dict(code=code, seconds=time.time()-start, time=time.time(),
                                 GPU_process_seconds=time.time()-start))
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
