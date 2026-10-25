"""Independent process, wall and whole-experiment disk guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ['V131_ROOT'])
METHOD = os.environ['V131_METHOD']
OUT = ROOT / METHOD
cfg = json.loads((ROOT / 'PROTOCOL_LOCK.json').read_text())


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def size(folder):
    total = 0
    for path in Path(folder).rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def main():
    assert cfg['launch_authorized'] is True and cfg['source_public_verified'] is True
    assert METHOD in ('M13P_160','STATISTICS','MEMIT_MM')
    with (OUT / 'GUARD_ONCE').open('x'):
        pass
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert uuid == os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'] and int(free) >= 45 * 1024
    child, started = None, time.time()
    try:
        write('GUARD_HEALTH.json', dict(time=started, process_seconds=0))
        with (OUT / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, os.environ['NEUTRAL_ENTRY']],
                env=dict(os.environ, V131_ACTION='worker'), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        write('WORKER_IDENTITY.json', dict(pid=child.pid,
            start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21], start=started))
        while child.poll() is None:
            now = time.time()
            used = size(ROOT) + size(Path(os.environ['NEUTRAL_ENTRY']).parent)
            free_disk = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            write('GUARD_HEALTH.json', dict(time=now, process_seconds=now-started,
                new_bytes=used, free_disk_bytes=free_disk))
            reason = None
            if now >= cfg['hard_end'] - 20:
                reason = 'ORIGINAL_24H_WALL'
            elif now-started >= cfg['method_process_seconds'][METHOD] - 20:
                reason = 'METHOD_PROCESS_BUDGET'
            elif used >= cfg['storage_pre_stop_bytes']:
                reason = 'WHOLE_EXPERIMENT_STORAGE'
            elif free_disk < 8 * 1024 ** 3:
                reason = 'DISK_RESERVE'
            if reason:
                write('STOP.json', dict(time=now, reason=reason))
                break
            time.sleep(5)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write('GUARD_EXIT.json', dict(time=time.time(), code=child.returncode,
                process_seconds=time.time()-started))


if __name__ == '__main__':
    main()
