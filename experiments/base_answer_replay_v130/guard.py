"""One neutral worker with independent wall, phase GPU-process and storage stops."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ['V130_ROOT'])
cfg = json.loads((ROOT / 'PROTOCOL_LOCK.json').read_text())


def write(name, value):
    path = ROOT / name
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
            # Atomic saves and the declared A retirement can remove a scanned file.
            continue
    return total


def main():
    with (ROOT / 'GUARD_ONCE').open('x'):
        pass
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert uuid == os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'] and int(free) >= 45 * 1024
    child, started = None, None
    try:
        started = time.time()
        write('GUARD_HEALTH.json', dict(time=started, gpu_seconds=cfg['prior_gpu_seconds']))
        with (ROOT / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, os.environ['NEUTRAL_ENTRY']],
                env=dict(os.environ, ACTION_107='worker'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write('WORKER_IDENTITY.json', dict(pid=child.pid,
            start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21], start=started))
        while child.poll() is None:
            now, elapsed = time.time(), time.time() - started
            whole = sum(size(folder) for folder in cfg['guarded_roots'])
            own = size(ROOT) + sum(size(folder) for folder in cfg['phase_extra_roots'])
            inherited = sum(Path(path).stat().st_size for path in cfg['inherited_Q_files'])
            free_disk = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            write('GUARD_HEALTH.json', dict(time=now, gpu_seconds=cfg['prior_gpu_seconds'] + elapsed,
                phase_gpu_seconds=elapsed, bytes=whole, phase_new_bytes=own,
                phase_bytes_including_inherited_Q=own + inherited, free_disk_bytes=free_disk))
            reason = None
            if now >= cfg['hard_end'] - 15:
                reason = 'ORIGINAL_WALL_LIMIT'
            elif elapsed >= cfg['stage_gpu_cap_seconds'] - 15:
                reason = 'FROZEN_PHASE_GPU_PROCESS_LIMIT'
            elif whole >= cfg['storage_limit_bytes'] - 100000000 or own + inherited >= cfg['phase_storage_limit_bytes']:
                reason = 'FROZEN_STORAGE_LIMIT'
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
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            elapsed = time.time() - started
            write('GUARD_EXIT.json', dict(time=time.time(), code=child.returncode,
                phase_gpu_seconds=elapsed, prior_gpu_seconds=cfg['prior_gpu_seconds'],
                total_gpu_seconds=cfg['prior_gpu_seconds'] + elapsed))


if __name__ == '__main__':
    main()
