"""Apply a stricter user storage limit without restarting a healthy worker."""
import json
import os
from pathlib import Path
import signal
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def identity(pid, ticks):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[19] == ticks
    except FileNotFoundError:
        return False


def size(roots):
    total = 0
    for root in roots:
        for path in Path(root).rglob('*'):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def stop(config, reason):
    out = Path(config['output'])
    write(out/'STOP.json', dict(time=time.time(), reason=reason,
        storage_hard_limit_bytes=config['hard_limit_bytes'],
        storage_pre_stop_bytes=config['pre_stop_bytes']))
    pid, ticks = config['worker_pid'], config['worker_start_ticks']
    for _ in range(12):
        if not identity(pid, ticks):
            return
        time.sleep(1)
    if identity(pid, ticks):
        assert os.getpgid(pid) == pid, 'Only the registered worker session may be stopped'
        os.killpg(pid, signal.SIGTERM)
        for _ in range(8):
            if not identity(pid, ticks):
                return
            time.sleep(1)
        if identity(pid, ticks):
            os.killpg(pid, signal.SIGKILL)


def main():
    config = read(os.environ['STORAGE_OVERRIDE_CONFIG'])
    root, out = Path(config['root']), Path(config['output'])
    assert config['hard_limit_bytes'] == 50_000_000_000
    assert config['pre_stop_bytes'] == 47_000_000_000
    assert config['hard_end'] == read(root/'PROTOCOL_LOCK.json')['hard_end']
    assert out.resolve().is_relative_to(root.resolve())
    with (root/'STORAGE_OVERRIDE_ONCE').open('x'):
        pass
    started, status = time.time(), 'WORKER_FINISHED'
    try:
        while identity(config['worker_pid'], config['worker_start_ticks']):
            health = read(out/'GUARD_HEALTH.json')
            # The original live guard already measures these exact roots every
            # five seconds. Fall back to a fresh count if its sample is stale.
            fresh = time.time()-health['time'] <= 10
            used = health['new_bytes'] if fresh else size(config['accounted_roots'])
            write(root/'STORAGE_OVERRIDE_HEALTH.json', dict(time=time.time(),
                new_bytes=used, source='original_guard' if fresh else 'fresh_count',
                hard_limit_bytes=config['hard_limit_bytes'], pre_stop_bytes=config['pre_stop_bytes'],
                worker_identity_matches=True, original_hard_end=config['hard_end']))
            if used >= config['pre_stop_bytes'] or time.time() >= config['hard_end']-30:
                status = 'USER_50GB_PRESTOP' if used >= config['pre_stop_bytes'] else 'ORIGINAL_24H_WALL'
                stop(config, status)
                break
            time.sleep(5)
    except BaseException:
        status = 'STORAGE_OVERRIDE_FAILURE'
        write(root/'STORAGE_OVERRIDE_FAILURE.json', dict(error=traceback.format_exc(), time=time.time()))
        stop(config, status)
        raise
    finally:
        write(root/'STORAGE_OVERRIDE_EXIT.json', dict(status=status, time=time.time(), seconds=time.time()-started))


if __name__ == '__main__':
    main()
