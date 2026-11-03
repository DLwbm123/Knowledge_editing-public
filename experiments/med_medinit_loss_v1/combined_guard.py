"""Independently bound shared run plus neutral scratch and original deadline."""
import json
import os
from pathlib import Path
import signal
import time
import traceback

root = Path(os.environ['EXPERIMENT_ROOT'])
neutral = Path(os.environ['NEUTRAL_ROOT'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, data):
    path = neutral / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def allocated():
    total = 0
    for base in (root, neutral):
        for p in base.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def active(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    try:
        return os.environ['NEUTRAL_MARKER'].encode() in path.read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    with (neutral / 'COMBINED_GUARD_ONCE').open('x'):
        pass
    start = json.loads((root / 'START.private.json').read_text())
    guards = [p['pid'] for p in start['processes']]
    try:
        while True:
            used = allocated()
            fs = os.statvfs(root)
            free = fs.f_bavail * fs.f_frsize
            write('COMBINED_HEALTH.json', {'time': time.time(), 'bytes': used, 'free': free, 'deadline': deadline})
            if used >= 30_000_000_000 or free < 8 * 1024 ** 3 or time.time() >= deadline - 60:
                raise RuntimeError('combined resource or original deadline')
            if not any(active(pid) for pid in guards):
                write('COMBINED_EXIT.json', {'code': 0, 'time': time.time()})
                return
            time.sleep(5)
    except BaseException:
        write('COMBINED_FAILURE.private.json', {'traceback': traceback.format_exc()})
        pids = set(guards)
        for arm in ('MCE', 'MDU'):
            for phase in ('TRAIN', 'EVAL'):
                path = root / arm / (phase + '_IDENTITIES.private.json')
                if path.exists():
                    pids.update(json.loads(path.read_text())['pids'])
        for pid in pids:
            if active(pid):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        write('COMBINED_EXIT.json', {'code': 1, 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
