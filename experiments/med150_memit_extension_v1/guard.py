"""Independent deadline, storage and GPU guard for one finite experiment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['EXTENSION_ROOT'])


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def used(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_blocks * 512
        except FileNotFoundError:
            pass
    return total


def main():
    lock, clock = read('LOCK.json'), read('CLOCK.json')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(lock['gpu'])
    raw = subprocess.check_output(['nvidia-smi', '-i', str(lock['gpu']), '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert raw[0].strip() == lock['gpu_uuid'] and int(raw[1]) >= lock['min_free_mib']
    assert read('CPU_ADMISSION.json')['status'] == 'PASS'
    assert time.time() < clock['hard_end'] - 620
    with (ROOT / 'GUARD_ONCE').open('x'):
        pass
    write('GUARD_HEALTH.json', dict(time=time.time(), bytes=used(ROOT)))
    with (ROOT / 'worker.log').open('a') as log:
        child = subprocess.Popen([sys.executable, os.environ['NEUTRAL_ENTRY']],
                                 env=dict(os.environ, ENTRY_MODE='worker'), stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while child.poll() is None:
            consumed = used(ROOT) + used(Path(os.environ['NEUTRAL_ENTRY']).parent)
            stat = os.statvfs(ROOT)
            free = stat.f_bavail * stat.f_frsize
            write('GUARD_HEALTH.json', dict(time=time.time(), bytes=consumed, free_bytes=free, hard_end=clock['hard_end']))
            assert consumed < 47_000_000_000 and free > 8 * 1024**3 and time.time() < clock['hard_end'] - 45
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
    write('GUARD_EXIT.json', dict(code=code, time=time.time(), clock=clock))
    raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            write('GUARD_FAILURE.json', dict(error=traceback.format_exc(), time=time.time()))
            if not (ROOT / 'GUARD_EXIT.json').exists():
                write('GUARD_EXIT.json', dict(code=1, time=time.time()))
        raise
