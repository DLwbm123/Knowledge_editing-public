"""Independent process, wall and whole-experiment disk guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(os.environ['V131_ROOT'])
METHOD = os.environ['V131_METHOD']
OUT = ROOT / METHOD
cfg = json.loads(Path(os.environ.get('V131_PROTOCOL_FILE', ROOT / 'PROTOCOL_LOCK.json')).read_text())


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def size(folder):
    total = 0
    pending = [str(folder)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    pass
    return total


def start_counter(roots, initial):
    state = {'sample': initial}
    stopped = threading.Event()
    def collect():
        while not stopped.is_set():
            try:
                used = sum(size(root) for root in roots)
                free = os.statvfs(roots[0]).f_bavail * os.statvfs(roots[0]).f_frsize
                state['sample'] = dict(time=time.time(), new_bytes=used, free_disk_bytes=free)
            except BaseException as error:
                state['sample'] = dict(time=time.time(), error=repr(error))
            stopped.wait(10)
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    return state, stopped


def main():
    sys.path.insert(0, str(ROOT/'tools'))
    import prefix_parallel_v131
    assert Path(prefix_parallel_v131.__file__).resolve() == (ROOT/'tools/prefix_parallel_v131.py').resolve()
    helper_seconds = prefix_parallel_v131.helper_seconds
    assert cfg['launch_authorized'] is True and cfg['source_public_verified'] is True
    assert METHOD in ('M13P_160','STATISTICS','MEMIT_MM')
    with (OUT / 'GUARD_ONCE').open('x'):
        pass
    uuid, free = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert uuid == os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'] and int(free) >= 45 * 1024
    child, started, counter_stop = None, time.time(), None
    prior = cfg.get('prior_process_seconds',{}).get(METHOD,0.)
    try:
        roots = [ROOT, Path(os.environ['NEUTRAL_ENTRY']).parent]
        used = sum(size(root) for root in roots)
        free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        sample = dict(time=time.time(), new_bytes=used, free_disk_bytes=free)
        assert used < cfg['storage_pre_stop_bytes'] and free >= 8*1024**3
        state, counter_stop = start_counter(roots, sample)
        write('GUARD_HEALTH.json', dict(time=time.time(), process_seconds=time.time()-started,
            new_bytes=used, free_disk_bytes=free, storage_sample_time=sample['time']))
        with (OUT / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, os.environ['NEUTRAL_ENTRY']],
                env=dict(os.environ, V131_ACTION='worker'), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        write('WORKER_IDENTITY.json', dict(pid=child.pid,
            start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21], start=started))
        while child.poll() is None:
            sample = state['sample']
            now = time.time()
            auxiliary, live_helpers = helper_seconds(OUT/'parallel', now, since=started)
            assert live_helpers <= 2, 'Human limit: at most three total GPUs'
            total = prior+now-started+auxiliary
            shutdown_reserve = 30*(1+live_helpers) if live_helpers else 20
            used, free_disk = sample.get('new_bytes',used), sample.get('free_disk_bytes',free)
            write('GUARD_HEALTH.json', dict(time=now, process_seconds=now-started,
                prior_process_seconds=prior,total_method_process_seconds=total,
                auxiliary_process_seconds=auxiliary,
                total_GPU_process_seconds=total,live_helpers=live_helpers,
                time_limits_enforced=not cfg.get('run_until_completion',False),
                new_bytes=used, free_disk_bytes=free_disk, storage_sample_time=sample['time'],
                storage_sample_age_seconds=now-sample['time'], accounting_error=sample.get('error'),
                storage_hard_limit_bytes=cfg['storage_limit_bytes'],storage_pre_stop_bytes=cfg['storage_pre_stop_bytes']))
            reason = None
            if sample.get('error') or now-sample['time'] >= 45:
                reason = 'STORAGE_COUNTER_UNAVAILABLE'
            elif not cfg.get('run_until_completion') and now >= cfg['hard_end'] - 20:
                reason = 'ORIGINAL_24H_WALL'
            elif not cfg.get('run_until_completion') and now >= cfg.get('method_original_deadline', {}).get(METHOD, cfg['hard_end']) - 20:
                reason = 'METHOD_ORIGINAL_WALL'
            elif not cfg.get('run_until_completion') and total >= cfg['method_process_seconds'][METHOD] - shutdown_reserve:
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
        if counter_stop is not None:
            counter_stop.set()
        if child is not None:
            try:
                _, live_helpers = helper_seconds(OUT/'parallel', time.time(), since=started)
            except BaseException:
                live_helpers = 1  # Unclosed helper evidence fails closed.
            if child.poll() is None or live_helpers:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if live_helpers:
                    limit = time.monotonic()+8
                    while time.monotonic() < limit:
                        try:
                            _, alive = helper_seconds(OUT/'parallel', time.time(), since=started)
                            if not alive:
                                break
                        except BaseException:
                            pass
                        time.sleep(.1)
                    else:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            auxiliary, _ = helper_seconds(OUT/'parallel', time.time(), strict=False, since=started)
            write('GUARD_EXIT.json', dict(time=time.time(), code=child.returncode,
                process_seconds=time.time()-started,prior_process_seconds=prior,
                auxiliary_process_seconds=auxiliary,
                total_method_process_seconds=prior+time.time()-started+auxiliary,
                total_GPU_process_seconds=prior+time.time()-started+auxiliary))
    return child.returncode if child is not None else 1


if __name__ == '__main__':
    sys.exit(main())
