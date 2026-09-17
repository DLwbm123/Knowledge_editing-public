"""Shared process-wall-time guard; every GPU attempt is charged by its supervisor."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remaining(ledger, now=None):
    now = time.time() if now is None else now
    used = sum(s.get('seconds', max(0, now-s['started_epoch'])) for s in ledger['sessions'])
    return max(0., ledger['limit_seconds']-used)


def check(cfg):
    if time.time() >= cfg['gpu_deadline_epoch']:
        raise TimeoutError('Cumulative GPU process budget exhausted')


def supervise(cfg):
    root = Path(cfg['run']); path = root/'public/GPU_BUDGET_LEDGER.json'
    with (root/'private/supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = read(path) if path.exists() else dict(limit_seconds=28800, sessions=[])
        # A supervisor lost without an end receipt cannot silently reset the clock.
        if any('seconds' not in s for s in ledger['sessions']):
            raise RuntimeError('Unclosed GPU session needs evidence-bound recovery')
        available = remaining(ledger)
        if available < 60:
            raise TimeoutError('Insufficient remaining GPU process budget')
        started = time.time()
        session = dict(started_epoch=started, phase=cfg['phase'], code=cfg['code_commit'])
        ledger['sessions'].append(session); write(path, ledger)
        worker = dict(cfg, gpu_deadline_epoch=started+available-10,
                      campaign_epoch=started, train_seconds=available)
        dispatch = root/'private'/('DISPATCH_'+cfg['phase']+'.json'); write(dispatch, worker)
        env = dict(os.environ, JOB_CONFIG=str(dispatch), JOB_ARGV=json.dumps([cfg['worker']]))
        command = [cfg['python'], cfg['entrypoint'], 'run']
        exit_code = None
        try:
            process = subprocess.Popen(command, env=env, start_new_session=True)
            session['pid'] = process.pid; write(path, ledger)
            try:
                exit_code = process.wait(timeout=available-10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); exit_code = process.wait()
                session['hard_deadline_reached'] = True
        finally:
            session.update(ended_epoch=time.time(), seconds=time.time()-started, exit_code=exit_code)
            write(path, ledger)
        write(root/'public'/('EXIT_'+cfg['phase']+'.json'), dict(exit_code=exit_code, remaining_seconds=remaining(ledger)))
        return exit_code


if __name__ == '__main__':
    sys.exit(supervise(read(os.environ['JOB_CONFIG'])))
