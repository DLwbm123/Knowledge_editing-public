"""Linux independent TERM/KILL guard; neutral entry path, PID birth identity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def birth_tick(stat_text):
    # comm may contain spaces and parentheses; fields after its final ')' start at state.
    return stat_text.rsplit(')', 1)[1].split()[19]


def identity(pid):
    return birth_tick(Path(f'/proc/{pid}/stat').read_text())


def verify_guard(receipt):
    now = time.time()
    pid = os.getpid()
    if (receipt['target_pid'] != pid or receipt['target_birth'] != identity(pid)
            or os.getpgid(pid) != pid or not now < receipt['TERM_epoch'] < receipt['KILL_epoch']
            or receipt['guard_birth'] != identity(receipt['guard_pid'])):
        raise PermissionError('Independent process guard missing/stale')
    return True


def spawn_guard(neutral_directory, start_epoch, config):
    """Call after storage admission, before loading a model, in a new process group.

    This helper is never called on import. No remote access or launch is implicit.
    """
    directory = Path(neutral_directory).resolve()
    if not directory.is_relative_to(Path('/data/bmw/tmp')) or os.getpgid(0) != os.getpid():
        raise PermissionError('Need admitted neutral directory and process-group leader')
    now = time.time()
    term = start_epoch + config['TERM_seconds']
    kill = start_epoch + config['KILL_seconds']
    if not math_finite_order(start_epoch, now, term, kill):
        raise ValueError('Original guard clock invalid or expired')
    entry, ready = directory / 'w.py', directory / 'r.json'
    argv = [sys.executable, str(entry)]
    if any(word in ' '.join(argv).lower() for word in ('wangbomin', 'knowledge_editing', 'purew', 'vigor', 'joint2', 'ind2')):
        raise PermissionError('Non-neutral visible argv')
    payload = {'target_pid': os.getpid(), 'target_birth': identity(os.getpid()), 'TERM_epoch': term, 'KILL_epoch': kill}
    code = '''import os,time,signal,json
from pathlib import Path
c=json.loads(os.environ['RUN_STATE'])
def born(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
c['guard_pid']=os.getpid();c['guard_birth']=born(os.getpid())
with Path(os.environ['READY_FILE']).open('x') as f:
 json.dump(c,f);f.flush();os.fsync(f.fileno())
for deadline,signum in [(c['TERM_epoch'],signal.SIGTERM),(c['KILL_epoch'],signal.SIGKILL)]:
 while time.time()<deadline:
  if born(c['target_pid'])!=c['target_birth']:raise SystemExit
  time.sleep(min(1.,max(0.,deadline-time.time())))
 if born(c['target_pid'])==c['target_birth']:
  try:os.killpg(c['target_pid'],signum)
  except ProcessLookupError:pass
'''
    with entry.open('x') as f:
        f.write(code)
    env = dict(os.environ, RUN_STATE=json.dumps(payload), READY_FILE=str(ready))
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + 3
    while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(.02)
    if not ready.exists():
        raise RuntimeError('Guard did not become ready; do not load model')
    receipt = json.loads(ready.read_text())
    verify_guard(receipt)
    receipt['argv_verified'] = subprocess.check_output(['ps', '-ww', '-p', str(proc.pid), '-o', 'args='], text=True).strip()
    if any(word in receipt['argv_verified'].lower() for word in ('wangbomin', 'knowledge_editing', 'purew', 'vigor', 'joint2', 'ind2')):
        raise RuntimeError('Guard argv check failed')
    return receipt


def math_finite_order(start, now, term, kill):
    import math
    return all(math.isfinite(x) for x in (start, now, term, kill)) and start <= now < term < kill
