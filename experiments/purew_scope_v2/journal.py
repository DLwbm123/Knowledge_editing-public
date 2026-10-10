"""Durable step receipts and identity-checked stage adoption."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.' + str(os.getpid()) + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def alive(start):
    path = Path('/proc') / str(start['pid']) / 'stat'
    if not path.exists(): return False
    # /proc comm may contain spaces; field22 starts after the closing parenthesis.
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[0] != 'Z' and fields[19] == str(start['start_ticks'])


def ensure_event(directory, state):
    """Checkpoint is authoritative; repair a crash between save and event publication."""
    event = state.get('last_event')
    if event is None: return
    path = Path(directory) / (str(event['update']).zfill(4) + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == event, 'committed diagnostic mismatch'
    else: write(path, event)


def completed(path, binding):
    path = Path(path)
    if not path.exists(): return False
    value = json.loads(path.read_text())
    assert value['lock'] == binding, 'completion receipt has another frozen binding'
    return True


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'done.json'
        assert not completed(path, 'a')
        write(path, {'lock':'a'}); assert completed(path, 'a')
        try: completed(path, 'b')
        except AssertionError: pass
        else: raise AssertionError('foreign receipt accepted')
        state = {'last_event':{'update':1,'step':1,'loss':.5}}
        ensure_event(tmp,state); ensure_event(tmp,state)
        assert len(list(Path(tmp).glob('0001.json'))) == 1
        assert not alive({'pid':999999999,'start_ticks':'0'})
    return dict(status='PASS',binding_rejection=True,checkpoint_log_gap_repaired=True,
                completed_receipt_skipped=True,dead_process_rejected=True)
