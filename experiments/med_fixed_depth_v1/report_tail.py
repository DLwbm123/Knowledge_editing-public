"""Finite dependency tail for the frozen aggregate and execution audit."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['FIXED_SCORE_ROOT'])
WORKTREE = Path(os.environ['FIXED_WORKTREE'])
DEADLINE = 1793447820


def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def main():
    with (ROOT / 'REPORT_TAIL_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen deadline')))
    signal.alarm(max(1, int(DEADLINE - time.time() - 30)))
    while True:
        if time.time() >= DEADLINE - 60:
            raise TimeoutError('Frozen deadline')
        if (ROOT / 'FAILURE.json').exists() or (ROOT / 'GUARD_FAILURE.json').exists() or (ROOT / 'STOP.json').exists():
            raise RuntimeError('Score dependency failed')
        if all((ROOT / n).is_file() for n in ('COMPLETE.json', 'WORKER_EXIT.json', 'GUARD_EXIT.json')):
            if json.loads((ROOT / 'WORKER_EXIT.json').read_text())['code'] != 0 or json.loads((ROOT / 'GUARD_EXIT.json').read_text())['code'] != 0:
                raise RuntimeError('Score dependency exited unsuccessfully')
            break
        time.sleep(20)
    sys.path.insert(0, str(WORKTREE / 'experiments/med_fixed_depth_v1'))
    import execution_audit
    import report
    execution_audit.main()
    report.main()
    write('REPORT_COMPLETE.json', dict(time=time.time(), status='AGGREGATED_PUBLICATION_PENDING'))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('REPORT_FAILURE.json', dict(time=time.time(), error=traceback.format_exc()))
        raise
    finally:
        write('REPORT_EXIT.json', dict(time=time.time(), code=code))
