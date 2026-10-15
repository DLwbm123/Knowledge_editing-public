"""One fresh model per fixed partition, on GPU2 only."""
import os
from pathlib import Path
import time
import traceback
import runtime146 as worker
import wording_pair as protocol

r, c, RUN = worker.r, worker.c, worker.RUN


def main():
    gate = protocol.check(RUN)
    parent = Path(gate['parent'])
    part = int(os.environ['PARTITION'])
    assert part in (0, 1) and os.environ['GPU'] == '2'
    with (RUN/f'private/WORDING_GPU_ONCE_{part}').open('x'): pass
    worker.point = lambda task, arm: parent/f"private/candidates/{arm}/{task['order']}.pt"
    worker.configure()
    r.NATIVE_FORWARD_CAP = 2000
    original_count = r.count
    def bounded(*args):
        original_count(*args)
        assert r.COUNTS['generation_attempts'] <= 117
        assert r.COUNTS['updates'] == r.COUNTS['backwards'] == r.COUNTS['operator_backwards'] == 0
        ledger = c.read(parent/'RESOURCE_LEDGER.json')
        used = ledger['gpu_seconds_used'] + sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if not s.get('ended_epoch'))
        own = c.used()-c.read(RUN/'private/INHERITED_COST.json')['gpu_seconds_used']
        assert used+own+1488.5244522094727 < c.read(parent/'RUN_MANIFEST.json')['GPU_seconds_limit']
    r.count = bounded
    import bank146
    bank146.evaluate(40, 2, lambda: protocol.check(RUN),
                     'WORDING_FROZEN1.json', 'WORDING_SCOPE1.json', 'WORDING_SOURCE1.json')
    assert r.COUNTS['generation_attempts'] == 117


if __name__ == '__main__':
    code = 0
    try: main()
    except BaseException as error:
        code = 1
        c.write(RUN/'private/FAILURE_wording_runtime1.json', dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        r.save_counts()
        c.write(RUN/f"private/WORDING_GPU_EXIT_{os.environ['PARTITION']}.json", dict(exit_code=code, epoch=time.time()))
