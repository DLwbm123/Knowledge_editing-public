"""Freeze the completed fixed-panel predictions before opening DEV references."""
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
FIXED = Path(os.environ['FIXED_DEPTH_ROOT'])
MED = Path(os.environ['RAW_MED_ROOT'])
DEADLINE = float(os.environ['EXPERIMENT_DEADLINE'])


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def main():
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    import score_packet
    import extract_reference
    with (ROOT / 'PACKET_WORKER_ONCE').open('x'):
        pass
    if time.time() >= DEADLINE - 660 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('Deadline or guard stop')
    for name in ('TRAIN_EXIT.json', 'TRAIN_GUARD_EXIT.json', 'EVAL_EXIT.json', 'EVAL_GUARD_EXIT.json'):
        if json.loads((ROOT / name).read_text())['code'] != 0:
            raise ValueError('Upstream execution failed')
    if any((ROOT / name).exists() for name in ('TRAIN_FAILURE.json', 'EVAL_FAILURE.json', 'GUARD_FAILURE.json')):
        raise ValueError('Upstream failure receipt exists')
    completion = json.loads((ROOT / 'EVAL_COMPLETE.json').read_text())
    inputs = json.loads((FIXED / 'EVAL_INPUTS_PRIVATE.json').read_text())
    prior = json.loads((FIXED / 'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())
    lock = json.loads((FIXED / 'PANEL_IDENTITY_LOCK.json').read_text())
    admission = json.loads((FIXED / 'ROLE_ADMISSION.json').read_text())
    files = list((ROOT / 'eval_outputs').glob('*.json'))
    predictions, frozen = score_packet.freeze(ROOT, inputs, prior, files, completion)
    # The source is opened by build only after the immutable freeze exists.
    result = score_packet.build(ROOT, predictions, frozen, MED / 'INPUTS_FROZEN.json',
                                lock, admission, extract_reference)
    write('PACKET_COMPLETE.json', result)


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('PACKET_FAILURE.json', dict(error=traceback.format_exc()))
        raise
    finally:
        write('PACKET_EXIT.json', dict(code=code))
