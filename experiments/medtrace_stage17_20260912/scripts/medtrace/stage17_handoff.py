"""CPU-only acceptance coordinator after removing a rented-host dependency."""
import fcntl
import os
from pathlib import Path
import time
from scripts.medtrace.astra_judge_bundle import read
from scripts.medtrace.stage17_external import state, validate_phase


def all_phases(root, cfg, methods):
    phases = [(method, mode) for method in methods for mode in ('single', 'sequential')]
    if not all((root/'private'/f'{method}_{mode}'/'CLEANUP.json').exists() for method, mode in phases):
        return False
    for method, mode in phases:
        validate_phase(root, cfg, mode, method=method)
    return True


def tick(cfg):
    baseline = Path(cfg['baseline'])
    campaign = Path(cfg['campaign'])
    bc = read(baseline/'private/DISPATCH.json')
    cc = read(campaign/'private/DISPATCH.json')
    if (cc['freeze_id'], cc['N']) != (bc['freeze_id'], bc['N']):
        raise ValueError('Transferred cohort mismatch')
    # The original campaign dispatch predates the external order field.
    cc = dict(cc, order=bc['order'])
    if all_phases(baseline, bc, ('grace', 'belora')):
        state(baseline/'public/PROGRESS.json', dict(status='GPU_GENERATED_NOT_SCORED', N=bc['N'],
            phases=[(m, mode) for m in ('grace', 'belora') for mode in ('single', 'sequential')]))
    else:
        state(baseline/'public/PROGRESS.json', dict(status='RUNNING', phase='WAITING_FOR_TRANSFERRED_PHASES', N=bc['N']))
    for assignment in ('EXTERNAL_LORA', 'EXTERNAL_BASELINES'):
        if (campaign/'private'/f'{assignment}_FAILURE.json').exists():
            raise RuntimeError('External transfer failed: ' + assignment)
        marker = campaign/'private'/f'{assignment}_IMPORTED.json'
        if not marker.exists():
            state(campaign/'public/PROGRESS.json', dict(status='RUNNING', phase='WAITING_FOR_ASSIGNED_EXTERNAL_WORKER',
                assignment=assignment, N=cc['N']))
            return False
        expected = read(campaign/'private'/f'{assignment}.json')
        if read(marker) != expected:
            raise ValueError('Assignment/import mismatch')
        methods = ('lora',) if assignment == 'EXTERNAL_LORA' else ('grace', 'belora')
        if not all_phases(campaign, expected, methods):
            raise ValueError('Imported phase incomplete')
    for method in ('C_NO_H', 'balancedit'):
        validate_phase(campaign, cc, 'sequential', method=method)
    state(campaign/'public/PROGRESS.json', dict(status='GPU_GENERATED_NOT_SCORED', N=cc['N'],
        provenance='Validated transferred phases; no training performed by handoff coordinator'))
    return True


def run(cfg):
    root = Path(cfg['campaign'])
    with (root/'private/handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not tick(cfg):
                if (root/'STOP').exists():
                    raise RuntimeError('Explicit handoff stop')
                time.sleep(60)
        except Exception as error:
            state(root/'public/PROGRESS.json', dict(status='STOPPED', error=repr(error)))
            raise


if __name__ == '__main__':
    run(read(os.environ['JOB_CONFIG']))
