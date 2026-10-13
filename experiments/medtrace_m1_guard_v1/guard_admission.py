"""Inherit the completed exact C reference; never reconstruct its training."""
import math
import os
from pathlib import Path
import torch
import module_run as r


def verify_reference():
    c, run = r.c, r.RUN
    parent = Path(os.environ['MODULE_PARENT'])
    retained = Path(os.environ['REFERENCE_PARENT'])
    assert c.read(parent/'EXIT.json')['exit_code'] == 0
    receipt = c.read(parent/'private/REFERENCE_EXACT.json')
    assert receipt['status'] == 'PASS' and receipt['candidates'] == 8
    assert receipt['exact_source_outputs'] == 56 and receipt['exact_retained_final_tensors']
    assert c.read(run/'private/SOURCE_COMMIT.json') == c.read(parent/'private/SOURCE_COMMIT.json')
    lock = c.digest(c.read(retained/'private/SCOPE_LOCK.json'))
    tasks = r.s.tasks()
    assert [t['order'] for t in tasks] == list(range(1, 9))
    outputs = 0
    for task in tasks:
        owner = str(task['order'])
        point = r.s.point(task, 'C')
        assert point.resolve() == (retained/'private/candidates/C'/f'{owner}.pt').resolve()
        state = r.p.load_state(point)
        assert state['arm'] == 'C' and state['order'] == task['order'] and state['lock'] == lock
        for name in ('BASE', 'FINAL'):
            tensors = state['states'][name]
            assert set(tensors) == set(r.mm.KEYS)
            assert sum(v.numel() for v in tensors.values()) == 7168
            assert all(torch.isfinite(v).all() for v in tensors.values())
        result_path = run/'private/results/C'/f'{owner}.json'
        assert result_path.resolve() == (parent/'private/results/C'/f'{owner}.json').resolve()
        result = c.read(result_path)
        assert result['status'] == 'COMPLETE' and result['save_load_exact']
        assert [row['step'] for row in result['curve']] == list(range(1, 161))
        assert all(math.isfinite(row['operator_step_norm']) and row['operator_step_norm'] > 0 for row in result['curve'])
        assert len(c.read(parent/'private/results/M1'/f'{owner}.json')['curve']) == 160
        calibration = c.read(parent/'private/calibration'/f'{owner}.json')
        assert calibration['fraction'] == .1 and not calibration['heldout_used']
        assert math.isfinite(calibration['C_operator_penalty']) and calibration['C_operator_penalty'] > 0
        sources = r.s.source(task)
        assert len(sources) == 7
        for row in sources:
            name = Path('private/outputs/C')/owner/(row['query_id']+'.json')
            actual, prior = c.read(parent/name), c.read(r.SCOPE/name)
            assert actual['binding'] == prior['binding'] and actual['R0'] == prior['R0']
            outputs += 1
    assert outputs == 56
    return dict(status='PASS', candidates=8, exact_source_outputs=56,
                exact_retained_final_tensors=receipt['exact_retained_final_tensors'],
                inherited=True, C_retrained=False, parent_receipt=receipt,
                source_binding=c.read(run/'private/SOURCE_COMMIT.json'),
                retained_lock=lock, seeds=[t['seed'] for t in tasks])


def require_reference():
    expected = verify_reference()
    assert r.c.read(r.RUN/'private/REFERENCE_EXACT.json') == expected, 'Missing or stale inherited exact reference admission'
    return expected


if __name__ == '__main__':
    receipt = verify_reference()
    path = r.RUN/'private/REFERENCE_EXACT.json'
    assert not path.exists(), 'No implicit admission replacement'
    r.c.write(path, receipt)
    print('INHERITED_REFERENCE_PASS', receipt['exact_source_outputs'])
