"""Small deployment regressions: schedule, shared caps, evaluation between edits."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import launch
from .executor import Budget, atomic_json
from .cpu_check import Runtime, make_record
from .worker import run_current
from .configuration import CAPS


def run():
    torch.set_num_threads(2)
    schedule = list(launch.plan())
    assert len(schedule) == 43
    assert max(i for i, (_, action, _) in enumerate(schedule) if action == 'admit') < min(i for i, (_, action, _) in enumerate(schedule) if action == 'learn')
    for arm in ('U', 'P'):
        assert [o for a, k, o in schedule if a == arm and k == 'learn'] == list(range(1, 9))
        assert [o for a, k, o in schedule if a == arm and k == 'prefix'] == [4, 8]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        folder = root/'private/fixed8/U'
        atomic_json(root/'RESOURCE_LEDGER.json', dict(gpu_sessions=[]))
        common = SimpleNamespace(read=lambda p: json.loads(Path(p).read_text()), budget=lambda: None)
        before_check, before_consume = Budget.check, Budget.consume
        try:
            with patch.object(launch, 'helpers', return_value=(common, root)):
                launch.install_budget('U')
                calibration = torch.zeros(32, 14336)
                first = run_current(Runtime(torch.float16), 'U', calibration, make_record(1), ['f0','f1','f2','f3'], 1, folder, 101)
                evaluation = Budget(folder/'evaluation-budget.json')
                evaluation.consume('forwards')
                evaluation.consume('generation_attempts')
                atomic_json(folder/'CONSUMED.json', dict(owner=1, insertion_complete=True, prefix_complete=False))
                second = run_current(Runtime(torch.float16), 'U', calibration, make_record(2), ['f0','f1','f2','f3'], 2, folder, 102)
                assert first['slots'] == second['slots'] == 20 and second['counters']['attempt_slots'] == 40
                edit = Budget(folder/'budget.json')
                evaluation.data['counts']['forwards'] = CAPS['forwards']-edit.data['counts']['forwards']
                atomic_json(evaluation.path, evaluation.data)
                try:
                    edit.consume('forwards')
                except TimeoutError:
                    pass
                else:
                    raise AssertionError('Combined cap allowed overspend')
                assert json.loads(edit.path.read_text()) == edit.data
        finally:
            Budget.check, Budget.consume = before_check, before_consume
    return dict(status='PASS', stages=43, both_admissions_before_learning=True,
                independent_eval_ledger_preserves_exact_optimizer_resume=True, combined_cap_checked_before_call=True)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
