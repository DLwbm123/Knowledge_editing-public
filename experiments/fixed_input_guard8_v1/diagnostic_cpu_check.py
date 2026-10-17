"""Synthetic CPU checks: unchanged learner trajectory and actual diagnostic consumer."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from . import diagnostic as d, launch
from .cpu_check import Runtime, make_record, exact, record_type
from .worker import run_current
from .executor import atomic_json, Budget


def run():
    torch.set_num_threads(2)
    z = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
    m = d.token_metrics(z, torch.tensor([0, 1]), 2)
    assert m['target_margin'] == [1., -1.] and m['eos_minus_target'] == [-2., 1.]
    assert d.generation_metrics([1]*12, [1, 2], 12)['most_frequent_4gram_count'] == 9
    assert d.generation_metrics([1], [1, 2], 12)['first_divergence_from_source6'] == 1
    delta, h = torch.randn(8, 17), torch.randn(70, 17)
    assert torch.allclose(d.full_response(delta, h, 'cpu'), h @ delta.T, atol=1e-5)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        calibration = torch.zeros(32, 14336)
        final = []
        for enabled in (False, True):
            folder = root/str(enabled)
            for owner in range(1, 9):
                run_current(Runtime(torch.float16), 'P', calibration, make_record(owner), ['f0','f1','f2','f3'],
                            owner, folder, 100+owner, diagnostic=enabled)
                atomic_json(folder/'CONSUMED.json', dict(owner=owner, insertion_complete=True, prefix_complete=owner in (4, 8)))
            final.append(torch.load(folder/'active.pt', weights_only=True, map_location='cpu'))
        # Runtime cost timestamps differ; all scientific state and counters must be exact.
        for state in final:
            state['budget'].pop('start_epoch')
        assert exact(*final), 'Diagnostic export changed learner/Adam/RNG/guard/counters'
        folder = root/'True'
        accepted = [(o, s) for o in (7, 8) for s in range(1, 21) if json.loads((folder/'steps'/str(o)/f'{s}.json').read_text())['accepted']]
        assert len(list((folder/'diagnostic_weights').glob('*.pt'))) == 1+len(accepted)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        folder = root/'private/fixed8/P'
        runtime = Runtime(torch.float16)
        runtime.adapter = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7))
        runtime.generation_config = dict(max_new_tokens=12)
        state = runtime.model.layer.weight.detach().clone()
        for owner, slot, change in ((6, 20, 0.), (7, 1, .1), (8, 2, .2)):
            d.save_snapshot(state+change, folder, owner, slot)
            atomic_json(folder/'steps'/str(owner)/f'{slot}.json', dict(accepted=True))
        row = dict(dataset='SYNTHETIC', question='source 6', reference='6', image_path='/synthetic/input', original_image_path='/synthetic/input')
        atomic_json(root/'private/BENCHMARK146_QUEUE.json', dict(tasks=[{}]*5+[dict(edit_id='6', native=row, fit_questions=['fit'])]))
        ledger = dict(gpu_sessions=[dict(pid=os.getpid(), started_epoch=0, arm='P')])
        atomic_json(root/'RESOURCE_LEDGER.json', ledger)
        @contextlib.contextmanager
        def resources():
            yield ledger
        c = SimpleNamespace(read=lambda p: json.loads(Path(p).read_text()), lease=lambda _: contextlib.nullcontext(),
            resources=resources, load=lambda _: (runtime, {}))
        def generate(runtime, bindings, row, destination, phase, budget):
            budget.consume('generation_attempts')
            ids = [1, 2] if phase['owner'] == 6 else [1]*12
            value = dict(R0=dict(raw_token_ids=ids, raw_answer='synthetic'))
            atomic_json(destination, value)
            return value
        module = SimpleNamespace(EditorRecord=record_type())
        with patch.dict(sys.modules, {'m3bench_repro.editors.llava_runtime': module}), \
             patch.object(launch, 'helpers', return_value=(c, root)), \
             patch.object(launch, 'install_budget', return_value=Budget), \
             patch.object(launch, 'image_path', side_effect=Path), patch.object(launch, 'generate', side_effect=generate):
            for owner in (6, 7, 8):
                d.evaluate(owner, dict(source6_diagnostic=True, physical_GPU=7, implementation_commit='CPU'))
        output = root/'private/evaluation/source6_diagnostic'
        assert not list((folder/'diagnostic_weights').glob('*.pt'))
        assert torch.count_nonzero(torch.load(output/'6_20/accepted_increment_deltaWh.pt', weights_only=True)) == 0
        assert torch.count_nonzero(torch.load(output/'7_01/accepted_increment_deltaWh.pt', weights_only=True)) > 0
        assert c.read(output/'8_02/METRICS.json')['generation']['cap_hit']
        assert c.read(output/'8_02/METRICS.json')['generation']['first_divergence_from_source6'] == 1
        assert c.read(output/'COMPLETE_8.json')['counters']['generation_attempts'] == 3
        assert c.read(output/'FIRST_CHANGE.json')['first_generation_divergence'] == dict(owner=7, slot=1)
        assert c.read(output/'FIRST_CHANGE.json')['first_new_cap_hit'] == dict(owner=7, slot=1)
    original_plan, original_aux, original_arms = launch.plan, launch.auxiliary, launch.ARMS
    try:
        d.install()
        schedule = list(launch.plan())
        assert all(a == 'P' for a, _, _ in schedule)
        assert [(k, o) for _, k, o in schedule if k == 'diagnostic'] == [('diagnostic', 6), ('diagnostic', 7), ('diagnostic', 8)]
        assert len(schedule) == 25
    finally:
        launch.plan, launch.auxiliary, launch.ARMS = original_plan, original_aux, original_arms
    return dict(status='PASS', synthetic_only=True, exact_P160_slot_trajectory_with_without_diagnostics=True,
        exact_native_master_optimizer_guard_rng_counters=True, diagnostic_snapshots_match_accepted_slots=True,
        actual_evaluator_6_7_8_full_context_token_generation_and_retirement=True,
        token_margin_sign_and_repetition_and_full_response_math=True, P_only_schedule_stages=25,
        real_model_calls=0, GPU_calls=0)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
