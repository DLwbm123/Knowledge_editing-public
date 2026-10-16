"""Actual executor/Adam/hook/checkpoint paths on a synthetic native CPU runtime."""
import ast
import copy
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from .configuration import VERSION, LAYER, CAPS, schedule, evaluation_panel
from .executor import Editor, Budget, rng_state, atomic_json
from .adapter import h_loss
from .worker import run_current, main, calibration_inputs, validate_calibration, load_native, CooperativeStop, admit_current


ROOT = Path(__file__).resolve().parents[2]


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def record_type():
    path = ROOT/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source/m3bench_repro/editors/llava_runtime.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'EditorRecord')
    scope = dict(dataclass=dataclass, Path=Path, Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['EditorRecord']


def make_record(owner):
    return record_type()(str(owner), 'SYNTHETIC', 'source '+str(owner), str(owner % 8), 'fit',
                         Path('/synthetic/input'), '/synthetic/input', owner, 'CPU_ONLY')


class Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.layer = torch.nn.Linear(14336, 8, bias=False, dtype=dtype)
        gen = torch.Generator().manual_seed(700)
        with torch.no_grad():
            self.layer.weight.copy_(torch.randn(self.layer.weight.shape, generator=gen)*.02)
        self.other = torch.nn.Parameter(torch.ones(2, dtype=dtype), requires_grad=False)

    def forward(self, inputs_embeds):
        return SimpleNamespace(logits=self.layer(inputs_embeds))


class Runtime:
    def __init__(self, dtype):
        self.model = Model(dtype)

    def get_module(self, name):
        assert name == LAYER
        return self.model.layer

    def build_edit_batch(self, record):
        gen = torch.Generator().manual_seed(1000+record.formal_sequence_position)
        source = torch.randn(14336, generator=gen)*.1
        noise = torch.Generator().manual_seed(sum(map(ord, record.question)))
        x = source[None, None].repeat(1, 4, 1)+torch.randn((1, 4, 14336), generator=noise)*.0001
        x = x.to(self.model.layer.weight.dtype)
        labels = torch.tensor([[-100, int(record.target), int(record.target), int(record.target)]])
        return SimpleNamespace(labels=labels, forward_kwargs=lambda: dict(inputs_embeds=x))


def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(16)
    def forbidden(*a, **k):
        raise AssertionError('No GPU/model-loader call authorized')
    torch.cuda.init = torch.cuda._lazy_init = forbidden
    checks = {}
    old = ROOT/'experiments/purew_he_hm146_v1/source_loss.py'
    names = ('split_logits', 'teacher', 'components')
    nodes = [n for n in ast.parse(old.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, F=F, math=math, ALPHA=.95)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(old), 'exec'), scope)
    z = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    y = torch.arange(7)
    anchor = scope['teacher'](z.detach()+.2, y, torch.zeros(2, 11))
    expected = scope['components'](z, y, torch.zeros(2, 11), anchor)['odds_hinge']
    actual = h_loss(z, y, anchor['log_a']-anchor['log_rest'])
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0)
    assert torch.allclose(torch.autograd.grad(actual, z, retain_graph=True)[0],
                          torch.autograd.grad(expected, z)[0], atol=1e-12, rtol=0)
    checks['original_H_value_and_gradient'] = True
    try:
        main(dict(version=VERSION, authorization={}))
    except PermissionError as error:
        assert str(error) == 'PREPARATION_ONLY'
    else:
        raise AssertionError('Real execution admitted')
    checks['entry_denied_before_runtime_import'] = True
    with tempfile.TemporaryDirectory(prefix='e.') as temporary:
        folder = Path(temporary)
        records = [make_record(i) for i in range(101, 133)]
        manifest = dict(role='CALIBRATION_TRAIN', calibration_ids=[r.record_id for r in records],
                        calibration_source_groups=['train'], edit_and_eval_ids=['eval'],
                        edit_and_eval_source_groups=['heldout'])
        calibration = calibration_inputs(Runtime(torch.float16), records, manifest, Budget(folder/'calibration.json'))
        assert calibration.shape == (32, 14336)
        leak = dict(manifest, edit_and_eval_source_groups=['train'])
        try:
            validate_calibration(leak)
        except ValueError:
            pass
        else:
            raise AssertionError('Calibration leakage admitted')
        checks['actual_activation_hook_and_calibration_separation'] = True
        admission = admit_current(Runtime(torch.float16), 'P', calibration, make_record(1), ['f0', 'f1', 'f2', 'f3'], folder/'admission')
        assert admission['status'] == 'PASS'
        checks['native_admission_roundtrip_restores_original_optimizer_and_weight'] = True
        summaries = []
        for dtype in (torch.float16, torch.bfloat16):
            for arm in ('U', 'P'):
                run = folder/(str(dtype)+'_'+arm)
                accepted = qualified = 0
                for owner in range(1, 9):
                    runtime = Runtime(dtype)
                    other = runtime.model.other.clone()
                    result = run_current(runtime, arm, calibration, make_record(owner), ['f0', 'f1', 'f2', 'f3'], owner, run, 100+owner)
                    accepted += result['accepted']
                    qualified += result['summary']['qualified']
                    state = torch.load(run/'active.pt', weights_only=True)
                    assert state['owner'] == owner and state['boundary'] and state['slot'] == 20
                    assert state['native'].dtype == dtype and exact(state['master'], state['native'].float())
                    assert int(state['optimizer']['state'][0]['step']) == accepted
                    assert int(state['guard']['counts'][3]) == (qualified if arm == 'P' else 0)
                    assert exact(runtime.model.other, other)
                    assert len(runtime.model._forward_pre_hooks) == 0 and len(runtime.model.layer._forward_pre_hooks) == 0
                    # Synthetic evaluation consumer acknowledgement; no real scoring occurs here.
                    atomic_json(run/'CONSUMED.json', dict(owner=owner, insertion_complete=True, prefix_complete=owner in (4, 8)))
                assert accepted > 0 and qualified > 0
                assert result['counters']['attempt_slots'] == 160
                fresh = Runtime(dtype)
                load_native(fresh, run/'native.pt', arm, 8)
                with torch.no_grad():
                    for owner in (1, 8):
                        x = fresh.build_edit_batch(make_record(owner)).forward_kwargs()
                        assert torch.equal(runtime.model(**x).logits, fresh.model(**x).logits)
                summaries.append(dict(arm=arm, dtype=str(dtype), slots=160, accepted=accepted,
                                      qualified_sources=qualified, checkpoint_bytes=(run/'active.pt').stat().st_size,
                                      protection_bytes=result['summary']['protection_bytes'], exact_reload_logits=2))
                print('Integrated 8-edit path passed:', arm, dtype, flush=True)
        checks['eight_edit_real_worker_path'] = summaries
        atomic_json(folder/'base_teacher_budget.json', state['budget'])
        base_check = Editor(Runtime(torch.bfloat16), 'P', calibration, Budget(folder/'base_teacher_budget.json'))
        base_check.load(run/'active.pt')
        # Exercise the same Base switch used by prepare_source, at a valid new source boundary.
        base_check.owner = 7
        native_before = base_check.weight.detach().clone()
        base_check.prepare_source(make_record(8), ['f0', 'f1', 'f2', 'f3'], 8)
        fresh_check = Editor(Runtime(torch.bfloat16), 'P', calibration, Budget(folder/'base_teacher_reference.json'))
        fresh_check.owner = 7
        fresh_check.prepare_source(make_record(8), ['f0', 'f1', 'f2', 'f3'], 8)
        assert exact(base_check.anchors, fresh_check.anchors) and exact(native_before, base_check.weight)
        base_check.adapter.close(); fresh_check.adapter.close()
        checks['H_teacher_uses_original_Base_after_trajectory_restore'] = True
        calls = [0]
        def stop_after_three():
            calls[0] += 1
            return calls[0] == 3
        partial = folder/'cooperative'
        try:
            run_current(Runtime(torch.float16), 'P', calibration, make_record(1), ['f0', 'f1', 'f2', 'f3'],
                        1, partial, 101, stop_requested=stop_after_three)
        except CooperativeStop:
            pass
        else:
            raise AssertionError('Cooperative stop was ignored')
        state = torch.load(partial/'active.pt', weights_only=True)
        assert state['slot'] == 3 and not state['boundary']
        resumed_run = run_current(Runtime(torch.float16), 'P', calibration, make_record(1), ['f0', 'f1', 'f2', 'f3'],
                                  1, partial, 101, resume=True)
        assert resumed_run['slots'] == 17 and resumed_run['counters']['attempt_slots'] == 20
        checks['production_worker_cooperative_stop_and_resume_no_replay'] = True
        try:
            run_current(Runtime(torch.float16), 'P', calibration, make_record(2), ['f0', 'f1', 'f2', 'f3'], 2, partial)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Unconsumed native checkpoint overwritten')
        assert torch.load(partial/'active.pt', weights_only=True)['owner'] == 1
        checks['consumer_barrier_prevents_next_edit'] = True
        editor = Editor(Runtime(torch.float16), 'P', calibration, Budget(folder/'transactions.json'))
        record, fits = make_record(1), ['f0', 'f1', 'f2', 'f3']
        editor.prepare_source(record, fits, 1)
        assert editor.attempt(1)['accepted']
        before = editor.state()
        call = [0]
        objective = editor.adapter.objective
        def reject(*args):
            call[0] += 1
            random.random(); np.random.rand(); torch.rand(1)
            value = objective(*args)
            return value if call[0] == 1 else value+100
        with patch.object(editor.adapter, 'objective', reject):
            rejected = editor.attempt(2)
        assert not rejected['accepted'] and rejected['reason'] == 'ACTUAL_SOURCE_ARMIJO_REJECTED'
        after = editor.state()
        for key in ('native', 'master', 'optimizer', 'guard', 'rng'):
            assert exact(before[key], after[key]), key
        assert after['budget']['counts']['attempt_slots'] == before['budget']['counts']['attempt_slots']+1
        checks['rejected_candidate_rolls_back_adam_native_master_rng_not_cost'] = True
        before = editor.state()
        call = [0]
        def explode(*args):
            call[0] += 1
            if call[0] == 2:
                raise RuntimeError('injected post-write failure')
            return objective(*args)
        with patch.object(editor.adapter, 'objective', explode):
            try:
                editor.attempt(3)
            except RuntimeError as error:
                assert str(error) == 'injected post-write failure'
            else:
                raise AssertionError('Missing injected candidate exception')
        after = editor.state()
        for key in ('native', 'master', 'optimizer', 'guard', 'rng'):
            assert exact(before[key], after[key]), key
        checks['post_native_write_exception_rolls_back_transaction'] = True
        before = editor.state()
        with patch.object(editor.guard, 'write', return_value=dict(accepted=False, reason='NATIVE_PROJECTION_REJECTED')):
            assert not editor.attempt(3)['accepted']
        after = editor.state()
        for key in ('native', 'master', 'optimizer', 'guard', 'rng'):
            assert exact(before[key], after[key]), key
        checks['native_certificate_rejection_rolls_back_optimizer'] = True
        before = editor.state()
        original_step = editor.optimizer.step
        def optimizer_failure():
            original_step()
            raise RuntimeError('injected optimizer failure')
        with patch.object(editor.optimizer, 'step', optimizer_failure):
            try:
                editor.attempt(4)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Optimizer exception not propagated')
        for key in ('native', 'master', 'optimizer', 'guard', 'rng'):
            assert exact(before[key], editor.state()[key]), key
        checks['partially_mutated_optimizer_exception_is_atomic'] = True
        checkpoint = folder/'resume.pt'
        editor.save(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        reference_event = editor.attempt(4)
        reference = editor.state()
        atomic_json(folder/'resume_budget.json', saved['budget'])
        resumed = Editor(Runtime(torch.float16), 'P', calibration, Budget(folder/'resume_budget.json'))
        resumed.load(checkpoint)
        resumed.attach_source(record, fits)
        resumed_event = resumed.attempt(4)
        assert exact(reference_event, resumed_event)
        for key in ('native', 'master', 'optimizer', 'guard', 'rng'):
            assert exact(reference[key], resumed.state()[key]), key
        checks['mid_source_resume_exact_next_update'] = True
        def bad_save(obj, handle):
            handle.write(b'incomplete')
            raise OSError('injected save failure')
        with patch('torch.save', bad_save):
            try:
                editor.save(checkpoint)
            except OSError:
                pass
            else:
                raise AssertionError('Injected save failed to fail')
        assert exact(torch.load(checkpoint, weights_only=True), saved)
        assert not checkpoint.with_suffix('.tmp').exists()
        checks['atomic_save_failure_preserves_previous_checkpoint'] = True
        invalid = resumed.state()
        invalid['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
        torch.save(invalid, folder/'bad.pt')
        before = resumed.state()
        try:
            resumed.load(folder/'bad.pt')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt Adam state admitted')
        assert exact(before, resumed.state())
        checks['corrupt_restore_has_no_partial_mutation'] = True
        editor.adapter.close(); resumed.adapter.close()
    tasks = [dict(order=i, edit_id='e'+str(i), events=[dict(task=name, all_probe_query_ids=[f'{name}-{i}-{j}' for j in range(4)]) for name in ('T1G', 'T2G')]) for i in range(1, 9)]
    protect = ['p'+str(i) for i in range(54)]
    ledger = dict.fromkeys(protect+[t['edit_id'] for t in tasks]+[q for t in tasks for e in t['events'] for q in e['all_probe_query_ids']])
    panels = {p: len(evaluation_panel(tasks, ledger, p, protect)) for p in (0, 4, 8)}
    assert panels == {0: 126, 4: 90, 8: 126}
    assert len([s for s in schedule() if s[1] == 'learn']) == 16
    checks['fixed_schedule_and_panel_upper_bounds'] = panels
    return dict(status='PASS', scope='CPU_SYNTHETIC_ACTUAL_EXECUTOR', checks=checks,
                elapsed_seconds=time.monotonic()-started, model_experiment_started=False,
                native_dtypes=['float16', 'bfloat16'], formal_slots_exercised=640)
