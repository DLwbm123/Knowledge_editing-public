"""Synthetic CPU integration and independent constrained-solver checks."""
import ast
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .editor import (Editor, calibrate, direction, editable_fp32, nll,
                     paired_batches, TRUST_BUDGET)


def batch_type():
    path = Path(__file__).resolve().parents[1] / 'medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'PreparedBatch')
    scope = dict(dataclass=dataclass, torch=torch, Any=Any)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['PreparedBatch']


class Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.layer = torch.nn.Linear(4, 6, bias=False, dtype=dtype)
        self.frozen = torch.nn.Parameter(torch.tensor(.7, dtype=dtype))

    def forward(self, inputs_embeds, **kwargs):
        x = torch.tanh(inputs_embeds.cumsum(1) * self.frozen)
        return SimpleNamespace(logits=self.layer(x))


class Runtime:
    def __init__(self, dtype=torch.float32):
        self.model = Model(dtype)
        self.adapter = SimpleNamespace(prepare_inputs=self.prepare_inputs)
        self.Batch = batch_type()
        self.seen_images = []

    def get_module(self, name):
        assert name == 'model.layers.0.mlp.down_proj'
        return self.model.layer

    def prepare_inputs(self, image_path, question, target):
        return dict(input_ids=torch.tensor([[-200, 1, 2, int(target)]]),
                    attention_mask=torch.ones(1, 4, dtype=torch.long),
                    images=torch.tensor([[[[float(image_path), .4], [-.2, .8]]]],
                                        dtype=self.model.frozen.dtype))

    def _expand_multimodal(self, *, raw_input_ids, attention_mask, labels, images):
        self.seen_images.append(images.clone())
        # A nonzero vision bias makes zero pixels different from zero embeddings.
        pixels = images.reshape(1, 1, 4)
        visual = torch.cat((pixels+.3, pixels*.5+.2), 1)
        text = torch.tensor([[[.2, -.4, .1, .7], [.3, .6, -.7, .1], [.5, .1, .3, -.2]]],
                            dtype=images.dtype)
        return (torch.cat((visual, text), 1), torch.ones(1, 5, dtype=torch.long),
                torch.arange(5)[None], torch.tensor([[-100, -100, -100, -100, int(raw_input_ids[0,-1])]]))

    def build_edit_batch(self, record):
        raw = self.prepare_inputs(record.image_path, record.question, record.target)
        raw_labels = torch.tensor([[-100, -100, -100, int(record.target)]])
        x, mask, positions, labels = self._expand_multimodal(
            raw_input_ids=raw['input_ids'], attention_mask=raw['attention_mask'],
            labels=raw_labels, images=raw['images'])
        return self.Batch(x, mask, positions, labels, raw['input_ids'], raw_labels,
                          (int(record.target),), 4, 2, 'q', 3, 2, 0, 2,
                          list(raw['images'].shape), 'synthetic', 'not-a-file', 0)

    def generate(self, record):
        with torch.no_grad():
            return self.model(**self.build_edit_batch(record).forward_kwargs()).logits.argmax(-1)


def record(key, image=1., target='2'):
    return SimpleNamespace(record_id=key, image_path=image, question='synthetic', target=target)


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Expected rejection')


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU is not authorized')
    original_cuda_init = torch.cuda._lazy_init
    torch.cuda._lazy_init = forbidden
    try:
        # Independent dense KKT solution, with nonidentity anisotropic geometry.
        gv, g0 = torch.randn(3, 4), torch.randn(3, 4)
        a, b = torch.rand(4)+.2, torch.rand(3)+.3
        t, geometry = direction(gv, g0, a, b)
        diagonal = (b[:, None]*a[None, :]).double().flatten()
        constraints = torch.stack((gv.flatten(), g0.flatten())).double()
        inverse = torch.diag(1/diagonal)
        d = min(.1, (2*TRUST_BUDGET*geometry['E'])**.5)
        expected = inverse @ constraints.T @ torch.linalg.solve(
            constraints @ inverse @ constraints.T, torch.tensor([-d, 0.], dtype=torch.float64))
        delta = (-d/geometry['E'])*t.flatten().double()
        assert torch.allclose(delta, expected, atol=1e-7, rtol=1e-5)
        assert abs(float(constraints[1]@delta)) < 1e-7
        assert float((delta.square()*diagonal).sum())/2 <= TRUST_BUDGET*(1+1e-5)
        assert direction(gv, gv, a, b)[0] is None
        assert direction(gv, torch.zeros_like(g0), a, b)[0] is None
        rejects(lambda: direction(gv*float('nan'), g0, a, b))

        rows = []
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            runtime = Runtime(dtype)
            name = 'model.layers.0.mlp.down_proj'
            binding = dict(model_revision='synthetic-v1', dataset_revision='synthetic-v1',
                           preprocessing_revision='synthetic-v1')
            train = [record('train-1', .8, '1'), record('train-2', 1.2, '3')]
            base = {k: v.clone() for k, v in runtime.model.state_dict().items()}
            calibration = calibrate(runtime, train, [name], train_ids={'train-1', 'train-2'},
                                    excluded_ids={'test'}, binding=binding)
            assert all(torch.equal(base[k], v) for k, v in runtime.model.state_dict().items())
            rejects(lambda: calibrate(runtime, train, [name], train_ids={'train-1'},
                                      excluded_ids=set(), binding=binding))
            rejects(lambda: calibrate(runtime, train, [name], train_ids={'train-1', 'train-2'},
                                      excluded_ids={'train-2'}, binding=binding))
            editor = Editor(runtime, calibration, binding=binding)
            rejects(lambda: editor.edit(train[0]))
            full, null = paired_batches(runtime, record('test'))
            assert torch.count_nonzero(runtime.seen_images[-1]) == 0
            assert torch.count_nonzero(null.inputs_embeds[:, :2]) > 0
            assert torch.equal(full.inputs_embeds[:, 2:], null.inputs_embeds[:, 2:])
            assert torch.equal(full.labels, null.labels)
            expand = runtime._expand_multimodal
            def broken_null(**kwargs):
                x, mask, positions, labels = expand(**kwargs)
                if not torch.count_nonzero(kwargs['images']):
                    positions = positions + 1
                return x, mask, positions, labels
            runtime._expand_multimodal = broken_null
            rejects(lambda: paired_batches(runtime, record('test')))
            runtime._expand_multimodal = expand
            logits = runtime.model(**full.forward_kwargs()).logits.float()
            expected_loss = -logits[0, 3].log_softmax(-1)[2]
            assert torch.allclose(nll(runtime.model, full), expected_loss)
            initial = editor.state_dict()
            result = editor.edit(record('test'))
            assert any(s['status'] == 'UPDATED' for s in result['steps'])
            assert len(result['steps']) <= 5
            assert all(s.get('fisher_cost', 0) <= TRUST_BUDGET*(1+1e-6) for s in result['steps'])
            assert len({s['visual_nll'] for s in result['steps']}) > 1
            assert not torch.equal(initial['weight'], runtime.model.layer.weight)
            assert torch.equal(base['frozen'], runtime.model.frozen)
            assert runtime.model.layer.weight.dtype == dtype
            assert runtime.model.training and runtime.model.frozen.requires_grad
            assert all(p.grad is None for p in runtime.model.parameters())
            assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
            state, generated = editor.state_dict(), runtime.generate(record('test'))
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            restored = torch.load(buffer, weights_only=True)
            editor.load_state_dict(initial)
            assert torch.equal(initial['weight'], runtime.model.layer.weight)
            editor.load_state_dict(restored)
            assert torch.equal(generated, runtime.generate(record('test')))
            rejects(lambda: editor.load_state_dict(dict(restored, binding={'wrong': True})))
            # A failing edit restores native weights, modes, flags and hooks.
            original_forward = runtime.model.forward
            calls = [0]
            def fail_after_update(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError('Synthetic failure after first update')
                return original_forward(*args, **kwargs)
            runtime.model.forward = fail_after_update
            rejects(lambda: editor.edit(record('test-2')))
            runtime.model.forward = original_forward
            assert torch.equal(restored['weight'], runtime.model.layer.weight)
            assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
            assert runtime.model.training and runtime.model.layer.weight.dtype == dtype
            # A satisfied request is a true no-op, including native export.
            editor.load_state_dict(initial)
            with torch.no_grad():
                runtime.model.layer.weight.zero_()
                runtime.model.layer.weight[2].fill_(100)
            satisfied = editor.state_dict()
            no_op = editor.edit(record('satisfied'))
            assert no_op['steps'][0]['status'] == 'TARGET_ALREADY_MET'
            assert torch.equal(satisfied['weight'], runtime.model.layer.weight)
            rows.append(dict(dtype=str(dtype), steps=len(result['steps']),
                             native_rounding_l2=result['native_rounding_l2'],
                             native_target_met=result['native_target_met']))
        return dict(status='PASS', synthetic_only=True, solver='dense-KKT agreement',
                    pairing='postprocessing zeros with unchanged structure',
                    calibration='train-only, immutable weights, fixed mean-E selection',
                    lifecycle='native export, generation, restore and exception rollback',
                    precision_cases=rows, real_model_loaded=False, experiment_started=False)
    finally:
        torch.cuda._lazy_init = original_cuda_init
