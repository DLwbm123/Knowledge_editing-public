"""Train the D objective through the original MED constrained step."""
import math
from pathlib import Path
import torch
from prefix_cache import PrefixCache
import training_math
import med133_curvature as curvature
import d_gradient
from train_pair import save_new, state_digest


def train(runtime, r, task, rows, config, admission, lock, cache, calibration, output, budget):
    owner = task['order']
    if admission.get('status') != 'TRAINING_SOURCE_ROLE_ADMITTED' or owner not in admission['owners']:
        raise ValueError('Training owner not admitted')
    if [v['audit_role'] for v in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Source roles differ')
    if lock['order'] != owner or lock['status'] != 'WEAK_TOKEN_MATCH':
        raise ValueError('Span identity mismatch')
    coefficient, targets = calibration['coefficient'], calibration['targets']
    if not math.isfinite(coefficient) or coefficient <= 0 or len(targets) != 160:
        raise ValueError('Original MED calibration invalid')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Backbone must remain frozen')
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(task['seed'])
    initial = r.p.tr.TT4(task['seed'], 8, 8).state_dict()
    if sum(v.numel() for v in initial.values()) != 7168 or torch.count_nonzero(initial['G1']):
        raise ValueError('Original initialization differs')
    initial = r.mm.canonicalize(initial)[0]
    expert = r.p.expert(initial, task['seed'], runtime.device)
    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module('model.layers.21.mlp.down_proj'), expert)
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, {'configuration_only': config}, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, p[1]) for p in prepared]
    old_penalty = r.mm.operator_penalty
    r.mm.operator_penalty = lambda state, c: curvature.penalty(state, c, r.mm.factors)
    hook.attach()
    try:
        scalar = dict(step=0, second_moments=[0., 0.])
        curve = []
        with PrefixCache(runtime, batches):
            for step in range(1, 161):
                budget()
                before = r.old.snapshot(expert)
                _, losses, robust = d_gradient.source_gradients(r, runtime, hook, expert, batches, rows, lock, step)
                gradients = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                selected, scalar, audit = training_math.step(r, before, gradients, cache, coefficient, scalar, targets[step-1], arm='M13P')
                expert.load_state_dict(selected)
                r.COUNTS['updates'] += 1
                r.COUNTS['operator_backwards'] += audit['operator_backwards']
                curve.append(dict(step=step, source_nll=losses, robust=robust, **audit))
        final = r.old.snapshot(expert)
        path = Path(output) / f'{owner}_D.pt'
        save_new(path, dict(state=final, module='model.layers.21.mlp.down_proj', steps=160, curve=curve))
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        if state_digest(loaded['state']) != state_digest(final):
            raise ValueError('Saved expert state mismatch')
        active, tokens = r.q.logits(runtime, hook, batches[0])
        expert.load_state_dict(loaded['state'])
        restored, restored_tokens = r.q.logits(runtime, hook, batches[0])
        if not torch.equal(active, restored) or not torch.equal(tokens, restored_tokens):
            raise ValueError('Reload output mismatch')
        return dict(owner=owner, steps=160, calibration_bound=True)
    finally:
        hook.detach()
        r.mm.operator_penalty = old_penalty
