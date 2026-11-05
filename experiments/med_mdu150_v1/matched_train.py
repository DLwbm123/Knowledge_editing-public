"""Matched ordinary CE / D-U training from MED states, with frozen inputs.

The caller must independently establish role admission, token bindings, runtime
identity and resource protection; this function is not a launch admission.
"""
import copy
from pathlib import Path

import torch
from robust_objective import view_loss, weights_and_value
from span_train import terms
from train_pair import save_new, state_digest


def train(runtime, r, task, rows, bindings, admission, lock, arm, root, budget, expected_initial, initial_state):
    owner = task['order']
    if admission.get('status') != 'SOURCE_49_INPUTS_CHECKED' or len(admission.get('owners', [])) != 49:
        raise ValueError('101-source admission missing')
    if owner not in admission['owners'] or len(set(admission['owners'])) != 49 or any(not 1 <= x <= 150 for x in admission['owners']):
        raise ValueError('Owner not admitted')
    if arm not in ('MCE', 'MDU') or lock['order'] != owner or len(lock['views']) != 5:
        raise ValueError('Arm or source identity differs')
    if lock['status'] != 'TOKEN_BOUND':
        raise ValueError('Bound universal span lock required')
    if len(rows) != 5 or [x['audit_role'] for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Source/FIT roles differ')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Backbone must remain frozen')
    n = lock['answer_tokens']
    expected = [(0, n)] if n == 1 else [(0, n // 2), (n // 2, n)]
    if n < 1 or [tuple(x) for x in lock['spans']] != expected:
        raise ValueError('Universal answer partition differs')
    from m3bench_repro.editors.llava_runtime import seed_everything
    path = 'model.layers.21.mlp.down_proj'
    seed_everything(task['seed'])
    initial = {k: v.detach().cpu().clone() for k, v in initial_state.items()}
    if sum(v.numel() for v in initial.values()) != 7168 or any(not torch.isfinite(v).all() for v in initial.values()):
        raise ValueError('MED expert state invalid')
    if state_digest(initial) != expected_initial:
        raise ValueError('MED initial state differs from owner binding')
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, bindings, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, x[1]) for x in prepared]
    versions = {name: p._version for name, p in runtime.model.named_parameters()}
    seed_everything(task['seed'])
    expert = r.p.expert(copy.deepcopy(initial), task['seed'], runtime.device)
    if state_digest(expert.state_dict()) != state_digest(initial):
        raise ValueError('Initial state differs')
    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(path), expert)
    hook.attach()
    try:
        optimizer = r.p.optimizer(expert, runtime.model)
        curve = []
        for step in range(1, 161):
            budget()
            chosen = (0, 1 + (step - 1) % 4)
            def calculate(i):
                view = lock['views'][i]
                spans = lock['spans']
                return terms(runtime, hook, batches[i], rows[i], n, spans, view)
            expert.zero_grad(set_to_none=True)
            with torch.no_grad():
                first = [calculate(i) for i in chosen]
            if arm == 'MDU':
                weights, robust = weights_and_value(first[0][1], first[1][1], 1.0)
            else:
                weights = torch.full((2, len(lock['spans'])), 0.5, device=first[0][1].device)
                robust = torch.stack((first[0][1], first[1][1])).mean()
            for j, i in enumerate(chosen):
                ce, spans = calculate(i)
                if not torch.allclose(ce.detach(), first[j][0], atol=1e-5, rtol=1e-5) or not torch.allclose(
                        spans.detach(), first[j][1], atol=1e-5, rtol=1e-5):
                    raise ValueError('View recomputation changed')
                view_loss(ce, spans, weights[j], 0.0 if arm == 'MCE' else 1.0).backward()
                r.COUNTS['backwards'] += 1
            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            if not torch.isfinite(norm) or not torch.isfinite(robust):
                raise ValueError('Nonfinite training state')
            before = r.old.snapshot(expert) if step == 1 else None
            optimizer.step()
            r.COUNTS['updates'] += 1
            if before is not None and all(torch.equal(v, before[k]) for k, v in r.old.snapshot(expert).items()):
                raise ValueError('First update did not change state')
            curve.append({'step': step, 'ce': [float(v[0]) for v in first],
                          'robust': float(robust), 'grad_norm': float(norm)})
        final = r.old.snapshot(expert)
        target = Path(root) / f'{owner}_{arm}.pt'
        save_new(target, {'state': final, 'initial_digest': state_digest(initial),
                          'module': path, 'steps': 160, 'arm': arm, 'curve': curve})
        loaded = torch.load(target, map_location='cpu', weights_only=True)
        if state_digest(loaded['state']) != state_digest(final):
            raise ValueError('Saved state differs')
        active, tokens = r.q.logits(runtime, hook, batches[0])
        expert.load_state_dict(loaded['state'])
        restored, tokens2 = r.q.logits(runtime, hook, batches[0])
        if not torch.equal(active, restored) or not torch.equal(tokens, tokens2):
            raise ValueError('Reload output differs')
        if any(p._version != versions[name] for name, p in runtime.model.named_parameters()):
            raise ValueError('Backbone changed')
        return {'owner': owner, 'arm': arm, 'steps': 160, 'final_digest': state_digest(final)}
    finally:
        hook.detach()
