"""Produce one owner's frozen source distributions; caller must enforce admission."""
import copy
import math
import torch
import teacher_cache
from train_pair import save_new, state_digest


def produce(runtime, r, task, rows, config, teacher_state, expected_digest, destination, byte_limit, budget):
    if [x['audit_role'] for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Unexpected source roles')
    if state_digest(teacher_state) != expected_digest:
        raise ValueError('Original teacher state differs from the edit binding')
    if runtime.model.training or any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Teacher base must be frozen in evaluation mode')
    module = runtime.get_module('model.layers.21.mlp.down_proj')
    if (module.in_features, module.out_features) != (14336, 4096):
        raise ValueError('Teacher layer dimensions differ')
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, {'configuration_only': config}, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
    expert = r.p.expert(copy.deepcopy(teacher_state), task['seed'], runtime.device).requires_grad_(False)
    if state_digest(expert.state_dict()) != expected_digest:
        raise ValueError('Loaded teacher state changed')
    hook = r.q.scoped.replay.MedTraceLayerHook(module, expert); hook.attach()
    try:
        budget()
        active, targets = r.q.logits(runtime, hook, batches[0])
        hook.enabled = False
        try:
            with torch.no_grad():
                output = runtime.model(**batches[0].forward_kwargs())
                mask = batches[0].labels[:, 1:] != -100
                base = output.logits[:, :-1][mask].detach().cpu().double()
                base_targets = batches[0].labels[:, 1:][mask].cpu()
        finally:
            hook.enabled = True
        if not torch.equal(targets, base_targets):
            raise ValueError('Teacher and base target positions differ')
        response = float((active - base).norm())
        if not math.isfinite(response) or response == 0:
            raise ValueError('Teacher activation is absent or nonfinite')
        cache, tensor_bytes = teacher_cache.build(runtime, hook, batches, byte_limit, budget)
        save_new(destination, dict(owner=task['order'], teacher_state_digest=expected_digest, cache=cache))
        return dict(owner=task['order'], teacher_state_digest=expected_digest, tensor_bytes=tensor_bytes,
                    cache_digest=teacher_cache.cache_digest(cache), activation_norm=response)
    finally:
        hook.detach()
