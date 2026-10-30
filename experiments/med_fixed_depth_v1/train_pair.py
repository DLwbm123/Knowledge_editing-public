"""Source-only paired TT training; the caller supplies admitted inputs and runtime."""
import copy
import hashlib
import math
from pathlib import Path

import torch
import trajectory


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    temporary.replace(path)


def train_pair(runtime, r, task, source_rows, bindings, admission, root, budget):
    """Train 2x160 updates, leaving evaluation inputs and reference scores outside this process."""
    owner = task['order']
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or owner not in admission.get('owners', []):
        raise ValueError('Fixed owner and input roles must be independently admitted')
    if len(source_rows) != 5 or [x['audit_role'] for x in source_rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Expected exactly the original source and four FIT inputs')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Base must be frozen')
    layers = runtime.model.model.layers
    if len(layers) < 3:
        raise ValueError('Invalid decoder depth')
    paths = ('model.layers.21.mlp.down_proj', f'model.layers.{len(layers)-2}.mlp.down_proj')
    if paths[0] == paths[1]:
        raise ValueError('Fixed layers coincide')
    modules = [runtime.get_module(path) for path in paths]
    if any((m.in_features, m.out_features) != (modules[0].in_features, modules[0].out_features) for m in modules):
        raise ValueError('Fixed layer module dimensions differ')
    if (modules[0].in_features, modules[0].out_features) != (14336, 4096):
        raise ValueError('Unexpected TT input or output dimensions')
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(task['seed'])
    initial = r.p.tr.TT4(task['seed'], 8, 8).state_dict()
    if sum(t.numel() for t in initial.values()) != 7168 or torch.count_nonzero(initial['G1']) != 0:
        raise ValueError('Unexpected initial TT structure or nonzero function')
    initial = {k: v.detach().cpu().clone() for k, v in initial.items()}
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, bindings, task, source_rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
    versions = {name: p._version for name, p in runtime.model.named_parameters()}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for arm, path in zip(('C21', 'CDEEP'), paths):
        budget()
        seed_everything(task['seed'])
        expert = r.p.expert(copy.deepcopy(initial), task['seed'], runtime.device)
        if state_digest(expert.state_dict()) != state_digest(initial):
            raise ValueError('Initial states differ')
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(path), expert)
        hook.attach()
        try:
            for batch in batches:
                budget()
                r.q.logits(runtime, hook, batch, verify_base=True)
            optimizer = r.p.optimizer(expert, runtime.model)
            curve = []
            for step in range(1, 161):
                budget()
                before = r.old.snapshot(expert)
                _, losses = trajectory.source_gradients(r, runtime, hook, expert, batches, step)
                norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
                r.COUNTS['updates'] += 1
                after = r.old.snapshot(expert)
                if step == 1 and all(torch.equal(after[k], before[k]) for k in before):
                    raise ValueError('First optimizer step did not update the expert')
                delta = math.sqrt(sum(float((after[k].double() - before[k].double()).square().sum()) for k in before))
                curve.append(dict(step=step, source_nll=losses, grad_norm=float(norm), update_norm=delta))
            final = r.old.snapshot(expert)
            budget()
            active, target_tokens = r.q.logits(runtime, hook, batches[0])
            hook.enabled = False
            try:
                base, base_tokens = r.q.logits(runtime, hook, batches[0])
            finally:
                hook.enabled = True
            if not torch.equal(target_tokens, base_tokens):
                raise ValueError('Token bindings differ')
            response_norm = float((active - base).norm())
            if not math.isfinite(response_norm) or response_norm == 0:
                raise ValueError('No finite expert response after training')
            target = root / f'{owner}_{arm}.pt'
            save_new(target, dict(state=final, initial_digest=state_digest(initial), module=path, steps=160, curve=curve))
            loaded = torch.load(target, map_location='cpu', weights_only=True)
            if any(not torch.equal(v.cpu(), loaded['state'][k]) for k, v in final.items()):
                raise ValueError('Save/reload mismatch')
            expert.load_state_dict(loaded['state'])
            restored, restored_tokens = r.q.logits(runtime, hook, batches[0])
            if not torch.equal(active, restored) or not torch.equal(target_tokens, restored_tokens):
                raise ValueError('Reload changed model output')
            results.append(dict(arm=arm, module=path, initial_digest=state_digest(initial), final_digest=state_digest(final),
                                parameters=sum(v.numel() for v in final.values()), steps=len(curve),
                                response_norm=response_norm, save_reload_output_equal=True))
        finally:
            hook.detach()
            del expert, hook
    if any(p._version != versions[name] for name, p in runtime.model.named_parameters()):
        raise ValueError('Base parameter version changed')
    return results
