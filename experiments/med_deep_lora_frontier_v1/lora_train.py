"""Train the fixed LoRA expert on the admitted source and FIT rows."""
import math
from pathlib import Path
import torch
import frontier_lora
import trajectory
from train_pair import save_new, state_digest


def train(runtime, r, task, rows, bindings, admission, root, budget):
    owner = task['order']
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or owner not in admission['owners']:
        raise ValueError('Owner not admitted')
    if len(rows) != 5 or [x['audit_role'] for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Expected source and four FIT rows')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Base must be frozen')
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(task['seed'])
    path = f'model.layers.{len(runtime.model.model.layers)-2}.mlp.down_proj'
    if path != 'model.layers.30.mlp.down_proj':
        raise ValueError('Unexpected model depth')
    expert = frontier_lora.create(runtime.get_module(path), runtime.device)
    if torch.count_nonzero(expert.B):
        raise ValueError('Initial expert is not zero')
    initial = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, bindings, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
    versions = {n: p._version for n, p in runtime.model.named_parameters()}
    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(path), expert)
    hook.attach()
    try:
        for batch in batches:
            budget()
            r.q.logits(runtime, hook, batch, verify_base=True)
        optimizer = torch.optim.Adam(expert.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
        for step in range(1, 161):
            budget()
            _, losses = trajectory.source_gradients(r, runtime, hook, expert, batches, step)
            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            r.COUNTS['updates'] += 1
            if step == 1 and torch.count_nonzero(expert.B) == 0:
                raise ValueError('First update produced no response')
        final = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
        state = dict(expert_kind='DEEP_LORA_R4', rank=4, alpha=4, module_path=path,
                     state_dict=final, seed=task['seed'], steps=160, state_digest=state_digest(final))
        target = Path(root) / f'{owner}.pt'
        save_new(target, state)
        active, tokens = r.q.logits(runtime, hook, batches[0])
        hook.enabled = False
        try:
            with torch.inference_mode():
                out = runtime.model(**batches[0].forward_kwargs())
                mask = batches[0].labels[:, 1:] != -100
                base = out.logits[:, :-1][mask].detach().cpu().double()
                base_tokens = batches[0].labels[:, 1:][mask].cpu()
        finally:
            hook.enabled = True
        response = float((active - base).norm())
        if not torch.equal(tokens, base_tokens) or not math.isfinite(response) or response == 0:
            raise ValueError('Invalid active response')
        loaded = torch.load(target, map_location='cpu', weights_only=True)
        if state_digest(loaded['state_dict']) != state['state_digest']:
            raise ValueError('State reload mismatch')
        expert.load_state_dict(loaded['state_dict'])
        restored, restored_tokens = r.q.logits(runtime, hook, batches[0])
        if not torch.equal(active, restored) or not torch.equal(tokens, restored_tokens):
            raise ValueError('Reload changed output')
    finally:
        hook.detach()
    if any(p._version != versions[n] for n, p in runtime.model.named_parameters()):
        raise ValueError('Base parameters changed')
    return dict(owner=owner, steps=160, module=path, initial_digest=state_digest(initial),
                final_digest=state['state_digest'], parameters=73728, response_norm=response)
