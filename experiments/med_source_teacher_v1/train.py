"""Fixed deep expert training after externally verified teacher-cache freeze."""
import copy
import math
import torch
import source_step
import teacher_cache
from train_pair import save_new, state_digest


def train(runtime, r, task, batches, teachers, frozen, output, budget):
    if frozen.get('status') != 'ALL_80_TEACHERS_FROZEN' or frozen.get('count') != 80:
        raise ValueError('All fixed teacher outputs must be frozen first')
    if len(batches) != 5 or len(teachers) != 5:
        raise ValueError('Expected native and four FIT batches')
    owner = task['order']
    if owner not in frozen.get('owners', {}) or frozen['owners'][owner]['cache_digest'] != teacher_cache.cache_digest(teachers):
        raise ValueError('Teacher cache differs from frozen owner receipt')
    if any(teacher_cache.input_digest(batch) != item['input_digest'] for batch, item in zip(batches, teachers)):
        raise ValueError('Teacher and student inputs differ')
    if any(p.requires_grad for p in runtime.model.parameters()) or len(runtime.model.model.layers) != 32:
        raise ValueError('Base or decoder binding invalid')
    module = runtime.get_module('model.layers.30.mlp.down_proj')
    if (module.in_features, module.out_features) != (14336, 4096):
        raise ValueError('Deep layer dimensions differ')
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(task['seed'])
    initial = r.p.tr.TT4(task['seed'], 8, 8).state_dict()
    if sum(t.numel() for t in initial.values()) != 7168 or torch.count_nonzero(initial['G1']):
        raise ValueError('TT initial state invalid')
    initial = {k: v.detach().cpu().clone() for k, v in initial.items()}
    expert = r.p.expert(copy.deepcopy(initial), task['seed'], runtime.device)
    if state_digest(expert.state_dict()) != state_digest(initial):
        raise ValueError('TT initial state differs')
    versions = {name: p._version for name, p in runtime.model.named_parameters()}
    hook = r.q.scoped.replay.MedTraceLayerHook(module, expert); hook.attach()
    try:
        optimizer = r.p.optimizer(expert, runtime.model)
        curve = []
        for step in range(1, 161):
            budget()
            before = r.old.snapshot(expert)
            rows = source_step.backwards(runtime, hook, expert, batches, teachers, step, budget, r.COUNTS)
            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step(); r.COUNTS['updates'] += 1
            after = r.old.snapshot(expert)
            if step == 1 and all(torch.equal(after[k], before[k]) for k in before):
                raise ValueError('First update did not change expert')
            curve.append(dict(step=step, terms=rows, grad_norm=float(norm)))
        final = r.old.snapshot(expert)
        save_new(output, dict(state=final, initial_digest=state_digest(initial), steps=160,
                              module='model.layers.30.mlp.down_proj', curve=curve))
        reloaded = torch.load(output, map_location='cpu', weights_only=True)
        if any(not torch.equal(v.cpu(), reloaded['state'][k]) for k, v in final.items()):
            raise ValueError('Saved candidate state differs')
        budget()
        active, targets = r.q.logits(runtime, hook, batches[0])
        hook.enabled = False
        try:
            with torch.no_grad():
                base_output = runtime.model(**batches[0].forward_kwargs())
                mask = batches[0].labels[:, 1:] != -100
                base = base_output.logits[:, :-1][mask].detach().cpu().double()
                base_targets = batches[0].labels[:, 1:][mask].cpu()
        finally:
            hook.enabled = True
        if not torch.equal(targets, base_targets):
            raise ValueError('Expert and base targets differ')
        response = float((active - base).norm())
        if not math.isfinite(response) or response == 0:
            raise ValueError('Expert activation is absent or nonfinite')
        expert.load_state_dict(reloaded['state'])
        restored, restored_targets = r.q.logits(runtime, hook, batches[0])
        if not torch.equal(active, restored) or not torch.equal(targets, restored_targets):
            raise ValueError('Reloaded expert output differs')
        if any(p._version != versions[name] for name, p in runtime.model.named_parameters()):
            raise ValueError('Frozen base parameter changed')
        return dict(steps=160, initial_digest=state_digest(initial), final_digest=state_digest(final),
                    activation_norm=response, reload_output_equal=True)
    finally:
        hook.detach()
