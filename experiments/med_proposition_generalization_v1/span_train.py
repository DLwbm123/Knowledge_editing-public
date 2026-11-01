"""Train B/C/D TT candidates using source-only batches and a frozen private span lock."""
import copy
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from train_pair import save_new, state_digest
from robust_objective import weights_and_value, view_loss


def terms(runtime, hook, batch, row, length, spans):
    hook.set_teacher_routing(batch.labels)
    output = runtime.model(**batch.forward_kwargs())
    ce = output.loss
    if ce is None or not torch.isfinite(ce):
        raise ValueError('Invalid CE')
    labels = batch.labels[:, 1:]
    mask = labels != -100
    ids = labels[mask]
    if ids.numel() < length or any(int(x) != runtime.adapter.tokenizer.eos_token_id for x in ids[length:]):
        raise ValueError('Expanded target length or suffix mismatch')
    decoded = runtime.adapter.tokenizer.decode(ids[:length].tolist(), skip_special_tokens=False).strip()
    if decoded != row['reference'].strip():
        raise ValueError('Expanded target does not match the source answer')
    per_token = F.cross_entropy(output.logits[:, :-1, :].float().reshape(-1, output.logits.shape[-1]),
                                labels.reshape(-1), ignore_index=-100, reduction='none').reshape(labels.shape)[mask][:length]
    if not spans or any(not (0 <= a < b <= length) for a, b in spans):
        raise ValueError('Invalid frozen answer span')
    losses = torch.stack([per_token[a:b].mean() for a,b in spans])
    if not torch.isfinite(losses).all():
        raise ValueError('Invalid span loss')
    return ce, losses


def train(runtime, r, task, rows, bindings, admission, lock, root, budget):
    owner = task['order']
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or owner not in admission['owners']:
        raise ValueError('Owner not admitted')
    if lock['order'] != owner or lock['status'] != 'WEAK_TOKEN_MATCH' or len(lock['views']) != 5:
        raise ValueError('Owner has no frozen matched spans')
    if len(rows) != 5 or [x['audit_role'] for x in rows] != ['NATIVE','FIT','FIT','FIT','FIT']:
        raise ValueError('Training roles differ')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Backbone not frozen')
    from m3bench_repro.editors.llava_runtime import seed_everything
    path = 'model.layers.21.mlp.down_proj'
    seed_everything(task['seed'])
    initial = {k:v.detach().cpu().clone() for k,v in r.p.tr.TT4(task['seed'],8,8).state_dict().items()}
    if sum(v.numel() for v in initial.values()) != 7168 or torch.count_nonzero(initial['G1']):
        raise ValueError('Unexpected initial TT')
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, bindings, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batches = [r.old.batch_on(runtime, x[1]) for x in prepared]
    versions = {n:p._version for n,p in runtime.model.named_parameters()}
    results=[]
    for arm in ('B','C','D'):
        seed_everything(task['seed'])
        expert = r.p.expert(copy.deepcopy(initial), task['seed'], runtime.device)
        if state_digest(expert.state_dict()) != state_digest(initial):
            raise ValueError('Initial state mismatch')
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(path), expert)
        hook.attach()
        try:
            optimizer=r.p.optimizer(expert,runtime.model)
            curve=[]
            for step in range(1,161):
                budget()
                selected=(0,1+(step-1)%4)
                def spans_for(i):
                    if arm=='B': return [(0,lock['answer_tokens'])]
                    if arm=='C': return lock['controls']
                    return lock['views'][i]['medical']
                expert.zero_grad(set_to_none=True)
                with torch.no_grad():
                    first=[terms(runtime,hook,batches[i],rows[i],lock['answer_tokens'],spans_for(i)) for i in selected]
                w,actual=weights_and_value(first[0][1],first[1][1],1.0)
                for j,i in enumerate(selected):
                    ce,span=terms(runtime,hook,batches[i],rows[i],lock['answer_tokens'],spans_for(i))
                    if not torch.allclose(ce.detach(), first[j][0], atol=1e-5, rtol=1e-5) or not torch.allclose(span.detach(),first[j][1],atol=1e-5,rtol=1e-5):
                        raise ValueError('Recomputed forward differs')
                    view_loss(ce,span,w[j],1.0).backward()
                    r.COUNTS['backwards']+=1
                norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                if not torch.isfinite(norm) or not torch.isfinite(actual):
                    raise ValueError('Nonfinite update')
                before = r.old.snapshot(expert) if step == 1 else None
                curve.append(dict(step=step, ce=[float(x[0]) for x in first], robust=float(actual), grad_norm=float(norm)))
                optimizer.step();r.COUNTS['updates']+=1
                if before is not None and all(torch.equal(v,before[k]) for k,v in r.old.snapshot(expert).items()):
                    raise ValueError('First update did not change expert')
            final=r.old.snapshot(expert)
            target=Path(root)/f'{owner}_{arm}.pt'
            save_new(target,dict(state=final,initial_digest=state_digest(initial),module=path,steps=160,arm=arm,curve=curve))
            loaded=torch.load(target,map_location='cpu',weights_only=True)
            if state_digest(loaded['state']) != state_digest(final):
                raise ValueError('Saved state mismatch')
            active,tok=r.q.logits(runtime,hook,batches[0])
            expert.load_state_dict(loaded['state'])
            restored,tok2=r.q.logits(runtime,hook,batches[0])
            if not torch.equal(active,restored) or not torch.equal(tok,tok2):
                raise ValueError('Reload output mismatch')
            results.append(dict(owner=owner,arm=arm,steps=160,final_digest=state_digest(final)))
        finally:
            hook.detach()
    if any(p._version != versions[n] for n,p in runtime.model.named_parameters()):
        raise ValueError('Backbone changed')
    return results
