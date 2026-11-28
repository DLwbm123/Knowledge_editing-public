"""Learn TT write-out factors with frozen read-in or a matched joint update."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def shifted_scores(runtime, batch):
    labels = batch['labels'][0, 1:]
    mask = labels != -100
    scores = runtime.forward(batch)[0, :-1][mask].float()
    return scores, labels[mask]


def positive(runtime, batch, with_eos):
    scores, tokens = shifted_scores(runtime, batch)
    losses = F.cross_entropy(scores, tokens, reduction='none')
    if with_eos:
        return losses.mean()
    losses = losses[tokens != runtime.eos]
    if not losses.numel():
        raise ValueError('Empty non-EOS answer')
    if losses.numel() == 1:
        return losses[0]
    return .5 * (losses[:len(losses)//2].mean() + losses[len(losses)//2:].mean())


def kl_terms(runtime, batch, base_logp):
    scores, tokens = shifted_scores(runtime, batch)
    edited = scores[tokens != runtime.eos].log_softmax(-1)
    return (base_logp.exp() * (base_logp - edited)).sum(-1).mean()


@contextmanager
def mounted(runtime, expert):
    hook = runtime.hook_type(runtime.layer, expert)
    hook.attach()
    try:
        yield hook
    finally:
        hook.detach()


def train(runtime, expert, solution, events, schedule, beta, joint, clip=1.):
    if not schedule or not 0 <= beta < float('inf'):
        raise ValueError('Invalid training contract')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Backbone must remain frozen')
    a, b = expert.parameter_groups()
    with torch.no_grad():
        expert.G3.copy_(solution['G3'])
        expert.G4.copy_(solution['G4'])
        expert.G1.zero_()
    for p in a:
        p.requires_grad_(joint)
    snapshots = [p.detach().clone() for p in a]
    groups = [{'params': b, 'lr': 1e-3}]
    if joint:
        groups.append({'params': a, 'lr': 1e-3})
    opt = torch.optim.Adam(groups)
    used = {i for step in schedule for i in step['views'] + step['negatives'] + [step['native']]}
    batches = {}
    base = {}
    with torch.no_grad():
        for i in used:
            row = events[i]
            negative = row['role'] in {'train_ind', 'train_vis'}
            if row['role'] not in {'native','fit','train_eq','train_ind','train_vis'}:
                raise ValueError('Training schedule contains an evaluation event')
            batch = runtime.teacher_batch(row, 'base' if negative else 'reference')
            batches[i] = batch
            if negative:
                logits, tokens = shifted_scores(runtime, batch)
                scores = logits[tokens != runtime.eos]
                if not len(scores):
                    raise ValueError('Empty negative prefix')
                base[i] = scores.log_softmax(-1)
    trace = []
    with mounted(runtime, expert) as hook:
        for step in schedule:
            runtime.budget()
            views, negatives, native = step['views'], step['negatives'], step['native']
            if len(views) != 4 or not 1 <= len(negatives) <= 2 or native in views:
                raise ValueError('Training step violates view/negative quota')
            if (events[native]['role'] != 'native' or any(events[i]['role'] not in {'fit','train_eq'} for i in views)
                    or any(i not in base for i in negatives)):
                raise ValueError('Training roles disagree with the schedule')
            opt.zero_grad(set_to_none=True)
            losses = []
            for i, weight in [(native, 1.)] + [(i, .25) for i in views]:
                hook.set_teacher_routing(batches[i]['labels'])
                loss = positive(runtime, batches[i], with_eos=i == native)
                (weight * loss).backward()
                losses.append(float(loss.detach()))
            locality = []
            for i in negatives:
                hook.set_teacher_routing(batches[i]['labels'])
                loss = kl_terms(runtime, batches[i], base[i])
                (beta * loss / len(negatives)).backward()
                locality.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_([p for p in expert.parameters() if p.requires_grad],
                                                  clip, error_if_nonfinite=True)
            opt.step()
            trace.append({'ce':losses,'kl':locality,'gradient_norm':float(norm)})
    if not joint and any(not torch.equal(p, s) for p, s in zip(a, snapshots, strict=True)):
        raise RuntimeError('Read-in changed in the frozen arm')
    return {k:v.detach().cpu() for k,v in expert.state_dict().items()}, trace


def validate_kl(runtime, row, expert, prefix='base'):
    with torch.no_grad():
        batch = runtime.teacher_batch(row,prefix)
        logits,tokens = shifted_scores(runtime,batch)
        base = logits[tokens!=runtime.eos].log_softmax(-1)
        if not len(base):
            raise ValueError('Empty locality comparison')
        with mounted(runtime,expert) as hook:
            hook.set_teacher_routing(batch['labels'])
            kl = float(kl_terms(runtime,batch,base))
    if kl < -2e-6:
        raise ValueError('Invalid negative KL')
    return max(0.,kl)


def gradient_ratio(runtime, expert, validation):
    # A zero residual gives a zero KL gradient. Estimate the scale after CE warmup.
    a,b = expert.parameter_groups()
    params = b
    pos,neg = [],[]
    for row in validation:
        runtime.budget()
        if row['role'] not in {'val_pos','val_neg'}:
            raise ValueError('Beta calibration got a non-validation event')
        is_neg = row['role']=='val_neg'
        batch = runtime.teacher_batch(row,'base' if is_neg else 'reference')
        if is_neg:
            with torch.no_grad():
                scores,labels = shifted_scores(runtime,batch)
                teacher = scores[labels!=runtime.eos].log_softmax(-1)
                if not len(teacher):
                    raise ValueError('Empty validation negative prefix')
        with mounted(runtime,expert) as hook:
            hook.set_teacher_routing(batch['labels'])
            loss = kl_terms(runtime,batch,teacher) if is_neg else positive(runtime,batch,False)
            g = torch.autograd.grad(loss,params)
        (neg if is_neg else pos).append(float(torch.linalg.vector_norm(torch.stack([x.norm() for x in g]))))
    return pos,neg
