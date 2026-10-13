"""Positive functional anchors. No backbone, data selection, or admission side effects."""
from contextlib import contextmanager
import math
import torch

ARMS = {m: (.1, .1 if m.startswith('EGP_A') or m in ['SMOKE_SP','SMOKE_GUARD','SMOKE_RESUME'] else 0.) for m in ['CAP_1','CAP_2','EGP_1','EGP_2','EGP_A_1','EGP_A_2','SMOKE_S','SMOKE_SP','SMOKE_GUARD','SMOKE_RESUME']}

def coefficients(method):
    return ARMS[method]  # Unknown names must fail, never fall through to AH.

def predictor_mask(labels, attention=None):
    if labels.ndim != 2 or (labels[:, 0] != -100).any():
        raise ValueError('Causal labels require a preceding prompt token')
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    if attention is not None:
        if attention.shape != labels.shape or ((labels != -100) & ~attention.bool()).any():
            raise ValueError('Padding cannot carry target labels')
        mask &= attention.bool()
    if not mask.any():
        raise ValueError('Empty predictor mask')
    return mask

def response(a, b, h, mask):
    if mask.dtype != torch.bool or mask.shape != h.shape[:-1] or not mask.any():
        raise ValueError('Nonempty aligned predictor mask required')
    z = h[mask].detach().float()
    z = z / (z.square().mean(-1, keepdim=True).sqrt() + 1e-6)
    r = (z @ a.T) @ b.T
    if not torch.isfinite(r).all():
        raise ValueError('Nonfinite residual')
    return r

def validate_scale(c):
    if c.numel() != 1 or c.requires_grad or not torch.isfinite(c) or c <= 0:
        raise ValueError('Fixed positive finite detached scale required')

def positive_loss(a, b, ref, h, mask, scale):
    validate_scale(scale)
    if any(p.requires_grad for p in ref):
        raise ValueError('Reference must be frozen')
    return (response(a, b, h, mask) - response(*ref, h, mask)).square().sum(-1).mean() / scale

def positive_scale(energies, base_energies):
    # Native=.5; the four inherited S_fit batches contribute .5 in total.
    if len(energies) != 5 or len(base_energies) != 5:
        raise ValueError('Inherited CE requires native plus exactly four S_fit batches')
    weights = [.5] + [.125] * 4
    e = sum(w * x.detach() for w, x in zip(weights, energies))
    v = sum(w * x.detach() for w, x in zip(weights, base_energies))
    c = torch.maximum(torch.maximum(e, 1e-6 * v), e.new_tensor(1e-12)).detach()
    validate_scale(c)
    return c

class PositiveCapture:
    def __init__(self, layer, expert, ref, mask, scale):
        validate_scale(scale)
        self.loss = None
        self.handle = layer.register_forward_pre_hook(
            lambda module, args: self.capture(expert, ref, args[0], mask, scale))
    def capture(self, expert, ref, h, mask, scale):
        self.loss = positive_loss(expert.A, expert.B, ref, h, mask, scale)
    def close(self):
        self.handle.remove()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()

class ScaledExpert(torch.nn.Module):
    """Scale the complete residual exactly once; gamma=0 is handled by Base-off."""
    def __init__(self, expert, gamma):
        super().__init__()
        if gamma not in (.25, .5, .75, 1.):
            raise ValueError('gamma=0 must use explicit Base-off, without an expert hook')
        self.expert, self.gamma = expert, gamma
    def residual(self, h):
        return self.expert.residual(h) * self.gamma

@contextmanager
def diagnostic_rng():
    import random
    state = random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(state)

def component_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [g.detach().clone() if g is not None else torch.zeros_like(p) for p, g in zip(params, gs)]

def grad_summary(parts):
    norms = {k: float(sum(g.float().square().sum() for g in v).sqrt()) for k, v in parts.items()}
    angles = {}
    for i, (k, v) in enumerate(parts.items()):
        for j, (other, w) in enumerate(parts.items()):
            if j <= i:
                continue
            den = norms[k] * norms[other]
            angles[k + '__' + other] = float(sum(a.float().mul(b.float()).sum() for a, b in zip(v, w))) / den if den else None
    return dict(norms=norms, cosine=angles)

def prepare_positive(runtime, layer, expert, batches, path, read, write):
    """Five shared Base-off forwards per edit; persist scalars, never activations."""
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise RuntimeError('Trainable upstream/Base parameters invalidate fixed h assumption')
    if layer._forward_hooks or layer._forward_pre_hooks:
        raise RuntimeError('Unexpected pre-existing editable-layer hook')
    ref=(expert.A.detach().clone(),expert.B.detach().clone())
    if path.exists():
        saved=read(path);c=expert.A.new_tensor(saved['c_plus']);validate_scale(c)
        return ref,c,0
    energies=[];base=[]
    for batch in batches:
        mask=predictor_mask(batch.labels,batch.attention_mask)
        def capture(module,args,out):
            energies.append(response(*ref,args[0],mask).square().sum(-1).mean().detach())
            base.append(out[mask].detach().float().square().sum(-1).mean())
        handle=layer.register_forward_hook(capture)
        try:
            with torch.no_grad():runtime.model(**batch.forward_kwargs())
        finally:handle.remove()
    c=positive_scale(energies,base)
    write(path,dict(c_plus=float(c),weights=[.5]+[.125]*4,ref_energy=[float(x) for x in energies],Base_module_energy=[float(x) for x in base],Base_off_forwards=5,layer_output_not_logits=True))
    return ref,c,5

def fixed_diagnostics(runtime, layer, expert, hook, batches, negatives, ref, cplus, cminus, kl_fn):
    """Fixed native/first S_fit and first U_old/first hard; gradients never touch .grad."""
    params=list(expert.parameters());parts={k:[torch.zeros_like(p) for p in params] for k in ['CE','U_KL','L_minus','L_plus']}
    values={k:0. for k in parts};energy=[];positive=[]
    def add(name, loss, weight):
        values[name]+=float(loss.detach())*weight
        for dst,g in zip(parts[name],component_grad(loss,params)):dst.add_(g,alpha=weight)
    with diagnostic_rng():
        for batch in batches[:2]:
            mask=predictor_mask(batch.labels,batch.attention_mask);hook.set_teacher_routing(batch.labels)
            def positive_capture(module,args):
                with torch.no_grad():
                    r=response(expert.A,expert.B,args[0],mask);rr=response(*ref,args[0],mask);en=float(r.square().sum(-1).mean());er=float(rr.square().sum(-1).mean())
                    positive.append(dict(energy=en,reference_energy=er,energy_ratio=en/er if er>1e-12 else None,degenerate=er<=1e-12,anchor_error=float((r-rr).square().sum(-1).mean()),direction_cosine=float(torch.nn.functional.cosine_similarity(r,rr,dim=-1).mean())))
            handle=layer.register_forward_pre_hook(positive_capture)
            try:
                with PositiveCapture(layer,expert,ref,mask,cplus) as cap:
                    output=runtime.model(**batch.forward_kwargs());loss=output.loss;add('CE',loss,.5);add('L_plus',cap.loss,.5)
                    index=int(torch.where(batch.labels[0]!=-100)[0][0]);logits=output.logits[0,index-1].detach().float();target=int(batch.labels[0,index]);target_logit=float(logits[target]);logits=logits.clone();logits[target]=-torch.inf
                    positive[-1].update(target_NLL=float(loss.detach()),first_target_margin=target_logit-float(logits.max()))
            finally:handle.remove()
        for kw,labels,mask,lp in negatives:
            hook.set_teacher_routing(labels);box=[]
            def capture(module,args):
                r=response(expert.A,expert.B,args[0],mask);rr=response(*ref,args[0],mask)
                box.append(r.square().sum(-1).mean()/cminus)
                en=float(r.square().sum(-1).mean());er=float(rr.square().sum(-1).mean())
                energy.append(dict(energy=en,reference_energy=er,energy_ratio=en/er if er>1e-12 else None,degenerate=er<=1e-12,anchor_error=float((r-rr).square().sum(-1).mean()),direction_cosine=float(torch.nn.functional.cosine_similarity(r,rr,dim=-1).mean())))
            handle=layer.register_forward_pre_hook(capture)
            try:
                logits=runtime.model(**kw).logits
                loss=kl_fn(logits[mask],lp.detach());add('U_KL',loss,.005);add('L_minus',box[0],1/len(negatives))
            finally:handle.remove()
    return dict(values=values,gradients=grad_summary(parts),negative_energy=energy,positive_energy=positive,model_forwards=2+len(negatives),autograd_grad_calls=4+2*len(negatives),batch='native, first S_fit; first U_old and first fixed hard',gradient_units='CE weighted .5/.5; KL .005 each; residual terms unmultiplied by lambda/beta')
