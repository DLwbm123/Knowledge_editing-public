"""Train only a fixed-owner residual mixing scalar from admitted source/FIT."""
from pathlib import Path
import torch
from torch import nn

class TrainableBridge(nn.Module):
    def __init__(self, med, du):
        super().__init__()
        self.med = med.requires_grad_(False)
        self.du = du.requires_grad_(False)
        self.logit = nn.Parameter(torch.zeros(()))
        self.endpoint_checked = False

    def residual(self, activation):
        a, b = self.med.residual(activation), self.du.residual(activation)
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('invalid frozen expert residuals')
        if not self.endpoint_checked:
            zero=torch.zeros((),device=a.device,dtype=a.dtype);one=torch.ones_like(zero)
            if not torch.equal((1-zero)*a+zero*b,a) or not torch.equal((1-one)*a+one*b,b):
                raise ValueError('real-layer endpoint identity failed')
            self.endpoint_checked=True
        alpha = self.logit.sigmoid().to(device=a.device, dtype=a.dtype)
        return (1-alpha)*a + alpha*b


def train(runtime, r, task, rows, config, admission, med_state, du_state, destination, budget):
    owner = task['order']
    if admission.get('status') != 'SOURCE_101_ROLES_ADMITTED' or owner not in admission.get('owners', []):
        raise ValueError('source admission missing')
    if [x['audit_role'] for x in rows] != ['NATIVE','FIT','FIT','FIT','FIT']:
        raise ValueError('source schedule mismatch')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('backbone not frozen')
    path = 'model.layers.21.mlp.down_proj'
    if set(med_state) != set(du_state) or any(med_state[k].shape != du_state[k].shape for k in med_state):
        raise ValueError('expert layouts differ')
    med = r.p.expert(med_state, task['seed'], runtime.device).requires_grad_(False)
    du = r.p.expert(du_state, task['seed'], runtime.device).requires_grad_(False)
    expert = TrainableBridge(med,du).to(runtime.device)
    if sum(p.numel() for p in expert.parameters() if p.requires_grad) != 1:
        raise ValueError('unexpected trainable state')
    runtime.model.half()
    try:
        prepared=r.s.prepare(runtime,{'configuration_only':config},task,rows,True)
    finally:
        r.old.fp32(runtime)
    batches=[r.old.batch_on(runtime,x[1]) for x in prepared]
    versions={n:p._version for n,p in runtime.model.named_parameters()}
    endpoints={n:p._version for n,p in expert.named_parameters() if not p.requires_grad}
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(path),expert)
    hook.attach()
    try:
        optimizer=torch.optim.Adam([expert.logit],lr=0.01)
        for step in range(1,41):
            budget();optimizer.zero_grad(set_to_none=True)
            for i in (0,1+(step-1)%4):
                hook.set_teacher_routing(batches[i].labels)
                output=runtime.model(**batches[i].forward_kwargs())
                loss=output.loss
                if loss is None or not torch.isfinite(loss):raise ValueError('nonfinite CE')
                (0.5*loss).backward();r.COUNTS['backwards']+=1
            norm=torch.nn.utils.clip_grad_norm_([expert.logit],1.)
            if not torch.isfinite(norm) or expert.logit.grad is None:raise ValueError('invalid gradient')
            optimizer.step();r.COUNTS['updates']+=1
        if not expert.endpoint_checked or not torch.isfinite(expert.logit):raise ValueError('real endpoints or learned mix invalid')
        if any(p._version!=versions[n] for n,p in runtime.model.named_parameters()):raise ValueError('backbone changed')
        if any(p._version!=endpoints[n] for n,p in expert.named_parameters() if not p.requires_grad):raise ValueError('expert changed')
        result={'owner':owner,'steps':40,'logit':float(expert.logit.detach().cpu()),'alpha':float(expert.logit.sigmoid().detach().cpu()),'module':path}
        with Path(destination).open('x') as f:
            import json
            json.dump(result,f)
        return result
    finally:
        hook.detach()
