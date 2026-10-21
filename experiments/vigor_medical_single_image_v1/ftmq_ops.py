"""Isolated, source-only single-matrix teacher fitting; no student distillation."""
import torch
from .editor import editable_fp32, finite, nll

AUTH = dict(samples=['E01','E02','E03'], layer='model.layers.6.mlp.down_proj',
            steps=25, lr=.0005, weight_decay=0., betas=[.9,.999], eps=1e-8,
            stop_nll=.01, objective='original_chat_full_source_target',
            gradient_clip=None, coordinate_clamp=None, forward_max=600,
            work_forward_max=560, seconds=900, work_seconds=840,
            experiment_seconds=14400, disk_max=251658240,
            generation_max=3, per_generation_forward_max=128,
            judge_payload_max=6, checkpoint=0, retries=0)

def fit(model, layer, batch, emit):
    """FP32 selected matrix and Adam state; frozen mixed-precision backbone."""
    steps=[]
    with editable_fp32(model,layer):
        opt=torch.optim.Adam([layer.weight],lr=AUTH['lr'],betas=tuple(AUTH['betas']),
                             eps=AUTH['eps'],weight_decay=AUTH['weight_decay'])
        try:
            for i in range(AUTH['steps']):
                opt.zero_grad(set_to_none=True)
                loss=nll(model,batch);value=float(loss.detach())
                row=dict(step=i+1,loss_before=value,optimizer_step=False)
                if value<AUTH['stop_nll']:
                    steps.append(row);emit(row);break
                loss.backward();finite(layer.weight.grad)
                row['gradient_norm']=float(layer.weight.grad.double().norm())
                opt.step();finite(layer.weight)
                if not all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)):
                    raise ValueError('Nonfinite Adam state')
                row['optimizer_step']=True;steps.append(row);emit(row)
        finally:
            opt.zero_grad(set_to_none=True)
    return steps

def run_checks():
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(4,4,bias=False).half()
        def forward(self,inputs_embeds):
            return SimpleNamespace(logits=self.layer(inputs_embeds))
    class Batch:
        labels=torch.tensor([[-100,1,2,3]])
        def forward_kwargs(self):return dict(inputs_embeds=torch.eye(4).half().unsqueeze(0))
    torch.manual_seed(0);m=Toy();m.train();old=m.layer.weight.detach().clone();flags=[p.requires_grad for p in m.parameters()]
    initial=float(nll(m,Batch()).detach());rows=fit(m,m.layer,Batch(),lambda x:None)
    assert len(rows)==25 and sum(r['optimizer_step'] for r in rows)==25
    assert float(nll(m,Batch()).detach())<initial and not torch.equal(m.layer.weight,old)
    assert m.layer.weight.dtype==old.dtype and m.training and [p.requires_grad for p in m.parameters()]==flags
    assert m.layer.weight.grad is None and not m.layer._forward_hooks and not m.layer._forward_pre_hooks
    fitted=m.layer.weight.detach().clone()
    def fail(row):raise RuntimeError('injected')
    try:fit(m,m.layer,Batch(),fail)
    except RuntimeError:pass
    else:raise AssertionError('Exception must propagate')
    assert torch.equal(m.layer.weight,fitted) and m.layer.weight.grad is None
    assert not m.layer._forward_hooks and not m.layer._forward_pre_hooks and m.training
    return dict(status='PASS',checks=['source_full_causal_loss_descent','fixed_25_Adam_steps',
        'native_commit','flags_modes_hooks_grad_cleanup','exception_rollback'])
