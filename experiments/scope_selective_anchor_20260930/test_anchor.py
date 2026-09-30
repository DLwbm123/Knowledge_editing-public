"""CPU scientific invariants; GPU execution parity is a separate mandatory smoke."""
import copy, io, random, sys
from pathlib import Path
import torch
from anchor import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'decomp_24h_20260927'))
from structures import LR4

def main():
    torch.manual_seed(41); random.seed(41)
    ex=LR4(12,8);ex.B.data.normal_();ref=(ex.A.detach().clone(),ex.B.detach().clone())
    h=torch.randn(2,5,12);labels=torch.tensor([[-100,-100,2,3,-100],[-100,1,2,-100,-100]])
    attention=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]])
    mask=predictor_mask(labels,attention)
    assert mask.tolist()==[[False,True,True,False,False],[True,True,False,False,False]]
    layer=torch.nn.Linear(12,8);layer.requires_grad_(False)
    c=positive_scale([response(*ref,h,mask).square().sum(-1).mean()]*5,[layer(h)[mask].square().sum(-1).mean()]*5)
    assert positive_loss(ex.A,ex.B,ref,h,mask,c)==0
    assert ref[0].data_ptr()!=ex.A.data_ptr() and ref[1].data_ptr()!=ex.B.data_ptr()
    rotated=(ref[0],-ref[1]);assert torch.allclose(response(*ref,h,mask).norm(),response(*rotated,h,mask).norm())
    assert positive_loss(ex.A,ex.B,rotated,h,mask,c)>0
    ex.B.data.add_(.03)
    before=positive_loss(ex.A,ex.B,ref,h,mask,c)
    q=torch.eye(4)+.03*torch.randn(4,4);qi=torch.linalg.inv(q)
    for student,teacher in [(True,False),(False,True),(True,True)]:
        a,b=(q@ex.A,ex.B@qi) if student else (ex.A,ex.B)
        rr=(q@ref[0],ref[1]@qi) if teacher else ref
        assert torch.allclose(before,positive_loss(a,b,rr,h,mask,c),rtol=5e-5,atol=2e-6)
    with PositiveCapture(layer,ex,ref,mask,c) as capture:
        layer(h);capture.loss.backward()
    assert not layer._forward_pre_hooks and all(p.grad is None for p in layer.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ex.parameters())
    assert all(p.grad is None and not p.requires_grad for p in ref) and not c.requires_grad
    try:
        with PositiveCapture(layer,ex,ref,mask,c):raise RuntimeError('injected')
    except RuntimeError: pass
    assert not layer._forward_pre_hooks
    for fn in [lambda:predictor_mask(labels*0-100),lambda:positive_loss(ex.A,ex.B,ref,h,mask*False,c),lambda:validate_scale(c*0),lambda:coefficients('TYPO'),lambda:ScaledExpert(ex,0)]:
        try:fn();raise AssertionError('invalid input accepted')
        except (ValueError,KeyError):pass
    assert torch.equal(ScaledExpert(ex,1.).residual(h),ex.residual(h))
    assert torch.equal(ScaledExpert(ex,.5).residual(h),ex.residual(h)*.5)
    snapshot=copy.deepcopy(ex.state_dict())
    def trajectory(beta=0.,diagnostic=False,resume=False,legacy=False):
        e=LR4(12,8);e.load_state_dict(snapshot)
        opt=torch.optim.Adam([dict(params=[e.A],lr=1e-4),dict(params=[e.B],lr=1e-3)])
        torch.manual_seed(113);random.seed(113)
        for step in range(4):
            opt.zero_grad(set_to_none=True)
            noise=torch.randn_like(h)*.01+random.random()*.001
            ce=e.residual(h+noise).square().mean();plus=positive_loss(e.A,e.B,ref,h,mask,c)
            if diagnostic:
                with diagnostic_rng():
                    state=torch.get_rng_state().clone();random.random();torch.randn(2)
                    parts=dict(CE=component_grad(ce,list(e.parameters())),plus=component_grad(plus,list(e.parameters())))
                    grad_summary(parts)
                assert torch.equal(state,torch.get_rng_state()) and all(p.grad is None for p in e.parameters())
            ce.backward()
            if not legacy and beta: (beta*plus).backward()
            torch.nn.utils.clip_grad_norm_(e.parameters(),1);opt.step();e.normalize_factors_()
            if resume and step==1:
                bio=io.BytesIO();torch.save(dict(expert=e.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate()),bio);bio.seek(0);d=torch.load(bio,weights_only=False)
                e=LR4(12,8);e.load_state_dict(d['expert']);opt=torch.optim.Adam([dict(params=[e.A],lr=1e-4),dict(params=[e.B],lr=1e-3)]);opt.load_state_dict(d['optimizer']);torch.set_rng_state(d['torch_rng']);random.setstate(d['python_rng'])
        return e.residual(h).detach()
    assert torch.equal(trajectory(),trajectory(legacy=True))
    assert torch.equal(trajectory(beta=.1),trajectory(beta=.1,diagnostic=True))
    assert torch.equal(trajectory(beta=.1),trajectory(beta=.1,resume=True))
    print('PASS CPU: vector/gauge/scale/frozen gradients/mask/hooks/beta0/scaling/diagnostic trajectory/optimizer+RNG resume; GPU smoke still required')
if __name__=='__main__':main()
