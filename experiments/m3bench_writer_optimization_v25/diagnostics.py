"""Detached, fixed-schedule observations; never participate in updates or stopping."""
import copy
import torch

def should_stop(losses,t):
    assert len(losses)==t+1
    return t>=160 and (all(x<=.01 for x in losses[-3:]) or t==640)

def token_summary(logits,labels,eos):
    with torch.no_grad():
        valid=labels[:,1:]!=-100;z=logits[:,:-1][valid].detach().float();y=labels[:,1:][valid]
        assert len(y)>0
        target=z.gather(1,y[:,None]).squeeze(1)
        top=z.topk(2,dim=-1);rival=torch.where(top.indices[:,0]==y,top.values[:,1],top.values[:,0])
        ce=torch.logsumexp(z,dim=-1)-target;is_eos=y==eos
        return dict(tokens=len(y),content_tokens=int((~is_eos).sum()),
                    content_CE=float(ce[~is_eos].mean()) if (~is_eos).any() else None,
                    EOS_supervised_tokens=int(is_eos.sum()),EOS_CE=float(ce[is_eos].mean()) if is_eos.any() else None,
                    top1_accuracy=float((z.argmax(-1)==y).float().mean()),
                    first_token_rank=int((z[0]>target[0]).sum())+1,
                    minimum_target_margin=float((target-rival).min()),logits_RMS=float(z.square().mean().sqrt()))

def gradient_summary(gs,before,after,scalar):
    with torch.no_grad():
        norms=[float(g.detach().double().norm()) for g in gs];total=sum(n*n for n in norms)**.5
        return dict(core_gradient_norms_before_clip=norms,gradient_norm=total,clip_triggered=total>1.,
                    scalar=copy.deepcopy(scalar),core_parameter_update_norms=[float((after[k]-before[k]).double().norm()) for k in before],
                    gauge='right-canonical input state; gradient before clipping; update after re-canonicalization')

class Observer:
    def __init__(self,runtime,expert,batch):
        self.batch=batch;self.runtime=runtime;self.expert=expert;self.enabled=False;self.value={};self.gate=None
        layer=runtime.model.model.layers[31]
        def gate(_m,_a,out):
            if self.enabled:self.gate=out.detach()
        def up(_m,args,_out):
            if self.enabled:
                with torch.no_grad():
                    x=args[0].detach();branch=expert.residual(x).detach()
                    mask=batch.labels[:,1:]!=-100
                    active=torch.zeros(batch.labels.shape,dtype=torch.bool,device=x.device);active[:,:-1]=mask
                    delta=layer.mlp.act_fn(self.gate)*branch
                    composed=torch.nn.functional.linear(delta[active],layer.mlp.down_proj.weight.detach())
                    self.value['function_response']=dict(TT_branch_RMS=float(branch[active].float().square().mean().sqrt()),
                         gated_branch_RMS=float(delta[active].float().square().mean().sqrt()),
                         gate_down_residual_RMS=float(composed.float().square().mean().sqrt()),
                         additional_gate_down_diagnostic_products=1,positions='native supervised predictors only')
        def model(_m,_a,out):
            if self.enabled:self.value['tokens']=token_summary(out.logits,batch.labels,runtime.adapter.tokenizer.eos_token_id)
        self.handles=[layer.mlp.gate_proj.register_forward_hook(gate),layer.mlp.up_proj.register_forward_hook(up),runtime.model.register_forward_hook(model)]
    def close(self):
        for h in self.handles:h.remove()
        self.gate=None

def generation_summary(tokens,eos):
    counts={}
    for i in range(max(0,len(tokens)-3)):
        k=tuple(tokens[i:i+4]);counts[k]=counts.get(k,0)+1
    return dict(length=len(tokens),at_cap=len(tokens)>=1024,empty=not tokens,EOS=bool(tokens) and tokens[-1]==eos,
                repeated_4gram_excess=sum(n-1 for n in counts.values()),repetition_is_not_a_correctness_verdict=True)

def selfcheck(r,seed):
    from alignment import TT,snapshot,next_step
    torch.set_num_threads(1)
    initial_rng=torch.get_rng_state().clone()
    def trajectory(observe):
        e=TT(seed,True);e.load_state_dict(r.mm.canonicalize(snapshot(e))[0]);scalar=dict(step=0,second_moments=[0.,0.]);anchor=None
        x=torch.ones(1,4096);history=[]
        for step in range(1,181):
            before=snapshot(e);loss=(e.residual(x)[0,:8]-1).square().mean()
            gs=torch.autograd.grad(loss,tuple(e.parameters()));gradient=torch.cat([g.detach().double().flatten() for g in gs])
            state,scalar=next_step(r,before,gradient,scalar)
            if observe:
                rng=torch.get_rng_state().clone();gradient_summary(gs,before,state,scalar)
                token_summary(torch.arange(24,dtype=torch.float).reshape(1,3,8),torch.tensor([[-100,2,7]]),7)
                assert torch.equal(rng,torch.get_rng_state())
            e.load_state_dict(state);history.append(float(loss.detach()))
            if step==160:anchor=(snapshot(e),copy.deepcopy(scalar))
        return snapshot(e),scalar,history,anchor
    a=trajectory(False);b=trajectory(True)
    assert all(torch.equal(a[0][k],b[0][k]) for k in a[0]) and a[1:3]==b[1:3]
    assert a[3][1]['step']==160 and b[1]['step']==180
    # Save/load must preserve both parameters and scalar state before continuing.
    import io
    buffer=io.BytesIO();torch.save(dict(state=a[3][0],scalar=a[3][1]),buffer);buffer.seek(0)
    restored=torch.load(buffer,weights_only=True)
    assert restored['scalar']==a[3][1] and all(torch.equal(v,restored['state'][k]) for k,v in a[3][0].items())
    for t,tail,expected in [(159,[0.,0.,0.],False),(160,[.01,.01,.01],True),(160,[.01,.01,.01001],False),(640,[1.,1.,1.],True)]:
        assert should_stop([1.]*(t-2)+tail,t)==expected
    z=token_summary(torch.arange(24,dtype=torch.float).reshape(1,3,8),torch.tensor([[-100,2,7]]),6)
    assert z['EOS_CE'] is None and z['EOS_supervised_tokens']==0
    assert torch.equal(initial_rng,torch.get_rng_state())
    return dict(status='PASS',C_180_step_diagnostics_on_off_parameters_scalar_losses_exact=True,
                continuous_scalar_step160_to180=True,save_load_state_scalar_exact=True,update_state_stop_boundaries=True,
                EOS_absent_is_NA=True,RNG_unchanged=True,real_model_calls=0)
