"""DPC/SCRO TT migrations with independently optimized source activation targets."""
import copy
import os
import route_common as c
v=c.v


def latent_class():
    torch=v.torch
    class Latent(torch.nn.Module):
        def __init__(self,factor):
            super().__init__();self.register_buffer('factor',factor)
            self.z=torch.nn.Parameter(torch.zeros(factor.shape[1],4096,device=factor.device))
            self.view=0
        def residual(self,x):return (self.factor[self.view]@self.z).expand(*x.shape[:-1],4096)
    return Latent


def optimize_residual(task,batches,data,spectral):
    torch=v.torch
    keys=torch.stack([data[id(b)]['key'] for b in batches]).double()
    scale=keys.square().mean(0).clamp_min(1e-4*float(keys.square().mean())).sqrt()
    u,s,_=torch.linalg.svd(keys/scale,full_matrices=False)
    rank=int((s>1e-6*s.max()).sum());assert rank>0
    factor=(u[:,:rank]*s[:rank]).float() if spectral else torch.eye(5,device=keys.device)
    expert=latent_class()(factor)
    hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
    opt=torch.optim.Adam(expert.parameters(),lr=.1)
    try:
        for step in range(64):
            opt.zero_grad(set_to_none=True)
            for view in (0,1+step%4):
                expert.view=view;hook.set_teacher_routing(batches[view].labels)
                loss=v.runtime.compute_loss(batches[view]);assert torch.isfinite(loss)
                (.5*loss).backward();v.counts['backwards']+=1
            opt.step();v.counts['updates']+=1
            if spectral:
                with torch.no_grad():
                    uu,ss,vv=torch.linalg.svd(expert.z,full_matrices=False)
                    expert.z.copy_((uu*ss.clamp_max(1.))@vv)
        residual=(expert.factor@expert.z).detach()
        # Native-plus-FIT CE sensitivity, not a teacher or a heldout signal.
        deltas=[]
        for view,b in enumerate(batches):
            expert.view=view;hook.set_teacher_routing(b.labels)
            output=[]
            handle=v.runtime.get_module(v.r.c.LAYER).register_forward_hook(lambda _m,_a,o:output.append(o))
            try:
                loss=v.runtime.compute_loss(b)
                delta=torch.autograd.grad(loss,output[0])[0][data[id(b)]['mask']].detach().float()
                v.counts['backwards']+=1;deltas.append(delta/delta.norm(dim=1,keepdim=True).clamp_min(1e-12))
            finally:handle.remove()
    finally:hook.detach()
    return residual,deltas,dict(spectral=spectral,key_rank=rank,alpha=1.,steps=64,whitening='SOURCE_KEY_DIAGONAL',same_relation_views=True)


def internalize(task,original,batches,data,residual,deltas,decision):
    torch=v.torch
    expert=v.r.p.expert(original,task['seed'],v.runtime.device)
    expert.load_state_dict(v.r.mm.canonicalize(original)[0])
    opt=torch.optim.Adam(expert.parameters(),lr=.01)
    x=torch.cat([data[id(b)]['x'] for b in batches]);target=torch.cat([r.expand(len(data[id(b)]['x']),-1) for b,r in zip(batches,residual)])
    delta=torch.cat(deltas);normalizer=target.square().mean().clamp_min(1e-8)
    for step in range(256):
        opt.zero_grad(set_to_none=True);pred=expert.residual(x)
        error=pred-target;loss=error.square().mean()/normalizer
        if decision:loss=loss+(error*delta).sum(-1).square().mean()/normalizer
        assert torch.isfinite(loss);loss.backward();opt.step();v.counts['operator_backwards']+=1;v.counts['updates']+=1
    return expert


def main():
    tasks,items,by_owner,batches=c.prepare();torch=v.torch
    from m3bench_repro.editors.llava_runtime import seed_everything
    route=os.environ['ROUTE'];arms=['CE','FREE_FROB','DPC'] if route=='DPC' else ['CE','FREE_FROB','SCRO']
    with c.CachedSource(batches) as cache:
        for task in tasks:
            owner=task['order'];bs=by_owner[owner]
            original=v.r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
            seed_everything(task['seed'])
            free,free_d,audit=optimize_residual(task,bs,cache.data,False)
            if route=='SCRO':constrained,constrained_d,constrained_audit=optimize_residual(task,bs,cache.data,True)
            for arm in arms:
                seed_everything(task['seed'])
                if arm=='CE':expert=v.r.p.expert(original,task['seed'],v.runtime.device);expert.load_state_dict(v.r.mm.canonicalize(original)[0])
                else:
                    use_constrained=arm=='SCRO'
                    expert=internalize(task,original,bs,cache.data,constrained if use_constrained else free,
                        constrained_d if use_constrained else free_d,decision=arm=='DPC')
                hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
                try:c.ce_train(expert,hook,bs)
                finally:hook.detach()
                state=v.r.old.snapshot(expert);path=v.ROOT/'candidates'/arm/f'{owner}.pt'
                v.save_state(path,dict(state=state,owner=owner,arm=arm,FP32=True))
                loaded=v.r.p.load_state(path)['state'];assert all(torch.equal(loaded[k].cpu(),state[k].cpu()) for k in state)
                v.write(v.ROOT/'training'/arm/f'{owner}.json',dict(status='PASS',CE_steps=160,compression_steps=0 if arm=='CE' else 256,
                    RO_steps=0 if arm=='CE' else 64,decision_preserving=arm=='DPC',spectral_constraint=arm=='SCRO',source_only=True))
                v.progress(owner=owner,arm=arm,stage='FULL_TRAINING_COMPLETE')
                del expert
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=24,time=__import__('time').time(),all8=True))
    def loader(arm,owner):
        state=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
        return v.r.p.expert(state,owner,v.runtime.device)
    c.evaluate(tasks,arms,loader);c.finish(arms)


if __name__=='__main__':c.run(main)
