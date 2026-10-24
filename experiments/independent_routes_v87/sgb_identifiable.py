"""Source-identifiable spectral TT routing with matched and separated head initialization."""
import copy
import route_common as c
v=c.v


def flat(g):return v.torch.cat([g[k].flatten() for k in v.r.mm.KEYS])


def state_direction(before,vector,target):
    assert float(vector.norm())>0
    vector=vector/vector.norm();offset=0;candidate={}
    for k in v.r.mm.KEYS:
        n=before[k].numel();candidate[k]=before[k]-vector[offset:offset+n].reshape_as(before[k]);offset+=n
    assert v.r.fm.map_norm(before,candidate)>1e-14
    candidate,_=v.r.match_proposal(before,candidate,target)
    return v.r.mm.canonicalize(candidate)[0]


def composite_class():
    torch=v.torch
    class Composite(torch.nn.Module):
        def __init__(self,meta,signed):
            super().__init__();self.signed=signed
            self.branches=torch.nn.ModuleList([v.r.p.expert(s,meta['seed'],v.runtime.device) for s in meta['branches']])
            self.reference=v.r.p.expert(meta['warm'],meta['seed'],v.runtime.device).requires_grad_(False)
            self.register_buffer('basis',meta['basis'].to(v.runtime.device));self.register_buffer('center',meta['center'].to(v.runtime.device))
            self.register_buffer('scale',meta['scale'].to(v.runtime.device))
            self.router=torch.nn.Linear(meta['basis'].shape[0],2,bias=True,device=v.runtime.device)
            self.router.weight.data.zero_();self.router.bias.data.zero_()
            if signed and meta.get('initialize_from_spectrum',False):self.router.weight.data.copy_(meta['router'].T)
            self.request(None)
        def request(self,prefix):self.prefix=prefix;self.coefficients=None
        def residual(self,x):
            if self.coefficients is None:
                index=self.prefix if self.prefix is not None else x.shape[1]-1
                key=x[0,index].detach().float()
                feature=((key-self.center)@self.basis.T)/self.scale
                if not self.signed:feature=torch.ones_like(feature)
                self.coefficients=2*torch.tanh(self.router(feature)/2)
            value=self.branches[0].residual(x)
            ref=self.reference.residual(x)
            for j in range(2):value=value+self.coefficients[j]*(self.branches[j+1].residual(x)-ref)
            return value
    return Composite


def main():
    tasks,items,by_owner,batches=c.prepare();torch=v.torch;r=v.r;seed=tasks[0]['seed']
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(seed)
    original=r.p.load_state(v.PARENT/'private/candidates/C'/'1.pt')['states']['BASE']
    expert=r.p.expert(original,seed,v.runtime.device);expert.load_state_dict(r.mm.canonicalize(original)[0])
    target=sum(v.read(v.PARENT/'private/results/C'/f'{i}.json')['curve'][0]['operator_step_norm'] for i in range(1,9))/8
    with c.CachedSource(batches) as cache:
        hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach()
        try:
            expert.zero_grad(set_to_none=True)
            for task in tasks:
                for view,b in enumerate(by_owner[task['order']]):
                    hook.set_teacher_routing(b.labels);loss=v.runtime.compute_loss(b)
                    (((.5 if view==0 else .125)/8)*loss).backward();v.counts['backwards']+=1
            before=r.old.snapshot(expert)
            warm,_=r.mm.canonical_rms_step(before,{k:p.grad.detach().clone() for k,p in expert.named_parameters()},dict(step=0,second_moments=[0.,0.]))
            warm,_=r.match_proposal(before,warm,target);warm=r.mm.canonicalize(warm)[0];expert.load_state_dict(warm)
            v.counts['updates']+=1
            rows=[];keys=[]
            for task in tasks:
                b=by_owner[task['order']][0];expert.zero_grad(set_to_none=True);hook.set_teacher_routing(b.labels)
                loss=v.runtime.compute_loss(b);loss.backward();v.counts['backwards']+=1
                rows.append(flat({k:p.grad.detach().clone() for k,p in expert.named_parameters()}));keys.append(cache.data[id(b)]['key'])
        finally:hook.detach()
        gradients=torch.stack(rows).double();mean=gradients.mean(0);centered=gradients-mean
        u,s,vh=torch.linalg.svd(centered,full_matrices=False)
        assert float(s[1])>1e-12
        modes=vh[:2]
        for j in range(2):
            if modes[j,int(modes[j].abs().argmax())]<0:modes[j]*=-1
        coords=centered@modes.T;coords=coords/coords.std(0,unbiased=False).clamp_min(1e-12)
        key=torch.stack(keys).double();center=key.mean(0)
        _,key_singular,basis=torch.linalg.svd(key-center,full_matrices=False)
        feature_rank=int((key_singular>1e-6*key_singular.max()).sum());assert 1<=feature_rank<=7
        basis=basis[:feature_rank]
        features=(key-center)@basis.T;scale=features.std(0,unbiased=False).clamp_min(1e-4)
        features=features/scale
        router=torch.linalg.lstsq(features.cpu(),coords.cpu(),rcond=1e-6).solution.to(features)
        branches=[state_direction(warm,g.float(),target) for g in [mean,*modes]]
        meta=dict(warm=warm,branches=branches,seed=seed,basis=basis.float(),center=center.float(),scale=scale.float(),router=router.float())
        v.save_state(v.ROOT/'SOURCE_SPECTRAL_META.pt',meta)
        v.write(v.ROOT/'SPECTRAL_ADMISSION.json',dict(status='PASS',source_edits=8,not_paraphrase_groups=True,
            residual_singular_values=s.tolist(),trainable_TT_parameters=21504,router_parameters=2*feature_rank+2,source_feature_rank=feature_rank,source_key_singular_values=key_singular.tolist(),
            fixed_reference_TT_parameters=7168,gradient_geometry='COMMON_CANONICAL_TT_COORDINATES',source_only=True))
        for arm in ('SHARED','IDENT_ZERO','IDENT_LS'):
            seed_everything(seed);arm_meta=dict(meta,initialize_from_spectrum=arm=='IDENT_LS');model=composite_class()(arm_meta,arm!='SHARED')
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8)
            hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),model);hook.attach()
            try:
                for step in range(1280):
                    task=tasks[step%8];local=step//8;bs=by_owner[task['order']];opt.zero_grad(set_to_none=True)
                    for view in (0,1+local%4):
                        b=bs[view];model.request(int(torch.where(b.labels[0]!=-100)[0][0])-1);hook.set_teacher_routing(b.labels)
                        loss=v.runtime.compute_loss(b);assert torch.isfinite(loss)
                        (.5*loss).backward();v.counts['backwards']+=1
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    opt.step();v.counts['updates']+=1
                    if (step+1)%80==0:
                        v.save_state(v.ROOT/'resume'/arm/'global.pt',dict(state=model.state_dict(),optimizer=opt.state_dict(),step=step+1,meta=arm_meta,counts=dict(v.counts)))
                        v.progress(arm=arm,step=step+1,stage='GLOBAL_FULL_TRAINING')
                path=v.ROOT/'candidates'/arm/'global.pt';v.save_state(path,dict(state=model.state_dict(),meta=arm_meta,signed=arm!='SHARED'))
                saved=r.p.load_state(path)['state'];assert all(torch.equal(saved[k].cpu(),x.cpu()) for k,x in model.state_dict().items())
            finally:hook.detach()
            del model
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=3,global_models=True,all8=True))
    def loader(arm,owner):
        saved=r.p.load_state(v.ROOT/'candidates'/arm/'global.pt')
        model=composite_class()(saved['meta'],saved['signed']);model.load_state_dict(saved['state']);return model
    c.evaluate(tasks,['SHARED','IDENT_ZERO','IDENT_LS'],loader);c.finish(['SHARED','IDENT_ZERO','IDENT_LS'])


if __name__=='__main__':c.run(main)
