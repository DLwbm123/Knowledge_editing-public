"""Rank/JB objectives on frozen reference prefixes; explicitly not on-policy OPD."""
import time
import route_common as c
v=c.v
if (v.ROOT/'ENGINEERING_CARRIED_COUNTS.json').exists():v.counts.update(v.read(v.ROOT/'ENGINEERING_CARRIED_COUNTS.json'))


def weights(scores):
    torch=v.torch;x=torch.cat([s[:-1] for s in scores if len(s)>1]).double().clamp_min(1e-12).log().cpu()
    assert len(x)>1
    mean=torch.quantile(x,torch.tensor([.25,.75],dtype=torch.double));var=torch.full((2,),max(float(x.var(unbiased=False)),1e-4));mix=torch.full((2,),.5)
    for _ in range(32):
        log=-.5*((x[:,None]-mean)**2/var+var.log())+mix.log();post=log.softmax(-1)
        mass=post.sum(0).clamp_min(1e-6);mean=(post*x[:,None]).sum(0)/mass
        var=(post*(x[:,None]-mean)**2).sum(0)/mass;var=var.clamp_min(1e-4);mix=mass/mass.sum()
    j=int(mean.argmax());result=[]
    for s in scores:
        z=s.double().clamp_min(1e-12).log().cpu();log=-.5*((z[:,None]-mean)**2/var+var.log())+mix.log()
        p=log.softmax(-1)[:,j].float();p[-1]=1.;result.append(p)
    return result,dict(means=mean.tolist(),variances=var.tolist(),mix=mix.tolist(),EM_steps=32,final_token_J=1.)


def token_log(out,b):return out.logits[:,:-1][b.labels[:,1:]!=-100].float().log_softmax(-1)


def rank_loss(log,q):
    torch=v.torch;tq,ids=q.topk(16,dim=-1);t=tq.exp();student=log.gather(1,ids)
    cumulative=t.cumsum(-1);head=(cumulative<.9).sum(-1).add(1).clamp_max(16)
    pair=torch.triu(torch.ones(16,16,device=log.device,dtype=torch.bool),diagonal=1)
    valid=pair[None]&(torch.arange(16,device=log.device)[None,:,None]<head[:,None,None])
    gap=tq[:,:,None]-tq[:,None,:];valid=valid&(gap>0)
    margin=torch.minimum(gap,1+4*(1-torch.exp(-gap/.4)))
    violation=(margin-(student[:,:,None]-student[:,None,:])).clamp_min(0)
    ranking=(violation*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    Q=t.sum(-1).clamp(1e-7,1-1e-7);P=student.exp().sum(-1).clamp(1e-7,1-1e-7)
    return ranking+.1*(torch.logit(Q)-torch.logit(P)).clamp_min(0)


def objective(log,q,z,arm):
    torch=v.torch;p=log.exp();rkl=(p*(log-q)).sum(-1)
    if arm=='RKL':return rkl.mean()
    if arm=='RANK':return rank_loss(log,q).mean()
    tq,ids=q.topk(16,-1);lp=log.gather(1,ids)
    # Original full-vocabulary teacher mass on Top16, not renormalized.
    fkl=(tq.exp()*(tq-lp)).sum(-1)
    qt=(1-tq.exp().sum(-1)).clamp_min(1e-7);pt=(1-lp.exp().sum(-1)).clamp_min(1e-7)
    fkl=fkl+qt*(qt.log()-pt.log())
    error=(lp[:,:,None]-lp[:,None,:])-(tq[:,:,None]-tq[:,None,:])
    rel=(error.square().sum((1,2)))/(16*15)
    return (z*rkl+(1-z)*(fkl+.1*rel)).mean()


def main():
    tasks,items,by_owner,batches=c.prepare();torch=v.torch;r=v.r
    from m3bench_repro.editors.llava_runtime import seed_everything
    import diagnosis
    teacher={};sens=[];ordered=[]
    for task in tasks:
        owner=task['order'];rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)];fact=v.read(v.ROOT/'FACTS.json')[str(owner)]
        conditioned=[dict(row,question=diagnosis.short_context(fact,'S3')+'\n\nQuestion: '+row['question']) for row in rows]
        ci=v.prepare(task,conditioned,True)
        for view,(row,item,b) in enumerate(zip(rows,ci,by_owner[owner])):
            key=f'{owner}:{view}';answer=diagnosis.greedy(item)
            v.write(v.ROOT/'qualification'/f'{owner}_{view}.json',dict(key=key,source=row,source_facts=fact['medical_record'],**answer))
            cb=r.old.batch_on(v.runtime,item[1])
            with torch.no_grad():q=token_log(v.runtime.model(**cb.forward_kwargs()),cb).detach().cpu()
            assert torch.equal(cb.labels[cb.labels!=-100].cpu(),b.labels[b.labels!=-100].cpu())
            teacher[key]=q;ordered.append((key,b))
            v.progress(stage='TEACHER_SOURCE_QUALIFICATION',owner=owner,view=view)
    v.save_state(v.ROOT/'FROZEN_TEACHER_LOGP.pt',teacher)
    v.write(v.ROOT/'SOURCE_REVIEW_READY.json',dict(count=40,full_source_required=True,time=time.time()))
    while not (v.ROOT/'SOURCE_REVIEW_FROZEN.json').exists():
        v.budget();time.sleep(3)
    labels=v.read(v.ROOT/'SOURCE_REVIEW_FROZEN.json');assert set(labels['pass'])==set(teacher)
    if not any(labels['pass'].values()):
        v.write(v.ROOT/'CLOSED_NO_QUALIFIED_TEACHER.json',dict(count=0,no_training=True,time=time.time()));return
    for key,b in ordered:
        kwargs=b.forward_kwargs();embedded=kwargs['inputs_embeds'].detach().clone().requires_grad_(True);kwargs=dict(kwargs,inputs_embeds=embedded)
        out=v.runtime.model(**kwargs);log=token_log(out,b);target=b.labels[b.labels!=-100]
        value=-log.gather(1,target[:,None]).sum()
        grad=torch.autograd.grad(value,embedded)[0];v.counts['backwards']+=1
        positions=torch.where(b.labels[0]!=-100)[0]
        top=log.detach().topk(2,-1).indices;alt=torch.where(top[:,0]==target,top[:,1],top[:,0])
        direction=v.runtime.model.get_input_embeddings()(alt).detach()-v.runtime.model.get_input_embeddings()(target).detach()
        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        score=(grad[0,positions]*direction).sum(-1).abs().detach().cpu();sens.append(score)
        del out,log,value,grad,embedded
    zz,audit=weights(sens);zmap={key:z for (key,_),z in zip(ordered,zz)}
    v.save_state(v.ROOT/'FROZEN_JUNCTION_POSTERIOR.pt',dict(weights=zmap,sensitivity=dict(zip(teacher,sens)),audit=audit))
    v.write(v.ROOT/'SUPERVISION_FROZEN.json',dict(qualified=sum(labels['pass'].values()),total=40,reference_prefix=True,on_policy=False,teacher='SOURCE_CONDITIONED_FROZEN_BASE',time=time.time()))
    arms=['CE','RKL','RANK','JB']
    with c.CachedSource(batches):
        for task in tasks:
            owner=task['order'];bs=by_owner[owner];original=r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
            for arm in arms:
                seed_everything(task['seed']);expert=r.p.expert(original,task['seed'],v.runtime.device);expert.load_state_dict(r.mm.canonicalize(original)[0])
                opt=torch.optim.Adam([{'params':[p for k,p in expert.named_parameters() if k in ('G1','G2')],'lr':1e-3},{'params':[p for k,p in expert.named_parameters() if k in ('G3','G4')],'lr':1e-4}],betas=(.9,.999),eps=1e-8)
                hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach()
                try:
                    for step in range(160):
                        opt.zero_grad(set_to_none=True)
                        for view in (0,1+step%4):
                            b=bs[view];key=f'{owner}:{view}';hook.set_teacher_routing(b.labels)
                            out=v.runtime.model(**b.forward_kwargs());log=token_log(out,b);target=b.labels[b.labels!=-100]
                            ce=-log.gather(1,target[:,None]).mean();loss=ce
                            if arm!='CE' and labels['pass'][key]:loss=.5*ce+.5*objective(log,teacher[key].to(log.device),zmap[key].to(log.device),arm)
                            assert torch.isfinite(loss);(.5*loss).backward();v.counts['backwards']+=1
                        torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);opt.step();v.counts['updates']+=1
                        if (step+1)%20==0:
                            v.save_state(v.ROOT/'resume'/arm/f'{owner}.pt',dict(state=r.old.snapshot(expert),optimizer=opt.state_dict(),step=step+1,counts=dict(v.counts),rng=r.c.rng()))
                            v.progress(stage='FULL_METHOD_TRAINING',owner=owner,arm=arm,step=step+1)
                    state=r.old.snapshot(expert);path=v.ROOT/'candidates'/arm/f'{owner}.pt';v.save_state(path,dict(state=state,owner=owner,arm=arm))
                    saved=r.p.load_state(path)['state'];assert all(torch.equal(state[k].cpu(),saved[k].cpu()) for k in state)
                finally:hook.detach()
                del expert
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=32,all8=True,time=time.time()))
    def loader(arm,owner):return r.p.expert(r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state'],owner,v.runtime.device)
    c.evaluate(tasks,arms,loader);c.finish(arms)


if __name__=='__main__':c.run(main)
