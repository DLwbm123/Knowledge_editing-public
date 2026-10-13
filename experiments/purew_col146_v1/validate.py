"""Small CPU checks for persistence, matched direction and dependency gates."""


def resume_check(state,units):
    assert sorted(units)==list(range(1,len(units)+1)) and all(v==160 for v in units.values())
    if state is None:
        assert not units, 'Completed panels without an active state require explicit recovery'
        return
    i,s=state['owner'],state['step']
    assert 1<=i<=146 and 1<=s<=160 and state['committed']==160*(i-1)+s
    assert i in (len(units),len(units)+1)
    if i==len(units):assert s==160
    if state['optimizer'] is not None:assert state['optimizer']['step']==s


def protected_counts(pairs,scores):
    protected=[b for a,b in pairs if scores[a]==1]
    return dict(denominator=len(protected),new_wrong=sum(scores.get(k)==0 for k in protected),
        unknown=sum(scores.get(k) is None for k in protected),passed=all(scores.get(k)==1 for k in protected))


def selfcheck():
    import io
    import torch
    import dependency
    import norm_math
    result=norm_math.selfcheck();dependency.selfcheck()
    resume_check(None,{})
    resume_check(dict(owner=2,step=4,committed=164,optimizer=dict(step=4)),{1:160})
    resume_check(dict(owner=1,step=160,committed=160,optimizer=dict(step=160)),{1:160})
    for state,units in [(None,{1:160}),(dict(owner=2,step=4,committed=164,optimizer=dict(step=3)),{1:160}),
                        (dict(owner=2,step=4,committed=165,optimizer=None),{1:160})]:
        try:resume_check(state,units)
        except AssertionError:pass
        else:raise AssertionError('Broken source/optimizer clock accepted')
    torch.manual_seed(31);w=torch.randn(8,8).half().float();opt=None
    grads=[torch.randn_like(w) for _ in range(5)]
    for g in grads[:3]:w,opt,_=norm_math.update(g,w,.003,'adam_direction',opt)
    state=dict(master=w,native=w.half(),optimizer=opt,rng=torch.get_rng_state(),owner=1,step=3,committed=3,event=dict(step=3))
    data=io.BytesIO();torch.save(state,data);data.seek(0);restored=torch.load(data,weights_only=True)
    assert torch.equal(restored['native'],state['native']) and torch.equal(restored['rng'],state['rng'])
    x=w;y=restored['master'];a=opt;b=restored['optimizer']
    for g in grads[3:]:
        x,a,_=norm_math.update(g,x,.003,'adam_direction',a);y,b,_=norm_math.update(g,y,.003,'adam_direction',b)
        assert torch.equal(x,y) and torch.equal(x.half(),y.half()) and torch.equal(a['m'],b['m'])
    assert protected_counts([('c','n')],{'c':1,'n':None})==dict(denominator=1,new_wrong=0,unknown=1,passed=False)
    assert protected_counts([('c','n')],{'c':1,'n':0})['new_wrong']==1
    result.update(dependency_barrier=True,serialized_native_master_Adam_exact=True,source_clock_corruption_rejected=True,unknown_protection_cannot_pass=True,real_VLM='PENDING_GPU5_AFTER_PREDECESSOR')
    return result


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
