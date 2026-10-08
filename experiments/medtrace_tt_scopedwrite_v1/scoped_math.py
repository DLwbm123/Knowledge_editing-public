"""Fixed-rank TT gauge and post-Adam write geometry; no additional model parameters."""
import torch

ARMS = ('A_FIXED_COORD', 'B_JOINT_HISTORY', 'C_SPLIT_HISTORY', 'D_SPLIT_CONST_GATE', 'E_SPLIT_SCOPE_GATE')
SHAPES = {'G1': (1,64,8), 'G2': (8,64,4), 'G3': (4,112,8), 'G4': (8,128,1)}


def factors(state):
    return (torch.einsum('ia,ajb->ijb',state['G1'][0],state['G2']).reshape(4096,4),
            torch.einsum('aib,bj->aij',state['G3'],state['G4'][:,:,0]).reshape(4,14336))


def canonicalize(state):
    assert {k:tuple(v.shape) for k,v in state.items()} == SHAPES
    old = {k:v.detach().double().clone() for k,v in state.items()}
    _, a = factors(old); gram = a@a.T; eigen = torch.linalg.eigvalsh(gram)
    assert float(eigen.min()) > float(eigen.max())*1e-10, 'Rank-deficient coordinates: stop without truncation'
    lower = torch.linalg.cholesky(gram)
    transform = torch.linalg.solve_triangular(lower,torch.eye(4,dtype=a.dtype,device=a.device),upper=False)
    new = dict(old)
    new['G3'] = (transform@old['G3'].reshape(4,-1)).reshape_as(old['G3'])
    new['G2'] = (old['G2'].reshape(-1,4)@lower).reshape_as(old['G2'])
    _, white = factors(new)
    assert torch.allclose(white@white.T,torch.eye(4,dtype=a.dtype,device=a.device),atol=1e-10,rtol=1e-10)
    return {k:v.to(state[k].dtype) for k,v in new.items()}, dict(
        cores={k:dict(shape=list(v.shape),parameters=v.numel(),side='output' if k in ('G1','G2') else 'input') for k,v in state.items()},
        total_parameters=sum(v.numel() for v in state.values()),trainable_parameters=state['G2'].numel(),
        connection_ranks=[1,8,4,8,1],C_shape=[512,4],local_dimension=2,shared_dimension=2,
        input_gram_eigenvalues=eigen.tolist(),transform_condition=float(torch.linalg.cond(transform)),
        whiten_error=float((white@white.T-torch.eye(4,dtype=a.dtype,device=a.device)).abs().max()))


def gate(text, visual, visual_present=True):
    # Both arguments use the same two shared coordinates, before any key rescaling.
    assert text.shape == visual.shape == (2,)
    nt,nv=text.norm(),visual.norm(); eps=1e-6
    if not visual_present or not torch.isfinite(text).all() or not torch.isfinite(visual).all() or nt<=eps or nv<=eps:
        return dict(gamma=0.,cosine=None,support=0.,closed_invalid=True)
    cos=torch.dot(text,visual)/(nt*nv+eps);support=torch.minimum(nt,nv)/(torch.maximum(nt,nv)+eps)
    return dict(gamma=float(torch.sigmoid(10*(cos-.2))*support),cosine=float(cos),support=float(support),closed_invalid=False)


def precondition(candidate, arm, joint, local, shared, gamma):
    assert arm in ARMS and candidate.shape == (512,4) and 0<=gamma<=1
    d=candidate.double()
    if arm==ARMS[0]: return candidate.clone()
    if arm==ARMS[1]: return (d@joint).to(candidate.dtype)
    return torch.cat((d[:,:2]@local,gamma*(d[:,2:]@shared)),1).to(candidate.dtype)


class History:
    def __init__(self):
        self.joint=torch.eye(4,dtype=torch.float64)
        self.local=torch.eye(2,dtype=torch.float64)
        self.shared=torch.eye(2,dtype=torch.float64)
        self.keys={}

    def absorb(self, binding, key, role):
        assert role in ('native','FIT'), 'Only allowed training inputs can shape the geometry'
        z=key.detach().cpu().double()
        assert z.shape==(4,) and torch.isfinite(z).all() and z.norm()<=1+1e-6
        if binding in self.keys:
            assert torch.equal(z,self.keys[binding]);return False
        for name, value in [('joint',z),('local',z[:2]),('shared',z[2:])]:
            old=getattr(self,name);pz=old@value
            new=old-torch.outer(pz,pz)/(1+torch.dot(value,pz))
            assert torch.allclose(new,new.T,atol=1e-12,rtol=0) and torch.linalg.eigvalsh(new).min()>0
            setattr(self,name,new)
        self.keys[binding]=z.clone();return True

    def state(self):return dict(joint=self.joint,local=self.local,shared=self.shared,keys=self.keys)
    def restore(self,state):
        for name in ('joint','local','shared','keys'):setattr(self,name,state[name])
    def spectrum(self):
        return {name:dict(eigenvalues=torch.linalg.eigvalsh(getattr(self,name)).tolist(),trace=float(getattr(self,name).trace())) for name in ('joint','local','shared')}


def selfcheck(TT4):
    rng=torch.Generator().manual_seed(20261008)
    expert=TT4(42,8,8).double()
    with torch.no_grad():expert.G1.copy_(torch.randn(expert.G1.shape,generator=rng,dtype=torch.float64)*.02)
    state=expert.state_dict();new,audit=canonicalize(state)
    h=torch.randn(5,14336,generator=rng,dtype=torch.float64)
    before=expert.residual(h); expert.load_state_dict(new); after=expert.residual(h)
    assert torch.allclose(before,after,atol=1e-10,rtol=1e-10)
    b,a=expert.factors();x=h/(h.square().mean(-1,keepdim=True).sqrt()+expert.epsilon)
    c=expert.G2.view(512,4);assert c.data_ptr()==expert.G2.data_ptr()
    response=(x@a.T)@c.T
    implicit=torch.einsum('ia,naj->nij',expert.G1[0],response.reshape(-1,8,64)).reshape(-1,4096)
    assert torch.allclose(implicit,after,atol=1e-10,rtol=1e-10)
    split=torch.einsum('ia,naj->nij',expert.G1[0],((x@a[:2].T)@c[:,:2].T+(x@a[2:].T)@c[:,2:].T).reshape(-1,8,64)).reshape(-1,4096)
    assert torch.allclose(split,after,atol=1e-10,rtol=1e-10)
    history=History();candidate=torch.randn(512,4,generator=rng)
    for arm in ARMS:assert torch.equal(precondition(candidate,arm,history.joint,history.local,history.shared,1.),candidate)
    zero=precondition(candidate,ARMS[4],history.joint,history.local,history.shared,0.)
    assert torch.equal(zero[:,:2],candidate[:,:2]) and torch.count_nonzero(zero[:,2:])==0
    zs=[]
    for i in range(5):
        z=torch.nn.functional.normalize(torch.randn(4,generator=rng,dtype=torch.float64),dim=0);zs.append(z)
        assert history.absorb(str(i),z,'FIT') and not history.absorb(str(i),z,'FIT')
    for name,sl in [('joint',slice(None)),('local',slice(0,2)),('shared',slice(2,4))]:
        matrix=torch.eye(len(zs[0][sl]),dtype=torch.float64)+sum(torch.outer(z[sl],z[sl]) for z in zs)
        assert torch.allclose(getattr(history,name),torch.linalg.inv(matrix),atol=1e-12,rtol=1e-12)
    try:history.absorb('bad',zs[0],'CHECK')
    except AssertionError:pass
    else:raise AssertionError('CHECK admitted')
    assert gate(torch.ones(2),torch.zeros(2))['gamma']==0
    assert gate(torch.ones(2),torch.ones(2),False)['gamma']==0
    assert gate(torch.ones(2),-torch.ones(2))['gamma']<1e-4
    # Adam maintains candidate momentum even if the applied shared update is closed.
    param=torch.nn.Parameter(torch.zeros(512,4));opt=torch.optim.Adam([param],lr=.001)
    for step in range(2):
        opt.zero_grad();param.square().sum().add(param.sum()).backward();old=param.detach().clone();opt.step()
        delta=precondition(param.detach()-old,ARMS[4],history.joint,history.local,history.shared,0.)
        with torch.no_grad():param.copy_(old+delta)
    assert torch.count_nonzero(param[:,2:])==0 and opt.state[param]['step']==2
    assert opt.state[param]['exp_avg'][:,2:].abs().sum()>0
    return dict(status='PASS',checks=['TT shapes and rank','invertible gauge double precision','implicit F C A contraction',
        'C shares original G2 storage','zero-effect split','identity P recovery','gamma zero actual shared update',
        'local remains active','Sherman Morrison direct inverse','positive definite symmetric P','training-role isolation',
        'unique-key deduplication','invalid gate closes','closed gate retains Adam candidate momentum'],audit=audit)
