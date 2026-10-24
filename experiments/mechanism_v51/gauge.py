"""No-model diagnosis of central TT modulation under the actual canonicalizer."""
import torch

GAMMA=(.55,.85,1.2,1.45)

def diagnose(state, math):
    before={k:v.detach().cpu().double() for k,v in state.items()}
    after,pivots=math.canonicalize(before)
    b,a=math.factors(before);bc,ac=math.factors(after)
    # a = T ac and bc = b T. Never materialize the full 4096 x 14336 map.
    transfer=torch.linalg.solve(ac@ac.T,ac@a.T).T
    diag=torch.diag(torch.tensor(GAMMA,dtype=torch.float64))
    transported=torch.linalg.solve(transfer,diag@transfer)
    rng=torch.Generator().manual_seed(510022)
    x=torch.randn(8,a.shape[1],generator=rng,dtype=torch.float64)
    x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
    def apply(left,right,middle):return (x@right.T)@middle.T@left.T
    identity=torch.eye(4,dtype=torch.float64)
    y0=apply(b,a,identity);y1=apply(bc,ac,identity)
    z0=apply(b,a,diag);z1=apply(bc,ac,diag);zt=apply(bc,ac,transported)
    def rel(left,right):return float((left-right).norm()/left.norm().clamp_min(1e-30))
    torch.testing.assert_close(y0,y1,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(z0,zt,rtol=1e-8,atol=1e-9)
    return dict(base_relative_error=rel(y0,y1),unchanged_diagonal_gamma_relative_change=rel(z0,z1),transported_full_matrix_relative_error=rel(z0,zt),transfer_condition=float(torch.linalg.cond(transfer)),transported_offdiagonal_norm=float((transported-torch.diag(transported.diag())).norm()),gamma=list(GAMMA),pivots=pivots,input_probes=8,model_calls=0,scope='Pure coordinate change only; not a clinical effect or a proposed trained fix')

def selfcheck(math,TT):
    torch.set_num_threads(2)
    base=TT(225034559).double().state_dict();rng=torch.Generator().manual_seed(510022)
    nonzero={k:v.clone() for k,v in base.items()};nonzero['G1']=torch.randn(nonzero['G1'].shape,generator=rng,dtype=torch.float64)*.01
    initial,_=math.canonicalize(nonzero)
    raw={k:v+.005*torch.randn(v.shape,generator=rng,dtype=torch.float64) for k,v in initial.items()}
    result={name:diagnose(s,math) for name,s in [('zero_root',base),('nonzero_initial',nonzero),('canonical_initial',initial),('perturbed_raw',raw)]}
    # A real V50-style RMS proposal from a synthetic differentiable objective.
    current={k:v.clone().requires_grad_() for k,v in initial.items()};b,a=math.factors(current)
    x=torch.randn(3,a.shape[1],generator=rng,dtype=torch.float64);target=torch.randn(3,b.shape[0],generator=rng,dtype=torch.float64)
    loss=((x@a.T)@torch.diag(torch.tensor(GAMMA,dtype=torch.float64))@b.T-target).square().mean()
    gs=torch.autograd.grad(loss,list(current.values()));grads=dict(zip(current,gs))
    total=torch.cat([g.flatten() for g in gs]).norm();scale=min(1.,1./(float(total)+1e-6));grads={k:g*scale for k,g in grads.items()}
    proposal,_=math.canonical_rms_step(initial,grads,dict(step=0,second_moments=[0.,0.]))
    result['actual_rms_synthetic_proposal']=diagnose(proposal,math)
    assert result['perturbed_raw']['unchanged_diagonal_gamma_relative_change']>1e-6
    assert result['canonical_initial']['unchanged_diagonal_gamma_relative_change']<1e-8
    return dict(status='PASS',synthetic_only=True,model_calls=0,medical_effect='NOT_TESTED',cases=result,interpretation='The actual canonicalizer can change nonidentity diagonal conditioning while preserving the unconditioned map. Full matrix transport restores equivalence in this diagnostic; diagonal-only transport generally cannot. No fix applied to V50 or localization writers.')

if __name__=='__main__':
    import sys,json
    from pathlib import Path
    repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(repo/'experiments/medtrace_module_transfer_v1'),str(repo/'.private/evidence_scope_v50_20261022/stage/tools')]
    import module_math
    from alignment import TT
    print(json.dumps(selfcheck(module_math,TT),indent=2))
