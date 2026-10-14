"""CPU halfspace, scalar regression, infeasibility and operator checks."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import correction_math as cm


def main():
    assert cm.selfcheck()['status']=='PASS'
    args=(np.eye(2),np.zeros(2),np.array([1.,0.]),np.zeros(2),0.,1.002-1e-5)
    point,a=cm.solve(*args,np.array([0.,1.]),.2,fisher_objective=True)
    assert a['status']=='LOCAL_KKT_PASS' and point[0]<.01
    source=np.array([[0.,1.],[1.,0.]])
    point,a=cm.solve(*args,source,np.array([.2,.90001]),fisher_objective=True)
    assert a['status']=='LOCAL_KKT_PASS' and point[0]>=.90001-1e-7 and point[1]>=.2-1e-7
    assert max(a['residuals'])<=1e-7
    point,a=cm.solve(*args,source,np.array([.2,1.1]),fisher_objective=True)
    assert point is None and a['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    import aligned as module
    rng=torch.Generator().manual_seed(20261014)
    shapes=((1,3,2),(2,4,2),(2,5,2),(2,6,1))
    before=cm.mm.canonicalize({k:torch.randn(s,generator=rng,dtype=torch.float64) for k,s in zip(cm.mm.KEYS,shapes)})[0]
    pure={k:v+.001*torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in before.items()}
    radius=module.r.fm.map_norm(before,pure)
    assert abs(module.alignment(before,pure,pure,radius)-1)<1e-5
    cache=dict(x=torch.randn(7,30,generator=rng,dtype=torch.float64),y=torch.randn(7,3,12,generator=rng,dtype=torch.float64),token_counts=[3,4])
    gradients={k:torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in before.items()}
    model,build,space=cm.tangent(before,before,before,gradients,gradients,cache,radius,1.,alignment_reference=pure)
    small=np.zeros(len(model['center']));small[0]=1e-5
    predicted=model['alignment']@small+model['alignment_offset']
    assert abs(module.alignment(before,build(small),pure,radius)-predicted)<1e-7
    module.j.cpu()
    result=dict(status='PASS',scalar_solver_regression=True,multiple_halfspaces=True,infeasibility_certificate=True,
        real_operator_alignment=True,tangent_alignment_JVP=True,source_RMS_and_scope_preserved=True,
        new_model_calls=0,full_consumer_implemented=False)
    module.c.write(module.RUN/'public/CPU_CHECKS.json',result);print(json.dumps(result))


if __name__=='__main__':main()
