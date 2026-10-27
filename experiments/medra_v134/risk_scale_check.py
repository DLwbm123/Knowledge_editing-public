"""Synthetic algebra check of the supplied normalized risk, not a model experiment."""
import sys
from pathlib import Path
import torch


def check(workspace):
    sys.path.insert(0,str(Path(workspace)/'experiments/medical_proposition_v133'))
    import med133_curvature as curvature
    generator=torch.Generator().manual_seed(134)
    block=dict(x=torch.randn(3,7,generator=generator,dtype=torch.float64),
               y=torch.randn(3,1,5,generator=generator,dtype=torch.float64))
    cache=dict(parts=[dict(block,token_counts=[3])])
    b=torch.randn(5,2,generator=generator,dtype=torch.float64)
    a=torch.randn(2,7,generator=generator,dtype=torch.float64)
    trace=curvature.energy(block)
    norm2=float(((b.T@b)*(a@a.T).T).sum())
    raw=float(curvature.penalty(dict(b=b,a=a),cache,lambda s:(s['b'],s['a'])))
    rows=[]
    for alpha in (1.10,1.,.90,.75,.50):
        energy=float(curvature.penalty(dict(b=b*alpha,a=a),cache,lambda s:(s['b'],s['a'])))
        assert abs(energy/(alpha*alpha*raw)-1)<1e-12
        q=energy/(trace*norm2*alpha*alpha)
        assert abs(q/(raw/(trace*norm2))-1)<1e-12
        rows.append(dict(alpha=alpha,quadratic_energy_ratio=energy/raw,
            normalized_q=q,normalized_q_epsilon1e12=energy/(trace*norm2*alpha*alpha+1e-12)))
    return dict(status='PASS',scope='Synthetic algebra only, existing V133 factorized penalty implementation',
        alpha_scaling_quadratic_energy=True,normalized_q_scale_invariant=True,
        no_current_MED_F_reconstruction=True,new_GPU_or_model_calls=0,rows=rows,
        consequence='For fixed F and negligible epsilon, shrinking alpha cannot make q exceedance disappear; high-q experts fall to the prescribed violation fallback')
