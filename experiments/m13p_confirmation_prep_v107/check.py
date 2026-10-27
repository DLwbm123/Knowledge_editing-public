"""CPU-only equivalence and equal-norm edge checks; no model load."""
import copy
import json
import torch
import module_run as r
import training_math as original
import matched_math as current
from balanced_fisher import remove_opposing


def main():
    for source,penalty in (([1.,0.],[-2.,3.]),([1.,0.],[2.,3.]),
                           ([1.,0.],[0.,3.]),([0.,0.],[2.,3.]),([1.,0.],[0.,0.])):
        g={'a':torch.tensor(source,dtype=torch.float64)}
        h={'a':torch.tensor(penalty,dtype=torch.float64)}
        p,_=remove_opposing(g,h,(('a',),))
        s,a=current.equal_norm(g,h,(('a',),))
        torch.testing.assert_close(s['a'].norm(),p['a'].norm(),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(s['a'],h['a']*a['groups'][0]['equal_norm_ratio'],rtol=0,atol=0)
    rng=torch.Generator().manual_seed(107024)
    cache=dict(x=torch.randn(2,14336,generator=rng),y=torch.randn(2,2,4096,generator=rng),token_counts=[1,1])
    for nonzero in (False,True):
        state=r.p.tr.TT4(225034559,8,8).state_dict()
        if nonzero:state['G1'].fill_(.001)
        state=r.mm.canonicalize(state)[0]
        source={k:torch.randn(v.shape,generator=rng)*.001 for k,v in state.items()}
        scalar=dict(step=0,second_moments=[0.,0.])
        for arm in ('M13B','M13P'):
            a=original.step(r,state,source,cache,.01,copy.deepcopy(scalar),.001,arm)
            b=current.step(r,state,source,cache,.01,copy.deepcopy(scalar),.001,arm)
            assert all(torch.equal(v,b[0][k]) for k,v in a[0].items())
            assert a[1:]==b[1:]
        selected,updated,audit=current.step(r,state,source,cache,.01,copy.deepcopy(scalar),.001,'S')
        assert updated['step']==1 and all(torch.isfinite(v).all() for v in selected.values())
        assert abs(r.fm.map_norm(state,selected)-.001)<=.000002+1e-10
    print(json.dumps(dict(status='PASS',B_P_exact_original=True,S_equal_norm_and_direction=True,
                         zero_source=True,zero_penalty=True,matched_step=True,model_loads=0)))


if __name__=='__main__':main()
