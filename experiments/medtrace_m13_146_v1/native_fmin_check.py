"""CPU checks for source weighting, real-guard branches, and Fisher selection."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import native_fmin as new

p,g,r,c=new.p,new.g,new.r,new.c


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Changed source shape accepted')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    assert g.SOURCE_WEIGHTS==(.25,.25,.5)
    assert g.mean_loss([4.,8.,12.])==9. and g.gain([4.,8.,12.],[2.,6.,10.])==2.
    assert g.source_batches(list(range(7)),1)==[0,1,5]
    new.configure()
    assert g.SOURCE_WEIGHTS==(.5,.5) and g.mean_loss([4.,8.])==6.
    for step in range(1,161):assert g.source_batches(list(range(5)),step)==[0,1+(step-1)%4]
    rejected(lambda:g.mean_loss([1.,2.,3.]));rejected(lambda:g.gain([1.,2.],[1.,2.,3.]))
    assert g.gain([4.,8.],[2.,6.])==2.
    before={'G1':torch.tensor([1.])};pure={'G1':torch.tensor([2.])}
    expert=torch.nn.Module();expert.register_parameter('G1',torch.nn.Parameter(torch.tensor([1.])))
    seen=[]
    def gradients(*args):
        expert.G1.grad=torch.ones_like(expert.G1);return None,[3.,7.]
    def solve(**kw):seen.append(kw['required']);return torch.zeros(1),{'status':'CPU_FIXTURE'}
    with patch.object(g,'source_batches',return_value=[0,1]), \
         patch.object(r,'losses_at',return_value=[2.,6.]), \
         patch.object(r,'source_gradients',side_effect=gradients), \
         patch.object(r.fm,'map_norm',return_value=1.), \
         patch.object(r.mm,'operator_penalty',side_effect=lambda state,cache:state['G1'].square().sum()), \
         patch.object(p.cm,'tangent',return_value=({},lambda _:pure,dict(same_TT_parameter_count=7168,root_dimensions=512))), \
         patch.object(p.cm,'solve',side_effect=solve):
        state,record=p.trial(None,None,expert,before,pure,1,[],[4.,8.],None,None,10.,'MULTI',1.)
        assert state is pure and record['original_rule_pass'] and abs(seen[0]-.799999)<1e-10
    baseline={'value':.5};low={'value':.2};high={'value':.6}
    with patch.object(c,'diagnostic_scope',side_effect=lambda _:nullcontext()), \
         patch.object(r.mm,'operator_penalty',side_effect=lambda state,_:state['value']), \
         patch.object(new.original,'original_correction',return_value=(baseline,{'original':{'pure_source_gain':1.}})):
        args=(None,None,None,None,None,1,None,None,None,None,1.,'FISHER_MIN')
        with patch.object(new.original,'trial',return_value=(low,{'original_rule_pass':True})):
            state,audit=new.original.correction(*args);assert state is low and audit['kind']=='FISHER_MIN_ACCEPTED'
        with patch.object(new.original,'trial',return_value=(high,{'original_rule_pass':True})) as trials:
            state,audit=new.original.correction(*args)
            assert state is baseline and audit['kind']=='MULTI_FALLBACK' and trials.call_count==4
    with patch.object(c,'diagnostic_scope',side_effect=lambda _:nullcontext()), \
         patch.object(r.mm,'operator_penalty',side_effect=lambda state,_:state['value']), \
         patch.object(g,'source_batches',return_value=[0,1]), \
         patch.object(g,'select_guard',return_value=(baseline,{'status':'SKIPPED_INFEASIBLE','pure_source_gain':1.})), \
         patch.object(p,'trial',return_value=(None,{})) as old_trials, \
         patch.object(new.original,'trial',return_value=(None,{})):
        state,audit=new.original.correction(None,None,None,baseline,None,1,None,None,None,None,1.,'FISHER_MIN')
        assert state is baseline and old_trials.call_count==8 and audit['kind']=='MULTI_FALLBACK'
    assert p.f.finite(1.,1.,.1,.1,1.,1.)
    assert not p.f.finite(.8,1.,.1,.1,1.,1.)
    assert not p.f.finite(1.,1.,.2,.1,1.,1.)
    assert not p.f.finite(1.,1.,.1,.1,1.01,1.)
    source=Path(__file__).with_name('native_fmin.py').read_text()
    assert "scalar['step']==step" in source and 'NATIVE_FMIN_FIRST_STEP_REAL' in source
    result=dict(status='PASS',historical_three_loss_defaults_exact=True,native_two_loss_anchor_gain_exact=True,
        cyclic_FIT_160_source_order=True,Fisher_lower_selection_and_fallback=True,MULTI_root_recursive_fallback=True,
        original_Fisher_gain_step_rules_enforced=True,source_shape_changes_rejected=True,new_GPU=0,new_VLM=0,new_Judge=0,
        actual_network_first_step_pending=True)
    Path(os.environ['NATIVE_FMIN_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
