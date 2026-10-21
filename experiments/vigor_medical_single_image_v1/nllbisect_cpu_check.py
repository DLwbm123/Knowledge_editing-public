"""Eight-candidate cap, deterministic bisection decisions, first match and safe stops."""
from unittest.mock import patch
from unittest import TestCase
import torch
from .nllmatch_cpu_check import run as prior_checks
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .finite_null import native_losses
from .nllbisect import search,UPPER


def run():
    prior_checks();torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('test');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('a'),record('b',image=.2,target='3')],[name],train_ids=['a','b'],excluded_ids=['test'],binding={'model':'synthetic'})['factors'][name]
    pair=paired_batches(rt,r);base=rt.model.layer.weight.detach().clone();base_losses=native_losses(rt,pair)
    with torch.no_grad():rt.model.layer.weight.add_(.002)
    snapshot=rt.model.layer.weight.detach().clone();losses=native_losses(rt,pair)
    with torch.no_grad():rt.model.layer.weight.copy_(base)
    for mode in ('success','cap','gate','nonmonotonic'):
        seen=[];target=losses[0]-.4
        def attempt(*args,**kw):
            assert args[11] is None and kw['correction_factor'] is None and 'native_norm_target' not in kw
            assert torch.equal(rt.model.layer.weight,snapshot)
            alpha=args[7]/.7;seen.append(alpha)
            assert alpha==UPPER if len(seen)==1 else 0<alpha<UPPER
            loss=losses[0]-.8*alpha
            if mode=='cap':loss=losses[0]-(.8 if alpha>=.5 else 0)
            if mode=='nonmonotonic' and len(seen)==2:loss=losses[0]+.1
            with torch.no_grad():rt.model.layer.weight.add_(.01)
            v=dict(native_nll=[loss,losses[1]],accepted=not(mode=='gate' and len(seen)==2))
            args[9]['final_verdict']=v
            return v
        state=dict(arm='D',baseline=dict(native=base_losses),endpoint_native_nll=losses,checks=[])
        with patch('experiments.vigor_medical_single_image_v1.nllbisect._attempt',side_effect=attempt),patch('experiments.vigor_medical_single_image_v1.nllbisect.control') as ctrl:
            selected=search(rt,r,f,snapshot,state,target,.7,{},lambda:None,None)
            assert ctrl.call_count==1
        q=state['search'];assert all(c['rollback_pass'] for c in q['candidates']) and state['search_base_restored']
        if mode=='success':assert selected is not None and q['accepted'] and len(seen)<8 and q['candidates'][-1]['error']<=.005
        elif mode=='cap':assert selected is None and len(seen)==8 and state['status']=='STOPPED_CANDIDATE_CAP'
        else:assert selected is None and len(seen)==2 and state['status']==('STOPPED_GATE_REJECTED' if mode=='gate' else 'STOPPED_NON_MONOTONIC')
        assert torch.equal(base,rt.model.layer.weight) and native_losses(rt,pair)==base_losses
    with TestCase().assertRaises(TimeoutError):search(rt,r,f,snapshot,{},losses[0]-.4,.7,{},lambda:None,0)
    assert torch.equal(base,rt.model.layer.weight)
    return dict(status='PASS',prior_checks=True,upper_counts_in_eight_candidate_cap=True,midpoint_only=True,
        first_match_stops=True,original_gate_failure_stops=True,observed_nonmonotonic_stops=True,
        all_candidates_same_start_and_roll_back=True,no_native_rescale_in_search=True,base_and_deadline_safe=True)
