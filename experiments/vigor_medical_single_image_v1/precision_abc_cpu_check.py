"""Check ABC isolation, unchanged correction, reconstruction and hard-stop lifecycle."""
from unittest.mock import patch
import copy
import torch
from .cpu_check import Runtime,record
from .editor import calibrate,paired_batches
from .batch8 import baseline_check
from .finite_null import trajectory,correction,native_losses
from .precision_abc import reconstruct,evaluate,compare,attribution


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    rt=Runtime(torch.float16);r=record('trial');name='model.layers.0.mlp.down_proj'
    f=calibrate(rt,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],binding={'model':'synthetic'})['factors'][name]
    f=dict(f,a=f['a']*.001);base=rt.model.layer.weight.detach().clone()
    baseline=baseline_check(rt,r,f)
    old=trajectory(rt,r,f,lambda rows:None)
    target=next(s['step'] for s in old['steps'] if s['correction'])
    prefix=copy.deepcopy(old);prefix['steps']=prefix['steps'][:target]
    prefix['steps'][-1]['status']='REJECTED_AND_STOPPED'
    history=dict(baseline=baseline,method=prefix)
    state=dict(checks=[],prefix=[])
    snap=reconstruct(rt,r,f,history,target,state,lambda:None)
    assert torch.equal(base,rt.model.layer.weight) and state['reconstruction']=='PASS'
    # Synthetic point need not reject: evaluate requires exact agreement with recorded native gates.
    evaluate(rt,r,f,snap,history,state,lambda:None)
    assert state['status']=='COMPLETE' and state['base_restored']
    assert state['correction']['rounded_weights_equal'] and state['correction']['native_weights_equal_rounded']
    assert state['correction']['native_after']==prefix['steps'][-1]['correction']['native_after']
    assert state['correction']['fp32_after']==prefix['steps'][-1]['correction']['fp32_after']
    broken=copy.deepcopy(history);broken['method']['steps'][0]['proposal']['native_after'][0]+=.001
    with patch('experiments.vigor_medical_single_image_v1.precision_abc.correction') as mock:
        try:reconstruct(rt,r,f,broken,target,dict(checks=[],prefix=[]),lambda:None)
        except ValueError as e:assert 'mismatch' in str(e)
        else:raise AssertionError('Mismatch must stop')
        assert mock.call_count==0
    assert torch.equal(base,rt.model.layer.weight)
    for field in ('E','null_correction_energy'):
        try:compare({field:1.01},{field:1.0},'test',[])
        except ValueError:pass
        else:raise AssertionError('Energy gate relaxed')
    out=attribution(dict(fp32_after=[2,.003],rounded_fp32_after=[2,.012],native_after=[2,.023]),
        dict(fp32=[3,0],native=[3,.01]),[3,0],[3,.01],.0001)
    assert out['branches']['A']['all_gates'] and not out['branches']['B']['all_gates']
    assert abs(out['computation_null_effect_adjusted']-.001)<1e-12
    assert abs(out['computation_null_effect_raw']-.011)<1e-12
    assert not rt.model.layer._forward_hooks and not rt.model.layer._forward_pre_hooks
    return dict(status='PASS',unchanged_original_correction=True,rounded_weight_equality=True,
        reconstructed_prefix=True,mismatch_stops_before_correction=True,original_tolerances=True,
        baseline_adjusted_precision_effect=True,base_restoration=True,generation_calls=0,judge_calls=0)
