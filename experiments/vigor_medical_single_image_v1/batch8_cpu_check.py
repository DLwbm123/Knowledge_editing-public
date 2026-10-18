"""Synthetic checks for frozen selection, paired baselines and failure accounting."""
from unittest.mock import patch
import torch
from .batch8 import run_sample,sample_metrics,stage_diagnostics
from .batch8_selection import select_records
from .cpu_check import Runtime,record,rejects
from .diagnose import DegenerateCandidateError
from .editor import calibrate
from .finite_null_cpu_check import run as check_trajectory


def run():
    check_trajectory()
    rows=[dict(qid=i,img_name=f'g{i//2}/a.png',q_lang='en',question='q',answer='a') for i in range(30)]
    manifest=dict(calibration_source_groups=['SLAKE:g0'],excluded_source_groups=['SLAKE:g1'],
                  calibration_ids=[],excluded_ids=[])
    pilot=dict(relative_image_path='g2/a.png',record_id='train.SLAKE.4')
    selected=select_records(list(reversed(rows)),manifest,pilot,'/synthetic')
    assert [r['record_id'] for r in selected]==[f'train.SLAKE.{i}' for i in range(6,22,2)]
    rejects(lambda:select_records(rows[:16],manifest,pilot,'/synthetic'))
    rows[6]['answer']=''
    rejects(lambda:select_records(rows,manifest,pilot,'/synthetic'))
    torch.manual_seed(42)
    runtime=Runtime(torch.float16);name='model.layers.0.mlp.down_proj'
    factor=calibrate(runtime,[record('train')],[name],train_ids=['train'],excluded_ids=['trial'],
                     binding={'model':'synthetic'})['factors'][name]
    # Only this synthetic fixture rescales F; production reuses the original saved F unchanged.
    factor=dict(factor,a=factor['a']*.001)
    base=runtime.model.layer.weight.detach().clone();snapshots=[]
    state=dict(sample='S01',status='RUNNING')
    run_sample(runtime,record('trial'),factor,state,lambda s:snapshots.append(s['phase']))
    assert state['baseline']['status']=='PASS' and state['base_restored']
    assert state['method']['accepted_steps']==4 and torch.equal(base,runtime.model.layer.weight)
    expected=state['method']['final_accepted_native_nll'][0]<=state['reference']['native_after'][0]
    assert state['metrics']['joint_pass']==expected
    assert snapshots[0]=='BASELINE' and 'REFERENCE' in snapshots and snapshots[-1]=='DONE'
    diag=stage_diagnostics(dict(fp32_after=[0,10.009],native_after=[0,20.011]),dict(fp32=[0,10],native=[0,20]))
    assert diag['null_budget_decision_flip']
    partial=dict(status='DEGENERATE_DIRECTION',method_candidates=[dict(status='ACCEPTED',correction=None,
        final_verdict=dict(cumulative_null_change=.004)),dict(status='PROPOSING',correction=None)])
    metrics=sample_metrics(partial)
    assert metrics['accepted_steps']==1 and abs(metrics['boundary_margin']-.006)<1e-12 and not metrics['joint_pass']
    failed=dict(sample='S02',status='RUNNING')
    with patch('experiments.vigor_medical_single_image_v1.batch8.trajectory',side_effect=DegenerateCandidateError('synthetic')):
        rejects(lambda:run_sample(runtime,record('trial'),factor,failed,lambda s:None))
    assert failed['base_restored'] and torch.equal(base,runtime.model.layer.weight)
    assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
    return dict(status='PASS',synthetic_only=True,frozen_unique_selection=True,no_replacement=True,
        per_sample_baseline=True,reference_restores_base=True,paired_reference_threshold=True,
        partial_failure_accounting=True,separate_precision_baselines=True,exception_restores_base=True,
        finite_trajectory_regression=True,generation_calls=0,judge_calls=0)
