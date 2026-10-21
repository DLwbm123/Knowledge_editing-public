"""Audit fixed precision/position grid and recompute finite-increment residuals."""
import json,math
from pathlib import Path


def summarize(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    result=read('RESULT.json');assert result['status']=='COMPLETE' and result['probe_count']==2
    assert read('PROBE_CONTROL_GATE.json')['both_F1_before_diagnostics']
    rows=[];checks=0
    def verdict(v,start,base):
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        if 'valid_correction' in v['gates']:expected['valid_correction']=False
        assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12
    def trial(t,start,base):
        assert t['proposal']['native_before']==start
        verdict(t['proposal_verdict'],start,base);verdict(t['final_verdict'],start,base)
        assert t['final_verdict']['accepted']
        return t['final_verdict']['native_nll']
    for s in result['arms']:
        assert len(s['steps'])==({'A':8,'D':4}[s['arm']]) and s['base_restored']
        prev=s['baseline']['native']
        for step in s['steps']:
            assert step['accepted'] and step['start_native_nll']==prev
            prev=trial(step['trial'],prev,s['baseline']['native'])
        if s['arm']=='D':prev=trial(s['replay'],prev,s['baseline']['native'])
        assert abs(prev[0]-read('FROZEN_TARGETS.json')['target_nll'])<=.005
        if s['arm']=='D':assert s['replay']['final_verdict']['accepted'] and s['replay_base_restored']
        p=s['probes']['F1'];verdict(p['verdict'],prev,s['baseline']['native'])
        assert p['matching_passed'] and p['endpoint_restored'] and s['probe_base_restored']
        assert p['proposal']['native_norm_matching']['applications']==1
        assert p['proposal']['native_norm_matching']['relative_error']<=.01
        grid=s['grid'];assert grid['base_restored'] and grid['endpoint_restored']
        assert grid['chain_audit']['non_fp32_outputs']==0 and grid['chain_audit']['floating_outputs']>0
        tau=max(1e-5,10*max(grid[k]['baseline_repeat_error'] for k in ('mixed','chain_fp32')))
        values={}
        for precision in ('mixed','chain_fp32'):
            g=grid[precision];assert g['positions']==[0,.5,1] and g['baseline_repeat_error']<=1e-5
            before=g['losses']['0'];values[precision]={}
            for t in (.5,1):
                after=g['losses'][str(t)];change=after[1]-before[1];linear=t*g['null_linear_full'];residual=change-linear
                assert all(math.isfinite(z) for z in [change,linear,residual])
                row=dict(endpoint=s['arm'],precision=precision,t=t,visual_gain=before[0]-after[0],null_change=change,
                    null_linear=linear,null_residual=residual,tau=tau)
                rows.append(row);values[precision][t]=row;checks+=1
            assert abs(g['fp32_update_norm']-p['proposal']['quantization']['fp32_delta_l2'])<1e-7
        for t in (.5,1):
            m=values['mixed'][t];f=values['chain_fp32'][t]
            m['precision_residual_mixed_minus_fp32']=m['null_residual']-f['null_residual']
        checks+=12
    effects={};scaling=[]
    index={(r['endpoint'],r['precision'],r['t']):r for r in rows}
    for endpoint in ('A','D'):
        for precision in ('mixed','chain_fp32'):
            h=index[endpoint,precision,.5];f=index[endpoint,precision,1]
            detectable=abs(h['null_residual'])>f['tau'] and abs(f['null_residual'])>f['tau']
            ratio=h['null_residual']/f['null_residual'] if detectable else None
            scaling.append(dict(endpoint=endpoint,precision=precision,residual_half_over_full=ratio,
                above_noise_floor=detectable,approximately_quadratic=detectable and .20<=ratio<=.30))
    for t in (.5,1):
        effects[str(t)]={}
        for precision in ('mixed','chain_fp32'):
            a=index['A',precision,t];d=index['D',precision,t]
            effects[str(t)][precision]={k:d[k]-a[k] for k in ('visual_gain','null_change','null_linear','null_residual')}
        effects[str(t)]['precision_residual_difference']=effects[str(t)]['mixed']['null_residual']-effects[str(t)]['chain_fp32']['null_residual']
    diffs=[v for s in result['arms'] for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    summary=dict(status='COMPLETE',reference_main=13,fixed_F1_probes=2,readonly_positions=[0,.5,1],precisions=2,
        null_gradient_calls=4,reference_checks=sum(len(s['checks']) for s in result['arms']),reference_max_difference=max(diffs),
        grid_rows=len(rows),all_base_restored=result['all_base_restored'],model_forward_calls=result['model_forward_calls'],
        elapsed_seconds=result['elapsed_seconds'],peak_allocated_bytes=result['peak_allocated_bytes'],scalar_structure_checks=checks)
    assert result['all_base_restored'] and result['generation_calls']==result['judge_calls']==result['half_candidates']==result['full16_runs']==0
    for n,v in [('SUMMARY.json',summary),('READOUTS.json',rows),('CONTRASTS.json',effects),('SCALING.json',scaling),('PROTOCOL_AUDIT.json',dict(status='PASS',checks=checks,no_new_direction=True,original_native_only_gates=True,full_executed_float_outputs_audited=True,private_weight_quadratic_not_independently_recomputed=True))]:
        (root/n).write_text(json.dumps(v,indent=2)+'\n')
    return summary,rows,effects,scaling


def summarize_failed(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    status=read('STATUS.json');states=read('ARMS.json');life=read('LIFECYCLE.json')
    assert status['status']=='FAILED' and status['error_type']=='ValueError'
    assert read('CONTROL_GATE.json')['selected_step_replayed_once']
    assert read('PROBE_CONTROL_GATE.json')['both_F1_before_diagnostics']
    diffs=[v for s in states for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    assert max(diffs)==0 and all(s['base_restored'] and s['probe_base_restored'] for s in states)
    assert all(p['endpoint_restored'] and p['matching_passed'] for s in states for p in s['probes'].values())
    a,d=states;assert len(a['steps'])==8 and len(d['steps'])==4 and len(a['probes'])==len(d['probes'])==1
    checks=0
    def gate(v,start,base):
        nonlocal checks
        drift=v['native_nll'][1]-base[1]
        expected=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        assert v['gates']==expected and v['accepted']==all(expected.values())
        assert abs(v['cumulative_null_change']-drift)<1e-12;checks+=3
    for s in states:
        prev=s['baseline']['native'];trials=[t['trial'] for t in s['steps']]+([s['replay']] if s['arm']=='D' else [])
        for t in trials:
            gate(t['proposal_verdict'],prev,s['baseline']['native']);gate(t['final_verdict'],prev,s['baseline']['native'])
            assert t['proposal']['native_before']==prev and t['final_verdict']['accepted'];prev=t['final_verdict']['native_nll']
        gate(s['probes']['F1']['verdict'],prev,s['baseline']['native'])
    assert a['grid']['base_restored'] and 'chain_fp32' not in a['grid'] and 'grid' not in d
    g=a['grid']['mixed'];b=g['losses']['0'][1];linear=g['null_linear_full']
    rows=[dict(endpoint='A',precision='mixed',t=t,null_change=g['losses'][str(t)][1]-b,null_linear=t*linear,
        null_residual=g['losses'][str(t)][1]-b-t*linear) for t in (.5,1)]
    summary=dict(status='STOPPED_QUALIFICATION_FAILED',reason='A full executed FP32 chain baseline repeat exceeded 1e-5',
        matched_reference_main=13,path_corrections=8,fixed_F1_probes=2,reference_checks=sum(len(s['checks']) for s in states),reference_max_difference=max(diffs),
        scalar_gate_checks=checks,completed_mixed_grids=1,qualified_complete_FP32_grids=0,D_grid_started=False,
        new_candidates_after_failure=0,retries=0,downgrades=0,all_recorded_base_restored=True,
        complete_scientific_contrast_available=False,peak_allocated_bytes=None,repeat_error_numeric_value=None,
        missing_reason='Full FP32 grid rejected before return; values and audit not serialized',
        mixed_A_half_full_residual_ratio=rows[0]['null_residual']/rows[1]['null_residual'])
    assert life['main_process_exited'] and life['own_gpu_process_exited'] and life['within_900_seconds']
    for n,v in [('SUMMARY.json',summary),('PARTIAL_READOUTS.json',rows),('PROTOCOL_AUDIT.json',dict(status='PASS_STOP_ENFORCEMENT',reference_checks_passed=True,scalar_gate_checks=checks,qualification_failed=True,not_scientific_success=True,private_weight_quadratic_not_independently_recomputed=True))]:
        (root/n).write_text(json.dumps(v,indent=2)+'\n')
    return summary,rows
