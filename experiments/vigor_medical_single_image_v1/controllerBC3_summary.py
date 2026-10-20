"""Scalar-only final audit; no inference, no best-checkpoint selection."""
import math
from .native_control import admission,response,next_radius,SCALES


def summarize(result,ended,score,lifecycle):
    checks=[]
    def ck(ok,label):
        checks.append(dict(field=label,passed=bool(ok)))
        if not ok:raise ValueError(label)
    ck(result['status']=='COMPLETE','GPU_terminal')
    ck(all(result['restoration'].values()),'global_restore')
    ck(lifecycle['process_exited'] and lifecycle['gpu_process_exited'],'process_exit')
    ck([c['sample'] for c in result['cases']]==['E01','E02','E03'],'case_order')
    for r in result['checks']:ck(r.get('passed',True) and r.get('absolute_difference',0)<=r.get('tolerance',1e-5),'runtime.'+r['field'])
    rows=[];editing=0;generation=0
    for case in result['cases']:
        ck([a['name'] for a in case['arms']]==['B','C'],'arm_order')
        for arm in case['arms']:
            key=case['sample']+'_'+arm['name'];previous=case['base']['native_nll'];pm=case['base']['margin']['margin'];radius=1.;accepted=[];secants=0;trial_count=0
            for step in arm['steps']:
                ck(abs(step['radius']-radius)<1e-12,key+'.radius')
                ck(max(abs(x-y) for x,y in zip(previous,step['start_native_nll']))<=1e-5,key+'.start_loss')
                ck(abs(step['start_margin']-pm)<=1e-5,key+'.start_margin')
                ck(len(step['trials'])<=3,key+'.trial_limit')
                for idx,t in enumerate(step['trials']):
                    trial_count+=1;ck(abs(t['scale']-radius*SCALES[idx])<1e-12,key+'.scale_order')
                    if 'main_gate' not in t:continue
                    ck(admission(t['cost'],t['linear'],5.368434395105091e-13*t['scale']**2,previous[1]-case['base']['native_nll'][1]),key+'.native_admission')
                    actual=response(previous[0],t['main_gate']['native_nll'][0],pm,t['main_margin']['margin'],t['linear']);ck(actual==t['actual_response'],key+'.main_response')
                    for g,want in [('visual',t['main_gate']['native_nll'][0]<previous[0]),('null',abs(t['main_gate']['native_nll'][1]-case['base']['native_nll'][1])<=.01),('trust',0<=t['main_gate']['combined_native_fisher_cost']<=.001)]:ck(t['main_gate']['gates'][g]==want,key+'.gate_'+g)
                    if t.get('correction') is not None:secants+=1
                    if t['status']=='ACCEPTED':
                        final=response(previous[0],t['final_gate']['native_nll'][0],pm,t['final_margin']['margin'],t['linear'])
                        ck(final==t['final_response'] and final['passed'] and t['final_gate']['accepted'] and all(t['final_gate']['gates'].values()),key+'.final_accept')
                        if t.get('correction') is not None:
                            ck(actual['passed'] and not t['main_gate']['gates']['null'] and t['correction']['status']=='APPLIED',key+'.secant_eligibility')
                        previous=t['final_gate']['native_nll'];pm=t['final_margin']['margin'];radius=next_radius(t['scale'],final)
                        accepted.append(dict(step=step['step'],editing_forward=step['editing_forward_at_accept'],native_nll=previous,margin=pm,rank=t['final_margin']['rank'],probability=t['final_margin']['probability'],original_base_null=previous[1]-case['base']['native_nll'][1],step_F0_cost=t['final_gate']['combined_native_fisher_cost'],endpoint_F0_cost=None,endpoint_F0_cost_note='not_recorded_at_intermediate_checkpoint'))
            ck(len(accepted)==arm['accepted_steps']<=64,key+'.step_count')
            ck(arm['secant_calls']<=64 and (secants==arm['secant_calls'] or (arm['status']=='STOPPED_EDIT_FORWARD_CAP' and arm['secant_calls']==secants+1)),key+'.secant_count_with_at_most_one_budget_interrupted_call')
            ck(arm['base_restored'] and arm['editing_forwards']<=512,key+'.arm_restore_budget')
            ck(previous==arm['last_accepted']['native_nll'] and pm==arm['last_accepted']['margin']['margin'],key+'.last_accepted')
            equal_steps={str(n):next((r for r in accepted if r['step']==n),None) for n in [8,16,32]}
            equal_budget={str(n):next((r for r in reversed(accepted) if r['editing_forward']<=n),None) for n in [128,256,512]}
            observed=score.get('scores',{}).get(key+'_final');g=arm['generations']['final']
            editing+=arm['editing_forwards'];generation+=arm.get('generation_forwards_total',0)
            rows.append(dict(sample=case['sample'],arm=arm['name'],status=arm['status'],accepted_steps=arm['accepted_steps'],editing_forward=arm['editing_forwards'],generation_forward=arm.get('generation_forwards_total',0),backward=arm.get('backward_calls'),secants=secants,proposals=trial_count,NLL=previous[0],margin=pm,original_base_null=previous[1]-case['base']['native_nll'][1],endpoint_F0_cost=arm['last_accepted']['cumulative_F0_cost'],generation=g,semantic_correct=observed,semantic_missing=observed is None,equal_steps=equal_steps,equal_edit_forward_budget=equal_budget))
    ck(editing<=3072 and generation<=1024,'phase_budget')
    ck(ended['model_forward_calls']<=4300 and ended['resident_seconds']<=1800,'global_budget')
    ck(lifecycle['checkpoint_count']==0 and lifecycle['new_disk_bytes']<=134217728,'disk_budget')
    metrics={}
    for arm in ['B','C']:
        metrics[arm]={}
        for label,subset in [('all',[r for r in rows if r['arm']==arm]),('non_E01',[r for r in rows if r['arm']==arm and r['sample']!='E01'])]:
            n=len(subset);correct=sum(r['semantic_correct'] is True for r in subset);missing=sum(r['semantic_missing'] for r in subset)
            metrics[arm][label]=dict(correct=correct,denominator=n,missing=missing,lower=correct/n,upper=(correct+missing)/n)
    return dict(status='COMPLETE',rows=rows,metrics=metrics,editing_forward=editing,generation_forward=generation,total_forward=ended['model_forward_calls'],GPU_seconds=ended['resident_seconds'],runtime_checks=len(result['checks']),offline_checks=len(checks),audit_checks=checks,restoration=result['restoration'],no_new_locality=True)
