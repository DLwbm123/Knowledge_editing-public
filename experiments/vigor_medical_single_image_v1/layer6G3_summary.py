"""Offline predictive-screen ablation audit, actual safety and pre-divergence prefix."""
import json,math,collections
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_layer6G3_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());s=json.loads((out/'SCORES.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE');check('scoring_terminal',s['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x.get('passed',math.isfinite(x.get('absolute_difference',float('inf'))) and x.get('absolute_difference',float('inf'))<=x.get('tolerance',0)))
 for k,t in v['restoration'].items():check('restore.'+k,t is True)
 rows=[];override_events=[];prefix=[]
 prior=json.loads((root/'reports/vigor_layer6N3_v1_20261020/completion/RESULT.json').read_text())
 for case in v['cases']:
  sample=case['sample'];a=case['arms'][0];check(sample+'.original_base',a['base_restored']);check(sample+'.edit_bound',a['editing_forwards']<=512)
  accepted=0;predictive=[]
  old=next(c['arms'][0] for c in prior['cases'] if c['sample']==sample);diverged=False;matched=0;first=None
  for step_index,step in enumerate(a['steps']):
   if not diverged:
    check(sample+'.prefix_step_exists',step_index<len(old['steps']))
    prev=old['steps'][step_index]
    check(sample+'.prefix_start_native',all(abs(x-y)<=1e-5 for x,y in zip(step['start_native_nll'],prev['start_native_nll'])))
    check(sample+'.prefix_radius',abs(step['radius']-prev['radius'])<=1e-12)
    for j,t in enumerate(step['trials']):
     if t.get('only_null_prediction_rejected'):
      diverged=True;first=dict(step=step['step'],trial=j+1,scale=t['scale']);break
     check(sample+'.prefix_trial_exists',j<len(prev['trials']))
     pt=prev['trials'][j]
     check(sample+'.prefix_trial',t['status']==pt['status'] and abs(t['scale']-pt['scale'])<=1e-12)
     for field in ['main_gate','final_gate']:
      if field in t:
       check(sample+'.prefix_'+field,field in pt and all(abs(x-y)<=1e-5 for x,y in zip(t[field]['native_nll'],pt[field]['native_nll'])))
    if not diverged:
     check(sample+'.prefix_accept_path',step['accepted']==prev['accepted'] and len(step['trials'])==len(prev['trials']));matched+=1
   for t in step['trials']:
    if not t.get('only_null_prediction_rejected'):continue
    measured='main_gate' in t
    outcome=('COMPENSATED_ACCEPTED' if t.get('correction') is not None else 'DIRECT_ACCEPTED') if t['status']=='ACCEPTED' else 'REJECTED' if measured else 'INCOMPLETE_UNKNOWN'
    override_events.append(dict(sample=sample,step=step['step'],scale=t['scale'],source='old_rule_applied_to_new_G_trajectory',predicted_null=t['predicted_original_base_null'],outcome=outcome,status=t['status'],main_gates=t.get('main_gate',{}).get('gates'),final_gates=t.get('final_gate',{}).get('gates'),correction_status=(t.get('correction') or {}).get('status')))
   for trial in step['trials']:
    if trial['status']=='NATIVE_ADMISSION_REJECTED':
     predictive.append(dict(step=step['step'],scale=trial['scale'],cost_pass=0<trial['cost']<=1.01*5.368434395105091e-13*trial['scale']**2,source_prediction_pass=trial['linear'][0]<-1e-5,null_prediction_pass=abs(trial['predicted_original_base_null'])<=.01,actual_unmeasured=True))
   tt=[t for t in step['trials'] if t['status']=='ACCEPTED'];check(sample+'.unique_accept',len(tt)==int(step['accepted']))
   if not tt:continue
   t=tt[0];accepted+=1;check(sample+'.source_native_prediction',t['linear'][0]<-1e-5)
   for name in ['actual_response','final_response']:
    r=t[name];check(sample+'.'+name,r['passed'] and r['NLL_gain']>1e-5 and r['rho_NLL']>=.1)
   gg=t['final_gate'];check(sample+'.original_gates',gg['accepted'] and all(gg['gates'].values()) and abs(gg['cumulative_null_change'])<=.01 and gg['combined_native_fisher_cost']<=.001)
   check(sample+'.final_native_repeat',abs(t['final_margin']['mean_nll']-gg['native_nll'][0])<=1e-5)
  prefix.append(dict(sample=sample,complete_matched_steps=matched,first_screen_divergence=first))
  check(sample+'.accept_count',accepted==a['accepted_steps'])
  ev=[e for e in v['evaluation'] if e['sample']==sample];check(sample+'.six_panel',len(ev)==6);values=[s['scores'].get(e['alias']) for e in ev];last=a['last_accepted'];t0=s['scores'].get(sample+'_L6G_T0')
  rows.append(dict(sample=sample,arm='L6G',semantic_T0=t0,locality_damage=sum(x is False for x in values),locality_denominator=6,locality_missing=sum(x is None for x in values),damaged_panel=[e['panel_id'] for e,x in zip(ev,values) if x is False],accepted_steps=accepted,editing_forwards=a['editing_forwards'],backward_calls=a['backward_calls'],secant_calls=a['secant_calls'],native_source_NLL=last['native_nll'][0],tracked_margin=last['margin']['margin'],original_base_null_drift=last['native_nll'][1]-case['base']['native_nll'][1],endpoint_F0_cost=last['cumulative_F0_cost'],status=a['status'],all_accepted_original_gates=True,trial_status_counts=dict(collections.Counter(t['status'] for st in a['steps'] for t in st['trials'])),predictive_rejections=predictive))
 check('source_subset',[r['sample'] for r in rows]==['E01','E02','E03']);check('21_endpoints',len(s['scores'])==21 and not s['missing']);check('18_locality',len(v['evaluation'])==18)
 for e in s['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=21);check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 originalN=json.loads((root/'reports/vigor_factF3_v1_20261020/completion/ANALYSIS.json').read_text());nr=[r for r in originalN['rows'] if r['arm']=='N']
 baseline=dict(semantic_correct=sum(r['semantic_T0'] is True for r in nr),semantic_denominator=3,damage=sum(r['damage'] for r in nr),locality_denominator=18,editing_forwards=1350,reconstruction_not_repeated=True,source='previous_N3_T0_and_F3_fixed_trace_locality_binding')
 summary=dict(semantic_correct=sum(r['semantic_T0'] is True for r in rows),semantic_denominator=3,semantic_missing=sum(r['semantic_T0'] is None for r in rows),damage=sum(r['locality_damage'] for r in rows),locality_denominator=18,locality_missing=sum(r['locality_missing'] for r in rows),editing_forwards=sum(r['editing_forwards'] for r in rows),backward_calls=sum(r['backward_calls'] for r in rows),secant_calls=sum(r['secant_calls'] for r in rows))
 edits=summary['editing_forwards'];gens=sum(e['generation_forwards'] for e in v['evaluation'])+sum(a['generation_forwards_total'] for c in v['cases'] for a in c['arms'])
 costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],editing=edits,final_T0_and_locality_generation=gens,control_restore=end['model_forward_calls']-edits-gens,peak_allocated_bytes=end['peak_allocated_bytes'],new_judge_payloads=s['new_payloads'],cached_endpoints=sum(x!='new_blind_score' for x in s['provenance'].values()),teacher_fitting_forwards=0,training_rollout_forwards=0,checkpoint=0,per_arm_seconds=None)
 check('global_budget',end['model_forward_calls']<=4500 and end['resident_seconds']<=1800 and costs['control_restore']>=0)
 damage_set={(r['sample'],x) for r in rows for x in r['damaged_panel']};useful=not summary['semantic_missing'] and not summary['locality_missing'] and next(r['semantic_T0'] for r in rows if r['sample']=='E03') is True and any(r['semantic_T0'] is True for r in rows if r['sample'] in ['E01','E02']) and damage_set<={('E01','L006')}
 analysis=dict(rows=rows,L6G=summary,L6N_baseline=dict(semantic_correct=0,semantic_denominator=3,damage=1,locality_denominator=18,editing_forwards=513),shared_prefix=prefix,override_events=override_events,override_outcomes=dict(collections.Counter(e['outcome'] for e in override_events)),historical_unmeasured_candidates_remain_NA=True,N_baseline=baseline,fixed_module='model.layers.6.mlp.down_proj',legacy_comparison_module='model.layers.0.mlp.down_proj',prespecified_useful=useful,original_per_layer_training_F0_reused=True,layer_scan=False,costs=costs,near_locality_support=0,distinct_unrelated_queries=6,scope='Three exposed development edits; only predictive null screening disabled, actual protection unchanged; no historical NA filled')
 audit=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),failed=[c for c in checks if not c['passed']]);(out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n');(out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
