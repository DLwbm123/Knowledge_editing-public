"""Offline audit of functional rewrites, native behavior, endpoint gates and real damage."""
import json,math
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_responseAnchors3_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());s=json.loads((out/'SCORES.json').read_text());tql=json.loads((root/'reports/vigor_teacherLocality3_v1_20261020/completion/ANALYSIS.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE');check('scoring_terminal',s['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x['passed'])
 for k,x in v['restoration'].items():check('restoration.'+k,x is True)
 rows=[];summaries={};spaces=[]
 for i,t in enumerate(v['teachers']):
  sample=t['sample'];check('source_order',sample==['E01','E02','E03'][i]);check(sample+'.teacher_binding',t['source_output_exact_binding']);check(sample+'.teacher_steps',t['fit_forwards']==[15,17,10][i] and t['backward_calls']==[14,16,9][i]);check(sample+'.base_restore',t['base_restored'])
  sp=t['basis'];coverage=t['key_coverage'];check(sample+'.all_source_positions',sp['positions']==coverage['all_positions']==coverage['prefill_positions']+coverage['all_decode_positions'] and coverage['decode_calls']==t['generation_forwards']-1)
  check(sample+'.space',sp['input_dimension']==14336 and sp['rank']+sp['remaining_input_freedom']==14336 and sp['rank']<=sp['positions'] and sp['orthogonality_residual']<=1e-8);spaces.append(dict(sample=sample,**sp,coverage=coverage))
  check(sample+'.fixed_candidates',[a['name'] for a in t['candidates']]==['A32'])
  for a in t['candidates']:
   name=a['name'];g=a['endpoint'];native=g['native_nll'];drift=native[1]-t['base_native_nll'][1];cost=g['combined_native_fisher_cost'];verdict=dict(null=math.isfinite(drift) and abs(drift)<=.01,visual=math.isfinite(native[0]) and native[0]<t['base_native_nll'][0],trust=math.isfinite(cost) and 0<=cost<=.001)
   check(sample+'.'+name+'.endpoint_recompute',g['gates']==verdict and abs(drift-g['cumulative_null_change'])<=1e-12 and g['accepted']==all(verdict.values())==a['allowed_endpoint_submission'])
   check(sample+'.final_source_action',a['source_action_error']<=1e-8)
   steps=t['repair_steps'];check(sample+'.repair_budget',len(steps)<=25 and t['repair_backwards']==len(steps) and t['accepted_updates']==sum(x['accepted_trust'] for x in steps))
   for j,row in enumerate(steps):
    check(sample+'.Adam_state',row['optimizer_step']==row['step']==j+1 and row['optimizer_state_FP32'])
    check(sample+'.projected_response',row['response_error']<=1e-8 and len(row['per_record_response'])==32)
    check(sample+'.projected_source_action',math.isfinite(row['source_action_error']) and row['source_action_error']<=1e-8)
    check(sample+'.trust_acceptance',row['accepted_trust']==row['endpoint']['gates']['trust'])
   check(sample+'.response_final',a['response_evidence']['response_error']<=1e-8 and len(a['response_evidence']['per_record_response'])==32)
   geo=t['geometry'];eigen=geo['gram_eigenvalues'];retained=[e for e in eigen if e>max(eigen)*1e-10]
   check(sample+'.spectrum',len(eigen)==32 and geo['active_anchors']<=32 and geo['gram_rank']==len(retained))
   check(sample+'.spectral_condition_recompute',(not retained and geo['nonzero_spectral_condition'] is None) or (bool(retained) and abs(geo['nonzero_spectral_condition']/(max(retained)/min(retained))-1)<1e-12))
   check(sample+'.compensation_identity_recompute',abs(geo['actual_compensation_squared']-geo['minimum_compensation_squared'])/max(abs(geo['minimum_compensation_squared']),1e-30)<=1e-8)
   check(sample+'.initial_constraints',t['initial_evidence']['source_action_error']<=1e-8 and t['initial_evidence']['response_error']<=1e-8)
   check(sample+'.native_responses',len(a['response_evidence']['native_response']['per_record'])==32 and all(math.isfinite(z) for z in a['response_evidence']['native_response']['per_record']))
   check(sample+'.Gram_consistency',t['geometry']['linear_residual']<=1e-8 and t['geometry']['compensation_identity_relative_error']<=1e-8)
   if t['repair_status'] in ['NULL_TARGET_REACHED','INITIAL_NULL_PASSED']:check(sample+'.target_stop',abs(drift)<=.01)
   if t['repair_status']=='STEP_CAP_REACHED':check(sample+'.step_cap',len(steps)==25)
   if t['repair_status']=='TRUST_REJECTED_STOP':check(sample+'.trust_stop',steps and not steps[-1]['accepted_trust'] and verdict['trust'])
   check(sample+'.'+name+'.native_finite',all(math.isfinite(a[k]) for k in ['native_action_relative_error','actual_layer_output_relative_error','actual_layer_output_max_abs','fixed_prefix_logits_max_abs','fixed_prefix_logits_relative_error']))
   ev=[x for x in v['evaluation'] if x['sample']==sample and x['arm']==name];check(sample+'.'+name+'.six_complete',len(ev)==6 and all(x['status']=='COMPLETE' for x in ev));damaged=[x['panel_id'] for x in ev if s['scores'].get(x['alias']) is False];missing=sum(s['scores'].get(x['alias']) is None for x in ev);semantic=s['scores'].get(sample+'_'+name+'_T0')
   rows.append(dict(sample=sample,arm=name,semantic_T0=semantic,source_exact_teacher=a['source_exact_teacher'],source_action_error=a['source_action_error'],geometry=t['geometry'],response_evidence=a['response_evidence'],repair_status=t['repair_status'],repair_steps=len(t['repair_steps']),repair_backwards=t['repair_backwards'],accepted_updates=t['accepted_updates'],native_action_relative_error=a['native_action_relative_error'],actual_layer_output_relative_error=a['actual_layer_output_relative_error'],actual_layer_output_max_abs=a['actual_layer_output_max_abs'],fixed_prefix_logits_max_abs=a['fixed_prefix_logits_max_abs'],fixed_prefix_logits_relative_error=a['fixed_prefix_logits_relative_error'],native_source_NLL=native[0],null_drift=drift,endpoint_F0_cost=cost,endpoint_gates=verdict,allowed_endpoint_submission=g['accepted'],damage=len(damaged),damaged_panel=damaged,locality_denominator=6,locality_missing=missing))
 for name in ['A32']:
  rr=[x for x in rows if x['arm']==name];damage={(x['sample'],p) for x in rr for p in x['damaged_panel']};source=sum(x['semantic_T0'] is True for x in rr);source_missing=sum(x['semantic_T0'] is None for x in rr);missing=sum(x['locality_missing'] for x in rr);eligible=sum(x['allowed_endpoint_submission'] for x in rr)
  summaries[name]=dict(semantic_correct=source,semantic_denominator=3,semantic_missing=source_missing,strict_function_preserved=sum(x['source_exact_teacher'] for x in rr),null_pass=sum(x['endpoint_gates']['null'] for x in rr),trust_pass=sum(x['endpoint_gates']['trust'] for x in rr),endpoint_eligible=eligible,damage=len(damage),locality_denominator=18,locality_missing=missing,prespecified_useful=source==3 and not source_missing and not missing and eligible==3 and damage<={('E01','L006')})
 check('21_complete_scores',len(s['scores'])==21 and not s['missing']);check('three_candidates',len(rows)==3)
 for e in s['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=25);check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 fit=sum(t['fit_forwards'] for t in v['teachers']);teacher_gen=sum(t['generation_forwards'] for t in v['teachers']);candidate_gen=sum(a['generation_forwards'] for t in v['teachers'] for a in t['candidates']);local_gen=sum(e['generation_forwards'] for e in v['evaluation']);repair_backward=sum(t['repair_backwards'] for t in v['teachers']);controls=end['model_forward_calls']-fit-teacher_gen-candidate_gen-local_gen-3*repair_backward-32
 check('global_budget',end['model_forward_calls']<=4500 and end['resident_seconds']<=1800 and controls>=0 and fit<=78)
 previous=json.loads((root/'reports/vigor_scalarNull3_v1_20261020/completion/ANALYSIS.json').read_text());old_damage={(x['sample'],p) for x in previous['rows'] for p in x['damaged_panel']};new_damage={(x['sample'],p) for x in rows for p in x['damaged_panel']}
 check('calibration_exact_cost',v['calibration']['records']==v['calibration']['forwards']==v['calibration']['backwards']==32 and v['calibration']['raw_records_released'] and v['calibration']['seconds']<=240)
 analysis=dict(S_baseline=previous['summary'],repaired_S_damage=[dict(sample=x,panel=p) for x,p in sorted(old_damage-new_damage)],new_vs_S_damage=[dict(sample=x,panel=p) for x,p in sorted(new_damage-old_damage)],same_damage_set_as_S=old_damage==new_damage,rows=rows,summary=summaries,source_spaces=spaces,T_baseline=tql['summary'],costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],calibration_forwards=32,calibration_backwards=32,calibration_seconds=v['calibration']['seconds'],teacher_fit=fit,teacher_backwards=sum(t['backward_calls'] for t in v['teachers']),repair_backwards=repair_backward,repair_gradient_forwards=repair_backward,repair_native_gate_forwards=2*repair_backward,teacher_source_binding_generation=teacher_gen,candidate_source_generation=candidate_gen,candidate_locality_generation=local_gen,control_binding_prefix_restore=controls,peak_allocated_bytes=end['peak_allocated_bytes'],new_judge_payloads=s['new_payloads'],cached_endpoints=sum(x!='new_blind_score' for x in s['provenance'].values()),checkpoint=0,old_T_panel_reconstruction_forwards=0),historical_NA_unchanged=True,near_locality_support=0,distinct_unrelated_queries=6,scope='32 fixed per-record first-order training NLL anchors with fixed 25-step projected Adam of scalar null loss inside the successful source-action class; not strict null hidden-action preservation or old source controller')
 audit=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',check_count=len(checks),failed=[x for x in checks if not x['passed']],checks=checks)
 for n,x in [('ANALYSIS.json',analysis),('AUDIT.json',audit)]:(out/n).write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
