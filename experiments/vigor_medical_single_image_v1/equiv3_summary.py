"""Offline audit of functional rewrites, native behavior, endpoint gates and real damage."""
import json,math
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_equivTIU3_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());s=json.loads((out/'SCORES.json').read_text());tql=json.loads((root/'reports/vigor_teacherLocality3_v1_20261020/completion/ANALYSIS.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE');check('scoring_terminal',s['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x['passed'])
 for k,x in v['restoration'].items():check('restoration.'+k,x is True)
 rows=[];summaries={};spaces=[]
 for i,t in enumerate(v['teachers']):
  sample=t['sample'];check('source_order',sample==['E01','E02','E03'][i]);check(sample+'.teacher_binding',t['source_output_exact_binding']);check(sample+'.teacher_steps',t['fit_forwards']==[15,17,10][i] and t['backward_calls']==[14,16,9][i]);check(sample+'.base_restore',t['base_restored'])
  sp=t['basis'];coverage=t['key_coverage'];check(sample+'.all_source_positions',sp['positions']==coverage['all_positions']==coverage['prefill_positions']+coverage['all_decode_positions'] and coverage['decode_calls']==t['generation_forwards']-1)
  check(sample+'.space',sp['input_dimension']==14336 and sp['rank']+sp['remaining_input_freedom']==14336 and sp['rank']<=sp['positions'] and sp['orthogonality_residual']<=1e-8);spaces.append(dict(sample=sample,**sp,coverage=coverage))
  check(sample+'.fixed_candidates',[a['name'] for a in t['candidates']]==['I','U'])
  for a in t['candidates']:
   name=a['name'];g=a['endpoint'];native=g['native_nll'];drift=native[1]-t['base_native_nll'][1];cost=g['combined_native_fisher_cost'];verdict=dict(null=math.isfinite(drift) and abs(drift)<=.01,visual=math.isfinite(native[0]) and native[0]<t['base_native_nll'][0],trust=math.isfinite(cost) and 0<=cost<=.001)
   check(sample+'.'+name+'.endpoint_recompute',g['gates']==verdict and abs(drift-g['cumulative_null_change'])<=1e-12 and g['accepted']==all(verdict.values())==a['allowed_endpoint_submission'])
   sol=a['solve'];check(sample+'.'+name+'.solve',sol['solve_residual']<=1e-8 and sol['source_action_relative_error']<=1e-8 and sol['candidate_metric_cost']<=sol['teacher_metric_cost']*(1+1e-8))
   check(sample+'.'+name+'.native_finite',all(math.isfinite(a[k]) for k in ['native_action_relative_error','actual_layer_output_relative_error','actual_layer_output_max_abs','fixed_prefix_logits_max_abs','fixed_prefix_logits_relative_error']))
   ev=[x for x in v['evaluation'] if x['sample']==sample and x['arm']==name];check(sample+'.'+name+'.six_complete',len(ev)==6 and all(x['status']=='COMPLETE' for x in ev));damaged=[x['panel_id'] for x in ev if s['scores'].get(x['alias']) is False];missing=sum(s['scores'].get(x['alias']) is None for x in ev);semantic=s['scores'].get(sample+'_'+name+'_T0')
   rows.append(dict(sample=sample,arm=name,semantic_T0=semantic,source_exact_teacher=a['source_exact_teacher'],solve=sol,native_action_relative_error=a['native_action_relative_error'],actual_layer_output_relative_error=a['actual_layer_output_relative_error'],actual_layer_output_max_abs=a['actual_layer_output_max_abs'],fixed_prefix_logits_max_abs=a['fixed_prefix_logits_max_abs'],fixed_prefix_logits_relative_error=a['fixed_prefix_logits_relative_error'],native_source_NLL=native[0],null_drift=drift,endpoint_F0_cost=cost,endpoint_gates=verdict,allowed_endpoint_submission=g['accepted'],damage=len(damaged),damaged_panel=damaged,locality_denominator=6,locality_missing=missing))
 for name in ['I','U']:
  rr=[x for x in rows if x['arm']==name];damage={(x['sample'],p) for x in rr for p in x['damaged_panel']};source=sum(x['semantic_T0'] is True for x in rr);source_missing=sum(x['semantic_T0'] is None for x in rr);missing=sum(x['locality_missing'] for x in rr);eligible=sum(x['allowed_endpoint_submission'] for x in rr)
  summaries[name]=dict(semantic_correct=source,semantic_denominator=3,semantic_missing=source_missing,strict_function_preserved=sum(x['source_exact_teacher'] for x in rr),null_pass=sum(x['endpoint_gates']['null'] for x in rr),trust_pass=sum(x['endpoint_gates']['trust'] for x in rr),endpoint_eligible=eligible,damage=len(damage),locality_denominator=18,locality_missing=missing,prespecified_useful=source==3 and not source_missing and not missing and eligible==3 and damage<={('E01','L006')})
 check('42_complete_scores',len(s['scores'])==42 and not s['missing']);check('six_candidates',len(rows)==6)
 for e in s['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=25);check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 fit=sum(t['fit_forwards'] for t in v['teachers']);teacher_gen=sum(t['generation_forwards'] for t in v['teachers']);candidate_gen=sum(a['generation_forwards'] for t in v['teachers'] for a in t['candidates']);local_gen=sum(e['generation_forwards'] for e in v['evaluation']);controls=end['model_forward_calls']-fit-teacher_gen-candidate_gen-local_gen
 check('global_budget',end['model_forward_calls']<=6800 and end['resident_seconds']<=1800 and controls>=0 and fit<=78)
 analysis=dict(rows=rows,summary=summaries,source_spaces=spaces,T_baseline=tql['summary'],costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],teacher_fit=fit,backwards=sum(t['backward_calls'] for t in v['teachers']),teacher_source_binding_generation=teacher_gen,candidate_source_generation=candidate_gen,candidate_locality_generation=local_gen,control_binding_prefix_restore=controls,peak_allocated_bytes=end['peak_allocated_bytes'],new_judge_payloads=s['new_payloads'],cached_endpoints=sum(x!='new_blind_score' for x in s['provenance'].values()),checkpoint=0,old_T_panel_reconstruction_forwards=0),historical_NA_unchanged=True,near_locality_support=0,distinct_unrelated_queries=6,scope='Development diagnosis; exact source layer action in float64 and native full source behavior separately measured; general diagonal activation protection only')
 audit=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',check_count=len(checks),failed=[x for x in checks if not x['passed']],checks=checks)
 for n,x in [('ANALYSIS.json',analysis),('AUDIT.json',audit)]:(out/n).write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
