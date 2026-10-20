"""Offline protected reverse-KL student audit with all teacher reconstruction costs."""
import json,math
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_alignA3_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());s=json.loads((out/'SCORES.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('terminal',v['status']==end['status']=='COMPLETE');check('scoring_terminal',s['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x.get('passed',math.isfinite(x.get('absolute_difference',float('inf'))) and x.get('absolute_difference',float('inf'))<=x.get('tolerance',0)))
 for k,t in v['restoration'].items():check('restore.'+k,t is True)
 rows=[];rollout_tokens=0;rollout_calls=0;teacher_total=0;teacher_fit=0;teacher_backwards=0
 qualified=json.loads((root/'reports/vigor_ftmq3_v1_20261020/completion/RESULT.json').read_text())
 for case in v['cases']:
  sample=case['sample'];a=case['arms'][0];check(sample+'.original_base',a['base_restored']);check(sample+'.edit_bound',a['editing_forwards']<=512)
  expected=next(t for t in qualified['teachers'] if t['sample']==sample);rebuild=a['teacher_reconstruction']
  check(sample+'.teacher_reconstruction_steps',rebuild['fit_forwards']==expected['fit_forwards'] and len(rebuild['steps'])==len(expected['steps']) and rebuild['backward_calls']==expected['backward_calls'])
  check(sample+'.teacher_native_binding',all(abs(x-y)<=1e-5 for x,y in zip(rebuild['native_nll'],expected['teacher_native_nll'])))
  teacher_total+=rebuild['total_forwards'];teacher_fit+=rebuild['fit_forwards'];teacher_backwards+=rebuild['backward_calls']
  accepted=0
  for step in a['steps']:
   rollout=step.get('rollout')
   if rollout:
    rollout_tokens+=rollout['token_count'];rollout_calls+=rollout['forward_calls']
    check(sample+'.complete_trajectory',rollout['token_count']>0 and rollout['complete'] and rollout['full_vocabulary'] and not rollout['partial_trajectory_used'] and 0<rollout['forward_calls']<=128)
   tt=[t for t in step['trials'] if t['status']=='ACCEPTED'];check(sample+'.unique_accept',len(tt)==int(step['accepted']))
   if not tt:continue
   t=tt[0];accepted+=1;check(sample+'.source_native_prediction',t['linear'][0]<-1e-5);check(sample+'.joint_native_prediction',t['joint_linear']<-1e-5)
   for name in ['actual_response','final_response']:
    r=t[name];check(sample+'.'+name,r['passed'] and r['NLL_gain']>1e-5 and r['joint_gain']>1e-5 and r['rho_NLL']>=.1 and r['rho_joint']>=.1)
   gg=t['final_gate'];check(sample+'.original_gates',gg['accepted'] and all(gg['gates'].values()) and abs(gg['cumulative_null_change'])<=.01 and gg['combined_native_fisher_cost']<=.001)
   check(sample+'.final_native_repeat',abs(t['final_margin']['mean_nll']-gg['native_nll'][0])<=1e-5)
  check(sample+'.accept_count',accepted==a['accepted_steps'])
  ev=[e for e in v['evaluation'] if e['sample']==sample];check(sample+'.six_panel',len(ev)==6);values=[s['scores'].get(e['alias']) for e in ev];last=a['last_accepted'];t0=s['scores'].get(sample+'_A_T0')
  rows.append(dict(sample=sample,arm='A',semantic_T0=t0,locality_damage=sum(x is False for x in values),locality_denominator=6,locality_missing=sum(x is None for x in values),damaged_panel=[e['panel_id'] for e,x in zip(ev,values) if x is False],accepted_steps=accepted,editing_forwards_including_teacher=a['editing_forwards'],backward_calls=a['backward_calls'],secant_calls=a['secant_calls'],native_source_NLL=last['native_nll'][0],tracked_margin=last['margin']['margin'],original_base_null_drift=last['native_nll'][1]-case['base']['native_nll'][1],endpoint_F0_cost=last['cumulative_F0_cost'],status=a['status'],all_accepted_original_gates=True,teacher_reconstruction_forwards=rebuild['total_forwards'],completed_rollout_forwards=sum(x.get('rollout',{}).get('forward_calls',0) for x in a['steps'])))
 check('source_subset',[r['sample'] for r in rows]==['E01','E02','E03']);check('21_endpoints',len(s['scores'])==21 and not s['missing']);check('18_locality',len(v['evaluation'])==18)
 for e in s['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=21);check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 originalN=json.loads((root/'reports/vigor_factF3_v1_20261020/completion/ANALYSIS.json').read_text());nr=[r for r in originalN['rows'] if r['arm']=='N']
 baseline=dict(semantic_correct=sum(r['semantic_T0'] is True for r in nr),semantic_denominator=3,damage=sum(r['damage'] for r in nr),locality_denominator=18,editing_forwards=1350,reconstruction_not_repeated=True,source='previous_N3_T0_and_F3_fixed_trace_locality_binding')
 summary=dict(semantic_correct=sum(r['semantic_T0'] is True for r in rows),semantic_denominator=3,semantic_missing=sum(r['semantic_T0'] is None for r in rows),damage=sum(r['locality_damage'] for r in rows),locality_denominator=18,locality_missing=sum(r['locality_missing'] for r in rows),editing_forwards=sum(r['editing_forwards_including_teacher'] for r in rows),backward_calls=sum(r['backward_calls'] for r in rows),secant_calls=sum(r['secant_calls'] for r in rows))
 edits=summary['editing_forwards'];gens=sum(e['generation_forwards'] for e in v['evaluation'])+sum(a['generation_forwards_total'] for c in v['cases'] for a in c['arms'])
 costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],editing_including_teacher=edits,final_T0_and_locality_generation=gens,control_restore=end['model_forward_calls']-edits-gens,peak_allocated_bytes=end['peak_allocated_bytes'],new_judge_payloads=s['new_payloads'],cached_endpoints=sum(x!='new_blind_score' for x in s['provenance'].values()),teacher_qualification_prior_GPU_seconds=25.037349700927734,teacher_qualification_prior_forwards=98,teacher_qualification_prior_judge_payloads=5,teacher_reconstruction_included=teacher_total,teacher_fit_included=teacher_fit,teacher_backward_included=teacher_backwards,completed_rollout_forwards_included=rollout_calls,checkpoint=0,per_arm_seconds=None)
 check('global_budget',end['model_forward_calls']<=5000 and end['resident_seconds']<=1800 and costs['control_restore']>=0)
 damage_set={(r['sample'],x) for r in rows for x in r['damaged_panel']};useful=not summary['semantic_missing'] and not summary['locality_missing'] and next(r['semantic_T0'] for r in rows if r['sample']=='E03') is True and any(r['semantic_T0'] is True for r in rows if r['sample'] in ['E01','E02']) and damage_set<={('E01','L006')}
 analysis=dict(rows=rows,A=summary,N_baseline=baseline,source_teacher_eligibility=dict(qualified=3,denominator=3,excluded=[]),prespecified_useful=useful,completed_trajectory_tokens=rollout_tokens,partial_rollout_call_breakdown='Private incomplete events retained; not all partial calls classified in public counters',costs=costs,near_locality_support=0,distinct_unrelated_queries=6,scope='Three exposed development edits; protected single-matrix adaptation, not independent generalization or full AlignEdit reproduction')
 audit=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),failed=[c for c in checks if not c['passed']]);(out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n');(out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
