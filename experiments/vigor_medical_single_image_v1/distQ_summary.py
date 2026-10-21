"""Independent scalar-only audit, including missing points and frozen qualification gates."""
import json,math
from pathlib import Path

def ranks(values):
 order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and values[order[j]]==values[order[i]]:j+=1
  for k in order[i:j]:result[k]=(i+j-1)/2+1
  i=j
 return result

def corr(x,y):
 if len(x)<3 or len(set(x))<2 or len(set(y))<2 or not all(math.isfinite(z) for z in x+y):return None
 x=ranks(x);y=ranks(y);mx=sum(x)/len(x);my=sum(y)/len(y);vx=sum((z-mx)**2 for z in x);vy=sum((z-my)**2 for z in y)
 return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy)

def main():
 root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_distributionQ_v1_20261020/completion';v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());checks=[]
 def check(name,ok):checks.append(dict(field=name,passed=bool(ok)))
 check('terminal_COMPLETE',v['status']==end['status']=='COMPLETE')
 for row in v['checks']:check('runtime.'+row['field'],row['passed'])
 for k,x in v.get('restoration',{}).items():
  if k=='restored_records':check('restored_32_records',x==32)
  else:check('restoration.'+k,x is True)
 rows=v['measurements'];check('72_fixed_points',len(rows)==72)
 expected=[(d,s,sign) for d in ['R1','R2','R3','R4','G1','G2','G3','G4','B1','B2','B3','B4'] for s in [.25,.5,1.] for sign in [-1,1]]
 for i,r in enumerate(rows):
  check('point_order',i<len(expected) and (r['direction'],r['scale'],r['sign'])==expected[i] and r['point']==i+1)
  check('full_32_KL',len(r['record_KL'])==32 and all(math.isfinite(x) and x>=-1e-12 for x in r['record_KL']))
  if len(r['record_KL'])==32:check('C_H_mean_recomputed',abs(r['C_KL']-sum(r['record_KL'][:24])/24)<=1e-12 and abs(r['H_KL']-sum(r['record_KL'][24:])/8)<=1e-12)
  check('point_W0_restored',r.get('W0_restored') is True)
  check('nonnegative_risk_and_norm',all(math.isfinite(r[k]) and r[k]>=0 for k in ['fit','audit','misaligned_audit','F0','norm_squared','native_norm']))
  check('actual_native_norm',abs(r['native_norm']**2-r['norm_squared'])<=1e-8*max(1,r['norm_squared']))
 work=[r for r in rows if r['scale']==1. and len(r['record_KL'])==32];valid=[r for r in work if r['native_norm']>0 and r['C_KL']>1e-10];accurate=[r for r in valid if .5<=r['audit']/r['C_KL']<=2];severe=[r['point'] for r in valid if r['audit']<r['C_KL']/4];coverage=[r['point'] for r in work if r['direction'].startswith('B') and r['H_KL']>1e-10 and r['audit']<r['H_KL']/4]
 correlations={s:{key:corr([r[key] for r in work],[r[s+'_KL'] for r in work]) for key in ['audit','F0','norm_squared','misaligned_audit']} for s in ['C','H']}
 N=len(work)==24 and len(valid)>=20 and len(accurate)>=.8*len(valid) and not severe;H=correlations['H']['audit'] is not None and correlations['H']['audit']>=.6 and not coverage;I=all(x['audit'] is not None and all(x[k] is not None for k in ['F0','norm_squared','misaligned_audit']) and x['audit']>=max(x[k] for k in ['F0','norm_squared','misaligned_audit'])+.1 for x in correlations.values())
 qualification=dict(complete=len(rows)==72 and all(len(r['record_KL'])==32 for r in rows),work_points=len(work),discernible=len(valid),factor_two=len(accurate),factor_two_fraction=len(accurate)/len(valid) if valid else None,serious_C_underestimates=severe,blind_H_underestimates=coverage,correlations=correlations,gates=dict(N=N,H=H,I=I),PASS=bool(len(rows)==72 and N and H and I),editor_started=False)
 check('qualification_recomputed',v['qualification']['gates']==qualification['gates'] and (v['qualification']['status']=='PASS')==qualification['PASS'])
 for s in ['C','H']:
  for k in correlations[s]:
   a=correlations[s][k];b=v['qualification']['correlations'][s][k];check('independent_rank_correlation',a is None and b is None or a is not None and b is not None and abs(a-b)<=1e-10)
 check('no_audit_before_direction_freeze',v['direction_freeze']['audit_queries_before_freeze']==0 and v['direction_freeze']['geometry']['only_fit_statistics_used'] and not v['direction_freeze']['heldout_used_for_directions'])
 check('all_768_formal_probes',len(v['construction'])==768 and len({(r['record'],r['domain'],r['probe']) for r in v['construction']})==768)
 check('VJP_total_smoke_and_ref',v['backward_calls']==end['backward_calls']==774)
 check('forward_budget',end['model_forward_calls']<=2600)
 check('no_Judge_or_editor',v['judge_calls']==v['editor_calls']==v['complete_generations']==end['judge_calls']==end['editor_calls']==0)
 costs=dict(resident_seconds=end['resident_seconds'],model_forwards=end['model_forward_calls'],formal_statistic_VJPs=768,smoke_VJPs=2,reference_VJPs=4,backward_total=end['backward_calls'],native_perturbation_forwards=sum(len(r['record_KL']) for r in rows),native_baseline_forwards=len(v['baseline']),proxy_statistic_forwards=24,proxy_smoke_forwards=2,smoke_restore_forwards=2,final_restore_forwards=v.get('restoration',{}).get('restored_records',0),vision_preparations=v['preprocessing']['vision_preparations'],vision_preprocessing_seconds=v['preprocessing']['seconds'],peak_allocated_bytes=end['peak_allocated_bytes'],Judge=0,editor=0,checkpoint=0)
 check('forward_ledger_sum',costs['model_forwards']==sum(costs[k] for k in ['native_perturbation_forwards','native_baseline_forwards','proxy_statistic_forwards','proxy_smoke_forwards','smoke_restore_forwards','final_restore_forwards']))
 audit=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',checks=len(checks),failed=[x for x in checks if not x['passed']],details=checks)
 result=dict(qualification=qualification,costs=costs,direction_geometry=v['direction_freeze']['geometry'],smoke=v['smoke'],partial_or_nonqualifying_preserved=True,heldout_is_training_validation=True,medical_behavior_protection_not_tested=True)
 for n,value in [('AUDIT.json',audit),('ANALYSIS.json',result)]:(out/n).write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**result),indent=2))
 if audit['status']!='PASS':raise SystemExit('Offline audit failed')
if __name__=='__main__':main()
