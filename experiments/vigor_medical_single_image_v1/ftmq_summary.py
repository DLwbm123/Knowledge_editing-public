"""Offline teacher qualification audit; never evaluates or regenerates an answer."""
import json,math
from pathlib import Path

def main():
 out=Path(__file__).resolve().parents[2]/'reports/vigor_ftmq3_v1_20261020/completion'
 v=json.loads((out/'RESULT.json').read_text());end=json.loads((out/'ENDED.json').read_text());q=json.loads((out/'QUALIFICATION.json').read_text());checks=[]
 def check(k,t):checks.append(dict(field=k,passed=bool(t)))
 check('terminal',v['status']==end['status']=='COMPLETE')
 check('qualification_terminal',q['status']=='COMPLETE')
 for c in v['checks']:check('runtime.'+c['field'],c['passed'])
 for k,t in v['restoration'].items():check('restoration.'+k,t is True)
 check('three_fixed_sources',[t['sample'] for t in v['teachers']]==['E01','E02','E03'])
 rows=[]
 for t in v['teachers']:
  name=t['sample'];steps=t['steps'];target=next(x for x in q['targets'] if x['alias']==name+'_TQ')
  check(name+'.steps',0<len(steps)<=25 and len(steps)==t['fit_forwards'])
  check(name+'.backwards',sum(s['optimizer_step'] for s in steps)==t['backward_calls'])
  for i,s in enumerate(steps):
   check(name+'.fixed_stop',s['step']==i+1 and math.isfinite(s['loss_before']) and s['optimizer_step']==(s['loss_before']>=.01))
   if not s['optimizer_step']:check(name+'.first_stop',i==len(steps)-1)
   else:check(name+'.finite_gradient',math.isfinite(s['gradient_norm']))
  check(name+'.no_early_stop_else',len(steps)==25 or steps[-1]['loss_before']<.01)
  check(name+'.retired_native_teacher',t['status']=='COMPLETE' and t['base_restored'] and t['native_changed'])
  check(name+'.complete_generation',0<t['generation_forwards']<=128 and 0<t['raw_token_count']<1024)
  check(name+'.finite_native_loss',all(math.isfinite(x) for x in t['teacher_native_nll']))
  vals=[target[k] for k in ['agreement','faithfulness']]
  check(name+'.qualification_AND',target['qualified']==(all(x is True for x in vals) if all(x is not None for x in vals) else None))
  rows.append(dict(sample=name,agreement=vals[0],faithfulness=vals[1],qualified=target['qualified'],fit_forwards=t['fit_forwards'],backwards=t['backward_calls'],generation_forwards=t['generation_forwards'],native_NLL=t['teacher_native_nll'][0],final_FP32_selected_layer_loss=steps[-1]['loss_before'],stop='SOURCE_NLL_THRESHOLD' if not steps[-1]['optimizer_step'] else 'STEP_CAP'))
 for e in q['judge_evidence']:
  check('judge_once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=3)
  check('judge_isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
 check('two_criteria',set(q['criteria'])=={'agreement','faithfulness'})
 for k,criterion in q['criteria'].items():check(k+'.full_denominator',len(criterion['scores'])==3 and not criterion['missing'])
 fit=sum(r['fit_forwards'] for r in rows);gen=sum(r['generation_forwards'] for r in rows);control=end['model_forward_calls']-fit-gen
 check('budgets',end['model_forward_calls']<=600 and end['resident_seconds']<=900 and sum(r['backwards'] for r in rows)<=75 and control>=0)
 analysis=dict(rows=rows,denominator=3,agreement=sum(r['agreement'] is True for r in rows),faithfulness=sum(r['faithfulness'] is True for r in rows),qualified=sum(r['qualified'] is True for r in rows),missing=sum(r['qualified'] is None for r in rows),source_context_P_prior_qualification=dict(correct=2,denominator=3,reused=True),student_edits=0,locality_support=0,locality_damage=None,costs=dict(GPU_seconds=end['resident_seconds'],total_forwards=end['model_forward_calls'],fit_forwards=fit,backwards=sum(r['backwards'] for r in rows),generation_forwards=gen,control_restore=control,peak_allocated_bytes=end['peak_allocated_bytes'],checkpoint=0,new_judge_payloads=sum(e['new_attempts'] for e in q['judge_evidence']),judge_seconds=sum(e['elapsed_seconds'] for e in q['judge_evidence']),per_teacher_seconds=None),scope='Source-only teacher qualification; no student distillation, independent generalization or locality claim')
 audit=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),failed=[c for c in checks if not c['passed']])
 for n,x in [('ANALYSIS',analysis),('AUDIT',audit)]:(out/(n+'.json')).write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Teacher qualification audit failed')

if __name__=='__main__':main()
