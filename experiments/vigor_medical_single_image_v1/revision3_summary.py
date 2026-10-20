"""Offline qualification audit: candidate quality must not be called editor performance."""
import json,math
from pathlib import Path

def main():
 r=Path(__file__).resolve().parents[2]/'reports/vigor_revision3_v1_20261020/completion';v=json.loads((r/'RESULT.json').read_text());end=json.loads((r/'ENDED.json').read_text());s=json.loads((r/'SCORES.json').read_text());checks=[]
 def check(k,ok):checks.append(dict(field=k,passed=bool(ok)))
 check('GPU_terminal',v['status']==end['status']=='COMPLETE');check('score_terminal',s['status']=='COMPLETE')
 for x in v['checks']:check('runtime.'+x['field'],x['passed'])
 for k,val in v['restoration'].items():check('restore.'+k,val is True)
 check('six_outputs',len(v['outputs'])==6 and len({e['alias'] for e in v['outputs']})==6)
 for e in v['outputs']:check(e['alias']+'.natural_complete',e['status']=='COMPLETE' and 0<e['generation_forwards']<=128 and e['raw_token_count']<1024)
 check('no_editing',v['editing_candidates']==v['backward_calls']==end['checkpoint']==0)
 gf=sum(e['generation_forwards'] for e in v['outputs']);check('forward_budget',end['model_forward_calls']<=820 and 0<=end['model_forward_calls']-gf<=20)
 check('resident_budget',end['resident_seconds']<=900)
 for criterion,data in s['criteria'].items():
  check(criterion+'.full_denominator',len(data['scores'])==6 and not data['missing'])
 for e in s['judge_evidence']:
  check(e['criterion']+'.once',e['attempts_per_payload']==1 and e['retry_count']==0 and e['payloads']<=6)
  check(e['criterion']+'.isolation',not e['tool_event_types'] and all(e['isolation_checks'].values()))
  check(e['criterion']+'.deadline',e['elapsed_seconds']<=600)
 for t in s['targets']:check(t['alias']+'.qualification_identity',t['qualified']==(t['agreement'] is True and t['faithfulness'] is True))
 summary={a:dict(agreement=sum(t['agreement'] is True for t in s['targets'] if t['alias'].endswith('_'+a)),faithful=sum(t['faithfulness'] is True for t in s['targets'] if t['alias'].endswith('_'+a)),qualified=sum(t['qualified'] is True for t in s['targets'] if t['alias'].endswith('_'+a)),denominator=3,missing=sum(t['qualified'] is None for t in s['targets'] if t['alias'].endswith('_'+a))) for a in ['P','R']}
 costs=dict(gpu_seconds=end['resident_seconds'],forward_calls=end['model_forward_calls'],target_generation_forward=gf,control_restore_forward=end['model_forward_calls']-gf,peak_allocated_bytes=end['peak_allocated_bytes'],new_Judge_payloads=sum(e['payloads'] for e in s['judge_evidence']),judge_seconds=sum(e['elapsed_seconds'] for e in s['judge_evidence']),checkpoint=0,new_edits=0)
 audit=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',checks=checks,check_count=len(checks),failed=[x for x in checks if not x['passed']]);(r/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 analysis=dict(summary=summary,same_P_R=s['same_P_R'],targets=s['targets'],costs=costs,qualification_is_edit_performance=False,clinical_image_verification=False,decision='Only jointly qualified source targets may enter a separately frozen writer comparison; no repair resampling')
 (r/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],**analysis),indent=2))
 if audit['status']!='PASS':raise SystemExit('Audit failed')
if __name__=='__main__':main()
