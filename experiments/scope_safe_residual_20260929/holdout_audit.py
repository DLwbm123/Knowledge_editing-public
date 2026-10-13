"""Close the frozen holdout after all generation and scoring have terminated."""
import json,os,collections
from pathlib import Path
from final_audit import summarize,paired
from auxiliary_audit import routing

def summary(rows,scores):
 d=summarize([dict(r,edit=r['source_group']) for r in rows],scores)
 d['source_macro_bounds']=d.pop('edit_macro_bounds');d.pop('edits');return d

def main(root):
 read=lambda p:json.loads(p.read_text());protocol=read(root/'HOLDOUT_PROTOCOL.json');queue=read(root/'HOLDOUT_QUEUE.json');assert len(queue)==10 and all(x['status']=='COMPLETE' for x in queue)
 scores={p.stem:read(p)['is_correct'] for p in (root/'private/judge/scores').glob('*.json')};failed=set(read(root/'private/judge/JUDGE_MISSING_LOCK.json')['keys']);groups={};bases={};required=set();costs=[]
 for j in queue:
  for n in protocol['prefixes']:
   folder=root/'auxiliary/holdout'/j['id']/f'p{n}';rows=read(folder/'CONSUMERS.json');assert len(rows)==protocol['candidates']==47
   groups[(j['id'],n)]=rows;costs.append(read(folder/'COST.json'))
   for row in rows:
    required.update([row['judge_key'],row['base_judge_key']])
    if row['input_id'] in bases:assert bases[row['input_id']]['judge_key']==row['base_judge_key']
    bases[row['input_id']]=dict(row,judge_key=row['base_judge_key'])
 assert len(bases)==47 and not required-set(scores)-failed
 ledger=read(root/'RESOURCE_LEDGER.json');assert all(x['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for x in ledger['judge_attempts'])
 assert not any(not x.get('ended_epoch') for x in ledger['gpu_sessions'])
 eligible={k for k,r in bases.items() if scores.get(r['judge_key']) is True};unknown={k for k,r in bases.items() if scores.get(r['judge_key']) is None}
 metrics={};comparisons={}
 for (job,n),rows in groups.items():
  known=[r for r in rows if r['input_id'] in eligible]
  metrics[f'{job}/{n}']=dict(unconditional_accuracy=summary(rows,scores),Base_correct_conditional_retention=summary(known,scores),qualification_unknown=len(unknown),routing=routing(rows,scores))
  if job.endswith('AHS_01'):
   control=groups[(job.rsplit('-',1)[0]+'-A0',n)]
   comparisons[f'{job}/{n}']=paired(control,rows,scores)
 result=dict(status='COMPLETE',candidate_inputs=47,source_groups=24,Base_accuracy=summary(list(bases.values()),scores),Base_correct_inputs=len(eligible),Base_unknown_inputs=len(unknown),target_inputs=200,target_sources=50,target_met=False,scope='Pre-frozen train-release acquisition modality/plane facts; new source groups, not a new edit confirmation panel',metrics=metrics,paired_AHS_vs_A0=comparisons,unique_required=len(required),scored=len(required&set(scores)),permanent_missing=len(required-set(scores)),pending=0,new_judge_attempts=ledger['judge_submission_attempt_items']-protocol['attempts_baseline'],lifetime_judge_attempts=ledger['judge_submission_attempt_items'],evaluation_GPU_hours=(ledger['current_gpu_seconds']-protocol['GPU_seconds_baseline'])/3600,scope_total_GPU_hours=ledger['current_gpu_seconds']/3600,measured_model_calls=sum(x['model_calls'] for x in costs),measured_prefix_seconds=sum(x['seconds'] for x in costs),separate_prefill_latency=None,retuning=False)
 (root/'public/HOLDOUT_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','paired_AHS_vs_A0']}))
if __name__=='__main__':main(Path(os.environ['RUN_ROOT']))
