"""Bind confirmed Base errors to unchanged C/MED/MDU and comparator pipelines."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['LONG_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False))
def main():
 cfg=read(ROOT/'CONFIG.private.json');clock=read(ROOT/'CLOCK.json');mask=read(ROOT/'BASE_MASK_FROZEN.private.json')
 source=read(ROOT/'SOURCE_INPUTS.private.json');tasks=[t for t in source['tasks'] if mask['native_core'][t['edit_id']] is False]
 write(ROOT/'COHORT_FROZEN.json',dict(admitted=len(tasks),known_correct=sum(v is True for v in mask['native_core'].values()),unknown=sum(v is None for v in mask['native_core'].values()),method_output_selection=False))
 if not tasks:return
 entries=[];plan=[];base={v['query_id']:v for v in [read(p) for p in (ROOT/'base_outputs').glob('*.json')]}
 for t in tasks:
  entries.append(dict(task=t,source_rows=[dict(q,audit_role='NATIVE' if i==0 else 'FIT') for i,q in enumerate(t['all_rows'][:5])]))
  for q in t['all_rows']:plan.append(dict(edit_id=t['edit_id'],role='T0' if q['audit_role']=='NATIVE' else q['audit_role'],query=q))
  for q in source['probes']:
   if q['label']==t['label']:plan.append(dict(edit_id=t['edit_id'],role='SAME_STATE' if q['progression']==t['progression'] else 'CONTRAST',query=q))
 expected={qid:{k:v['judge_input'][k] for k in ['question','reference','image_sha256','prompt_ids','attention_mask']} for qid,v in base.items()}
 # The ordered pair identity is available after the real preprocessing pass.
 for t in tasks:
  for q in t['all_rows']:q['image_sha256']=expected[q['query_id']]['image_sha256']
 for q in source['probes']:q['image_sha256']=expected[q['query_id']]['image_sha256']
 for row in plan:row['query']['image_sha256']=expected[row['query']['query_id']]['image_sha256']
 m=ROOT/'MDU';m.mkdir();(m/'tools').symlink_to(ROOT/'tools',target_is_directory=True)
 for name,value in [('CLOCK.json',clock),('TRAIN_SOURCE_BOUND_PRIVATE.json',entries),('ROLE_ADMISSION.json',dict(status='TRAINING_SOURCE_ROLE_ADMITTED',owners=[t['order'] for t in tasks])),('SPAN_LOCK.private.json',dict(records=[]))]:write(m/name,value)
 env=dict(os.environ,SOURCE_ROOT=str(m),OUTPUT_ROOT=str(m),OFFICIAL_SOURCE=os.environ['M3BENCH_LLAVA_SOURCE'],TOKENIZER=os.environ['M3BENCH_MODEL_PATH'],DEADLINE=str(clock['hard_end']),CURRENT_SOURCE=str(ROOT/'tools/coverage.py'))
 result=subprocess.run([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=env,capture_output=True,text=True,timeout=600)
 write(ROOT/'TOKEN_EXIT.private.json',dict(code=result.returncode,stderr=result.stderr[-4000:]));assert result.returncode==0,result.stderr[-4000:]
 tokens=read(m/'TOKEN_LOCK.private.json');assert all(v['status']=='TOKEN_BOUND' for v in tokens['records']),'Do not drop a task on training-token results'
 binding='LONGITUDINAL_DEV_V1_FIXED_RECIPES'
 write(m/'ADDITIONAL_INPUTS.json',dict(status='T5_SOURCE_ADMITTED',candidate_binding=binding,tasks=entries))
 write(m/'EXPECTED_SOURCE_BINDINGS.json',[dict(base[t['native']['query_id']]['judge_input'],query_id=t['native']['query_id']) for t in tasks])
 write(m/'TRAIN_LOCK.json',dict(physical_GPU=int(cfg['gpu']['index']),owners=[t['order'] for t in tasks],candidate_binding=binding,source_commit=cfg['source_commit'],budget_binding=binding,
 budgets=dict(forwards=20000000,non_generation_forwards=20000000,backwards=1200*len(tasks),operator_backwards=160*len(tasks),updates=480*len(tasks),generation_attempts=0)))
 write(m/'TASK_INPUTS.private.json',dict(tasks=tasks,plan=plan,owner_shards={str(t['order']):'.' for t in tasks},expected_consumers=2*len(plan)))
 write(m/'BASE_EXPECTATIONS.private.json',expected);write(m/'BASE_CONFIGURATION.private.json',read(ROOT/'BASE_CONFIGURATION.private.json'))
 write(ROOT/'T5_INPUTS.private.json',dict(tasks=tasks,plan=plan,expected=expected,generation=read(ROOT/'BASE_CONFIGURATION.private.json')['generation'],base_native_tokens={t['native']['query_id']:base[t['native']['query_id']]['answer']['raw_token_ids'] for t in tasks}))
 write(ROOT/'CPU_ADMISSION.json',dict(status='PASS',admitted_edits=len(tasks),consumers_per_method=2*len(plan),updates_MDU=480*len(tasks),patient_disjoint_probes=True))
if __name__=='__main__':main()
