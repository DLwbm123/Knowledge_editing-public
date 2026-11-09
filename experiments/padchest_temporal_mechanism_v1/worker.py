"""Fifty fixed MDU deployment generations; no fitting, new labels or Judge."""
import json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['DIAG_ROOT']);CFG=json.loads((ROOT/'CONFIG.private.json').read_text());CALLS=0

def write(name,x):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def budget():
 assert time.time()<CFG['clock']['hard_end']-120 and not (ROOT/'STOP.json').exists()
 assert time.time()-json.loads((ROOT/'HEALTH.json').read_text())['time']<30 and CALLS<=50

def main():
 global CALLS
 (ROOT/'WORKER_ONCE').touch(exist_ok=False)
 parent=Path(os.environ['PARENT_RUN']);corrected=Path(CFG['corrected_root'])
 sys.path[:0]=[str(corrected/'tools'),CFG['med150_tools'],str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
 import torch,numpy as np,module_run as r
 from pair_input import install
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
 from types import SimpleNamespace
 install();r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(p.requires_grad for p in runtime.model.parameters())
 runtime.run_root=ROOT/'runtime';runtime.run_root.mkdir()
 library=torch.load(corrected/'MDU/T5_CANONICAL.pt',map_location='cpu',weights_only=True)
 tasks=library['tasks'];byid={t['edit_id']:t for t in tasks};assert len(tasks)==4
 inputs=json.loads((corrected/'T5_INPUTS.private.json').read_text());queries={}
 for t in tasks:queries[t['native']['query_id']]=dict(task=t,query=t['native'])
 for row in inputs['plan']:
  if row['role'] in ['CONTRAST','SAME_STATE']:queries.setdefault(row['query']['query_id'],dict(task=byid[row['edit_id']],query=row['query']))
 assert len(queries)==10
 baseline={}
 for p in (corrected/'MDU/outputs').glob('*.json'):
  v=json.loads(p.read_text())
  if v['mode']=='bank' and v['query_id'] in queries:
   if v['query_id'] in baseline:assert baseline[v['query_id']]['answer']==v['answer']
   baseline[v['query_id']]=v
 assert len(baseline)==10
 rng=torch.load(Path(os.environ['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
 router=MemoryRouter.from_state(dict(distance='euclidean',entries=library['R0_entries']),device=runtime.device)
 config=json.loads((corrected/'BASE_CONFIGURATION.private.json').read_text());assert runtime.generation_config==config['generation']
 records=[]
 def prepare(task,q):
  runtime.model.half()
  try:
   batch=runtime.build_question_batch(r.s.record(task,q))
   with torch.inference_mode():key=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
   decision=router.route(key)
   item=r.s.prepare(runtime,{'configuration_only':config},task,[q])[0]
  finally:r.old.fp32(runtime)
  return item,decision
 def generate(qid,condition,task,item,chosen):
  global CALLS
  budget();out=ROOT/'outputs'/f'{condition}_{qid}_{chosen or "OFF"}.json';out.parent.mkdir(exist_ok=True);out.with_suffix('.attempt').touch(exist_ok=False)
  hook=None
  try:
   if chosen:
    expert=r.p.expert(library['states'][chosen],byid[chosen]['seed'],runtime.device).requires_grad_(False)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
   r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng']);CALLS+=1
   answer,_=r.s.fit.short.generate(runtime,hook,item[2],item[3]);tokens=list(answer.raw_token_ids)
  finally:
   if hook:hook.detach()
  assert 0<len(tokens)<=1024
  value=dict(query_id=qid,condition=condition,selected_expert=chosen,source_class=byid[chosen]['progression'] if chosen else None,
    answer=answer.decoded_text,tokens=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,
    pair_identity=item[4]['image_sha256'],prompt_ids=item[4]['prompt_ids'])
  write('outputs/'+out.name,value);records.append(value);write('PROGRESS.json',dict(time=time.time(),generations=CALLS,total=50))
  return value
 # Each fixed expert is crossed with all ten original source/probe inputs.
 # Its own source is first: original natural deployment tokens are a gate.
 for task in tasks:
  chosen=task['edit_id'];own=task['native']['query_id'];ordered=[own]+[q for q in sorted(queries) if q!=own]
  for index,qid in enumerate(ordered):
   row=queries[qid];item,decision=prepare(row['task'],row['query'])
   natural=decision.logical_edit_id if decision.activated else None
   assert natural==baseline[qid]['effective_expert'],'Original R0 replay differs'
   assert item[4]['prompt_ids']==baseline[qid]['judge_input']['prompt_ids'] and item[4]['image_sha256']==baseline[qid]['judge_input']['image_sha256']
   value=generate(qid,'FORCED_ORIGINAL',row['task'],item,chosen)
   if index==0:assert value['tokens']==baseline[qid]['answer']['raw_token_ids'],'Own-source saved expert deployment differs'
 # Reversed chronology changes only image order. No reversal gold is invented.
 (ROOT/'pairs').mkdir()
 reverse_routes=[]
 for qid,row in sorted(queries.items()):
  q=dict(row['query']);pair=json.loads(Path(q['image_path']).read_text());pair['paths'].reverse();dest=ROOT/'pairs'/f'{qid}.json';dest.write_text(json.dumps(pair))
  q['image_path']=str(dest);q.pop('image_sha256',None);item,decision=prepare(row['task'],q)
  chosen=decision.logical_edit_id if decision.activated else None
  value=generate(qid,'REVERSED_NATURAL',row['task'],item,chosen)
  assert value['pair_identity']!=baseline[qid]['judge_input']['image_sha256']
  reverse_routes.append(dict(same_route=chosen==baseline[qid]['effective_expert'],same_tokens=value['tokens']==baseline[qid]['answer']['raw_token_ids']))
 assert CALLS==50
 per_expert={}
 for task in tasks:
  values=[v for v in records if v['condition']=='FORCED_ORIGINAL' and v['selected_expert']==task['edit_id']]
  per_expert[str(task['order'])]=dict(queries=10,unique_token_sequences=len({tuple(v['tokens']) for v in values}),
   exact_source_class_answers=sum(v['answer'].strip().lower().strip('.')==task['progression'] for v in values))
 summary=dict(status='COMPLETE',full_generations=50,updates=0,Judge_calls=0,own_source_token_replays_equal=4,
   forced_expert_cross_input=per_expert,reverse_image_order=dict(queries=10,same_route=sum(v['same_route'] for v in reverse_routes),same_output_tokens=sum(v['same_tokens'] for v in reverse_routes)),
   at_cap=sum(v['at_cap'] for v in records),EOS=sum(v['EOS'] for v in records),
   limitation='Deployment dependency diagnostic on exposed DEV. Forced expert crossing changes question and images jointly; reverse-order outputs have no newly adjudicated medical gold. Not an accuracy improvement or independent clinical validation.')
 write('SUMMARY.json',summary);write('COMPLETE.json',dict(time=time.time(),generations=50))
if __name__=='__main__':
 try:main()
 except BaseException:write('FAILURE.private.json',dict(time=time.time(),generations=CALLS,error=traceback.format_exc()));raise
