"""One clean Base pass and dual-input mechanical admission, before all editors."""
import json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['LONG_ROOT'])
def write(name,x):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def budget():
 assert time.time()<json.loads((ROOT/'CLOCK.json').read_text())['hard_end']-1200
 assert not (ROOT/'STOP.json').exists() and time.time()-json.loads((ROOT/'HEALTH.json').read_text())['time']<45

def main():
 (ROOT/'BASE_ONCE').touch(exist_ok=False)
 cfg=json.loads((ROOT/'CONFIG.private.json').read_text());p=Path(os.environ['PARENT_RUN'])
 sys.path[:0]=[str(ROOT/'tools'),cfg['med150_tools'],str(p/'private/tools'),str(p/'private/source'),str(p/'private/official_llava')]
 import torch,numpy as np,module_run as r
 from pair_input import install
 install();r.c.budget=budget
 from types import SimpleNamespace
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=p/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 runtime.run_root=ROOT/'base_runtime';runtime.run_root.mkdir()
 data=json.loads((ROOT/'SOURCE_INPUTS.private.json').read_text());tasks={t['edit_id']:t for t in data['tasks']}
 rng=torch.load(Path(os.environ['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
 generation=runtime.generation_config;assert generation['max_new_tokens']==1024
 write('BASE_CONFIGURATION.private.json',dict(generation=generation,runtime={'task':'dual_image_DEV'}))
 for i,row in enumerate(data['base_plan']):
  budget();task=tasks[row['edit_id']];q=row['query'];runtime.model.half()
  try:
   item=r.s.prepare(runtime,{'configuration_only':{'runtime':{'task':'dual_image_DEV'}}},task,[q])[0]
   if i==0:
    batch=runtime.build_edit_batch(r.s.record(task,q));question=runtime.build_question_batch(r.s.record(task,q))
    assert batch.image_token_count==question.image_token_count==2
    assert batch.image_token_end-batch.image_token_start==2*(item[2]['images'].shape[-1]//runtime.model.get_vision_tower().config.patch_size)**2
    visual=batch.labels[:,batch.image_token_start:batch.image_token_end];assert (visual==-100).all()
    assert batch.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id
    assert batch.inputs_embeds.shape[1]==batch.labels.shape[1] and batch.key_token_index==batch.target_start_expanded-1
    raw=item[2];reverse=json.loads(Path(q['image_path']).read_text());reverse['paths'].reverse();rp=ROOT/'pairs/reverse_gate.json';rp.write_text(json.dumps(reverse))
    swapped=runtime.adapter.prepare_inputs(rp,q['question']);assert torch.equal(raw['images'][0],swapped['images'][1]) and torch.equal(raw['images'][1],swapped['images'][0])
    assert raw['image_sha256']!=swapped['image_sha256']
    write('MECHANICAL_INPUT.json',dict(status='PASS',native_image_tokens=2,visual_tokens=batch.image_token_end-batch.image_token_start,
        pair_order_preserved=True,target_only_labels=True,visual_labels_ignored=True,question_batch=True,patches_per_image=(batch.image_token_end-batch.image_token_start)//2))
  finally:r.old.fp32(runtime)
  r.c.restore_rng(rng['rng']);np.random.set_state(rng['numpy_rng'])
  dest=ROOT/'base_outputs'/f'{i:03d}.json';dest.with_suffix('.attempt').parent.mkdir(exist_ok=True);dest.with_suffix('.attempt').touch(exist_ok=False)
  answer,_=r.s.fit.short.generate(runtime,None,item[2],item[3]);tokens=list(answer.raw_token_ids)
  assert 0<len(tokens)<=1024
  write('base_outputs/'+dest.name,dict(consumer_id=f'base_{i}',owner=row['owner'],edit_id=row['edit_id'],role=row['role'],query_id=q['query_id'],judge_input=item[4],
       answer=dict(raw_answer=answer.decoded_text,raw_token_ids=tokens),EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
  write('BASE_PROGRESS.json',dict(completed=i+1,total=48,time=time.time()))
 write('BASE_COMPLETE.json',dict(status='COMPLETE',generations=48,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write('BASE_FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
