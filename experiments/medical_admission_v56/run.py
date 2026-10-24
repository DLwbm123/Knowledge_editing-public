"""Matched CE versus Base-relative increment constraints on fixed source-local medical packs."""
import os,json,time,signal,sys,traceback,shutil
from pathlib import Path
ROOT=Path(os.environ['V56_ROOT']);PART=int(os.environ['PARTITION']);OUT=ROOT/str(PART)
P=Path(os.environ['PARENT_RUN']);START=time.time()
counts=dict(forwards=0,backwards=0,updates=0,generations=0,admission_backwards=0,vision_forwards=0,cached_base_outputs=0)
limits=json.loads((ROOT/'RESOURCE_LIMITS.json').read_text());last_space=0

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def budget():
 global last_space
 assert time.time()<limits['hard_end']-300 and not (ROOT/'STOP').exists(),'Original deadline or STOP'
 assert counts['forwards']<80000 and counts['backwards']<10000 and counts['generations']<150
 assert time.time()-read(ROOT/'RESOURCE_GUARD_HEALTH.json')['time']<120
 if time.time()-last_space>30:
  assert shutil.disk_usage(ROOT).free>8*1024**3
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())<1500000000
  last_space=time.time()

def main():
 with (OUT/'ONCE').open('x'):pass
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded Base qualification')))
 signal.alarm(max(1,min(1800,int(limits['hard_end']-time.time()-300))))
 sys.path[:0]=[str(ROOT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
 import torch
 import module_run as r
 from scripts.medtrace.run_realmodel_core import load_real_runtime
 from types import SimpleNamespace
 r.c.budget=budget
 torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
 runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
 assert not any(v.requires_grad for v in runtime.model.parameters())
 rows=read(ROOT/'DATA_ADMISSION.json')['rows'][PART::2]
 template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
 bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
 def count(*_):budget();counts['forwards']+=1
 fh=runtime.model.register_forward_pre_hook(count)
 write(OUT/'MODEL_READY.json',dict(time=time.time(),generation=runtime.generation_config))
 try:
  for row in rows:
   runtime.model.half()
   try:item=r.s.prepare(runtime,bindings,template,[row],False)[0]
   finally:r.old.fp32(runtime)
   budget();counts['generations']+=1
   answer,trace=r.s.fit.short.generate(runtime,None,item[2],item[3]);tokens=list(answer.raw_token_ids)
   write(OUT/'outputs'/(row['query_id']+'.json'),dict(binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
   write(OUT/'PROGRESS.json',dict(counts=counts,time=time.time()))
  write(OUT/'MODEL_COMPLETE.json',dict(status='PASS',counts=counts,time=time.time()))
 finally:fh.remove()
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,time=time.time()));write(ROOT/'STOP',dict(reason='V56_FAILURE',time=time.time()));raise
 finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
