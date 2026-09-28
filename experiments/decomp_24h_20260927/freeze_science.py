"""One-time actual model/dependency lock; never overwrites an existing identity."""
import os,sys,json,hashlib,time
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(r),str(r/'source_patch'),str(r/'source')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
p=r/'SCIENCE_LOCK.json';assert not p.exists();assets={}
for root in [x.resolve() for x in (r/'models').iterdir() if x.is_dir()]:
 for f in root.rglob('*'):
  if f.is_file() and f.suffix in ['.json','.model','.safetensors','.bin','.txt']:assets[str(f)]=dict(bytes=f.stat().st_size,sha256=sha(f))
import importlib,torch,transformers
modules=['methods.medtrace.core','methods.medtrace.selective_write','scripts.medtrace.run_dev16','scripts.medtrace.stage15','scripts.medtrace.run_selective_write','m3bench_repro.editors.llava_runtime','m3bench_repro.editors.routing','freshstart.runtime','worker_v3','router_r3']
imports={}
for n in modules:
 mod=importlib.import_module(n);f=Path(mod.__file__);imports[n]=dict(path=str(f),sha256=sha(f))
lock=dict(model_assets=assets,imports=imports,backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__),protocol=json.loads((r/'EXPERIMENT_LOCK.json').read_text()),roles={name:sha(r/'private'/name) for name in ['TASKS_R2_LOCKED.json','G_SUPPORTS.json','BASE_MASKS.json']},scientific_recipe='CP/TK/Direct native<=200 A2=80 W0=320 continuation=80; P/S/SG/SGD fixed coefficients and samplers; no scientific changes in PR5 repair')
lock['id']=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest();from storage import Store
Store(r).write('SCIENCE_LOCK.json',json.dumps(lock,indent=2).encode(),exclusive=True,pin=True);print('SCIENCE_LOCK',lock['id'],'assets',len(assets),flush=True)
