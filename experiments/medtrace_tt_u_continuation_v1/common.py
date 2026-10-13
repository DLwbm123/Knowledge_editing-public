"""Bounded run I/O, runtime binding and exclusive idle GPU admission."""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

RUN=Path(os.environ['RUN_ROOT'])
LAYER='model.layers.21.mlp.down_proj'
sys.path.insert(0,str(RUN/'private/source'))

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def digest(d):
    from scripts.medtrace.stage17_prepare import digest as shared
    return shared(d)
@contextmanager
def resources():
    with (RUN/'RESOURCE_LEDGER.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);d=read(RUN/'RESOURCE_LEDGER.json');yield d;write(RUN/'RESOURCE_LEDGER.json',d)
def used():
    d=read(RUN/'RESOURCE_LEDGER.json')
    return d['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if not s.get('ended_epoch'))
def budget():
    m=read(RUN/'RUN_MANIFEST.json')
    if (RUN/'STOP').exists() or time.time()>=m['deadline_epoch'] or used()>=m['GPU_seconds_limit']:raise TimeoutError('Persisted run cap reached')
    if shutil.disk_usage(RUN).free<m['min_free_bytes']:raise OSError('Disk reserve reached')
def save(p,d):
    import torch
    budget();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.'+str(os.getpid())+'.tmp');torch.save(d,temp)
    size=sum(x.stat().st_size for x in (RUN/'private').rglob('*.pt') if not x.is_symlink())+temp.stat().st_size
    if size>=read(RUN/'RUN_MANIFEST.json')['owned_weight_limit_bytes']:
        temp.unlink();raise OSError('Generated checkpoint/teacher cap')
    temp.replace(p)
    with resources() as r:r['weights_observed_peak_bytes']=max(r.get('weights_observed_peak_bytes',0),size)
def state_hash(e):
    h=hashlib.sha256()
    for k,v in sorted(e.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def rng():
    import torch
    return dict(torch_rng=torch.get_rng_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def restore_rng(r):
    import torch
    torch.set_rng_state(r['torch_rng'].cpu());random.setstate(r['python_rng'])
    if r['cuda_rng'] is not None:torch.cuda.set_rng_state(r['cuda_rng'].cpu())
@contextmanager
def diagnostic_scope(hook):
    import copy
    r=rng();fields=('enabled','token_mask','generation_routing','generation_boundary','generation_trace','last_generation_trace','anchor_reference','anchor_loss')
    state={k:copy.deepcopy(getattr(hook,k)) for k in fields}
    try:yield
    finally:
        restore_rng(r)
        for k,v in state.items():setattr(hook,k,v)
def local_path(p):
    p=str(p)
    if '/derived_inputs/' in p:return str(RUN/'private/derived_inputs'/p.split('/derived_inputs/',1)[1])
    if '/DataP/' in p:return os.environ['DATA_ROOT']+'/'+p.split('/DataP/',1)[1]
    return p
def record(t):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    n=t['native'];return EditorRecord(t['edit_id'],n['dataset'],n['question'],n['reference'],t['fit_questions'][0],Path(local_path(n['image_path'])),n['original_image_path'],t['order'],'VERIFIED_SOURCE_ANSWER','NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')
def available(gpu):
    budget();u=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert row[0]==u
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    return int(row[1])>=60000 and int(row[2])<100 and int(row[3])==0 and u not in apps
@contextmanager
def lease(gpu):
    u=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    with (Path(os.environ['TMPDIR'])/('lease.'+u)).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert available(gpu),'Idle GPU admission failed'
        entry=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_uuid=u,started_epoch=time.time(),action=os.environ['ACTION'])
        with resources() as r:
            assert not any(s['gpu_uuid']==u and not s.get('ended_epoch') for s in r['gpu_sessions']);r['gpu_sessions'].append(entry)
        try:yield
        finally:
            with resources() as r:
                s=next(x for x in r['gpu_sessions'] if x['pid']==entry['pid'] and x['started_epoch']==entry['started_epoch']);s.update(ended_epoch=time.time(),resident_seconds=time.time()-entry['started_epoch']);r['gpu_seconds_used']+=s['resident_seconds']
def process_audit(gpu):
    rows=[x.strip().split(None,2) for x in subprocess.check_output(['ps','-ww','-eo','pid=,ppid=,args='],text=True).splitlines()];family={os.getpid()}
    while True:
        new=family|{int(p) for p,parent,*_ in rows if int(parent) in family}
        if new==family:break
        family=new
    cmds={p:args for p,parent,args in rows if int(p) in family};assert not any(s in cmd for cmd in cmds.values() for s in ('wangbomin','Knowledge_editing','medtrace','TT-U'))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines();own=[x for x in apps if int(x.split(',')[0]) in family]
    u=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)];assert own and all(x.split(',')[1].strip()==u for x in own)
    write(RUN/'private'/('PROCESS_AUDIT_'+os.environ['ACTION']+'_'+str(gpu)+'.json'),dict(commands=cmds,GPU_processes=own,neutral_argv=True))
def load(gpu):
    import torch
    u=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=u,M3BENCH_LLAVA_SOURCE=str(RUN/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(RUN/'private/official_llava'))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUN/'private/source',text=True).strip()==read(RUN/'private/SOURCE_COMMIT.json')['commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=RUN/'private/source',text=True).strip()
    torch.use_deterministic_algorithms(True)
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=RUN/'private/cpu_gate'));runtime.model.eval();assert not any(p.requires_grad for p in runtime.model.parameters())
    b=read(RUN/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(b.values()))['generation']
    runtime.run_root=RUN/'private/work'/str(gpu);runtime.run_root.mkdir(parents=True,exist_ok=True);process_audit(gpu)
    return runtime,b
