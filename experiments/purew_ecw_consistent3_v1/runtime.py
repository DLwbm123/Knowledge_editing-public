"""Shared native runtime/counts and neutral finite IPC; no learning from evaluation."""
import os
from pathlib import Path
import subprocess
import time
import common as c
import base_run as b

R=c.RUN
ARM=os.environ.get('ARM','M')
WINDOW=R/'private/window'/ARM
CONTROL=R/'private/control'/ARM
b.ARM=ARM
_original_counts=b.counts
def counts():
    _original_counts()
    slots=sum(c.read(p).get('attempt_slots',0) for p in (R/'private/attempts').glob('*.json'))
    protocol=c.read(R/'private/FROZEN_PROTOCOL.json')
    assert slots<=protocol['budgets']['maximum_formal_attempt_slots']
    if protocol.get('execution_revision')=='J_ONLY_DECOUPLED_REVISION1':
        assert slots<=60
        assert sum(c.read(p).get('committed_updates',0) for p in (R/'private/attempts').glob('*.json'))<=61
b.counts=counts


def available(gpu):
    c.budget()
    assert gpu == 2
    u, free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert u == c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['2']
    return int(free) >= 45*1024


b.available=c.available=available


def load_learner():
    import torch
    from types import SimpleNamespace
    gpu=2;u=c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['2']
    os.environ.update(CUDA_VISIBLE_DEVICES='2',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES='2',
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='2',M3BENCH_FORMAL_EXPECTED_GPU_UUID=u,
        M3BENCH_LLAVA_SOURCE=str(R/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(R/'private/official_llava'))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'private/source',text=True).strip()==c.read(R/'private/SOURCE_COMMIT.json')['commit']
    torch.use_deterministic_algorithms(True)
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=R/'private/learner_gate'))
    runtime.model.eval();assert not any(p.requires_grad for p in runtime.model.parameters())
    runtime.run_root=R/'private/learner_work'/ARM;runtime.run_root.mkdir(parents=True,exist_ok=True)
    c.process_audit(gpu)
    return runtime


def track(runtime):
    from contextlib import ExitStack
    stack=ExitStack()
    def forward(*_):b.COUNTS['forwards']+=1;b.call('F')
    def vision(*_):
        b.COUNTS['forwards']+=1;b.COUNTS['vision_forwards']=b.COUNTS.get('vision_forwards',0)+1;b.call('V')
    stack.callback(runtime.model.register_forward_pre_hook(forward).remove)
    stack.callback(runtime.model.get_vision_tower().register_forward_pre_hook(vision).remove)
    return stack


def backward(loss,weight):
    import torch
    b.COUNTS['backwards']+=1;b.call('B');b.counts()
    return torch.autograd.grad(128*loss,weight)[0].float()/128


def wait(path):
    while not Path(path).exists():
        c.budget()
        if list((R/'private/failures').glob('*.json')):raise RuntimeError('Retained stage failure; inspect before continuing')
        time.sleep(.2)
    return c.read(path)


def peak_memory():
    import torch
    own=c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions']
    ids={x['pid'] for x in own if not x.get('ended_epoch')}
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    used=sum(int(s.split(',')[2].strip()) for s in rows if int(s.split(',')[0]) in ids)
    assert used<=45*1024,'Owned GPU residency ceiling'
    with c.resources() as r:r['GPU_memory_observed_peak_MiB']=max(r.get('GPU_memory_observed_peak_MiB',0),used)
    return dict(owned_gpu_memory_MiB=used,process_peak_allocated_bytes=torch.cuda.max_memory_allocated())


def lease():
    # A controller owns the experiment lock; learner/evaluator have separate counted residencies.
    from contextlib import contextmanager
    @contextmanager
    def managed():
        row=dict(pid=os.getpid(),gpu_uuid=c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['2'],
                 start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                 started_epoch=time.time(),action=os.environ['ACTION'])
        with c.resources() as r:r['gpu_sessions'].append(row)
        try:yield
        finally:
            with c.resources() as r:
                s=next(x for x in r['gpu_sessions'] if x['pid']==row['pid'] and x['started_epoch']==row['started_epoch'])
                s.update(ended_epoch=time.time(),resident_seconds=time.time()-row['started_epoch']);r['gpu_seconds_used']+=s['resident_seconds']
    return managed()
