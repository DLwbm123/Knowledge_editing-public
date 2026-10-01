"""Inference-only resident worker: no training, loss, optimizer, or expert mutation."""
import os,sys,json,time,io,fcntl,traceback,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session,check
def configure():
    import torch
    from freshstart import runtime as rt
    import worker_v3 as old
    from scripts.medtrace import stage15,run_selective_write
    from storage import Store
    from judge_protocol import request
    torch.set_grad_enabled(False)
    def forbidden(*a,**k):raise RuntimeError('Router-only contract forbids backward')
    torch.Tensor.backward=torch.autograd.backward=forbidden
    store=Store(ROOT);rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check();rt.gpu_session=session
    def guarded(p,d):write(Path(p),d)
    def tensor_save(p,x):
        b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
    old.request=request;old.write=stage15.write=guarded;run_selective_write.save=tensor_save
    runtime=rt.load_runtime(ROOT/'jobs'/('worker'+os.environ['PHYSICAL_GPU']),20260929);counter=[0]
    def count(*a):check();counter[0]+=1
    runtime.model.register_forward_pre_hook(count)
    return runtime,counter
def weight_bindings():
    from storage import digest_file
    p=ROOT/'private/H_WEIGHT_BINDINGS.json'
    if p.exists():return read(p)
    old=Path(read(ROOT/'PREDECESSOR.json')['root']);tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];result={}
    for t in tasks:
        path=old/f'adapters/s20260929/AH/e{t["order"]:03d}.pt'
        assert path.exists(),'Frozen H expert unavailable; never retrain'
        result[str(t['order'])]=dict(adapter=str(path),sha256=digest_file(path),actual_steps=80,origin='INIT_POST80')
    write(p,result);return result
def load_expert(t,weights):
    import torch
    from structures import LR4
    from storage import digest_file
    b=weights[str(t['order'])];path=Path(b['adapter']);assert digest_file(path)==b['sha256']
    x=torch.load(path,map_location='cpu',weights_only=False);meta=next(m for m in read(ROOT/'private/INITIALIZERS.json') if m['order']==t['order'])
    assert x['step']==80 and x['binding']['edit']==t['canonical_edit_id'] and x['binding']['initialization_sha256']==meta['origin_sha256']
    with torch.random.fork_rng(devices=[0]):ex=LR4().cuda();ex.load_state_dict(x['expert'])
    ex.requires_grad_(False);assert all(torch.equal(v.cpu(),x['expert'][k]) for k,v in ex.state_dict().items())
    return ex
def exposed_task():
    rows=read(ROOT/'private/LOCALITY_STRESS_HOLDOUT.json')['rows']
    return dict(canonical_edit_id='EXPOSED_REGRESSION',order=0,native=rows[0],fit_questions=[rows[0]['question']],evaluation=[dict(r,task='EXPOSED_REGRESSION',query_id='exposed-'+hashlib.sha256((r['image_sha256']+'\0'+r['question']).encode()).hexdigest()) for r in rows])
def main():
    import torch
    from routers import ScopeRouter,tensors_to
    from evaluation import evaluate
    from phases import references
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];weights=weight_bindings()
    features=torch.load(ROOT/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False)
    with session('ROUTER_INFERENCE_ONLY',7200):
        runtime,counter=configure();began=time.time();protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
        while time.time()-began<6600:
            with (ROOT/'QUEUE.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');j=next((j for j in q if j['status']=='PENDING'),None)
                if j is None:break
                j.update(status='RUNNING',pid=os.getpid(),gpu=int(os.environ['PHYSICAL_GPU']),started=time.time());write(ROOT/'QUEUE.json',q)
            selected=[t for t in tasks if t['order'] in j['orders']];bank={t['canonical_edit_id']:load_expert(t,weights) for t in selected};snapshot={e:{k:v.clone() for k,v in ex.state_dict().items()} for e,ex in bank.items()};n=counter[0];start=time.time()
            if j['mode']=='single':
                for t in selected:
                    es=tensors_to([features['singles'][str(t['order'])]],runtime.device);router=ScopeRouter(es,j['method'])
                    evaluate(runtime,[t],{t['canonical_edit_id']:bank[t['canonical_edit_id']]},router,j['method'],'single',1,ROOT/'jobs'/j['id']/str(t['order']),protocol,references(j['phase']),features,weights)
            else:
                es=tensors_to(features['banks'][j['phase']],runtime.device)
                for prefix in j['prefixes']:
                    router=ScopeRouter(es[:prefix],j['method']);panel=[exposed_task()] if j.get('exposed') else selected[:prefix]
                    evaluate(runtime,panel,{e:bank[e] for e in router.logical_ids},router,j['method'],'EXPOSED_REGRESSION' if j.get('exposed') else 'sequential',prefix,ROOT/'jobs'/j['id']/f'p{prefix}',protocol,references(j['phase']),features,weights)
            assert all(torch.equal(v,snapshot[e][k]) for e,ex in bank.items() for k,v in ex.state_dict().items()),'Expert mutated'
            assert runtime.base_guard.verify()['unchanged'];assert not runtime.get_module('model.layers.30.mlp.down_proj')._forward_hooks
            write(ROOT/'jobs'/j['id']/'COMPUTE_COUNTS.json',dict(model_calls=counter[0]-n,seconds=time.time()-start,training_steps=0,backward_calls=0,optimizer_steps=0,expert_weights_unchanged=True));del bank,snapshot;torch.cuda.empty_cache()
            with (ROOT/'QUEUE.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');next(x for x in q if x['id']==j['id']).update(status='COMPLETE',ended=time.time());write(ROOT/'QUEUE.json',q)
    write(ROOT/'jobs'/f'WORKER_{os.environ["PHYSICAL_GPU"]}.json',dict(status='IDLE',pid=os.getpid()))
if __name__=='__main__':
    try:main()
    except Exception as e:write(ROOT/'jobs'/f'WORKER_{os.environ["PHYSICAL_GPU"]}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc()));raise
