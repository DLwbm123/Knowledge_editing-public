"""Resident paired worker; all writable artifacts belong to the new namespace."""
import os,sys,json,time,hashlib,io,fcntl,traceback,copy
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,check,session
from storage import Store,digest_file

def configure():
    import torch
    from freshstart import runtime as rt
    import worker_v3 as old
    from scripts.medtrace import stage15,run_selective_write
    import evaluation,diagnostics
    store=Store(ROOT);rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False));rt.gpu_session=session
    diagnostics.selected=lambda t:False;diagnostics.diagnose=lambda *a,**k:None
    def guarded(p,d):store.write(str(Path(p).relative_to(ROOT)),json.dumps(d,ensure_ascii=False).encode())
    def tensor_save(p,x):
        assert torch.is_tensor(x) and x.numel()<1000000
        b=io.BytesIO();torch.save(x,b);store.write(str(Path(p).relative_to(ROOT)),b.getvalue())
    from judge_protocol import request
    old.request=request
    old.write=stage15.write=evaluation.write=guarded;run_selective_write.save=tensor_save;evaluation.SEED=20260929
    runtime=rt.load_runtime(ROOT/'jobs'/('worker'+os.environ['PHYSICAL_GPU']),20260929)
    counter=[0]
    def count(*a):check();counter[0]+=1
    runtime.model.register_forward_pre_hook(count)
    return runtime,counter

def setup(runtime,t):
    import torch
    from dataclasses import replace
    from collections import OrderedDict
    import worker_v3 as old
    from scripts.medtrace.run_selective_write import teacher_batch
    from scripts.medtrace.stage18_cfact import assert_base_off
    from methods.medtrace.selective_write import full_vocab_kl
    from structures import LR4
    from training import u_teachers,derive_seed,LAYER
    from protection import protector
    from anchor import prepare_positive,coefficients,fixed_diagnostics
    meta=next(x for x in read(ROOT/'private/INITIALIZERS.json') if x['order']==t['order'])
    p=Path(meta['relative_path']);assert digest_file(p)==meta['sha256']
    initial=torch.load(p,map_location='cpu',weights_only=False)
    ex=LR4().cuda();ex.load_state_dict(initial['expert']);ex.requires_grad_(False)
    rec=old.record(t);bg=read(ROOT/'private/U_bg.json')['rows'];byid={old.input_id(r):r for r in bg}
    hardmeta=read(ROOT/f'private/hard/{t["order"]}.json');hard=[byid[r['input_hash']] for r in hardmeta['rows']]
    groups,size=u_teachers(runtime,t);cache=OrderedDict()
    def teacher(row,hook=None):
        k=old.input_id(row)
        if k in cache:cache.move_to_end(k);return cache[k]
        if hook:hook.detach()
        try:
            assert_base_off(runtime);out,_=old.base(runtime,row,rec,score=False);kw,labels,mask,b=teacher_batch(runtime,dict(row,eqkey=k),out['raw_token_ids'])
            with torch.no_grad():lp=runtime.model(**kw).logits[mask].float().log_softmax(-1).cpu()
            value=(kw,labels,mask,lp)
        finally:
            if hook:hook.attach()
        cache[k]=value
        while len(cache)>2:cache.popitem(last=False)
        return value
    class LazyRows:
        def __init__(self,rows):self.rows=rows;self.hook=None
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return teacher(self.rows[i],self.hook)
    background=LazyRows(bg);hardrows=LazyRows(hard)
    scale=ex.A.new_tensor(meta['c_minus'])
    batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['semantic_fit_questions']]
    start=time.time();ref,cplus,prep_calls=prepare_positive(runtime,runtime.get_module(LAYER),ex,batches,ROOT/f'private/positive_scales/{t["order"]}.json',read,write)
    prep_seconds=time.time()-start
    fixedneg=[groups[0][0],teacher(hard[0])]
    ss=derive_seed(t['canonical_edit_id'],20260929)+1
    def make_protect(method):
        lam,beta=coefficients(method)
        apply=protector(runtime,runtime.get_module(LAYER),ex,*groups,background,hardrows,'AHS' if lam else 'AH',ss,full_vocab_kl,scale,lam)
        def protect(hook,step):
            background.hook=hardrows.hook=hook
            return apply(hook,step)
        return protect
    def diagnostic(hook,batch):
        torch.cuda.synchronize();start=time.time()
        result=fixed_diagnostics(runtime,runtime.get_module(LAYER),ex,hook,batch,fixedneg,ref,cplus,scale,full_vocab_kl)
        torch.cuda.synchronize();result['seconds']=time.time()-start
        return result
    return dict(ex=ex,initial=initial,ref=ref,cplus=cplus,seed=ss,protect=make_protect,diagnostic=diagnostic,prep=dict(model_calls=prep_calls,seconds=prep_seconds),batches=batches)

def load_expert(t,method):
    import torch
    from training import load
    from structures import LR4
    from bindings import expected
    from anchor import ScaledExpert
    oldroot=Path(read(ROOT/'PREDECESSOR.json')['root'])
    gamma=float(method.split('_',1)[1]) if method.startswith('G_') else 1.
    inherited={'E_orig':'E_orig','A0':'A0','H':'AH','S':'AHS_01'}
    if method in inherited or method.startswith('G_'):
        m='AH' if method.startswith('G_') else inherited[method]
        p=oldroot/f'adapters/s20260929/{m}/e{t["order"]:03d}.pt';x=torch.load(p,map_location='cpu',weights_only=False)
        meta=next(x for x in read(ROOT/'private/INITIALIZERS.json') if x['order']==t['order'])
        assert x['binding']['initialization_sha256']==meta['origin_sha256'] and x['binding']['edit']==t['canonical_edit_id']
        ex=LR4().cuda();ex.load_state_dict(x['expert'])
    else:
        p=ROOT/f'adapters/s20260929/{method}/e{t["order"]:03d}.pt'
        ex,x=load(str(p.relative_to(ROOT)),expected(t,20260929,'LR',method,'continuation',80))
    ex.requires_grad_(False)
    binding=dict(adapter=str(p),sha256=digest_file(p),actual_steps=x['step'],origin='INIT_POST80',gamma=gamma,hook='complete_residual_once' if method.startswith('G_') else 'unscaled')
    return (ScaledExpert(ex,gamma) if method.startswith('G_') else ex),binding

def holdout_task():
    rows=read(ROOT/'private/LOCALITY_STRESS_HOLDOUT.json')['rows']
    assert len(rows)==47 and all(x['scope']=='negative' for x in rows)
    return dict(canonical_edit_id='EXPOSED_HOLDOUT',order=0,native=rows[0],fit_questions=[rows[0]['question']],evaluation=[dict(x,task='EXPOSED_HOLDOUT',query_id='holdout-'+hashlib.sha256((x['image_sha256']+'\0'+x['question']).encode()).hexdigest()) for x in rows])

def evaluate_job(runtime,j,tasks,protocol):
    import evaluation,worker_v3 as old
    from router_r3 import RejectRouter
    method=j['method'];selected=[next(t for t in tasks if t['order']==o) for o in j['orders']]
    bank={};router=RejectRouter();bindings=[]
    for i,t in enumerate(selected,1):
        ex,bind=load_expert(t,method);bank[t['canonical_edit_id']]=ex;bindings.append(bind)
        key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius)
        if i in j['prefixes']:
            panel=[holdout_task()] if j.get('holdout') else selected[:i]
            evaluation.evaluate(runtime,panel,bank,router,method,'holdout' if j.get('holdout') else 'sequential',i,ROOT/'jobs'/j['id']/f'p{i}',bindings,protocol)

def main():
    import torch,evaluation,worker_v3 as old
    from training import stage,save_final
    from bindings import expected
    from anchor import coefficients
    from router_r3 import RejectRouter
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];slot=int(os.environ['PHYSICAL_GPU']);store=Store(ROOT)
    with session('PAIRED_WORKER',7200):
        runtime,counter=configure();began=time.time()
        protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
        while time.time()-began<6600:
            with (ROOT/'QUEUE.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');j=next((x for x in q if x['status']=='PENDING' and all((ROOT/p).exists() for p in x.get('requires',[]))),None)
                if j is None:break
                j.update(status='RUNNING',pid=os.getpid(),gpu=slot,started=time.time());write(ROOT/'QUEUE.json',q)
            n=counter[0];start=time.time()
            if j['mode']=='train':
                t=next(t for t in tasks if t['order']==j['order']);o=t['order'];s=setup(runtime,t);ex=s['ex']
                key,radius=old.router_entry(runtime,t);router=RejectRouter();router.add(t['canonical_edit_id'],key,radius)
                for method in j['methods']:
                    dst=f'adapters/s20260929/{method}/e{o:03d}.pt'
                    if not (ROOT/dst).exists():
                        ex.load_state_dict(s['initial']['expert']);ex.requires_grad_(False);lam,beta=coefficients(method)
                        stage(runtime,t,ex,'LR','continuation',s['seed'],method,s['protect'](method),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],beta),diagnostic=s['diagnostic'])
                        save_final(store,dst,ex,20260929,t['canonical_edit_id'],'LR',80,expected(t,20260929,'LR',method,'continuation',80))
                    expert,bind=load_expert(t,method)
                    evaluation.evaluate(runtime,[t],{t['canonical_edit_id']:expert},router,method,'single',1,ROOT/'jobs'/j['id']/method/'single',[bind],protocol)
                    chk=copy.deepcopy(t);chk['evaluation']=read(ROOT/'private/CHECK_POS.json').get(str(o),[])+[dict(r,task='CHECK_NEG',query_id='check-'+old.input_id(r)) for r in read(ROOT/'private/CHECK_NEG.json')['rows']]
                    if j.get('check',False):evaluation.evaluate(runtime,[chk],{t['canonical_edit_id']:expert},router,method,'CHECK_FORCED_ON',1,ROOT/'jobs'/j['id']/method/'check',[bind],protocol,forced=True)
                    del expert
                write(ROOT/'jobs'/j['id']/'PREPARATION_COST.json',s['prep']);del s,ex
            elif j['mode'] in ['check','single']:
                for o in j['orders']:
                    t=next(t for t in tasks if t['order']==o);expert,bind=load_expert(t,j['method']);router=RejectRouter();key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius)
                    chk=copy.deepcopy(t)
                    if j['mode']=='check':chk['evaluation']=read(ROOT/'private/CHECK_POS.json')[str(o)]+[dict(r,task='CHECK_NEG',query_id='check-'+old.input_id(r)) for r in read(ROOT/'private/CHECK_NEG.json')['rows']]
                    evaluation.evaluate(runtime,[chk],{t['canonical_edit_id']:expert},router,j['method'],'CHECK_FORCED_ON' if j['mode']=='check' else 'single',1,ROOT/'jobs'/j['id']/str(o),[bind],protocol,forced=j['mode']=='check')
                    del expert
            else:evaluate_job(runtime,j,tasks,protocol)
            assert runtime.base_guard.verify()['unchanged'];torch.cuda.empty_cache()
            write(ROOT/'jobs'/j['id']/'COMPUTE_COUNTS.json',dict(model_calls=counter[0]-n,seconds=time.time()-start))
            with (ROOT/'QUEUE.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');next(x for x in q if x['id']==j['id']).update(status='COMPLETE',ended=time.time());write(ROOT/'QUEUE.json',q)
    write(ROOT/'jobs'/f'WORKER_{slot}.json',dict(status='IDLE',pid=os.getpid()))
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'jobs'/f'WORKER_{os.environ["PHYSICAL_GPU"]}.json',dict(status='FAILED',pid=os.getpid(),error=str(e),traceback=traceback.format_exc()));raise
