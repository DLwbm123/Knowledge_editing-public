"""Qualified-device kernels and real frozen-backbone/hook/save-reload canary."""
import os,sys,time,traceback,json,hashlib,inspect,importlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'source_patch'))
from resources import session,write,check

def main():
 import torch
 from structures import TuckerC4,LR4,convert
 from storage import Store
 from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
 from freshstart import runtime as rt
 rt.ROOT=ROOT;rt.GPU=os.environ['CUDA_VISIBLE_DEVICES'];rt.check_budget=lambda **kw:check(kw.get('training',False))
 slot=int(os.environ['PHYSICAL_GPU']);result=dict(slot=slot,pid=os.getpid(),uuid=rt.GPU,torch=torch.__version__,backend_cuda=torch.version.cuda,argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode())
 try:
  with session('CANARY_'+str(slot),1200):
   assert torch.cuda.device_count()==1;result['capability']=torch.cuda.get_device_capability(0)
   x=torch.randn(4,14336,device='cuda');t=TuckerC4().cuda();assert not t.residual(x).count_nonzero();loss=(t.residual(x)-1).square().mean();loss.backward();assert t.L.grad.norm()>0
   with torch.no_grad():t.L.normal_(0,.01)
   free=convert(t,20260927);error=float((t.residual(x)-free.residual(x)).abs().max());assert error<1e-5;result['Tucker_conversion_max_abs']=error
   cp=AsymmetricCPExpert(14336,4096,4).cuda()
   with torch.no_grad():cp.rho.fill_(.03)
   freecp=convert(cp,20260927);assert torch.allclose(cp.residual(x),freecp.residual(x),atol=1e-6,rtol=1e-4)
   del t,free,cp,freecp,x;torch.cuda.empty_cache()
   if slot==5:
    from dataclasses import replace
    import worker_v3 as old
    from scripts.medtrace import stage15
    run=ROOT/'jobs/canary5';runtime=rt.load_runtime(run,20260927)
    assert runtime.generation_config['max_new_tokens']==1024 and all(not p.requires_grad for p in runtime.model.parameters())
    runtime.model.register_forward_pre_hook(lambda m,a:check())
    task=json.loads((ROOT/'private/TASKS_R2_LOCKED.json').read_text())['tasks'][0];rec=old.record(task);batch=runtime.build_edit_batch(rec)
    base,key=old.base(runtime,task['native'],rec)
    stats=[]
    for kind,expert in [('CP',AsymmetricCPExpert(14336,4096,4)),('TK-C4',TuckerC4()),('Direct-LR4',LR4())]:
     expert=expert.cuda();hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert);hook.attach();opt=torch.optim.AdamW(expert.parameters(),lr=.001,weight_decay=0)
     try:
      for step in range(1,3):
       opt.zero_grad();hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(expert.parameters(),1));assert torch.isfinite(loss) and norm>0;opt.step();expert.normalize_factors_(verify_dense=False)
      hook.clear_request_routing();payload=dict(expert=expert.state_dict(),optimizer=opt.state_dict(),step=2,seed=20260927,kind=kind,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
      store=Store(ROOT);rel='checkpoints/canary/'+kind+'/latest.pt';store.save(rel,payload,consumers=['reload'])
      restored=torch.load(ROOT/rel,map_location='cuda',weights_only=True);assert all(torch.equal(v,restored['expert'][k]) for k,v in expert.state_dict().items());store.consumed(rel,'reload');store.delete(rel)
      raw,_,binding=stage15.prepared(runtime,task['native'],rec);output=stage15.generate(runtime,raw,binding,hook=hook);judge=old.request(task['native'],output)
      stats.append(dict(kind=kind,steps=2,CE=float(loss),gradient_norm=norm,judge_key=judge,save_reload=True))
     finally:hook.detach()
    assert runtime.base_guard.verify()['unchanged'];result.update(real_model='PASS',experts=stats,Base_judge_key=key,teacher_cap=1024)
    modules={}
    for name in ['freshstart.runtime','methods.medtrace.core','methods.medtrace.selective_write','scripts.medtrace.stage15','scripts.medtrace.run_selective_write','m3bench_repro.editors.llava_runtime']:
     p=Path(importlib.import_module(name).__file__).resolve();modules[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    write(ROOT/'MODULE_PATHS.json',modules)
   result['status']='PASS'
 except Exception as e:result.update(status='FAILED',error=str(e),traceback=traceback.format_exc());raise
 finally:write(ROOT/'jobs'/('CANARY_GPU'+str(slot)+'.json'),result)
if __name__=='__main__':main()
