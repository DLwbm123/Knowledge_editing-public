"""Bounded CPU and real GPU checks before scientific execution."""
import copy
import importlib.util
import io
import os
from pathlib import Path
import torch
from structures import TT4,RANKS,COUNTS,optimizer_for
import updates


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def cpu_check():
    base=torch.nn.Linear(1,1).requires_grad_(False);checks=[]
    torch.manual_seed(17)
    refpath=Path(os.environ['RUN_ROOT'])/'private/tools/pr26_structures.py'
    spec=importlib.util.spec_from_file_location('pr26_ref',refpath);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    for name,ranks in RANKS.items():
        e=TT4(19,*ranks);assert sum(p.numel() for p in e.parameters())==COUNTS[name]
        x=torch.randn(3,14336,requires_grad=True);target=torch.randn(3,4096)
        assert torch.count_nonzero(e.residual(x))==0
        (e.residual(x)-target).square().mean().backward();assert e.G1.grad.norm()>0 and all(p.grad.norm()==0 for p in (e.G2,e.G3,e.G4))
        if name=='TT44':
            old=ref.TT4(19);assert equal(old.state_dict(),e.state_dict());o=optimizer_for(e,base);ro=ref.optimizer_for(old,base)
            (old.residual(x)-target).square().mean().backward();o.step();ro.step();assert equal(old.state_dict(),e.state_dict()) and equal(ro.state_dict(),o.state_dict())
        with torch.no_grad():e.G1.normal_(0,.1)
        e.zero_grad(set_to_none=True);b,a=e.factors();xn=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
        y=e.residual(x);y2=xn@a.T@b.T;torch.testing.assert_close(y,y2,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(y.sum(),x,retain_graph=True)[0],torch.autograd.grad(y2.sum(),x,retain_graph=True)[0],rtol=0,atol=0)
        (y-target).square().mean().backward();assert all(p.grad.norm()>0 and torch.isfinite(p.grad).all() for p in e.parameters())
        opt=optimizer_for(e,base);opt.step();stream=io.BytesIO();torch.save(dict(expert=e.state_dict(),optimizer=opt.state_dict(),rng=updates.rng()),stream);stream.seek(0)
        saved=torch.load(stream,weights_only=True);ee=TT4(19,*ranks);oo=optimizer_for(ee,base);ee.load_state_dict(saved['expert']);oo.load_state_dict(saved['optimizer']);assert equal(e.state_dict(),ee.state_dict()) and equal(opt.state_dict(),oo.state_dict())
        checks.append(dict(structure=name,parameters=COUNTS[name],zero_and_nonzero_gradients=True,temporary_forward_input_gradient=True,save_load_optimizer=True))
    # Exercise accepted alpha1, shrink committing moments once, rejected moments rollback, replay.
    initial=TT4(73,8,8);x=torch.randn(2,14336);target=torch.randn(2,4096)
    def run(mode):
        e=copy.deepcopy(initial);o=optimizer_for(e,base);count=[0]
        def step():
            count[0]+=1;o.zero_grad();(e.residual(x)-target).square().mean().backward();torch.nn.utils.clip_grad_norm_(e.parameters(),1);o.step();return {}
        before=copy.deepcopy(o.state_dict());calls=[0]
        def check():calls[0]+=1;return [0.] if mode=='open' or (mode=='shrink' and calls[0]==2) else [2.]
        item=updates.guarded_candidate(e,o,step,check,[1.]);assert count[0]==1
        if mode=='reject':assert equal(e.state_dict(),initial.state_dict()) and equal(o.state_dict(),before)
        return e,o,item
    full,fo,fi=run('open');small,so,si=run('shrink');rej,ro,ri=run('reject')
    assert fi['guard']['alpha']==1 and si['guard']['alpha']==.5 and equal(fo.state_dict(),so.state_dict())
    for k,v in initial.state_dict().items():assert torch.equal(small.state_dict()[k],v+.5*(full.state_dict()[k]-v))
    # Reject leaves step0 Adam and params intact; serialized resume reproduces next update.
    stream=io.BytesIO();torch.save(dict(e=rej.state_dict(),o=ro.state_dict(),r=updates.rng()),stream);stream.seek(0);saved=torch.load(stream,weights_only=True)
    replay=TT4(73,8,8);replay.load_state_dict(saved['e']);op=optimizer_for(replay,base);op.load_state_dict(saved['o']);updates.restore_rng(saved['r'])
    for e,o in [(rej,ro),(replay,op)]:
        o.zero_grad();(e.residual(x)-target).square().mean().backward();torch.nn.utils.clip_grad_norm_(e.parameters(),1);o.step()
    assert equal(rej.state_dict(),replay.state_dict()) and equal(ro.state_dict(),op.state_dict())
    return dict(structures=checks,PR26_TT44_initialization_and_single_update='PASS',guard_one_candidate_shrink_reject_resume='PASS')


def gpu_check(runtime,t,cfg,w):
    from audit import read,write
    from scripts.medtrace import stage18_cfact as cf
    from methods.medtrace import MedTraceLayerHook
    from dataclasses import replace
    import fcntl
    record=w.legacy.record_for(t);task=dict(t,U_fit=[w.legacy.local_row(u) for u in t['U_fit']],H_fit=w.H[t['edit_id']]);rows=task['H_fit']
    with (w.RUN/'private/teacher/WRITE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);teachers=cf.teachers_for(runtime,w.RUN,cfg,task,record)
    batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    hb=cf.source_batch(runtime,record,rows[0]);checks=[]
    base=[(p,p._version,p.data_ptr(),p.requires_grad) for p in runtime.model.parameters()]
    raw,b=w.legacy.check_input(runtime,t['native'],read(w.RUN/'private/legacy_stage17/BINDINGS.json'))
    for name,ranks in RANKS.items():
        w.budget();e=TT4(t['seed'],*ranks).to(runtime.device);reference=copy.deepcopy(e)
        opt=optimizer_for(e,runtime.model);ro=optimizer_for(reference,runtime.model);h=MedTraceLayerHook(runtime.get_module(w.LAYER),e);h.attach()
        try:
            with torch.inference_mode(),h.generation_request():z=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            assert list(z.raw_token_ids)==b['output']['raw_generated_token_ids']
            r=updates.rng();h.set_teacher_routing(batches[0].labels);hs=updates.hook_state(h)
            act=updates.capture_training_activations(runtime,h,batches,teachers,w.LAYER)
            diag,_=updates.diagnostic(runtime,h,e,[batches[0],batches[1]],teachers,hb,1.,act)
            assert equal(r,updates.rng()) and equal(hs,updates.hook_state(h)) and all(p.grad is None for p in e.parameters())
            item=w.logged_update(runtime,h,e,opt,batches[0],batches[1],teachers[0],(hb,rows[0]),1.,True);after=updates.rng()
            h.expert=reference;updates.restore_rng(r)
            cf.update(runtime,h,reference,ro,batches[0],batches[1],teachers[0],(hb,rows[0]));assert equal(e.state_dict(),reference.state_dict()) and equal(opt.state_dict(),ro.state_dict())
            updates.restore_rng(after);h.expert=e
            # Nonzero real-model gradients for all four cores after the G1 opening step.
            cf.update(runtime,h,e,opt,batches[0],batches[1],teachers[0],(hb,rows[0]));assert all(p.grad.norm()>0 and torch.isfinite(p.grad).all() for p in e.parameters())
            snapshot=copy.deepcopy(e.state_dict());osnap=copy.deepcopy(opt.state_dict());rng=updates.rng();hs=updates.hook_state(h)
            stream=io.BytesIO();torch.save(dict(e=snapshot,o=osnap,r=rng,h=hs),stream);stream.seek(0);saved=torch.load(stream,map_location=runtime.device,weights_only=True)
            cf.update(runtime,h,e,opt,batches[0],batches[1],teachers[0],(hb,rows[0]));expected=copy.deepcopy(e.state_dict());expectedopt=copy.deepcopy(opt.state_dict());expected_rng=updates.rng()
            e.load_state_dict(saved['e']);opt.load_state_dict(saved['o']);updates.restore_rng(saved['r']);updates.restore_hook(h,saved['h'])
            cf.update(runtime,h,e,opt,batches[0],batches[1],teachers[0],(hb,rows[0]));assert equal(expected,e.state_dict()) and equal(expectedopt,opt.state_dict()) and equal(expected_rng,updates.rng())
            checks.append(dict(structure=name,zero_native_tokens=True,diagnostic_rng_hook_gradient_state_parity=True,actual_update_parity=True,save_load_next_step=True,all_core_gradients=True))
        finally:h.detach()
        del e,reference,opt,ro;torch.cuda.empty_cache()
    assert all(p._version==v and p.data_ptr()==ptr and p.requires_grad==req==False for p,v,ptr,req in base)
    write(w.RUN/'private/GPU_P0_DETAILS.json',dict(status='PASS',checks=checks,Base_frozen=True,temporary_qualification_updates_not_formal_continuations=True))
