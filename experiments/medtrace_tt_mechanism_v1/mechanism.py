"""Fixed same-state probes along the already frozen RAW pilot trajectory."""
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import direction as d
import mechanism_math as mm

p,c,RUN=d.p,d.c,d.RUN
GPUS=(2,3,4,5,6,7)
STEPS=(2,80,160,320)
PILOT=Path(os.environ['DIRECTION_PARENT'])


def logits(runtime,hook,batches):
    result=[]
    with torch.inference_mode():
        for b in batches:
            hook.set_teacher_routing(b.labels)
            out=runtime.model(**b.forward_kwargs());mask=b.labels[:,1:]!=-100
            result.append(out.logits[:,:-1][mask].double().log_softmax(-1))
    return result


def probe(runtime,hook,expert,opt,batches,x,before,adam,actual,t,step):
    rng=c.rng();states,groups=mm.candidate_states(expert,opt,before,adam)
    assert all(torch.equal(states['RAW'][k],actual[k]) for k in actual)
    path=RUN/'private/snapshots'/f"{t['order']}_{step}.pt"
    c.save(path,dict(parameters_before=before,Adam_candidate=adam,RAW_actual=actual,optimizer_after_moment_update=opt.state_dict(),
        gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()},step=step,expert_order=t['order'],rng=rng,
        phase='prewrite_parameters_with_current_gradient_and_postupdate_moments',execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
    saved=p.load_state(path);assert all(torch.equal(saved['parameters_before'][k],v.cpu()) for k,v in before.items())
    records=[]
    try:
        expert.load_state_dict(before);base_logp=logits(runtime,hook,batches);base_f=p.tr.functional(expert,x)
        maps={mode:mm.map_delta(before,state) for mode,state in states.items()}
        for mode in mm.MODES:
            expert.load_state_dict(states[mode]);lp=logits(runtime,hook,batches);fx=p.tr.functional(expert,x)-base_f
            stats=[]
            for i,(b,old,new) in enumerate(zip(batches,base_logp,lp)):
                target=b.labels[:,1:][b.labels[:,1:]!=-100];pos=torch.arange(len(target),device=target.device)
                kl=float((old.exp()*(old-new)).sum(-1).mean());assert kl>=-1e-10
                stats.append(dict(role='native' if i==0 else 'FIT',slot=i,tokens=len(target),KL_from_prewrite=max(0.,kl),
                    target_NLL_before=float(-old[pos,target].mean()),target_NLL_after=float(-new[pos,target].mean()),
                    argmax_changes=int((old.argmax(-1)!=new.argmax(-1)).sum()),argmax_correct=int((new.argmax(-1)==target).sum())))
            left,right=maps[mode];a,b=maps['ADAM'];norm2=mm.lowrank_inner(left,right,left,right);anorm2=mm.lowrank_inner(a,b,a,b)
            assert norm2>=-1e-10 and anorm2>=-1e-10
            records.append(dict(mode=mode,map_delta_frobenius=max(0.,norm2)**.5,
                map_delta_cosine_to_Adam=mm.lowrank_inner(left,right,a,b)/(norm2*anorm2)**.5 if norm2>0 and anorm2>0 else None,
                predictor_residual_delta_norm=float(fx.norm()),native_FIT=stats))
        return dict(expert_order=t['order'],step=step,groups=groups,candidates=records,snapshot_save_load_exact=True,
            trajectory='RAW',moment_timing='Same current clipped gradient; post-update first/second moments; all directions start at prewrite parameters')
    finally:
        expert.load_state_dict(actual);c.restore_rng(rng)


def trajectory(runtime,t):
    from m3bench_repro.editors.llava_runtime import seed_everything
    root=RUN/'private/experts'/str(t['order']);done=root/'COMPLETE.json'
    if done.exists():return
    expert=p.expert(d.start_state(t),t['seed'],runtime.device)
    batches=[runtime.build_edit_batch(r) for r in [c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]]
    assert len(batches)==5 and all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
    hook=d.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    seed_everything(t['seed']);opt=p.optimizer(expert,runtime.model);latest=root/'resume.pt';curve=[];probes=[];start=0
    try:
        x=p.tr.activations(runtime,hook,batches,[])
        if latest.exists():
            saved=p.load_state(latest);assert saved['execution']==c.read(RUN/'private/GPU_SOURCE_VERSION.json')
            expert.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);c.restore_rng(saved);start=saved['step'];curve=saved['curve'];probes=saved['probes']
        for step in range(start+1,321):
            c.budget();opt.zero_grad(set_to_none=True);before={k:v.detach().clone() for k,v in expert.state_dict().items()};losses=[]
            for i in [0,1+(step-1)%4]:
                hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss);(.5*loss).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in expert.parameters()) and not any(x.grad is not None for x in runtime.model.parameters())
            opt.step();adam={k:v.detach().clone() for k,v in expert.state_dict().items()}
            d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW')
            actual={k:v.detach().clone() for k,v in expert.state_dict().items()}
            if step in STEPS:probes.append(probe(runtime,hook,expert,opt,batches,x,before,adam,actual,t,step))
            curve.append(dict(step=step,CE=losses,gradient_norm=float(norm)))
            if step%20==0:
                c.save(latest,dict(execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),expert=expert.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,probes=probes,**c.rng()));print('STEP',t['order'],step,flush=True)
        historical=p.load_state(PILOT/'private/weights'/t['anonymous_edit']/'RAW_DIR_540/FINAL.pt')['expert']
        assert all(torch.equal(v.detach().cpu(),historical[k]) for k,v in expert.state_dict().items()),'Diagnostic reconstruction changed the frozen RAW endpoint'
        assert len(probes)==4
        c.write(done,dict(status='COMPLETE',expert_order=t['order'],endpoint_exact=True,steps=320,training_forwards=640,training_backwards=640,
            diagnostic_forwards=105,probes=probes,curve=curve,execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
        d.consume_resume(latest)
    finally:hook.detach()


def worker():
    part=int(os.environ['PARTITION']);gpu=GPUS[part]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in p.tasks()[:8][part::6]:trajectory(runtime,t)
    p.done('WORKER_'+str(part))


def plan():
    checks=mm.selfcheck()
    assert (Path(os.environ['MECHANISM_PARENT'])/'private/REPORT_COMPLETE.json').exists()
    for t in p.tasks()[:8]:
        assert (d.PARENT/'private/weights'/t['anonymous_edit']/'STAGED_220/FINAL.pt').is_file()
        assert (PILOT/'private/weights'/t['anonymous_edit']/'RAW_DIR_540/FINAL.pt').is_file()
    lock=dict(experts=8,steps=320,probe_steps=list(STEPS),directions=list(mm.MODES),GPUs=list(GPUS),optimizer_steps=2560,
        training_forwards=5120,training_backwards=5120,diagnostic_forwards=840,new_generations=0,new_Judge=0,snapshots=32,
        maximum_snapshot_bytes=16777216,roles=['native','FIT'],CHECK_training=False,independent_confirmation=False)
    c.write(RUN/'private/MECHANISM_LOCK.json',lock);c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**lock,selfcheck=checks));p.done('PLAN_COMPLETE')


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('mechanism.py','mechanism_worker',g,i) for i,g in enumerate(GPUS) if not (RUN/'private'/f'WORKER_{i}.json').exists()])
    pipeline.wait([pipeline.launch('mechanism_report.py','mechanism_report')])


if __name__=='__main__':
    try:{'mechanism_plan':plan,'mechanism_worker':worker,'mechanism_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
