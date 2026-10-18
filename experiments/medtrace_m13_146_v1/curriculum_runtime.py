"""Old source curriculum using the admitted current runtime and M13P math."""
import os
import time
import traceback
import torch
import runtime146 as w
import curriculum_protocol as f
from prefix_cache import PrefixCache
r,c,RUN=w.r,w.c,w.RUN

def gradients(runtime,hook,expert,batches,phase,step):
    expert.zero_grad(set_to_none=True);losses=[]
    for index,weight in f.selected(phase,step):
        hook.set_teacher_routing(batches[index].labels);loss=runtime.compute_loss(batches[index]);assert torch.isfinite(loss)
        (weight*loss).backward();r.COUNTS['backwards']+=1;losses.append(float(loss.detach()))
    return losses

def train(arm):
    from m3bench_repro.editors.llava_runtime import seed_everything
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU'])
    with r.p.lease(gpu):
        runtime,bindings,handle=r.load(gpu)
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()} if arm=='M13P' else None
        try:
            for n,task in enumerate(w.tasks()[part::2]):
                rows=w.trajectory.source_rows(w.packet(),task['order'])
                runtime.model.half();prepared=r.s.prepare(runtime,bindings,task,rows,True)
                r.old.fp32(runtime);batches=[r.old.batch_on(runtime,v[1]) for v in prepared]
                initial=r.p.tr.TT4(task['seed'],8,8).state_dict();expert=r.p.expert(initial,task['seed'],runtime.device)
                canonical=arm!='C';base=r.mm.canonicalize(initial)[0] if canonical else initial
                expert.load_state_dict(base);assert sum(v.numel() for v in base.values())==7168
                hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                assert not w.point(task,arm).exists() and not (RUN/'private/active'/f'{arm}_{task["order"]}.pt').exists()
                try:
                    refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                    targets=None if arm=='C' else c.read(RUN/'private/results/C'/f'{task["order"]}.json')['curve']
                    if targets is not None:assert len(targets)==540
                    coefficient=c.read(RUN/'private/PRIOR_CALIBRATIONS.json')[str(task['order'])]['coefficient'] if arm=='M13P' else 0.
                    start=dict(r.COUNTS);curve=[];total=0;scalar=None;opt=None
                    if arm=='C' and n==0:
                        rng=c.rng()
                        with c.diagnostic_scope(hook):
                            a=gradients(runtime,hook,expert,batches,'native',1);g={k:v.grad.clone() for k,v in expert.named_parameters()}
                            c.restore_rng(rng);b=gradients(runtime,hook,expert,batches,'native',1)
                            assert a==b and all(torch.equal(g[k],v.grad) for k,v in expert.named_parameters())
                        c.restore_rng(rng);c.write(RUN/'private'/f'CURRICULUM_PARITY_{part}.json',dict(status='PASS',epoch=time.time()))
                    with PrefixCache(runtime,batches) as prefix:
                        for phase,steps in f.PHASES:
                            seed_everything(task['seed'])
                            opt=None if canonical else (torch.optim.AdamW(expert.parameters(),lr=1e-3,weight_decay=0) if phase!='W0' else r.p.optimizer(expert,runtime.model))
                            scalar=dict(step=0,second_moments=[0.,0.]) if canonical else None
                            for step in range(1,steps+1):
                                c.budget();before=r.old.snapshot(expert);losses=gradients(runtime,hook,expert,batches,phase,step)
                                total+=1
                                if canonical:
                                    g={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                                    selected,scalar,audit=w.training_math.step(r,before,g,cache,coefficient,scalar,targets[total-1]['operator_step_norm'],arm)
                                    expert.load_state_dict(selected);assert scalar['step']==step
                                else:
                                    norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
                                    opt.step();assert all(torch.isfinite(v).all() for v in expert.parameters())
                                    audit=dict(gradient_norm=float(norm),operator_backwards=0)
                                r.COUNTS['updates']+=1;r.COUNTS['operator_backwards']+=audit['operator_backwards']
                                curve.append(dict(step=total,phase=phase,phase_step=step,source_NLL=losses,operator_step_norm=r.fm.map_norm(before,r.old.snapshot(expert)),**audit))
                                if step%20==0 or step==steps:
                                    r.save_resume(task,arm,expert,opt,scalar,total,curve);r.save_counts()
                        assert prefix.hits==935 and prefix.misses==5
                    assert total==540 and r.COUNTS['backwards']-start['backwards']==940+(2 if arm=='C' and n==0 else 0)
                    r.finish_candidate(runtime,hook,expert,task,arm,base,batches,refs,curve,start,scalar,mechanism=dict(coefficient=coefficient,recipe='native140_A2_80_W0_320',fisher_reference='original_Base',coefficient_source='frozen_160',stage_optimizer_resets=True))
                    r.retire_resume(task,arm)
                finally:hook.detach()
        finally:handle.remove();r.save_counts()

def main():
    f.admit(RUN);w.configure();r.NATIVE_FORWARD_CAP=250000
    original=c.budget
    def budget():
        original();f.project_budget(RUN)
        assert not list((RUN/'private').glob('FAILURE_curriculum*.json'))
    c.budget=budget
    action=os.environ['ACTION'];part=int(os.environ['PARTITION']);arm=os.environ['CURRICULUM_ARM'];assert arm in f.METHODS
    with (RUN/'private'/f'ONCE_{action}_{part}').open('x'):pass
    if action.startswith('curriculum_train_'):train(arm)
    else:
        import bank146
        # Reuse the original single-generation worker; method labels are unchanged inside this isolated run.
        os.environ['ACTION']='m13_146_single'
        bank146.evaluate(146,int(os.environ.get('GENERATION_PARTITIONS','2')),lambda:f.admit(RUN),f'FROZEN_{arm}.json','SCOPE_LOCK.json','GPU_SOURCE_VERSION.json',f'PLAN_{arm}.json')
        os.environ['ACTION']=action
    r.save_counts();c.write(RUN/'private'/f'COMPLETE_{action}_{part}.json',dict(status='COMPLETE',counts=r.COUNTS,epoch=time.time()))
if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private'/f'FAILURE_curriculum_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:r.save_counts();c.write(RUN/'private'/f'EXIT_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(exit_code=code,epoch=time.time()))
