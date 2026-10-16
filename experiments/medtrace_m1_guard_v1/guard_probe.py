"""Fixed eight-owner prefix experiment and counterfactual update diagnostics."""
import copy
import math
import os
import subprocess
from pathlib import Path
import time
import traceback
os.environ['M3BENCH_LLAVA_SOURCE'] = str(Path(os.environ['RUN_ROOT'])/'private/official_llava')
import torch
import module_run as r
import guard_math as gm
from module_recovery import admitted_stage
c, RUN = r.c, r.RUN
PARENT = Path(os.environ['MODULE_PARENT'])
NODES = (20, 80)
r.GPUS = (2, 4, 7)


def count(*_):
    c.budget()
    r.COUNTS['forwards'] += 1
    r.COUNTS['non_generation_forwards'] += 1
    own = f"COUNTS_{os.environ['ACTION']}_{os.environ.get('PARTITION','0')}.json"
    total = r.COUNTS['non_generation_forwards'] + sum(c.read(p)['non_generation_forwards'] for p in (RUN/'private').glob('COUNTS_*.json') if p.name != own)
    assert total <= 24000, 'Frozen phase-A forward ceiling'
r.count = count


def memory_record(gpu, stage, complete=False, prefix='guard_probe'):
    path=RUN/'private'/f"MEMORY_{prefix}_{os.environ['PARTITION']}.json"
    previous=c.read(path) if path.exists() else {}
    reserved=torch.cuda.memory_reserved()
    try:
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader,nounits'],text=True).splitlines()
        own=[int(row.split(',')[1].strip())*1024**2 for row in rows if row.split(',')[0].strip()==str(os.getpid())]
    except (subprocess.SubprocessError,OSError,ValueError):
        own=[]
    c.write(path,dict(pid=os.getpid(),gpu=gpu,stage=stage,epoch=time.time(),complete=complete,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        sampled_process_peak_bytes=max([previous.get('sampled_process_peak_bytes',0)]+own),
        sampled_nonallocator_max_bytes=max([previous.get('sampled_nonallocator_max_bytes',0),0]+[v-reserved for v in own]),
        process_sample_valid=len(own)==1,
        peak_counters_reset=False,includes_loading=True,process_peak_continuous=False))


def canonical_match(before, proposed, target):
    proposed, _ = r.match_proposal(before, proposed, target)
    proposed, _ = r.mm.canonicalize(proposed)
    assert abs(r.fm.map_norm(before, proposed)-target) <= 1e-10 + .002*target
    return proposed


SOURCE_WEIGHTS=(.25,.25,.5)


def mean_loss(losses):
    assert len(losses)==len(SOURCE_WEIGHTS)
    return sum(w*x for w,x in zip(SOURCE_WEIGHTS,losses))


def gain(losses, after):
    assert len(losses)==len(after)==len(SOURCE_WEIGHTS)
    return sum(w*(a-b) for w,a,b in zip(SOURCE_WEIGHTS, losses, after))


def select_guard(runtime, hook, expert, before, pure, chosen, before_losses, cache, budget):
    pure_losses = r.losses_at(runtime, hook, expert, pure, chosen)
    pure_gain = gain(before_losses, pure_losses)
    limit = r.fm.map_norm(before, pure)
    traces = []
    displacement = r.q.flatten(pure)-r.q.flatten(before)
    for fraction in (1., .5, .25, .125):
        candidate = pure if fraction == 1 else r.mm.canonicalize(r.q.add(before, fraction*displacement))[0]
        candidate, projection = gm.project(candidate, cache, budget)
        norm = r.fm.map_norm(before, candidate)
        values = pure_losses if projection['status']=='UNCHANGED' and fraction==1 else r.losses_at(runtime,hook,expert,candidate,chosen)
        progress = gain(before_losses, values)
        progress_ok = progress >= (.9*pure_gain-1e-6 if pure_gain>0 else pure_gain-1e-6)
        accepted = norm <= limit*(1+.002)+1e-10 and progress_ok
        traces.append(dict(fraction=fraction, projection=projection, operator_norm=norm,
                           pure_norm=limit, source_gain=progress, pure_source_gain=pure_gain, accepted=accepted))
        if accepted:
            return candidate, dict(status='ACCEPTED', checks=traces, source_gain=progress,
                                   pure_source_gain=pure_gain, norm_ratio=norm/max(limit,1e-30))
    return before, dict(status='SKIPPED_INFEASIBLE', checks=traces, source_gain=0.,
                        pure_source_gain=pure_gain, norm_ratio=0.)


def source_batches(batches, step):
    if len(batches)==5:
        assert SOURCE_WEIGHTS==(.5,.5)
        return [batches[0],batches[1+(step-1)%4]]
    assert len(batches)==7 and SOURCE_WEIGHTS==(.25,.25,.5)
    return [batches[0], batches[1+(step-1)%4], batches[5+(step-1)%2]]


def measure(runtime, hook, expert, state, sources, protection, refs):
    expert.load_state_dict(state)
    values = []
    for batch, (base, target) in zip(sources+protection, refs):
        logits, actual = r.q.logits(runtime,hook,batch)
        assert torch.equal(target,actual)
        values.append(r.q.compare(base,logits,target))
    source = values[:7]; basis = values[7:]
    weights = (.25,.0625,.0625,.0625,.0625,.25,.25)
    return dict(source_gain=-sum(w*v['NLL_change'] for w,v in zip(weights,source)),
                source_values=source, basis_mean_KL=sum(v['KL'] for v in basis)/61,
                basis_lost_correct_tokens=sum(v['lost_correct_tokens'] for v in basis),
                basis_all_argmax_correct=sum(v['all_argmax_correct'] for v in basis), basis_values=basis)


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(a@b)/denominator if denominator>0 else None


def diagnostics(runtime, hook, expert, task, node, before, scalar, sources, protection, refs, cache, reference, budget, coefficient):
    # All four proposals use this same state and source-only history, not old M13's path.
    expert.load_state_dict(before)
    _, losses = r.source_gradients(runtime,hook,expert,sources,node+1,False)
    raw_source={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
    torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
    gs={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
    expert.zero_grad(set_to_none=True)
    penalty=r.mm.operator_penalty(dict(expert.named_parameters()),cache)
    (coefficient*penalty).backward();r.COUNTS['operator_backwards']+=1
    gp={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
    mixed={k:raw_source[k]+gp[k] for k in gs}
    norm=torch.sqrt(sum(v.square().sum() for v in mixed.values()))
    mixed={k:v*min(1.,1./(float(norm)+1e-6)) for k,v in mixed.items()}
    target=reference['curve'][node]['operator_step_norm']
    pure, clock=r.mm.canonical_rms_step(before,gs,copy.deepcopy(scalar))
    pure=canonical_match(before,pure,target)
    mix, _=r.mm.canonical_rms_step(before,mixed,copy.deepcopy(scalar))
    mix=canonical_match(before,mix,target)
    separated=canonical_match(before,gm.source_moments_mixed_step(before,mixed,clock),target)
    guarded,audit=select_guard(runtime,hook,expert,before,pure,source_batches(sources,node+1),losses,cache,budget)
    proposals=dict(M1=pure,MIXED_M13=mix,SOURCE_RMS_MIXED=separated,GUARD=guarded)
    metrics={name:measure(runtime,hook,expert,state,sources,protection,refs) for name,state in proposals.items()}
    src=r.q.flatten(raw_source);protect=r.q.flatten(gp);delta={k:r.q.flatten(v)-r.q.flatten(before) for k,v in proposals.items()}
    maps={name:r.fm.map_delta(before,state) for name,state in proposals.items()}
    def operator_cosine(name):
        a,b=maps['M1'];x,y=maps[name]
        denominator=r.fm.map_norm(before,pure)*r.fm.map_norm(before,proposals[name])
        return r.fm.lowrank_inner(a,b,x,y)/denominator if denominator>0 else None
    data=dict(owner=task['order'],node=node,shared_state='GUARD_PREFIX',source_clock=scalar['step'],
              source_protection_gradient_cosine=cosine(src,protect),protection_source_gradient_ratio=float(protect.norm()/src.norm().clamp_min(1e-30)),
              direction_cosines={name:cosine(delta['M1'],d) for name,d in delta.items()},
              operator_direction_cosines={name:operator_cosine(name) for name in proposals},
              M13_state_vs_source_state_cosine=cosine(delta['MIXED_M13'],delta['SOURCE_RMS_MIXED']),
              guard=audit,metrics=metrics,heldout_accessed=False,teacher_forcing_only=True)
    c.write(RUN/'private/diagnostics'/f"{task['order']}_{node}.json",data)
    expert.load_state_dict(before);expert.zero_grad(set_to_none=True)


def probe():
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);tasks=r.s.tasks()[part::3]
    assert not any((RUN/'private/PREFIX_COMPLETE'/f"{t['order']}.json").exists() for t in tasks)
    from m3bench_repro.editors.llava_runtime import seed_everything
    with r.p.lease(gpu):
        policy=RUN/'private/MEMORY_ADMISSION.json'
        if policy.exists():
            admission=c.read(policy)
            assert admission['status']=='PASS' and admission['required_free_MiB']==39*1024
            assert gpu in c.read(RUN/'RUN_MANIFEST.json')['physical_GPUs']
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            assert free>=admission['required_free_MiB'], 'Measured memory admission reserve unavailable'
            runtime,_=c.load(gpu);handle=runtime.model.register_forward_pre_hook(count)
        else:
            runtime,_,handle=r.load(gpu)
        memory_record(gpu,'LOADED')
        basis,_=r.q.split();assert len(basis)==61
        frozen_sources={t['order']:[r.old.cpu(runtime.build_edit_batch(r.s.record(t,row))) for row in r.s.source(t)] for t in tasks}
        frozen_basis=[r.old.cpu(r.q.protection_batch(runtime,row,tasks[0])) for row in basis]
        r.old.fp32(runtime)
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        memory_record(gpu,'FP32_AND_FISHER_READY')
        try:
            for task in tasks:
                saved=r.p.load_state(r.s.point(task,'C'))['states'];initial=r.mm.canonicalize({k:v.to(runtime.device) for k,v in saved['BASE'].items()})[0]
                expert=r.p.expert(initial,task['seed'],runtime.device)
                hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                sources=[r.old.batch_on(runtime,v) for v in frozen_sources[task['order']]]
                protection=[r.old.batch_on(runtime,v) for v in frozen_basis]
                reference=c.read(RUN/'private/results/C'/f"{task['order']}.json")
                calibration=c.read(PARENT/'private/calibration'/f"{task['order']}.json")
                penalty=float(r.mm.operator_penalty({k:v.to(runtime.device) for k,v in saved['FINAL'].items()},cache))
                assert abs(penalty-calibration['C_operator_penalty']) <= 1e-12+1e-4*abs(calibration['C_operator_penalty']), 'Frozen C Fisher calibration drift'
                c.write(RUN/'private/C_FISHER_PARITY'/f"{task['order']}.json",dict(status='PASS',actual=penalty,expected=calibration['C_operator_penalty']))
                budget=.1*calibration['C_operator_penalty'];coefficient=calibration['coefficient']
                try:
                    refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(sources+protection)]
                    for arm in ('M1REF','GUARD'):
                        seed_everything(task['seed']);expert.load_state_dict(initial)
                        scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                        for step in range(1,81):
                            before=r.old.snapshot(expert)
                            _,losses=r.source_gradients(runtime,hook,expert,sources,step,False)
                            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                            assert torch.isfinite(norm)
                            pure,scalar=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in expert.named_parameters()},scalar)
                            pure=canonical_match(before,pure,reference['curve'][step-1]['operator_step_norm'])
                            if arm=='GUARD':
                                chosen,audit=select_guard(runtime,hook,expert,before,pure,source_batches(sources,step),losses,cache,budget)
                            else:
                                chosen,audit=pure,dict(status='M1_REFERENCE',norm_ratio=1.)
                                prior=c.read(PARENT/'private/results/M1'/f"{task['order']}.json")['curve'][step-1]['source_NLL']
                                assert all(abs(a-b)<=1e-5+1e-4*abs(b) for a,b in zip(losses,prior)), 'Original M1 prefix source trajectory drift'
                            expert.load_state_dict(chosen);r.COUNTS['updates']+=1
                            curve.append(dict(step=step,source_NLL=losses,decision=audit,operator_step_norm=r.fm.map_norm(before,chosen)))
                            if step in NODES:
                                with c.diagnostic_scope(hook):
                                    measured=measure(runtime,hook,expert,chosen,sources,protection,refs)
                                    c.write(RUN/'private/nodes'/f"{arm}_{task['order']}_{step}.json",dict(owner=task['order'],node=step,arm=arm,metrics=measured))
                                    if arm=='GUARD':diagnostics(runtime,hook,expert,task,step,chosen,scalar,sources,protection,refs,cache,reference,budget,coefficient)
                                expert.load_state_dict(chosen);expert.zero_grad(set_to_none=True)
                            if step%20==0:
                                r.save_resume(task,arm,expert,None,scalar,step,curve);r.save_counts()
                                memory_record(gpu,'TRAIN_AND_NODE_CHECKPOINT')
                        if arm=='GUARD':
                            c.write(RUN/'private/prefix_curve'/f"{task['order']}.json",curve)
                            c.save(RUN/'private/prefix/GUARD'/f"{task['order']}.pt",dict(completed_step=80,states=dict(BASE=r.s.previous.weights(initial),FINAL=r.s.previous.weights(expert.state_dict())),
                                scalar_state=scalar,rng=c.rng(),seed=task['seed'],curve=curve,lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),
                                protection_budget=budget,source_order=[row['query_id'] for row in r.s.source(task)]))
                        r.retire_resume(task,arm)
                    c.write(RUN/'private/PREFIX_COMPLETE'/f"{task['order']}.json",dict(status='COMPLETE',owner=task['order'],steps_each=80,arms=['M1REF','GUARD']))
                finally:hook.detach()
            memory_record(gpu,'ALL_ASSIGNED_PREFIXES_COMPLETE',complete=True)
        finally:handle.remove();r.save_counts()


def controller():
    from guard_admission import require_reference
    require_reference()
    assert not (RUN/'private/GUARD_START.json').exists()
    c.write(RUN/'private/GUARD_START.json',dict(epoch=time.time(),pid=os.getpid()))
    run_stages()


def run_stages():
    import pipeline
    c.write(RUN/'public/PROGRESS.json',dict(status='PHASE_A_FISHER_REBUILD',complete=False))
    admitted_stage('guard_probe.py','module_fisher',[0])
    c.write(RUN/'public/PROGRESS.json',dict(status='PHASE_A_FIXED_PREFIX_VALIDATION',complete=False))
    admitted_stage('guard_probe.py','guard_probe',range(3))
    pipeline.wait([pipeline.launch('guard_report.py','guard_report')])


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='guard_cpu':
            from guard_admission import require_reference
            require_reference()
            print(gm.selfcheck())
        elif action=='module_fisher':
            r.GPUS=(int(os.environ['GPU']),4,7);r.fisher()
        else:
            {'guard_controller':controller,'guard_probe':probe}[action]()
    except BaseException as error:
        c.write(RUN/'private'/f"FAILURE_guard_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time()))
        c.write(RUN/'public/PROGRESS.json',dict(status='STOPPED_RETAINED',complete=False));raise
