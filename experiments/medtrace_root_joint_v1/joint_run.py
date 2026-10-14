"""Finite reconstruction to eight preregistered rejections, then root diagnostics."""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import numpy as np
import torch
import guard_probe as g
import joint_math as jm
import joint_geometry as jg

r,c,RUN=g.r,g.c,g.RUN
PARENT=Path(os.environ['GUARD_PARENT'])
FRACTIONS=(1.,.5,.25,.125)


def count(*_):
    c.budget()
    r.COUNTS['forwards']+=1
    r.COUNTS['non_generation_forwards']+=1
    # Per-partition reservations sum to the global 12000 ceiling.
    cap=128 if os.environ['ACTION']=='module_fisher' else (4452 if int(os.environ['PARTITION'])<2 else 2968)
    assert r.COUNTS['non_generation_forwards']<=cap, 'Frozen forward reservation exhausted'
r.count=count


def close(a,b,atol=1e-5,rtol=1e-4):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.all(np.abs(a-b)<=atol+rtol*np.abs(b)), 'Reconstructed trajectory drift'


def check_decision(actual, expected):
    assert actual['status']==expected['status'] and len(actual['checks'])==len(expected['checks'])
    close(actual['norm_ratio'],expected['norm_ratio'],atol=1e-8)
    for a,b in zip(actual['checks'],expected['checks']):
        assert a['fraction']==b['fraction'] and a['accepted']==b['accepted']
        assert a['projection']['status']==b['projection']['status']
        close([a['source_gain'],a['pure_source_gain']],[b['source_gain'],b['pure_source_gain']])
        close(a['projection']['stored_penalty'],b['projection']['stored_penalty'],atol=1e-9)


def cpu():
    from guard_admission import require_reference
    require_reference()
    rows=c.read(RUN/'private/TARGETS.json')['rows']
    assert len(rows)==8 and sum(x['step'] for x in rows)==341
    for item in rows:
        curve=c.read(PARENT/'private/prefix_curve'/f"{item['owner']}.json")
        first=next(row for row in curve if row['decision']['status']=='SKIPPED_INFEASIBLE'
                   and row['decision']['pure_source_gain']>0)
        assert first['step']==item['step']
    result=dict(status='PASS',solver=jm.selfcheck(),geometry=jg.selfcheck(),reference_owners=8,
                fixed_reconstruction_steps=341,eligible_states=8,new_model_calls=0)
    c.write(RUN/'public/CPU_ADMISSION.json',result)
    print(json.dumps(result),flush=True)


def measure(runtime,hook,expert,state,sources,protection,refs):
    return g.measure(runtime,hook,expert,state,sources,protection,refs)


def diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar_after,
                sources,protection,refs,cache,budget,losses,old_decision,before_rng,numpy_before,consumer=None):
    numpy_rng=np.random.get_state()
    payload=dict(before=r.s.previous.weights(before),pure=r.s.previous.weights(pure),
                 scalar_before=scalar_before,scalar_after=scalar_after,rng=before_rng,
                 numpy_rng=dict(kind=numpy_before[0],keys=numpy_before[1].tolist(),position=numpy_before[2],
                                has_gauss=numpy_before[3],cached_gaussian=numpy_before[4]),
                 seed=task['seed'],step=step,source_order=[x['query_id'] for x in r.s.source(task)],
                 budget=budget,scope_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),
                 target_lock=c.digest(c.read(RUN/'private/TARGETS.json')),old_decision=old_decision,
                 source_losses=losses,counts=dict(r.COUNTS))
    path=RUN/'private/probes'/f"{task['order']}.pt"
    assert not path.exists()
    c.save(path,payload)
    loaded=r.p.load_state(path)
    for kind in ('before','pure'):
        assert all(torch.equal(loaded[kind][k],payload[kind][k]) for k in r.mm.KEYS)
    for key in ('scalar_before','scalar_after','seed','step','source_order','budget','scope_lock','target_lock','numpy_rng'):
        assert loaded[key]==payload[key]
    assert torch.equal(loaded['rng']['torch_rng'],before_rng['torch_rng'])
    assert torch.equal(loaded['rng']['cuda_rng'],before_rng['cuda_rng'])
    assert loaded['rng']['python_rng']==before_rng['python_rng']
    chosen=g.source_batches(sources,step)
    pure_gain=old_decision['pure_source_gain'];assert pure_gain>0
    target=r.fm.map_norm(before,pure)
    required=.9*pure_gain-1e-6
    before_loss=g.mean_loss(losses)
    records=[]
    with c.diagnostic_scope(hook):
        base_metrics=measure(runtime,hook,expert,before,sources,protection,refs)
        pure_metrics=measure(runtime,hook,expert,pure,sources,protection,refs)
        for fraction in FRACTIONS:
            candidate=pure if fraction==1 else r.mm.canonicalize(r.q.add(before,fraction*(r.q.flatten(pure)-r.q.flatten(before))))[0]
            expert.load_state_dict(candidate)
            _,anchor_losses=r.source_gradients(runtime,hook,expert,sources,step,False)
            gradient=expert.G1.grad.detach().cpu().double().flatten()
            gain_at_proposal=before_loss-g.mean_loss(anchor_losses)
            transform,inverse,center,orthogonal=jg.coordinates(before,candidate)
            rows=g.gm.root_rows(candidate,cache)
            root=candidate['G1'].detach().cpu().double().flatten()
            value,audit=jm.solve((rows@inverse).numpy(),(transform@root).numpy(),center.numpy(),orthogonal,
                                target*(1+.002)+1e-10,(inverse@gradient).numpy(),gain_at_proposal,required,budget,target)
            rec=dict(fraction=fraction,solver=audit,anchor_gain=gain_at_proposal,pure_gain=pure_gain,
                     required_gain=required,operator_radius=target,orthogonal_squared=orthogonal,
                     root_gradient_norm=float(gradient.norm()),original_rule_pass=False,matched_step_pass=False)
            if value is not None:
                updated={k:v.clone() for k,v in candidate.items()}
                final_root=inverse@torch.from_numpy(value)
                updated['G1']=final_root.reshape_as(updated['G1']).to(updated['G1'])
                actual_losses=r.losses_at(runtime,hook,expert,updated,chosen)
                gain=g.gain(losses,actual_losses)
                penalty=float(r.mm.operator_penalty(updated,cache))
                norm=r.fm.map_norm(before,updated)
                predicted=gain_at_proposal-float(gradient@(updated['G1'].detach().cpu().double().flatten()-root))
                source_ok=gain>=required
                budget_ok=penalty<=budget+1e-9+1e-5*budget
                upper_ok=norm<=target*(1+.002)+1e-10
                passed=source_ok and budget_ok and upper_ok
                rec.update(actual_source_gain=gain,predicted_source_gain=predicted,linearization_error=gain-predicted,
                           Fisher_ratio=penalty/budget,step_ratio=norm/target,source_pass=source_ok,
                           budget_pass=budget_ok,step_upper_pass=upper_ok,original_rule_pass=passed,
                           matched_step_pass=passed and norm>=.9*target-1e-10,
                           metrics=measure(runtime,hook,expert,updated,sources,protection,refs))
                if consumer is not None:
                    consumer(rec, updated, candidate, transform, inverse, center, orthogonal)
            records.append(rec)
    np.random.set_state(numpy_rng)
    expert.load_state_dict(before);expert.zero_grad(set_to_none=True)
    c.write(RUN/'private/results_joint'/f"{task['order']}.json",dict(owner=task['order'],step=step,
            status='COMPLETE',save_load_exact=True,full_state_binding=True,
            source_clock_before=scalar_before['step'],source_clock_after=scalar_after['step'],
            pure_metrics=pure_metrics,before_and_original_guard_metrics=base_metrics,old_decision=old_decision,
            fractions=records,heldout_accessed=False,generation_attempts=0))


def probe():
    from m3bench_repro.editors.llava_runtime import seed_everything
    from guard_admission import require_reference
    require_reference()
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU'])
    tasks=r.s.tasks()[part::3]
    targets={x['owner']:x['step'] for x in c.read(RUN/'private/TARGETS.json')['rows']}
    with r.p.lease(gpu):
        runtime,_,handle=r.load(gpu)
        g.memory_record(gpu,'LOADED')
        basis,_=r.q.split();assert len(basis)==61
        frozen_sources={t['order']:[r.old.cpu(runtime.build_edit_batch(r.s.record(t,row))) for row in r.s.source(t)] for t in tasks}
        frozen_basis=[r.old.cpu(r.q.protection_batch(runtime,row,tasks[0])) for row in basis]
        r.old.fp32(runtime)
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        try:
            for task in tasks:
                owner=task['order'];stop=targets[owner]
                assert not (RUN/'private/results_joint'/f'{owner}.json').exists(), 'No duplicate owner'
                saved=r.p.load_state(r.s.point(task,'C'))['states']
                initial=r.mm.canonicalize({k:v.to(runtime.device) for k,v in saved['BASE'].items()})[0]
                expert=r.p.expert(initial,task['seed'],runtime.device)
                hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                sources=[r.old.batch_on(runtime,x) for x in frozen_sources[owner]]
                protection=[r.old.batch_on(runtime,x) for x in frozen_basis]
                reference=c.read(RUN/'private/results/C'/f'{owner}.json')
                calibration=c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f'{owner}.json')
                penalty=float(r.mm.operator_penalty({k:v.to(runtime.device) for k,v in saved['FINAL'].items()},cache))
                close(penalty,calibration['C_operator_penalty'],atol=1e-12)
                c.write(RUN/'private/C_FISHER_PARITY'/f'{owner}.json',dict(status='PASS',actual=penalty,expected=calibration['C_operator_penalty']))
                budget=.1*calibration['C_operator_penalty']
                historical=c.read(PARENT/'private/prefix_curve'/f'{owner}.json')
                try:
                    refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(sources+protection)]
                    seed_everything(task['seed']);expert.load_state_dict(initial)
                    scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                    for step in range(1,stop+1):
                        before=r.old.snapshot(expert);scalar_before=copy.deepcopy(scalar)
                        before_rng=c.rng();numpy_before=np.random.get_state()
                        _,losses=r.source_gradients(runtime,hook,expert,sources,step,False)
                        close(losses,historical[step-1]['source_NLL'])
                        torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                        pure,scalar=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in expert.named_parameters()},scalar)
                        pure=g.canonical_match(before,pure,reference['curve'][step-1]['operator_step_norm'])
                        selected,audit=g.select_guard(runtime,hook,expert,before,pure,g.source_batches(sources,step),losses,cache,budget)
                        check_decision(audit,historical[step-1]['decision'])
                        r.COUNTS['updates']+=1
                        expert.load_state_dict(selected)
                        curve.append(dict(step=step,source_NLL=losses,decision=audit))
                        if step%20==0 or step==stop:
                            c.save(RUN/'private/active'/f'{owner}.pt',dict(state=r.s.previous.weights(selected),scalar=scalar,
                                   rng=c.rng(),step=step,seed=task['seed'],curve=curve,counts=r.COUNTS))
                            c.write(RUN/'private/reconstruction'/f'{owner}.json',dict(status='PREFIX_PARITY_PASS',steps=step,curve=curve))
                            r.save_counts();g.memory_record(gpu,'RECONSTRUCTION_CHECKPOINT')
                        if step==stop:
                            assert audit['status']=='SKIPPED_INFEASIBLE' and scalar_before['step']==step-1
                            diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar,sources,
                                        protection,refs,cache,budget,losses,audit,before_rng,numpy_before)
                            r.save_counts();print('FIXED_STATE_COMPLETE',owner,flush=True)
                finally:hook.detach()
            g.memory_record(gpu,'ALL_FIXED_STATES_COMPLETE',complete=True)
        finally:handle.remove();r.save_counts()


def report():
    results=[c.read(p) for p in sorted((RUN/'private/results_joint').glob('*.json'))]
    assert len(results)==8 and {x['owner'] for x in results}==set(range(1,9))
    trials=[t for row in results for t in row['fractions']]
    assert len(trials)==32 and all(tuple(t['fraction'] for t in row['fractions'])==FRACTIONS for row in results)
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(x[k] for x in counts) for k in r.COUNTS}
    assert totals['updates']==341 and totals['backwards']==3071 and totals['non_generation_forwards']<=12000
    assert totals['generation_attempts']==0 and totals['operator_backwards']==0
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==4 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    def summary(values):
        return dict(n=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values)) if values else dict(n=0)
    witnesses=sum(any(t['original_rule_pass'] for t in row['fractions']) for row in results)
    matched=sum(any(t['matched_step_pass'] for t in row['fractions']) for row in results)
    out=dict(status='COMPLETE',decision='SEARCH_MISS_WITNESS_FOUND' if witnesses else 'NO_VERIFIED_SEARCH_MISS_WITNESS',
             fixed_states=8,local_problems=32,original_rule_witness_states=witnesses,matched_step_witness_states=matched,
             original_rule_witness_trials=sum(t['original_rule_pass'] for t in trials),
             matched_step_witness_trials=sum(t['matched_step_pass'] for t in trials),
             solver_statuses={s:sum(t['solver']['status']==s for t in trials) for s in sorted({t['solver']['status'] for t in trials})},
             finite_check_failures={k:sum(not t[k] for t in trials if k in t) for k in ['source_pass','budget_pass','step_upper_pass']},
             distributions={k:summary([t[k] for t in trials if k in t]) for k in
                            ['actual_source_gain','predicted_source_gain','linearization_error','Fisher_ratio','step_ratio']},
             per_fraction=[dict(fraction=f,solver_KKT=sum(t['fraction']==f and t['solver']['status']=='LOCAL_KKT_PASS' for t in trials),
                                original_pass=sum(t['fraction']==f and t['original_rule_pass'] for t in trials),
                                matched_pass=sum(t['fraction']==f and t['matched_step_pass'] for t in trials)) for f in FRACTIONS],
             counts=totals,resources=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
                                          cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=0,
                                          cumulative_Judge=ledger['Judge_attempts']),
             reconstruction_parity_owners=8,C_Fisher_parity_owners=8,save_load_owners=8,independent_confirmation=False,
             lower_norm_bound_in_solver=False,no_generation_or_heldout=True,next_consumer='NONE')
    assert ledger['Judge_attempts']==inherited['Judge_attempts']
    for key in ('source_gain','basis_mean_KL','basis_lost_correct_tokens'):
        out.setdefault('teacher_forced_metrics',{})[key]=dict(
            before=summary([row['before_and_original_guard_metrics'][key] for row in results]),
            pure=summary([row['pure_metrics'][key] for row in results]),
            joint_all_locally_feasible=summary([t['metrics'][key] for t in trials if 'metrics' in t]))
    c.write(RUN/'public/RESULTS.json',out)
    text=f'''# 同根空间联合可行性：终结报告

结论：{out['decision']}。八个固定首拒绝状态、32个局部问题完成。
原规则真实通过见证：{witnesses}/8状态、{out['original_rule_witness_trials']}/32候选；另加90%步幅下界通过：{matched}/8状态、{out['matched_step_witness_trials']}/32候选。
局部求解状态：{json.dumps(out['solver_statuses'],ensure_ascii=False)}。
真实有限检查失败计数：{json.dumps(out['finite_check_failures'],ensure_ascii=False)}。

完整重建341提案，3071 NN反传，{totals['non_generation_forwards']}非生成前向，0生成/Judge。八例历史前缀、C惩罚、完整保存加载绑定通过。所有四个GPU租约结束。
新增{out['resources']['new_GPU_process_hours']:.9f} GPU进程小时，累计{out['resources']['cumulative_GPU_process_hours']:.9f}小时；Judge累计{ledger['Judge_attempts']}，保留历史失败成本。

见证只能证明存在部分受限搜索漏解；没有见证不证明真实模型或全TT不可行。局部证书只针对固定尾、预算、线性化源约束和步幅上界；下界仅验收，未通过不构成无解证明。全部DEV/SOURCE/BASIS，无T2G/自由生成或医学安全结论，无自动完整训练消费者。
每个比例和所有局部/真实检查汇总见RESULTS.json；逐owner、原题、tokens、参数和私有路径不公开。
'''
    (RUN/'public/REPORT_ZH.md').write_text(text)
    paths=[RUN/'private/OPERATOR_FISHER.pt']+list((RUN/'private/probes').glob('*.pt'))+list((RUN/'private/active').glob('*.pt'))
    assert len(paths)==17 and all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    receipt=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),copies=False,
                 reason='All fixed diagnostics, parity, save-load and report consumers complete',reconstruction_requires_training=True)
    c.write(RUN/'private/CONSUMED.json',dict(receipt,paths=[str(p) for p in paths]))
    for p in paths:p.unlink()
    c.write(RUN/'public/EXECUTION_AUDIT.json',dict(status='PASS',all_GPU_leases_ended=True,counts=totals,lifecycle=receipt,
                                                owned_pt_remaining=0,full_validation_started=False))
    c.write(RUN/'private/REPORT_COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status=out['decision'],complete=True,publication_complete=False))
    print(json.dumps({k:out[k] for k in ['decision','original_rule_witness_states','matched_step_witness_states','resources']}),flush=True)


def stage(action,parts,file='joint_run.py'):
    import pipeline
    waiting=list(parts);jobs=[];busy=set()
    expected=c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs']
    while waiting or any(job.poll() is None for job,gpu in jobs):
        c.budget()
        assert all(job.poll() in (None,0) for job,gpu in jobs), 'Child failure; no automatic retry'
        busy={gpu for job,gpu in jobs if job.poll() is None}
        for gpu in [0,1,4,7,2,3]:
            if not waiting or len(busy)>=3:break
            if gpu in busy:continue
            uuid,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free',
                                               '--format=csv,noheader,nounits'],text=True).strip().split(', ')
            assert uuid==expected[str(gpu)]
            if int(free)<45*1024:continue
            part=waiting.pop(0)
            assert not (RUN/'private'/f'START_CHAIN_{action}_{part}.json').exists(), 'No duplicate dispatch'
            jobs.append((pipeline.launch(file,action,gpu,part),gpu));busy.add(gpu)
        if waiting:c.write(RUN/'public/RESOURCE_WAIT.json',dict(action=action,waiting=len(waiting),required_free_MiB=45*1024))
        if waiting or any(job.poll() is None for job,gpu in jobs):time.sleep(10)
    assert all(job.returncode==0 for job,gpu in jobs)


def controller():
    from guard_admission import require_reference
    require_reference()
    assert c.read(RUN/'public/CPU_ADMISSION.json')['status']=='PASS'
    with (RUN/'private/ONCE.json').open('x') as handle:json.dump(dict(pid=os.getpid(),epoch=time.time()),handle)
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_REBUILD',complete=False))
    stage('module_fisher',[0])
    c.write(RUN/'public/PROGRESS.json',dict(status='FIXED_PREFIX_AND_JOINT_DIAGNOSTICS',complete=False))
    stage('joint_probe',range(3))
    report()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='joint_cpu':cpu()
        elif action=='module_fisher':r.GPUS=(int(os.environ['GPU']),);r.fisher()
        elif action=='joint_probe':probe()
        elif action=='joint_controller':controller()
        else:raise ValueError('Unregistered action')
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
                dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        c.write(RUN/'public/PROGRESS.json',dict(status='STOPPED_RETAINED',complete=False))
        raise
