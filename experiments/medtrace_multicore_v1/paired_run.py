"""One paired frozen-state diagnostic, then a gated two-arm full consumer."""
import copy
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import full_run as f
import correction_math as cm

j,g,r,c,RUN = f.j,f.g,f.r,f.c,f.RUN
ARMS = ('ROOT_FIX','MULTI')
r.mm.ARMS = ('C',)+ARMS
r.GPUS = (0,1,7)
r.COUNTS['local_solves']=0
_root_solver,_trust_solver = j.jm.solve,f.rm.solve


def root_solver(*args,**kwargs):
    solve_count()
    args=list(args)
    if args:args[4]-=cm.INNER_MARGIN*args[-1]
    else:kwargs['radius']-=cm.INNER_MARGIN*kwargs['scale']
    return _root_solver(*args,**kwargs)


def trust_solver(*args,**kwargs):
    solve_count()
    args=list(args)
    if args:args[4]-=cm.INNER_MARGIN*args[-2]
    else:kwargs['radius']-=cm.INNER_MARGIN*kwargs['scale']
    return _trust_solver(*args,**kwargs)

j.jm.solve=root_solver
f.rm.solve=trust_solver


def solve_count():
    r.COUNTS['local_solves']+=1
    if os.environ['ACTION']=='paired_probe':assert r.COUNTS['local_solves']<=96
    if os.environ['ACTION']=='paired_train':assert r.COUNTS['local_solves']<=17000


def count(*_):
    c.budget();r.COUNTS['forwards']+=1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards']+=1
        limits=dict(module_fisher=128,paired_probe=15000,paired_train=160000,
                    paired_evaluate=200,paired_route=1024)
        assert r.COUNTS['non_generation_forwards']<=limits[os.environ['ACTION']]
r.count=count


def cache_cpu(cache):
    return {k:v.detach().cpu().double() if isinstance(v,torch.Tensor) else v for k,v in cache.items()}


def trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm,fraction):
    radius=r.fm.map_norm(before,pure)
    chosen=g.source_batches(sources,step)
    pure_gain=g.gain(losses,r.losses_at(runtime,hook,expert,pure,chosen))
    before_loss=sum(w*x for w,x in zip((.25,.25,.5),losses))
    proposal=pure if fraction==1 else r.mm.canonicalize(r.q.add(before,
        fraction*(r.q.flatten(pure)-r.q.flatten(before))))[0]
    current=proposal;current_gain=None;attempts=[]
    for iteration in range(4):
        expert.load_state_dict(current)
        _,anchor_losses=r.source_gradients(runtime,hook,expert,sources,step,False)
        gradient={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
        anchor_gain=before_loss-sum(w*x for w,x in zip((.25,.25,.5),anchor_losses))
        if current_gain is not None:j.close(anchor_gain,current_gain)
        rho=None if iteration==0 else (.25,.125,.0625)[iteration-1]*radius
        required=.9*pure_gain-(1e-6 if iteration==0 else 0.)
        if arm=='ROOT_FIX':
            transform,inverse,center,orthogonal=j.jg.coordinates(before,proposal)
            rows=g.gm.root_rows(proposal,cache)@inverse
            root=proposal['G1'].detach().cpu().double().flatten()
            grad=gradient['G1'].detach().cpu().double().flatten()
            if iteration==0:
                value,audit=j.jm.solve(rows.numpy(),(transform@root).numpy(),center.numpy(),orthogonal,
                    radius*1.002+1e-10,(inverse@grad).numpy(),anchor_gain,required,budget,radius)
            else:
                anchor=current['G1'].detach().cpu().double().flatten()
                value,audit=f.rm.solve(rows.numpy(),(transform@root).numpy(),center.numpy(),orthogonal,
                    radius*1.002+1e-10,(inverse@grad).numpy(),(transform@anchor).numpy(),
                    anchor_gain,required,budget,radius,rho)
            candidate=None if value is None else {k:v.clone() for k,v in proposal.items()}
            if candidate is not None:candidate['G1']=(inverse@torch.from_numpy(value)).reshape_as(candidate['G1']).to(candidate['G1'])
            space=dict(root_dimensions=root.numel(),tail_directions=0)
        else:
            params=dict(expert.named_parameters())
            protection=torch.autograd.grad(r.mm.operator_penalty(params,cache),tuple(params.values()))
            r.COUNTS['operator_backwards']+=1
            geometry,build,space=cm.tangent(current,proposal,before,gradient,
                dict(zip(params,protection)),cpu_cache,radius,budget)
            assert space['same_TT_parameter_count']==7168 and space['root_dimensions']==512
            solve_count()
            value,audit=cm.solve(**geometry,radius=1.002-cm.INNER_MARGIN+1e-10/radius,
                                required=required-anchor_gain,trust=None if rho is None else rho/radius)
            candidate=None if value is None else build(value)
        entry=dict(iteration=iteration,solver=audit,space=space,passed=False)
        attempts.append(entry)
        if candidate is None:break
        after=r.losses_at(runtime,hook,expert,candidate,chosen)
        gain=g.gain(losses,after);penalty=float(r.mm.operator_penalty(candidate,cache))
        norm=r.fm.map_norm(before,candidate)
        trust_ok=rho is None or r.fm.map_norm(current,candidate)<=rho*(1+1e-5)+1e-10
        passed=f.finite(gain,pure_gain,penalty,budget,norm,radius) and trust_ok
        entry.update(passed=passed,source_gain=gain,pure_source_gain=pure_gain,
            Fisher_ratio=penalty/budget,norm_ratio=norm/radius,trust_pass=trust_ok,
            matched_step_pass=passed and norm>=.9*radius-1e-10)
        if passed:return candidate,dict(fraction=fraction,original_rule_pass=True,attempts=attempts)
        if iteration==0 or (trust_ok and penalty<=budget+1e-9+1e-5*budget
            and norm<=radius*1.002+1e-10 and gain>=anchor_gain):
            current=candidate;current_gain=gain
    return None,dict(fraction=fraction,original_rule_pass=False,attempts=attempts)


def correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm):
    guarded,original=g.select_guard(runtime,hook,expert,before,pure,g.source_batches(sources,step),losses,cache,budget)
    if original['status']=='ACCEPTED':
        return guarded,dict(kind='ORIGINAL_GUARD',original=original)
    if original['pure_source_gain']<=0:return before,dict(kind='SKIPPED_NONPOSITIVE',original=original)
    records=[]
    with c.diagnostic_scope(hook):
        for fraction in j.FRACTIONS:
            candidate,record=trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm,fraction)
            records.append(record)
            if candidate is not None:return candidate,dict(kind=arm+'_ACCEPTED',original=original,trials=records)
        if arm=='MULTI':
            # Root fallback keeps the paired arm's additional search from losing a root witness.
            selected,baseline=correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,'ROOT_FIX')
            return selected,dict(kind='ROOT_FALLBACK',original=original,trials=records,baseline=baseline)
    return before,dict(kind='SKIPPED_INFEASIBLE',original=original,trials=records)


def diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar_after,
                sources,protection,refs,cache,budget,losses,old_decision,before_rng,numpy_before,consumer=None):
    owner=task['order'];cpu_cache=cache_cpu(cache)
    payload=dict(before=r.s.previous.weights(before),pure=r.s.previous.weights(pure),
        scalar_before=scalar_before,scalar_after=scalar_after,rng=before_rng,old_decision=old_decision,
        numpy_rng=dict(kind=numpy_before[0],keys=numpy_before[1].tolist(),position=numpy_before[2],
                       has_gauss=numpy_before[3],cached_gaussian=numpy_before[4]),seed=task['seed'],step=step,
        source_order=[x['query_id'] for x in r.s.source(task)],budget=budget,source_losses=losses,
        scope_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),target_lock=c.digest(c.read(RUN/'private/TARGETS.json')))
    path=RUN/'private/probes'/f'{owner}.pt';assert not path.exists();c.save(path,payload)
    saved=r.p.load_state(path)
    assert all(torch.equal(saved[k][p],payload[k][p]) for k in ('before','pure') for p in r.mm.KEYS)
    assert saved['scalar_after']==scalar_after and saved['scalar_before']==scalar_before
    assert all(torch.equal(saved['rng'][k],before_rng[k]) for k in ('torch_rng','cuda_rng'))
    assert saved['rng']['python_rng']==before_rng['python_rng']
    output={};numpy_state=np.random.get_state()
    try:
        with c.diagnostic_scope(hook):
            for arm in ARMS:
                rows=[];selected=None
                for fraction in j.FRACTIONS:
                    state,record=trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm,fraction)
                    if state is not None:
                        p=RUN/'private/diagnostic_states'/f'{arm}_{owner}_{fraction}.pt'
                        c.save(p,dict(state=r.s.previous.weights(state),seed=task['seed'],step=step,fraction=fraction))
                        loaded=r.p.load_state(p);assert all(torch.equal(loaded['state'][k],state[k].cpu()) for k in r.mm.KEYS)
                        record['metrics']=g.measure(runtime,hook,expert,state,sources,protection,refs)
                        if selected is None:selected=dict(path=str(p),fraction=fraction,kind=arm+'_WITNESS')
                    rows.append(record)
                output[arm]=dict(trials=rows,selected=selected)
            if output['MULTI']['selected'] is None and output['ROOT_FIX']['selected'] is not None:
                output['MULTI']['selected']=dict(output['ROOT_FIX']['selected'],kind='ROOT_FALLBACK_WITNESS')
    finally:
        np.random.set_state(numpy_state);expert.load_state_dict(before);expert.zero_grad(set_to_none=True)
    c.write(RUN/'private/paired'/f'{owner}.json',dict(owner=owner,step=step,arms=output,
            save_load_exact=True,state_rng_RMS_fixed=True,heldout_accessed=False))
j.diagnostics=diagnostics


def mechanism_report():
    rows=[c.read(RUN/'private/paired'/f'{i}.json') for i in range(1,9)]
    assert all(row['save_load_exact'] and row['state_rng_RMS_fixed'] for row in rows)
    assert all(tuple(t['fraction'] for t in row['arms'][arm]['trials'])==j.FRACTIONS for row in rows for arm in ARMS)
    added=sum(not a['original_rule_pass'] and b['original_rule_pass'] for row in rows
        for a,b in zip(row['arms']['ROOT_FIX']['trials'],row['arms']['MULTI']['trials']))
    selected_benefit=sum(row['arms']['MULTI']['selected'] is not None and
        (row['arms']['ROOT_FIX']['selected'] is None or
         row['arms']['MULTI']['selected']['fraction']>row['arms']['ROOT_FIX']['selected']['fraction']) for row in rows)
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    assert sum(v['updates'] for v in counts)==341 and sum(v['backwards'] for v in counts)<=3743
    assert sum(v['operator_backwards'] for v in counts)<=128
    assert sum(v['local_solves'] for v in counts)<=256
    assert sum(v['non_generation_forwards'] for v in counts)<=45128
    assert all(c.read(RUN/'private/C_FISHER_PARITY'/f'{i}.json')['status']=='PASS' for i in range(1,9))
    passed=added>0 and selected_benefit>0
    out=dict(status='COMPLETE',mechanism_gate=passed,additional_real_paired_trials=added,
        improved_first_selected_owners=selected_benefit,fixed_states=8,paired_fraction_cases=32,
        root_witness_states=sum(row['arms']['ROOT_FIX']['selected'] is not None for row in rows),
        multi_witness_states=sum(row['arms']['MULTI']['selected'] is not None for row in rows),
        all_parity_owners=8,all_save_load_exact=True,new_generation=0,new_Judge=0,
        independent_confirmation=False,full_training_not_yet_complete=True)
    c.write(RUN/'public/MECHANISM_RESULTS.json',out)
    c.write(RUN/'private/MECHANISM_COMPLETE.json',dict(out,epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='MECHANISM_PASS_FULL_PENDING' if passed else 'STOP_NO_PAIRED_MECHANISM_BENEFIT'))
    return passed


def train():
    assert c.read(RUN/'private/MECHANISM_COMPLETE.json')['mechanism_gate']
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);tasks=r.s.tasks()[part::3]
    with r.p.lease(gpu):
        runtime,_,handle=r.load(gpu)
        frozen={t['order']:[r.old.cpu(runtime.build_edit_batch(r.s.record(t,x))) for x in r.s.source(t)] for t in tasks}
        r.old.fp32(runtime)
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        cpu_cache=cache_cpu(cache)
        try:
            for task in tasks:
                owner=task['order'];active=r.p.load_state(RUN/'private/active'/f'{owner}.pt')
                probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt');pair=c.read(RUN/'private/paired'/f'{owner}.json')
                assert active['step']==probe['step']==active['scalar']['step'] and active['scalar']==probe['scalar_after']
                assert all(torch.equal(active['state'][k],probe['before'][k]) for k in r.mm.KEYS)
                assert probe['seed']==task['seed'] and probe['source_order']==[x['query_id'] for x in r.s.source(task)]
                base=r.mm.canonicalize({k:v.to(runtime.device) for k,v in r.p.load_state(r.s.point(task,'C'))['states']['BASE'].items()})[0]
                sources=[r.old.batch_on(runtime,b) for b in frozen[owner]]
                reference=c.read(RUN/'private/results/C'/f'{owner}.json')
                for arm in ARMS:
                    assert not r.s.point(task,arm).exists()
                    expert=r.p.expert(base,task['seed'],runtime.device)
                    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                    try:
                        refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(sources)]
                        before={k:v.to(runtime.device) for k,v in probe['before'].items()}
                        selected=pair['arms'][arm]['selected']
                        state=before if selected is None else {k:v.to(runtime.device) for k,v in r.p.load_state(Path(selected['path']))['state'].items()}
                        j.close(r.losses_at(runtime,hook,expert,before,g.source_batches(sources,probe['step'])),probe['source_losses'])
                        pure={k:v.to(runtime.device) for k,v in probe['pure'].items()}
                        pg=g.gain(probe['source_losses'],r.losses_at(runtime,hook,expert,pure,g.source_batches(sources,probe['step'])))
                        j.close(pg,probe['old_decision']['pure_source_gain'])
                        sg=g.gain(probe['source_losses'],r.losses_at(runtime,hook,expert,state,g.source_batches(sources,probe['step'])))
                        if selected is not None:assert f.finite(sg,pg,float(r.mm.operator_penalty(state,cache)),probe['budget'],r.fm.map_norm(before,state),r.fm.map_norm(before,pure))
                        scalar=copy.deepcopy(active['scalar']);curve=copy.deepcopy(active['curve'])
                        for x in curve[:-1]:x['operator_step_norm']=x['decision']['norm_ratio']*reference['curve'][x['step']-1]['operator_step_norm']
                        curve[-1].update(decision=dict(kind='FROZEN_PAIRED_SELECTION',selected=selected is not None),operator_step_norm=r.fm.map_norm(before,state))
                        expert.load_state_dict(state);c.restore_rng(active['rng']);f.numpy_restore(probe['numpy_rng'])
                        now=c.rng();assert all(torch.equal(now[k],active['rng'][k].cpu()) for k in ('torch_rng','cuda_rng'))
                        assert now['python_rng']==active['rng']['python_rng']
                        c.write(RUN/'private/RESUME_REAL'/f'{arm}_{owner}.json',dict(status='PASS',source_state_RNG_RMS=True))
                        start=dict(r.COUNTS)
                        for step in range(probe['step']+1,161):
                            before=r.old.snapshot(expert)
                            _,losses=r.source_gradients(runtime,hook,expert,sources,step,False)
                            torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                            pure,scalar=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in expert.named_parameters()},scalar)
                            pure=g.canonical_match(before,pure,reference['curve'][step-1]['operator_step_norm'])
                            selected,audit=correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,probe['budget'],arm)
                            expert.load_state_dict(selected);r.COUNTS['updates']+=1
                            curve.append(dict(step=step,source_NLL=losses,decision=audit,operator_step_norm=r.fm.map_norm(before,selected)))
                            if step%20==0:
                                npstate=np.random.get_state()
                                c.save(RUN/'private/full_resume'/f'{arm}_{owner}.pt',dict(state=r.s.previous.weights(selected),scalar=scalar,
                                    rng=c.rng(),numpy_rng=dict(kind=npstate[0],keys=npstate[1].tolist(),position=npstate[2],has_gauss=npstate[3],cached_gaussian=npstate[4]),
                                    seed=task['seed'],step=step,curve=curve,source_order=probe['source_order'],budget=probe['budget'],counts=dict(r.COUNTS)))
                                r.save_counts();g.memory_record(gpu,'PAIRED_FULL_CHECKPOINT',prefix='paired_full')
                        r.finish_candidate(runtime,hook,expert,task,arm,base,sources,refs,curve,start,scalar,
                            dict(budget=probe['budget'],final_penalty=float(r.mm.operator_penalty(r.old.snapshot(expert),cache))))
                    finally:hook.detach()
            g.memory_record(gpu,'PAIRED_FULL_COMPLETE',complete=True,prefix='paired_full')
        finally:handle.remove();r.save_counts()


def retire(reason):
    folders=('probes','active','diagnostic_states','full_resume','candidates/ROOT_FIX','candidates/MULTI')
    paths=[RUN/'private/OPERATOR_FISHER.pt']+[p for folder in folders for p in (RUN/'private'/folder).glob('*.pt')]
    assert all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    receipt=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),reason=reason)
    c.write(RUN/'private/CONSUMED.json',dict(receipt,paths=[str(p) for p in paths]))
    for p in paths:p.unlink()
    c.write(RUN/'public/EXECUTION_AUDIT.json',dict(status='PASS',lifecycle=receipt,owned_pt_remaining=0))


def controller():
    with (RUN/'private/ONCE').open('x'):pass
    from guard_admission import require_reference
    require_reference();assert c.read(RUN/'public/CPU_CHECKS.json')['status']=='PASS'
    assert c.read(RUN/'public/CPU_CHECKS.json')['two_arm_queue_and_report_integration']
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_REBUILD_RUNNING'))
    j.stage('module_fisher',[0],file='paired_run.py')
    c.write(RUN/'public/PROGRESS.json',dict(status='FROZEN_PREFIX_AND_PAIRED_DIAGNOSTICS_RUNNING'))
    j.stage('paired_probe',range(3),file='paired_run.py')
    if not mechanism_report():
        c.write(RUN/'private/FULL_NOT_STARTED.json',dict(reason='NO_PAIRED_MECHANISM_BENEFIT'))
        c.write(RUN/'public/PROGRESS.json',dict(status='STOP_NO_PAIRED_MECHANISM_BENEFIT',full_started=False))
        retire('All frozen diagnostics complete; mechanism gate failed')
    else:
        c.write(RUN/'private/FULL_READY.json',dict(status='PASS',paired_mechanism_benefit=True))
        c.write(RUN/'public/PROGRESS.json',dict(status='PAIRED_FULL160_TRAINING_RUNNING'))
        j.stage('paired_train',range(3),file='paired_run.py')
        for task in r.s.tasks():
            for arm in ARMS:
                v=c.read(RUN/'private/results'/arm/f"{task['order']}.json")
                assert len(v['curve'])==160 and v['scalar_optimizer_state']['step']==160 and v['save_load_exact']
        c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=16,selection=False))
        c.write(RUN/'public/PROGRESS.json',dict(status='PAIRED_FULL_FROZEN_EVALUATION_RUNNING'))
        j.stage('paired_evaluate',range(3),file='paired_run.py')
        j.stage('paired_route',[0],file='paired_run.py')
        outputs=list((RUN/'private/outputs').glob('*/*/*.json'));assert len(outputs)==2316
        counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
        assert sum(v['updates'] for v in counts)==2219
        assert sum(v['backwards'] for v in counts)<=144593 and sum(v['operator_backwards'] for v in counts)<=15152
        assert sum(v['local_solves'] for v in counts)<=45328
        assert sum(v['non_generation_forwards'] for v in counts)<=600000 and sum(v['generation_attempts'] for v in counts)<=2316
        c.write(RUN/'private/MODULE_GENERATION_COMPLETE.json',dict(outputs=2316,counts=counts))
        import pipeline
        pipeline.wait([pipeline.launch('paired_queue.py','ingest')])
        c.write(RUN/'public/PROGRESS.json',dict(status='PAIRED_GENERATED_SCORING_PENDING'))
        root=RUN/'private/judge_multicore_astra_medium'
        while not (root/'ALL_WORKERS_COMPLETE.json').exists():
            c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(20)
        pipeline.wait([pipeline.launch('paired_report.py','paired_report')])
        retire('Both full arms and all paired consumers complete')
        c.write(RUN/'public/PROGRESS.json',dict(status='PAIRED_FULL_COMPLETE_PUBLICATION_PENDING'))
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    mechanism=c.read(RUN/'public/MECHANISM_RESULTS.json')
    lines=['# 同冻结状态根核与同秩多核验证','',
        f"机制门槛：{mechanism['mechanism_gate']}；新增真实配对见证{mechanism['additional_real_paired_trials']}，预定首选改善状态{mechanism['improved_first_selected_owners']}。",'',
        '两臂共用八个冻结状态、原秩/7168参数、0.1C硬预算、90%完整纯提案收益和原步幅/trust要求。多核包含完整根空间加六个预定尾方向；证书只覆盖局部切线搜索。','']
    if mechanism['mechanism_gate']:
        full=c.read(RUN/'public/PAIRED_FULL_RESULTS.json')
        lines+=[f"两臂完整八例160步/2316消费者结束；完整无缺失：{full['complete_no_missing']}；MULTI科学兼顾条件：{full['scientific_success']}。",'',
            '| 方法 | Native/8 | FIT/32 | T2G/32 | 旧知识新损伤/496 |','|---|---:|---:|---:|---:|']
        for arm in ARMS:
            panels=full['arms'][arm]['panels']
            lines.append('| '+arm+' | '+' | '.join(str(panels[k]['candidate_correct']) for k in ('NATIVE','FIT','T2G'))+' | '+str(panels['FP32_QUALIFIED']['new_damage'])+' |')
        lines+=['','完整历史M1/M3/M13/E比较、自然R0与成对区间见各臂FULL_RESULTS；根核/多核直接配对见PAIRED_FULL_RESULTS。']
    else:lines+=['机制门槛未通过，完整训练/生成/新Judge均未启动；保留全部阴性分母。']
    lines+=['',f"新增GPU进程时{(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600:.6f}h，累计{ledger['gpu_seconds_used']/3600:.6f}h；新增Judge {ledger['Judge_attempts']-inherited['Judge_attempts']}，累计{ledger['Judge_attempts']}。",'',
        '本轮自有包在声明的最后消费者后释放，历史C/D只读未动。全部DEV，无独立确认。八例T2L仅2探针/Base正确支持0，保持率NA；本轮不进入146或新增研究。']
    (RUN/'public/FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(RUN/'private/COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='module_fisher':r.GPUS=(int(os.environ['GPU']),);r.fisher()
        elif action=='paired_probe':j.probe()
        elif action=='paired_train':train()
        elif action=='paired_evaluate':r.evaluate(expected_candidates=16)
        elif action=='paired_route':
            import module_route
            module_route.main(expected_candidates=16)
        elif action=='paired_controller':controller()
        else:raise ValueError(action)
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
