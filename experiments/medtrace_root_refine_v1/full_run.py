"""One eight-owner continuation; only rejected positive-gain proposals are repaired."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
os.environ['M13_PARENT']=json.loads((Path(os.environ['RUN_ROOT'])/'private/LAUNCH_ENV.json').read_text())['M13_PARENT']
import joint_run as j
import refine_math as rm

r,c,RUN,g=j.r,j.c,j.RUN,j.g
ARM='REFINE'
r.mm.ARMS=('C',ARM)
r.GPUS=(0,1,7)


def count(*_):
    c.budget();r.COUNTS['forwards']+=1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards']+=1
        action=os.environ['ACTION']
        if action=='full_train':
            part=int(os.environ['PARTITION'])
            steps=[x['step'] for x in c.read(RUN/'private/TARGETS.json')['rows']][part::3]
            limit=sum(160-x for x in steps)*114+len(steps)*128
        else:limit={'full_evaluate':100,'full_route':512}[action]
        assert r.COUNTS['non_generation_forwards']<=limit,'Finite full-consumer forward cap'
r.count=count


def numpy_restore(state):
    np.random.set_state((state['kind'],np.asarray(state['keys'],dtype=np.uint32),state['position'],
                         state['has_gauss'],state['cached_gaussian']))


def finite(gain,pure_gain,penalty,budget,norm,radius):
    values=(gain,pure_gain,penalty,budget,norm,radius)
    return all(np.isfinite(x) for x in values) and gain>=.9*pure_gain-1e-6 and \
        penalty<=budget+1e-9+1e-5*budget and norm<=radius*1.002+1e-10


def cpu():
    assert finite(.9,1.,.1,.1,1.,1.)
    for args in ((.899,1.,.1,.1,1.,1.),(.9,1.,.2,.1,1.,1.),(.9,1.,.1,.1,1.003,1.),
                 (float('nan'),1.,.1,.1,1.,1.)):
        assert not finite(*args)
    assert j.jm.selfcheck()['status']=='PASS' and rm.selfcheck()['status']=='PASS'
    out=c.read(RUN/'public/RESULTS.json')
    assert out['original_rule_witness_states']>0
    assert all(out[k]==8 for k in ('reconstruction_parity_owners','C_Fisher_parity_owners','parent_joint_parity_owners'))
    assert c.read(RUN/'public/EXECUTION_AUDIT.json')['owned_pt_remaining']==23
    assert c.read(RUN/'private/REPORT_COMPLETE.json')['status']=='COMPLETE'
    rows=[]
    for task in r.s.tasks():
        active,probe,record,selected=resume(task)
        c.restore_rng(dict(active['rng'],cuda_rng=None))
        assert torch.equal(torch.get_rng_state(),active['rng']['torch_rng'])
        numpy_restore(probe['numpy_rng'])
        assert np.random.get_state()[2:]==(probe['numpy_rng']['position'],probe['numpy_rng']['has_gauss'],probe['numpy_rng']['cached_gaussian'])
        assert np.array_equal(np.random.get_state()[1],probe['numpy_rng']['keys'])
        rows.append(dict(owner=task['order'],step=probe['step'],full_state=True,source_RMS_clock=active['scalar']['step'],
                         first_decision_reused=selected['kind'],source_order_bound=True))
    assert sum(x['step'] for x in rows)==341
    c.write(RUN/'private/FULL_RESUME_ADMISSION.json',dict(status='PASS',rows=rows,
        new_model_calls=0,remaining_proposals=939,NumPy='Before snapshot; source gradient and deterministic projection consume no NumPy RNG'))


def resume(task):
    owner=task['order']
    active=r.p.load_state(RUN/'private/active'/f'{owner}.pt')
    probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt')
    record=c.read(RUN/'private/results_joint'/f'{owner}.json')
    assert active['step']==probe['step']==record['step']==active['scalar']['step']
    assert active['seed']==probe['seed']==task['seed']
    assert active['scalar']==probe['scalar_after'] and probe['scalar_before']['step']==probe['step']-1
    assert len(active['scalar']['second_moments'])==2 and all(np.isfinite(active['scalar']['second_moments']))
    assert len(active['curve'])==probe['step']
    assert all(torch.equal(active['state'][k],probe['before'][k]) for k in r.mm.KEYS)
    assert probe['source_order']==[x['query_id'] for x in r.s.source(task)]
    assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
    assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
    assert record['old_decision']==probe['old_decision'] and record['old_decision']['status']=='SKIPPED_INFEASIBLE'
    assert c.read(RUN/'private/PARENT_PARITY'/f'{owner}.json')['status']=='PASS'
    chosen=dict(kind='SKIPPED_INFEASIBLE',state=probe['before'],source_gain=0.)
    for trial in record['fractions']:
        assert not trial['original_rule_pass']
        if trial.get('refinement',{}).get('original_rule_pass'):
            saved=r.p.load_state(RUN/'private/refinement_states'/f"{owner}_{trial['fraction']}.pt")
            assert saved['seed']==task['seed'] and saved['step']==probe['step'] and saved['fraction']==trial['fraction']
            assert saved['attempts']==trial['refinement']['attempts']
            assert saved['source_state_binding']==str(RUN/'private/probes'/f'{owner}.pt')
            chosen=dict(kind='REUSED_REFINEMENT_WITNESS',state=saved['current'],fraction=trial['fraction'],
                        source_gain=trial['refinement']['final']['actual_source_gain'])
            break
    return active,probe,record,chosen


def choose(runtime,hook,expert,before,pure,step,sources,losses,cache,budget):
    selected,original=g.select_guard(runtime,hook,expert,before,pure,g.source_batches(sources,step),losses,cache,budget)
    if original['status']=='ACCEPTED' or original['pure_source_gain']<=0:
        return selected,dict(kind='ORIGINAL_GUARD',original=original)
    radius=r.fm.map_norm(before,pure);pure_gain=original['pure_source_gain']
    before_loss=g.mean_loss(losses);records=[]
    numpy_state=np.random.get_state()
    try:
        with c.diagnostic_scope(hook):
            for fraction in j.FRACTIONS:
                proposal=pure if fraction==1 else r.mm.canonicalize(r.q.add(before,fraction*(r.q.flatten(pure)-r.q.flatten(before))))[0]
                transform,inverse,center,orthogonal=j.jg.coordinates(before,proposal)
                rows=g.gm.root_rows(proposal,cache)@inverse
                original_root=proposal['G1'].detach().cpu().double().flatten()
                current=proposal;current_gain=None;attempts=[]
                for iteration in range(4):
                    expert.load_state_dict(current)
                    _,anchor_losses=r.source_gradients(runtime,hook,expert,sources,step,False)
                    anchor=current['G1'].detach().cpu().double().flatten()
                    gradient=expert.G1.grad.detach().cpu().double().flatten()
                    anchor_gain=before_loss-g.mean_loss(anchor_losses)
                    if current_gain is not None:j.close(anchor_gain,current_gain)
                    if iteration==0:
                        value,audit=j.jm.solve(rows.numpy(),(transform@original_root).numpy(),center.numpy(),orthogonal,
                            radius*1.002+1e-10,(inverse@gradient).numpy(),anchor_gain,.9*pure_gain-1e-6,budget,radius)
                        rho=None
                    else:
                        rho=(.25,.125,.0625)[iteration-1]*radius
                        value,audit=rm.solve(rows.numpy(),(transform@original_root).numpy(),center.numpy(),orthogonal,
                            radius*1.002+1e-10,(inverse@gradient).numpy(),(transform@anchor).numpy(),
                            anchor_gain,.9*pure_gain,budget,radius,rho)
                    entry=dict(iteration=iteration,solver=audit,passed=False)
                    attempts.append(entry)
                    if value is None:break
                    candidate={k:v.clone() for k,v in proposal.items()}
                    candidate['G1']=(inverse@torch.from_numpy(value)).reshape_as(candidate['G1']).to(candidate['G1'])
                    after=r.losses_at(runtime,hook,expert,candidate,g.source_batches(sources,step))
                    gain=g.gain(losses,after);penalty=float(r.mm.operator_penalty(candidate,cache))
                    norm=r.fm.map_norm(before,candidate)
                    trust_ok=rho is None or r.fm.map_norm(current,candidate)<=rho*(1+1e-5)+1e-10
                    passed=finite(gain,pure_gain,penalty,budget,norm,radius) and trust_ok
                    entry.update(passed=passed,source_gain=gain,pure_source_gain=pure_gain,
                        Fisher_ratio=penalty/budget,norm_ratio=norm/radius,trust_pass=trust_ok,
                        matched_step_pass=passed and norm>=.9*radius-1e-10)
                    if passed:
                        records.append(dict(fraction=fraction,attempts=attempts))
                        return candidate,dict(kind='JOINT_REFINEMENT_ACCEPTED',original=original,trials=records)
                    if iteration==0 or (trust_ok and penalty<=budget+1e-9+1e-5*budget and norm<=radius*1.002+1e-10 and gain>=anchor_gain):
                        current=candidate;current_gain=gain
                records.append(dict(fraction=fraction,attempts=attempts))
    finally:np.random.set_state(numpy_state)
    return before,dict(kind='SKIPPED_INFEASIBLE',original=original,trials=records)


def train():
    from guard_admission import require_reference
    require_reference()
    assert c.read(RUN/'private/FULL_CONSUMER_READY.json')['status']=='PASS'
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);tasks=r.s.tasks()[part::3]
    with r.p.lease(gpu):
        runtime,_,handle=r.load(gpu)
        g.memory_record(gpu,'FULL_LOADED',prefix='full_train')
        frozen={t['order']:[r.old.cpu(runtime.build_edit_batch(r.s.record(t,row))) for row in r.s.source(t)] for t in tasks}
        r.old.fp32(runtime)
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        try:
            for task in tasks:
                owner=task['order'];assert not r.s.point(task,ARM).exists()
                active,probe,record,selected=resume(task)
                base=r.mm.canonicalize({k:v.to(runtime.device) for k,v in r.p.load_state(r.s.point(task,'C'))['states']['BASE'].items()})[0]
                expert=r.p.expert(base,task['seed'],runtime.device)
                hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                sources=[r.old.batch_on(runtime,b) for b in frozen[owner]]
                reference=c.read(RUN/'private/results/C'/f'{owner}.json')
                scalar=copy.deepcopy(active['scalar']);curve=copy.deepcopy(active['curve']);start=dict(r.COUNTS)
                budget=probe['budget']
                try:
                    refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(sources)]
                    before={k:v.to(runtime.device) for k,v in probe['before'].items()}
                    pure={k:v.to(runtime.device) for k,v in probe['pure'].items()}
                    state={k:v.to(runtime.device) for k,v in selected['state'].items()}
                    original_losses=r.losses_at(runtime,hook,expert,before,g.source_batches(sources,probe['step']))
                    j.close(original_losses,probe['source_losses'])
                    pure_gain=g.gain(original_losses,r.losses_at(runtime,hook,expert,pure,g.source_batches(sources,probe['step'])))
                    j.close(pure_gain,record['old_decision']['pure_source_gain'])
                    source_gain=g.gain(original_losses,r.losses_at(runtime,hook,expert,state,g.source_batches(sources,probe['step'])))
                    j.close(source_gain,selected['source_gain'])
                    if selected['kind']=='REUSED_REFINEMENT_WITNESS':
                        assert finite(source_gain,pure_gain,float(r.mm.operator_penalty(state,cache)),budget,
                                      r.fm.map_norm(before,state),r.fm.map_norm(before,pure))
                    for row in curve[:-1]:
                        old=row['decision'];row['operator_step_norm']=old['norm_ratio']*reference['curve'][row['step']-1]['operator_step_norm']
                    curve[-1].update(decision={k:v for k,v in selected.items() if k!='state'},operator_step_norm=r.fm.map_norm(before,state))
                    expert.load_state_dict(state);c.restore_rng(active['rng']);numpy_restore(probe['numpy_rng'])
                    live_rng=c.rng()
                    assert all(torch.equal(live_rng[k],active['rng'][k].cpu()) for k in ('torch_rng','cuda_rng'))
                    assert live_rng['python_rng']==active['rng']['python_rng']
                    c.write(RUN/'private/FULL_RESUME_REAL'/f'{owner}.json',dict(status='PASS',step=probe['step'],state_rng_RMS_bound=True))
                    for step in range(probe['step']+1,161):
                        before=r.old.snapshot(expert)
                        _,losses=r.source_gradients(runtime,hook,expert,sources,step,False)
                        torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                        pure,scalar=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in expert.named_parameters()},scalar)
                        assert scalar['step']==step
                        pure=g.canonical_match(before,pure,reference['curve'][step-1]['operator_step_norm'])
                        chosen,audit=choose(runtime,hook,expert,before,pure,step,sources,losses,cache,budget)
                        expert.load_state_dict(chosen);r.COUNTS['updates']+=1
                        curve.append(dict(step=step,source_NLL=losses,decision=audit,operator_step_norm=r.fm.map_norm(before,chosen)))
                        assert r.COUNTS['backwards']<=51*r.COUNTS['updates']
                        if step%20==0:
                            npstate=np.random.get_state()
                            c.save(RUN/'private/full_resume'/f'{owner}.pt',dict(state=r.s.previous.weights(chosen),
                                scalar=scalar,rng=c.rng(),numpy_rng=dict(kind=npstate[0],keys=npstate[1].tolist(),position=npstate[2],
                                has_gauss=npstate[3],cached_gaussian=npstate[4]),step=step,seed=task['seed'],curve=curve,
                                scope_lock=probe['scope_lock'],target_lock=probe['target_lock'],source_order=probe['source_order'],budget=budget,counts=dict(r.COUNTS)))
                            r.save_counts();g.memory_record(gpu,'FULL_TRAIN_CHECKPOINT',prefix='full_train')
                    r.finish_candidate(runtime,hook,expert,task,ARM,base,sources,refs,curve,start,scalar,
                        mechanism=dict(budget=budget,final_penalty=float(r.mm.operator_penalty(r.old.snapshot(expert),cache))))
                    print('FULL160_COMPLETE',owner,flush=True)
                finally:hook.detach()
            g.memory_record(gpu,'FULL_PARTITION_COMPLETE',complete=True,prefix='full_train')
        finally:handle.remove();r.save_counts()


def controller():
    import pipeline
    with (RUN/'private/FULL_ONCE').open('x'):pass
    assert c.read(RUN/'private/FULL_RESUME_ADMISSION.json')['status']=='PASS'
    j.stage('full_train',range(3),file='full_run.py')
    totals={k:sum(c.read(p)[k] for p in (RUN/'private').glob('COUNTS_full_train_*.json')) for k in r.COUNTS}
    assert totals['updates']==939 and totals['backwards']<=47889 and totals['operator_backwards']==0
    for task in r.s.tasks():
        result=c.read(RUN/'private/results'/ARM/f"{task['order']}.json")
        assert result['save_load_exact'] and len(result['curve'])==160 and result['scalar_optimizer_state']['step']==160
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=8,epoch=time.time(),selection=False))
    c.write(RUN/'public/PROGRESS.json',dict(status='FULL160_FROZEN_EVALUATION_RUNNING',complete=False))
    j.stage('full_evaluate',range(3),file='full_run.py')
    j.stage('full_route',[0],file='full_run.py')
    outputs=list((RUN/'private/outputs').glob('*/*/*.json'));assert len(outputs)==1158
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_full_*.json')]
    assert sum(x['non_generation_forwards'] for x in counts)<=110000
    assert sum(x['generation_attempts'] for x in counts)<=1158
    c.write(RUN/'private/MODULE_GENERATION_COMPLETE.json',dict(outputs=1158,counts=counts,epoch=time.time()))
    pipeline.wait([pipeline.launch('full_queue.py','full_ingest')])
    c.write(RUN/'public/PROGRESS.json',dict(status='FULL_GENERATED_SCORING_PENDING',complete=False))
    root=RUN/'private/judge_full_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(20)
    pipeline.wait([pipeline.launch('full_report.py','full_report')])
    paths=[RUN/'private/OPERATOR_FISHER.pt']+[p for d in ('probes','active','refinement_states','full_resume') for p in (RUN/'private'/d).glob('*.pt')]
    paths += [r.s.point(t,ARM) for t in r.s.tasks()]
    assert len(paths)==39 and all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    receipt=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),reason='All authorized full training/evaluation/scoring/report consumers complete')
    c.write(RUN/'private/FULL_CONSUMED.json',dict(receipt,paths=[str(p) for p in paths]))
    for p in paths:p.unlink()
    c.write(RUN/'public/FULL_EXECUTION_AUDIT.json',dict(status='PASS',lifecycle=receipt,owned_pt_remaining=0,
        original_reconstruction_proposals=341,new_continuation_proposals=939,full_scientific_trajectory_steps=1280))
    c.write(RUN/'private/FULL_COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))


def admit():
    # An exclusive CPU admission waits for the complete fixed diagnostic, never a subset.
    with (RUN/'private/FULL_ADMISSION_ONCE').open('x'):pass
    while not (RUN/'private/REPORT_COMPLETE.json').exists():
        c.budget();assert not list((RUN/'private').glob('FAILURE*'));time.sleep(10)
    if c.read(RUN/'public/RESULTS.json')['original_rule_witness_states']==0:
        c.write(RUN/'private/FULL_NOT_STARTED.json',dict(reason='NO_VERIFIED_REFINEMENT_WITNESS',epoch=time.time()))
        return
    cpu()
    manifest=c.read(RUN/'RUN_MANIFEST.json')
    assert manifest['Judge_limit']==17177
    c.write(RUN/'private/full_authorization1/RUN_MANIFEST_BEFORE_FULL.json',manifest)
    manifest['Judge_limit']=18335
    c.write(RUN/'RUN_MANIFEST.json',manifest)
    plan=c.read(RUN/'PLAN_CONFIG.json')
    plan['full_consumer']=dict(arm=ARM,steps=160,owners=8,new_proposals=939,maximum_NN_backwards=47889,
        maximum_non_generation_forwards=110000,maximum_local_solves=15024,maximum_generations=1158,
        maximum_Judge=1158,comparators=['M1','M3','M13','E'],source_commit=os.environ['FULL_SOURCE_COMMIT'])
    c.write(RUN/'PLAN_CONFIG.json',plan)
    c.write(RUN/'private/FULL_CONSUMER_READY.json',dict(status='PASS',source_commit=os.environ['FULL_SOURCE_COMMIT'],
        mechanism_gate='ORIGINAL_REAL_RULE_WITNESS_AND_ALL_EIGHT_PARITY',new_model_calls=0,epoch=time.time()))


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='full_cpu':cpu()
        elif action=='full_admit':admit()
        elif action=='full_train':train()
        elif action=='full_evaluate':r.evaluate(expected_candidates=8)
        elif action=='full_route':
            import module_route
            module_route.main(expected_candidates=8)
        elif action=='full_controller':controller()
        else:raise ValueError('Unknown full action')
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f"FAILURE_full_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time()))
        raise
