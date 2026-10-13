"""At most three fixed-trust relinearizations of six previously frozen candidates."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import joint_run as j
import refine_math as rm
from prior_pairs import read_parent,summary

r,c,RUN,g=j.r,j.c,j.RUN,j.g
PARENT=Path(os.environ['JOINT_PARENT'])
TRUST_FRACTIONS=(.25,.125,.0625)
parent_diagnostics=j.diagnostics


def verify_trial(actual,expected):
    assert actual['fraction']==expected['fraction'] and actual['solver']['status']==expected['solver']['status']
    assert actual['original_rule_pass']==expected['original_rule_pass']
    assert actual['matched_step_pass']==expected['matched_step_pass']
    for key in ('pure_gain','required_gain','anchor_gain','actual_source_gain','Fisher_ratio','step_ratio'):
        if key in expected:j.close(actual[key],expected[key])
    for key in ('source_pass','budget_pass','step_upper_pass'):
        if key in expected:assert actual[key]==expected[key]


def diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar_after,
                sources,protection,refs,cache,budget,losses,old_decision,before_rng,numpy_before):
    prior=c.read(PARENT/'private/results_joint'/f"{task['order']}.json")
    selection=next(x for x in c.read(RUN/'private/PARENT_SELECTION.json')['rows'] if x['owner']==task['order'])
    assert step==prior['step']==selection['step']
    j.check_decision(old_decision,prior['old_decision'])
    clocks=copy.deepcopy((scalar_before,scalar_after))

    def consume(rec,initial,proposal,transform,inverse,center,orthogonal):
        expected=next(t for t in prior['fractions'] if t['fraction']==rec['fraction'])
        verify_trial(rec,expected)
        assert rec['fraction'] in selection['fractions'] and not rec['original_rule_pass']
        rng,numpy_rng=c.rng(),np.random.get_state()
        try:
            with c.diagnostic_scope(hook):
                rec['refinement']=repair(runtime,hook,expert,task,step,before,pure,initial,proposal,
                    transform,inverse,center,orthogonal,sources,protection,refs,cache,budget,losses,rec)
        finally:
            c.restore_rng(rng);np.random.set_state(numpy_rng)
            expert.load_state_dict(initial);expert.zero_grad(set_to_none=True)
        assert (scalar_before,scalar_after)==clocks

    parent_diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar_after,
        sources,protection,refs,cache,budget,losses,old_decision,before_rng,numpy_before,consumer=consume)
    actual=c.read(RUN/'private/results_joint'/f"{task['order']}.json")
    for a,b in zip(actual['fractions'],prior['fractions']):verify_trial(a,b)
    assert [t['fraction'] for t in actual['fractions'] if 'refinement' in t]==selection['fractions']
    c.write(RUN/'private/PARENT_PARITY'/f"{task['order']}.json",dict(status='PASS',trials=4,
            source_clocks_unchanged=True,all_prior_statuses_and_finite_checks_match=True))


def repair(runtime,hook,expert,task,step,before,pure,initial,proposal,transform,inverse,center,
           orthogonal,sources,protection,refs,cache,budget,losses,base):
    radius=r.fm.map_norm(before,pure)
    original_root=proposal['G1'].detach().cpu().double().flatten()
    rows=g.gm.root_rows(proposal,cache)@inverse
    nominal=.9*base['pure_gain'];required=nominal-1e-6
    before_loss=sum(w*x for w,x in zip((.25,.25,.5),losses))
    current={k:v.clone() for k,v in initial.items()}
    current_result={k:base[k] for k in ('actual_source_gain','Fisher_ratio','step_ratio','metrics',
                                      'original_rule_pass','matched_step_pass')}
    attempts=[];terminal='MAXIMUM_REPAIRS_REACHED'
    state_path=RUN/'private/refinement_states'/f"{task['order']}_{base['fraction']}.pt"
    for iteration,trust_fraction in enumerate(TRUST_FRACTIONS,1):
        expert.load_state_dict(current)
        before_counts=r.COUNTS['backwards']
        _,anchor_losses=r.source_gradients(runtime,hook,expert,sources,step,False)
        assert r.COUNTS['backwards']==before_counts+3
        anchor=current['G1'].detach().cpu().double().flatten()
        gradient=expert.G1.grad.detach().cpu().double().flatten()
        anchor_gain=before_loss-sum(w*x for w,x in zip((.25,.25,.5),anchor_losses))
        j.close(anchor_gain,current_result['actual_source_gain'])
        trust_radius=trust_fraction*radius
        value,audit=rm.solve(rows.numpy(),(transform@original_root).numpy(),center.numpy(),orthogonal,
            radius*(1+.002)+1e-10,(inverse@gradient).numpy(),(transform@anchor).numpy(),
            anchor_gain,nominal,budget,radius,trust_radius)
        trial=dict(iteration=iteration,trust_fraction=trust_fraction,solver=audit,anchor_gain=anchor_gain,
                   local_nominal_required=nominal,finite_required=required,original_rule_pass=False,
                   matched_step_pass=False,anchor_advanced=False)
        updated=None
        if value is not None:
            updated={k:v.clone() for k,v in proposal.items()}
            updated['G1']=(inverse@torch.from_numpy(value)).reshape_as(updated['G1']).to(updated['G1'])
            actual_losses=r.losses_at(runtime,hook,expert,updated,g.source_batches(sources,step))
            gain=g.gain(losses,actual_losses)
            penalty=float(r.mm.operator_penalty(updated,cache));norm=r.fm.map_norm(before,updated)
            distance=r.fm.map_norm(current,updated)
            predicted=anchor_gain-float(gradient@(updated['G1'].detach().cpu().double().flatten()-anchor))
            source_ok=gain>=required;budget_ok=penalty<=budget+1e-9+1e-5*budget
            upper_ok=norm<=radius*(1+.002)+1e-10
            trust_ok=distance<=trust_radius*(1+1e-5)+1e-10
            passed=source_ok and budget_ok and upper_ok and trust_ok
            trial.update(actual_source_gain=gain,predicted_source_gain=predicted,linearization_error=gain-predicted,
                Fisher_ratio=penalty/budget,step_ratio=norm/radius,anchor_distance_over_full_step=distance/radius,
                source_pass=source_ok,budget_pass=budget_ok,step_upper_pass=upper_ok,trust_pass=trust_ok,
                original_constraints_pass=source_ok and budget_ok and upper_ok,
                original_rule_pass=passed,matched_step_pass=passed and norm>=.9*radius-1e-10,
                metrics=j.measure(runtime,hook,expert,updated,sources,protection,refs))
            if budget_ok and upper_ok and trust_ok and gain>=current_result['actual_source_gain']:
                current=updated;current_result=copy.deepcopy(trial);trial['anchor_advanced']=True
            if passed:terminal='VERIFIED_ORIGINAL_RULE_WITNESS'
        else:terminal='CORRECTION_'+audit['status']
        attempts.append(trial)
        c.save(state_path,dict(current=r.s.previous.weights(current),initial=r.s.previous.weights(initial),
            last_trial=r.s.previous.weights(updated) if updated is not None else None,
            step=step,seed=task['seed'],fraction=base['fraction'],attempts=attempts,
            source_state_binding=str(RUN/'private/probes'/f"{task['order']}.pt"),rng=c.rng(),counts=dict(r.COUNTS)))
        loaded=r.p.load_state(state_path)
        assert all(torch.equal(loaded['current'][k],current[k].detach().cpu()) for k in current)
        r.save_counts()
        if terminal!='MAXIMUM_REPAIRS_REACHED':break
    return dict(status=terminal,attempts=attempts,final=current_result,
                original_rule_pass=terminal=='VERIFIED_ORIGINAL_RULE_WITNESS',
                matched_step_pass=terminal=='VERIFIED_ORIGINAL_RULE_WITNESS' and current_result['matched_step_pass'],
                save_load_exact=True,objective_center_fixed=True,pure_reference_fixed=True)


j.diagnostics=diagnostics


def cpu():
    j.cpu()
    prior=read_parent(PARENT)
    selected=c.read(RUN/'private/PARENT_SELECTION.json')['rows']
    assert len(selected)==8 and sum(len(x['fractions']) for x in selected)==6
    for row,item in zip(prior,selected):
        assert row['owner']==item['owner'] and row['step']==item['step']
        assert item['fractions']==[t['fraction'] for t in row['fractions'] if t['solver']['status']=='LOCAL_KKT_PASS']
    admission=c.read(RUN/'public/CPU_ADMISSION.json')
    admission.update(refinement_solver=rm.selfcheck(),eligible_candidates=6,all_state_denominator=8,
                     maximum_repairs=3,trust_fractions=TRUST_FRACTIONS,new_model_calls=0)
    c.write(RUN/'public/CPU_ADMISSION.json',admission)


def report():
    rows=[c.read(p) for p in sorted((RUN/'private/results_joint').glob('*.json'))]
    assert len(rows)==8 and {x['owner'] for x in rows}==set(range(1,9))
    assert len(list((RUN/'private/PARENT_PARITY').glob('*.json')))==8
    trials=[t for row in rows for t in row['fractions']];assert len(trials)==32
    candidates=[t for t in trials if 'refinement' in t];assert len(candidates)==6
    attempts=[a for t in candidates for a in t['refinement']['attempts']]
    assert all(1<=len(t['refinement']['attempts'])<=3 for t in candidates)
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(x[k] for x in counts) for k in r.COUNTS}
    assert totals['updates']==341 and totals['backwards']==3071+3*len(attempts)<=3125
    assert totals['non_generation_forwards']<=12000 and totals['generation_attempts']==totals['operator_backwards']==0
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==4 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    assert ledger['Judge_attempts']==inherited['Judge_attempts']
    witnesses=sum(any(t.get('refinement',{}).get('original_rule_pass',False) for t in row['fractions']) for row in rows)
    matched=sum(any(t.get('refinement',{}).get('matched_step_pass',False) for t in row['fractions']) for row in rows)
    out=dict(status='COMPLETE',decision='REFINEMENT_WITNESS_FOUND' if witnesses else 'NO_VERIFIED_REFINEMENT_WITNESS',
        fixed_states=8,original_local_problems=32,prior_local_infeasible=sum(t['solver']['status']=='CERTIFIED_LOCAL_INFEASIBLE' for t in trials),
        eligible_candidates=6,eligible_unique_states=5,repair_attempts=len(attempts),original_rule_witness_states=witnesses,
        matched_step_witness_states=matched,original_rule_witness_candidates=sum(t['refinement']['original_rule_pass'] for t in candidates),
        matched_step_witness_candidates=sum(t['refinement']['matched_step_pass'] for t in candidates),
        correction_solver_statuses={s:sum(a['solver']['status']==s for a in attempts) for s in sorted({a['solver']['status'] for a in attempts})},
        counts=totals,resources=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=0,cumulative_Judge=ledger['Judge_attempts']),
        reconstruction_parity_owners=8,C_Fisher_parity_owners=8,parent_joint_parity_owners=8,independent_confirmation=False,
        no_generation_or_heldout=True,next_consumer='NONE',per_fraction=[])
    assert out['prior_local_infeasible']==26
    for f in j.FRACTIONS:
        ts=[t for t in candidates if t['fraction']==f]
        out['per_fraction'].append(dict(fraction=f,eligible=len(ts),
            witnesses=sum(t['refinement']['original_rule_pass'] for t in ts),
            baseline_gain_ratio=summary([t['actual_source_gain']/t['pure_gain'] for t in ts]),
            final_gain_ratio=summary([t['refinement']['final']['actual_source_gain']/t['pure_gain'] for t in ts]),
            paired_gain_ratio_change=summary([(t['refinement']['final']['actual_source_gain']-t['actual_source_gain'])/t['pure_gain'] for t in ts]),
            paired_basis_KL_change=summary([t['refinement']['final']['metrics']['basis_mean_KL']-t['metrics']['basis_mean_KL'] for t in ts])))
    c.write(RUN/'public/RESULTS.json',out)
    (RUN/'public/REPORT_ZH.md').write_text(f'''# 有限再线性化终结报告

结论：{out['decision']}。固定八状态/32原局部问题，原26局部不可行保留；只修正预先选定的六候选（五状态），完成{len(attempts)}次、每候选最多三次。
真实原规则见证{witnesses}/8状态，另加90%步幅下界见证{matched}/8；候选通过{out['original_rule_witness_candidates']}/6。所有成功/失败与条件分母均保留。
八例轨迹、C惩罚、原联合结果及保存加载核验通过。341重建提案、{totals['backwards']}反传、{totals['non_generation_forwards']}非生成前向，0生成/Judge。
新增{out['resources']['new_GPU_process_hours']:.9f}GPU进程小时，累计{out['resources']['cumulative_GPU_process_hours']:.9f}，Judge累计{ledger['Judge_attempts']}。
本轮为开发暴露后固定的机制诊断，不是独立确认。局部证书仅针对各自线性模型/信赖范围；未找到见证不证明全TT不可行，找到见证也不证明完整训练或T2G保护成功。无自动后续消费者。
逐比例的配对变化见RESULTS；旧投影配对见PAIRED_PRIOR_RESULTS。不得公开逐题身份、tokens、私有状态或路径。
''')
    paths=[RUN/'private/OPERATOR_FISHER.pt']+[p for d in ('probes','active','refinement_states') for p in (RUN/'private'/d).glob('*.pt')]
    assert len(paths)==23 and all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    lifecycle=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),copies=False,
                   reason='All fixed repair, parity, save-load and report consumers complete',reconstruction_requires_training=True)
    c.write(RUN/'private/CONSUMED.json',dict(lifecycle,paths=[str(p) for p in paths]))
    for p in paths:p.unlink()
    c.write(RUN/'public/EXECUTION_AUDIT.json',dict(status='PASS',all_GPU_leases_ended=True,counts=totals,
            lifecycle=lifecycle,owned_pt_remaining=0,full_validation_started=False))
    c.write(RUN/'private/REPORT_COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status=out['decision'],complete=True,publication_complete=False))
    print(json.dumps({k:out[k] for k in ('decision','original_rule_witness_states','resources')}),flush=True)


def controller():
    from guard_admission import require_reference
    require_reference()
    assert c.read(RUN/'public/CPU_ADMISSION.json')['refinement_solver']['status']=='PASS'
    with (RUN/'private/ONCE.json').open('x') as handle:json.dump(dict(pid=os.getpid(),epoch=time.time()),handle)
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_REBUILD',complete=False))
    j.stage('module_fisher',[0],file='refine_run.py')
    c.write(RUN/'public/PROGRESS.json',dict(status='PREFIX_RECONSTRUCTION_AND_BOUNDED_REFINEMENT',complete=False))
    j.stage('refine_probe',range(3),file='refine_run.py')
    report()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='refine_cpu':cpu()
        elif action=='module_fisher':r.GPUS=(int(os.environ['GPU']),);r.fisher()
        elif action=='refine_probe':j.probe()
        elif action=='refine_controller':controller()
        else:raise ValueError('Unregistered action')
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
                dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        c.write(RUN/'public/PROGRESS.json',dict(status='STOPPED_RETAINED',complete=False))
        raise
