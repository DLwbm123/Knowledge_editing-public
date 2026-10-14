"""Fixed160 training loop; no launcher, implicit resume, or source-gate bypass."""
import copy
import numpy as np
import torch
import contract
import training_math


def source_rows(packet,owner):
    admission=contract.source_admission(packet)
    assert owner in admission['qualified_owners'],'Original-version source evidence missing'
    task=packet['original_tasks'][owner-1]
    rows=[dict(task['native'],query_id=f'edit_{owner}_{i}',question=question,
        audit_role='NATIVE' if i==0 else 'FIT')
        for i,question in enumerate([task['native']['question']]+task['fit_questions'])]
    rows += [dict(row,audit_role='GFIT') for row in packet['reviews']['qualified_anchors'] if row['owner']==owner]
    assert len(rows)==7 and len({row['query_id'] for row in rows})==7
    return rows


def snapshot(r,expert,optimizer,scalar,step,curve,binding):
    return dict(state=r.old.snapshot(expert),optimizer=copy.deepcopy(optimizer.state_dict()) if optimizer else None,
        scalar=copy.deepcopy(scalar),step=step,curve=copy.deepcopy(curve),binding=copy.deepcopy(binding),
        counts=copy.deepcopy(r.COUNTS),rng=copy.deepcopy(r.c.rng()),numpy_rng=copy.deepcopy(np.random.get_state()))


def validate_snapshot(r,saved,binding):
    assert saved['binding']==binding and 0<saved['step']<=160
    assert len(saved['curve'])==saved['step'] and [v['step'] for v in saved['curve']]==list(range(1,saved['step']+1))
    assert set(saved['state'])==set(r.mm.KEYS) and sum(v.numel() for v in saved['state'].values())==7168
    assert all(torch.isfinite(v).all() for v in saved['state'].values())
    assert all(k in saved['rng'] for k in ('torch_rng','python_rng','cuda_rng')) and len(saved['numpy_rng'])==5
    assert all(v>=0 for v in saved['counts'].values())
    assert saved['counts']['updates']>=saved['step'] and saved['counts']['backwards']>=3*saved['step']
    if binding['arm'] in ('M3','M13'):assert saved['counts']['operator_backwards']>=saved['step']
    if binding['arm'] in ('M1','M13'):
        assert saved['optimizer'] is None and saved['scalar']['step']==saved['step']
        assert len(saved['scalar']['second_moments'])==2 and all(np.isfinite(v) and v>=0 for v in saved['scalar']['second_moments'])
    else:
        assert saved['scalar'] is None and saved['optimizer'] is not None
        opt=saved['optimizer'];groups=opt['param_groups']
        assert len(groups)==2 and [g['lr'] for g in groups]==[1e-4,1e-3]
        assert all(tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in groups)
        ids=[k for g in groups for k in g['params']]
        assert len(ids)==len(set(ids))==4 and set(opt['state'])==set(ids)
        for value in opt['state'].values():
            assert int(value['step'])==saved['step']
            assert all(k in value and torch.isfinite(value[k]).all() for k in ('exp_avg','exp_avg_sq'))
            assert value['exp_avg'].shape==value['exp_avg_sq'].shape and (value['exp_avg_sq']>=0).all()
    return dict(status='PASS',automatic_resume=False)


def run(r,runtime,hook,expert,batches,arm,cache,regularization,targets,binding,save_checkpoint):
    assert arm in contract.METHODS and binding['arm']==arm and len(batches)==7
    assert len(binding['source_order'])==len(set(binding['source_order']))==7
    assert binding['steps']==160 and binding['parameters']==7168
    assert all(binding.get(k) for k in ('source_binding','scientific_lock','source_version','budget_binding'))
    assert type(binding['seed']) is int
    initial=r.old.snapshot(expert)
    assert sum(v.numel() for v in initial.values())==7168 and torch.count_nonzero(initial['G1'])==0
    assert all(torch.count_nonzero(initial[k])>0 for k in r.mm.KEYS[1:])
    canonical=arm in ('M1','M13')
    if canonical:
        expert.load_state_dict(r.mm.canonicalize(initial)[0])
    assert (arm=='C' and targets is None and regularization==0) or (
        arm!='C' and len(targets)==160 and all(np.isfinite(v) and v>=0 for v in targets))
    assert (arm in ('C','M1') and regularization==0) or (arm in ('M3','M13') and np.isfinite(regularization) and regularization>0)
    optimizer=None if canonical else r.p.optimizer(expert,runtime.model)
    scalar=dict(step=0,second_moments=[0.,0.]) if canonical else None
    start=copy.deepcopy(r.COUNTS);curve=[]
    for step in range(1,161):
        r.c.budget()
        before=r.old.snapshot(expert)
        _,losses=r.source_gradients(runtime,hook,expert,batches,step,False)
        gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
        if canonical:
            selected,scalar,audit=training_math.step(r,before,gradients,cache,regularization,scalar,targets[step-1],arm)
            expert.load_state_dict(selected)
        else:
            selected,audit=training_math.raw_step(r,expert,optimizer,gradients,cache,regularization,
                None if arm=='C' else targets[step-1],arm)
        r.COUNTS['operator_backwards']+=audit['operator_backwards'];r.COUNTS['updates']+=1
        assert r.COUNTS['backwards']-start['backwards']==3*step
        assert r.COUNTS['operator_backwards']-start['operator_backwards']==step*int(arm in ('M3','M13'))
        curve.append(dict(step=step,source_NLL=losses,operator_step_norm=r.fm.map_norm(before,selected),**audit))
        if step%20==0:
            saved=snapshot(r,expert,optimizer,scalar,step,curve,binding)
            validate_snapshot(r,saved,binding);save_checkpoint(saved)
    return snapshot(r,expert,optimizer,scalar,160,curve,binding)
