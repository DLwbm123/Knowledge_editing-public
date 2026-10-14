"""One anomalous baseline replay from an audited immutable frozen state."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import paired_run

# Legacy imports prepend their frozen tools; bind this repair's versions explicitly.
for name in ('correction_math','aligned'):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if name=='correction_math':paired_run.cm=module
    else:a=module

p,j,r,c,RUN=a.p,a.j,a.r,a.c,a.RUN


def historical_solver():
    path=Path(os.environ['BASE_PARENT'])/'private/tools/correction_math.py'
    tree=ast.parse(path.read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='solve')
    from scipy.optimize import minimize
    scope=dict(np=np,minimize=minimize)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope['solve']


def cpu():
    old=historical_solver();rng=np.random.default_rng(20261014)
    for _ in range(16):
        args=dict(matrix=rng.normal(size=(7,5)),offset=rng.normal(size=7)*.1,
            center=rng.normal(size=5),step_offset=np.zeros(5),orthogonal=0.,radius=1.002-1e-5,
            source=rng.normal(size=5),required=.1,trust=None,fisher_objective=True)
        x,u=old(**args);y,v=p.cm.solve(**args)
        assert u==v and ((x is None and y is None) or np.array_equal(x,y))
    assert p.cm.selfcheck()['status']=='PASS'
    # The new two-halfspace branch remains covered by the original CPU admission.
    args=(np.eye(2),np.zeros(2),np.array([1.,0.]),np.zeros(2),0.,1.002-1e-5)
    x,v=p.cm.solve(*args,np.array([[0.,1.],[1.,0.]]),np.array([.2,.90001]),fisher_objective=True)
    assert v['status']=='LOCAL_KKT_PASS' and x[0]>=.90001-1e-7 and x[1]>=.2-1e-7
    c.write(RUN/'private/BASELINE_REPAIR_CPU1.json',dict(status='PASS',exact_scalar_regressions=16,new_model_calls=0))


def audit():
    from guard_admission import require_reference
    require_reference()
    assert c.read(RUN/'private/EXIT1.json')['exit_code']==1
    assert all(x.get('ended_epoch') for x in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    cache=r.p.load_state(RUN/'private/OPERATOR_FISHER.pt')
    changed=[]
    for task in r.s.tasks():
        owner=task['order'];probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt')
        active=r.p.load_state(RUN/'private/active'/f'{owner}.pt')
        assert active['step']==probe['step']==active['scalar']['step']
        assert active['scalar']==probe['scalar_after'] and probe['scalar_before']['step']==probe['step']-1
        assert active['seed']==probe['seed']==task['seed']
        assert probe['source_order']==[x['query_id'] for x in r.s.source(task)]
        assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
        assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
        cal=c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f'{owner}.json')
        assert probe['budget']==.1*cal['C_operator_penalty']
        assert c.read(RUN/'private/reconstruction'/f'{owner}.json')['status']=='PREFIX_PARITY_PASS'
        final=r.p.load_state(r.s.point(task,'C'))['states']['FINAL']
        j.close(float(r.mm.operator_penalty(final,cache)),cal['C_operator_penalty'],atol=1e-12)
        assert all(torch.equal(active['state'][k],probe['before'][k]) for k in r.mm.KEYS)
        assert all(torch.equal(active['rng'][k],probe['rng'][k]) for k in ('torch_rng','cuda_rng'))
        assert active['rng']['python_rng']==probe['rng']['python_rng']
        current=c.read(RUN/'private/paired'/f'{owner}.json')['arms'][a.m.BASE]['trials']
        old=c.read(Path(os.environ['BASE_PARENT'])/'private/paired'/f'{owner}.json')['arms'][a.m.BASE]['trials']
        for x,y in zip(current,old):
            if x['original_rule_pass']!=y['original_rule_pass']:changed.append([owner,x['fraction']]);continue
            if x['original_rule_pass']:
                j.close([x['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')],
                    [y['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')])
    assert changed==[[5,.25]]
    c.write(RUN/'private/BASELINE_REPAIR_ADMISSION1.json',dict(status='PASS',states_RNG_RMS_source_budget_locks=8,
        changed=changed,all_prior_leases_ended=True,new_proposals=0,new_Fisher=0,new_direction_cases=0,
        maximum_added_NN_backwards=12,maximum_added_operator_backwards=4,maximum_added_forwards=100))


def replay():
    cpu();audit()
    with (RUN/'private/BASELINE_REPAIR_ONCE1').open('x'):pass
    gpu=int(os.environ['GPU']);task=next(t for t in r.s.tasks() if t['order']==5)
    probe=r.p.load_state(RUN/'private/probes/5.pt')
    with r.p.lease(gpu):
        runtime,_,handle=r.load(gpu)
        r.old.fp32(runtime)
        sources=[r.old.batch_on(runtime,r.old.cpu(runtime.build_edit_batch(r.s.record(task,x)))) for x in r.s.source(task)]
        before={k:v.to(runtime.device) for k,v in probe['before'].items()}
        pure={k:v.to(runtime.device) for k,v in probe['pure'].items()}
        cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        expert=r.p.expert(before,task['seed'],runtime.device)
        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
        try:
            c.restore_rng(probe['rng']);p.f.numpy_restore(probe['numpy_rng'])
            now=c.rng();assert all(torch.equal(now[k].cpu(),probe['rng'][k].cpu()) for k in ('torch_rng','cuda_rng'))
            j.close(r.losses_at(runtime,hook,expert,before,a.p.g.source_batches(sources,probe['step'])),probe['source_losses'])
            with c.diagnostic_scope(hook):
                state,record=a.trial(runtime,hook,expert,before,pure,probe['step'],sources,
                    probe['source_losses'],cache,p.cache_cpu(cache),probe['budget'],a.m.BASE,.25)
            old=c.read(Path(os.environ['BASE_PARENT'])/'private/paired/5.json')['arms'][a.m.BASE]['trials'][2]
            assert state is None and not record['original_rule_pass'] and not old['original_rule_pass']
            assert len(record['attempts'])==len(old['attempts'])==4
            for x,y in zip(record['attempts'],old['attempts']):
                assert x['passed']==y['passed'] and x['solver']['status']==y['solver']['status']
                j.close([x[k] for k in ('source_gain','Fisher_ratio','norm_ratio')],
                    [y[k] for k in ('source_gain','Fisher_ratio','norm_ratio')])
            assert r.COUNTS['backwards']<=12 and r.COUNTS['operator_backwards']<=4 and r.COUNTS['forwards']<=100
            assert r.COUNTS['updates']==r.COUNTS['generation_attempts']==0
            c.write(RUN/'private/BASELINE_REPAIR1.json',dict(status='PASS',only_baseline_case=[5,.25],trial=record,
                new_direction_cases_recomputed=0,new_source_updates=0,counts=dict(r.COUNTS),epoch=time.time()))
        finally:hook.detach();handle.remove();r.save_counts()
    a.report()
    c.write(RUN/'private/BASELINE_REPAIR_EXIT1.json',dict(exit_code=0,epoch=time.time()))


if __name__=='__main__':
    try:
        if os.environ['ACTION']=='baseline_repair_cpu':cpu()
        else:replay()
    except BaseException as error:
        r.save_counts();c.write(RUN/'private/FAILURE_baseline_repair1.json',dict(error=repr(error),
            traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
