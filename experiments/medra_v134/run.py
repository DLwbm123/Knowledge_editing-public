"""Reuse frozen raw MED160 states, rebuild its operator, scale once, then screen."""
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT'])
PARENT=Path(os.environ['PARENT_RUN'])
RAW=Path(os.environ['RAW_MED_ROOT'])
LOCK=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
CLOCK=json.loads((ROOT/'CLOCK.json').read_text())
COUNTS=dict(forwards=0,backwards=0,operator_backwards=0,source_forwards=0,updates=0,
            alpha_selections=0,generations=0,vision_forwards=0,LM_input_positions=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'STOP.json').exists()
    assert time.time()-json.loads((ROOT/'GUARD_HEALTH.json').read_text())['time']<45
    assert all(COUNTS.get(k,0)<=v for k,v in LOCK['budgets'].items() if k in COUNTS)


def save(name,value):
    import torch
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(p)


def main():
    if (ROOT/'WORKER_ONCE').exists():
        request=json.loads((ROOT/'RECOVERY_REQUEST.json').read_text())
        assert request['old_worker_exited'] and request['old_guard_exited']
        assert request['clock']==CLOCK and request['input_binding']==LOCK['input_binding']
        COUNTS.update(request['prior_counts'])
    else:
        with (ROOT/'WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Resource guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original wall clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import med133_curvature as curvature
    import ra_risk_math as risk_math
    import ra_curvature_rebuild as curvature_rebuild
    import ra_evaluate as evaluate
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget
    torch.set_num_threads(1);torch.manual_seed(LOCK['seed']);torch.cuda.manual_seed_all(LOCK['seed'])
    torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['selected_physical_GPU'])
    torch.cuda.set_per_process_memory_fraction(.78)
    inputs=r.c.read(ROOT/'INPUTS_FROZEN.json');assert r.c.digest(inputs)==LOCK['input_binding']
    raw=r.p.load_state(RAW/'retained/MED146.pt')
    assert len(raw['states'])==len(raw['R0_entries'])==146
    for key,state in raw['states'].items():assert evaluate.state_id(state)==raw['state_hashes'][key]
    write('RAW_BASELINE_ADMISSION.json',dict(status='PASS',actual_states146_hash_match=True,
        inherited_updates=23360,new_gradient_updates=0,raw_source_commit=raw['source_version']['commit']))
    write('RISK_CPU_CHECK.json',risk_math.selfcheck(r.mm.factors))
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config==next(iter(bindings.values()))['generation']
    def count(_module,args,kwargs):
        budget();COUNTS['forwards']+=1
        x=kwargs.get('inputs_embeds',kwargs.get('input_ids'))
        if x is None and args:x=args[0]
        if isinstance(x,torch.Tensor):COUNTS['LM_input_positions']+=x.shape[0]*x.shape[1]
    fh=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    caches=curvature_rebuild.rebuild(runtime,r,inputs,ROOT,PARENT,LOCK,COUNTS,budget,write,save)
    med=caches['MED'];del caches
    for part in med['parts']:
        for key in ('x','y'):part[key]=part[key].to(runtime.device)
    trace=risk_math.total_trace(med,curvature)
    write('F_MED_TRACE.json',dict(trace=trace,anchor_count=61,mapped=31,fallback=30,
        trace_is_sum_of_anchor_traces=True,original_lambda_and_C_endpoints_unchanged=True))
    for task in inputs['tasks']:
        budget();owner=task['order'];dest=ROOT/'raw_risk'/f'{owner}.json'
        if dest.exists():
            assert r.c.read(dest)['raw_state_hash']==raw['state_hashes'][task['edit_id']];continue
        state=raw['states'][task['edit_id']]
        norm2=risk_math.norm2(state,r.mm.factors)
        with torch.no_grad():
            rows=risk_math.projections({k:v.to(runtime.device) for k,v in state.items()},med,r.mm.factors)
            energy=sum(float(x.square().sum()) for x in rows)
        q=energy/(trace*norm2+risk_math.EPSILON)
        assert math.isfinite(q) and 0<=q<=1+3e-5
        anchors=r.c.read(ROOT/'CURVATURE_ADMISSION.json')['anchors']
        linear=[float(value[0])/math.sqrt(a['full_trace']/a['MED_unscaled_direction_square'])
                for value,a in zip(rows[30:],anchors)]
        write(f'raw_risk/{owner}.json',dict(owner=owner,raw_state_hash=raw['state_hashes'][task['edit_id']],
            raw_operator_norm2=norm2,energy=energy,trace=trace,q=q,medical_score_linear_proxy=linear,
            raw_lambda=r.c.read(RAW/'calibration'/f'{owner}.json')['coefficient'],
            common_mask_positions=sum(med['parts'][0]['token_counts']),
            mapped_mask_positions=sum(med['parts'][1]['token_counts']),input_binding=LOCK['input_binding']))
    values=[r.c.read(ROOT/'raw_risk'/f"{t['order']}.json")['q'] for t in inputs['tasks']]
    q0=float(torch.quantile(torch.tensor(values,dtype=torch.float64),.75,interpolation='linear'))
    if (ROOT/'Q0_FROZEN.json').exists():assert r.c.read(ROOT/'Q0_FROZEN.json')['raw_q']==values
    else:write('Q0_FROZEN.json',dict(q0=q0,raw_q=values,quantile=.75,interpolation='linear',training_calibration_only=True))
    prepared,batches=None,None
    for task in inputs['tasks']:
        owner=task['order'];original=raw['states'][task['edit_id']]
        final=ROOT/'candidates'/f'MED_RA_{owner}.pt';ledgerfile=ROOT/'selection'/f'{owner}.json'
        if final.exists() and ledgerfile.exists():
            ledger=r.c.read(ledgerfile)
            assert ledger['q0']==q0 and evaluate.state_id(r.p.load_state(final))==ledger['state_hash'];continue
        runtime.model.half()
        try:prepared=r.s.prepare(runtime,bindings,task,inputs['source_rows'][str(owner)],True)
        finally:r.old.fp32(runtime)
        batches=[r.old.batch_on(runtime,v[1]) for v in prepared]
        expert=r.p.expert(original,task['seed'],runtime.device).requires_grad_(False)
        hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
        support_counts=[]
        for b in batches:
            hook.set_teacher_routing(b.labels)
            expected=torch.zeros_like(b.labels,dtype=torch.bool);expected[:,:-1]=b.labels[:,1:]!=-100
            assert torch.equal(hook.token_mask,expected);support_counts.append(int(expected.sum()))
        def losses(key,state):
            path=ROOT/'source_eval'/f'{owner}_{key}.json'
            prior=r.c.read(path) if path.exists() else dict(losses=[],input_binding=LOCK['input_binding'])
            assert prior['input_binding']==LOCK['input_binding']
            hook.enabled=state is not None
            if state is not None:expert.load_state_dict(state)
            result=list(prior['losses'])
            with torch.no_grad():
                for b in batches[len(result):]:
                    budget();hook.set_teacher_routing(b.labels);COUNTS['source_forwards']+=1
                    value=float(runtime.compute_loss(b));assert math.isfinite(value)
                    result.append(value);write(str(path.relative_to(ROOT)),dict(losses=result,input_binding=LOCK['input_binding']))
                    write('PROGRESS.json',dict(phase='SOURCE_CALIBRATION',owner=owner,state=key,
                        completed_sources=len(result),counts=COUNTS,time=time.time()))
            return result
        try:
            with PrefixCache(runtime,batches):
                base=losses('Base',None);end=losses('raw',original)
                risk=r.c.read(ROOT/'raw_risk'/f'{owner}.json')
                alpha,trials,violation=risk_math.select(risk['q'],q0,base,end,
                    lambda a:losses(str(a),risk_math.scaled(original,a)))
                selected=risk_math.scaled(original,alpha)
                selected_losses=end if alpha==1 else losses(str(alpha),selected)
                with torch.no_grad():
                    responses=risk_math.projections({k:v.to(runtime.device) for k,v in selected.items()},med,r.mm.factors)
                    actual_energy=sum(float(x.square().sum()) for x in responses)
                assert abs(actual_energy/(alpha*alpha*risk['energy'])-1)<=3e-5 if risk['energy']>0 else actual_energy==0
                final_norm2=risk_math.norm2(selected,r.mm.factors)
                assert abs(final_norm2/(risk['raw_operator_norm2']*alpha*alpha)-1)<=3e-5
                state_hash=evaluate.state_id(selected)
                ledger=dict(owner=owner,raw_state_hash=risk['raw_state_hash'],state_hash=state_hash,
                    alpha=alpha,raw_lambda=risk['raw_lambda'],q=risk['q'],q0=q0,trace=trace,
                    raw_operator_norm=math.sqrt(risk['raw_operator_norm2']),final_operator_norm=math.sqrt(final_norm2),
                    raw_energy=risk['energy'],final_energy=actual_energy,
                    final_risk=actual_energy/(trace*risk['raw_operator_norm2']+risk_math.EPSILON),
                    base_losses=base,raw_losses=end,final_losses=selected_losses,
                    source_gain=[b-f for b,f in zip(base,selected_losses)],source_floor_pass=risk_math.source_ok(base,end,selected_losses),
                    risk_pass=actual_energy/(trace*risk['raw_operator_norm2']+risk_math.EPSILON)<=q0*(1+3e-5)+1e-20,
                    violation_fallback=violation,trials=trials,source_mask_counts=support_counts,
                    common_mask_positions=risk['common_mask_positions'],mapped_mask_positions=risk['mapped_mask_positions'],
                    medical_score_linear_proxy_raw=risk['medical_score_linear_proxy'],
                    medical_score_linear_proxy_final=[alpha*v for v in risk['medical_score_linear_proxy']],
                    actual_direction_tensor_identity=evaluate.state_id(selected),input_binding=LOCK['input_binding'])
                if final.exists():assert evaluate.state_id(r.p.load_state(final))==state_hash
                else:save(str(final.relative_to(ROOT)),selected)
                assert evaluate.state_id(r.p.load_state(final))==state_hash
                write(str(ledgerfile.relative_to(ROOT)),ledger)
                COUNTS['alpha_selections']+=1
                write('PROGRESS.json',dict(phase='SELECT',owner=owner,alpha=alpha,counts=COUNTS,time=time.time()))
                print('SELECT',owner,alpha,violation,flush=True)
        finally:hook.detach()
    assert len(list((ROOT/'candidates').glob('MED_RA_*.pt')))==146
    write('ALL_CANDIDATES_FROZEN.json',dict(time=time.time(),experts=146,counts=COUNTS,new_gradient_updates=0,
                                         inherited_raw_training_updates=23360))
    del med,raw,prepared,batches
    torch.cuda.empty_cache()
    evaluate.run(runtime,r,inputs,COUNTS,budget,write,ROOT,PARENT,LOCK)
    fh.remove();vh.remove()
    write('WORKER_COMPLETE.json',dict(time=time.time(),counts=COUNTS,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write('FAILURE.json',dict(time=time.time(),error=repr(error),traceback=traceback.format_exc(),counts=COUNTS));raise
    finally:write('WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
