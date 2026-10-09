"""FP32-consistent source solving, frozen-candidate barrier, then heldout evaluation."""
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import torch
import fisher_probe as prior
import calibration_probe as calibration
import finite_math

q,c,p,RUN = prior.q,prior.c,prior.p,prior.RUN
BASELINE = Path(os.environ['FP32_BASELINE_PARENT'])
ARMS = ('RAW','BOUNDED','MATCHED_RAW')


def package(task):
    return RUN/'private/candidates'/f"{task['order']}.pt"


def snapshot(expert):
    return {k:v.detach().clone() for k,v in expert.state_dict().items()}


def cpu(batch):
    kwargs={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in batch.forward_kwargs().items()}
    assert kwargs['inputs_embeds'].dtype==torch.float16
    return kwargs,batch.labels.detach().cpu(),tuple(batch.target_token_ids)


def batch_on(runtime, value):
    kwargs,labels,tokens=value;dtype=next(runtime.model.parameters()).dtype
    kwargs={k:v.to(device=runtime.device,dtype=dtype if v.is_floating_point() else v.dtype) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
    labels=labels.to(runtime.device);kwargs['labels']=labels
    return SimpleNamespace(labels=labels,target_token_ids=tokens,forward_kwargs=lambda:kwargs)


def fp32(runtime):
    runtime.model.float()
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert all(v.dtype==torch.float32 for v in runtime.model.parameters() if v.is_floating_point())


def unedited(runtime,batch):
    assert not runtime.get_module(c.LAYER)._forward_hooks
    with torch.inference_mode():out=runtime.model(**batch.forward_kwargs())
    mask=batch.labels[:,1:]!=-100
    return out.logits[:,:-1][mask].detach().cpu().double(),batch.labels[:,1:][mask].cpu()


def evaluate(runtime,hook,expert,state,batches,references):
    expert.load_state_dict(state);values=[]
    for b,(ref,target) in zip(batches,references):
        value,target2=q.logits(runtime,hook,b);assert torch.equal(target,target2)
        values.append(q.compare(ref,value,target))
    return values


def progress(values):
    assert len(values)==5
    return -sum((.5 if i==0 else .125)*v['NLL_change'] for i,v in enumerate(values))


def source_one(runtime,task,basis,batches,native16,forward_count):
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert not package(task).exists(),'No implicit expert retry'
    seed_everything(task['seed']);expert=p.expert(q.d.start_state(task),task['seed'],runtime.device)
    assert all(v.dtype==torch.float32 for v in expert.parameters())
    before=snapshot(expert);hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    started=forward_count[0]
    try:
        matrix,backwards,basis_audit=prior.basis_gradients(runtime,hook,basis,task,dict(expert.named_parameters()))
        seed_everything(task['seed']);opt=p.optimizer(expert,runtime.model);opt.zero_grad(set_to_none=True)
        for b in batches[:2]:
            hook.set_teacher_routing(b.labels);(.5*runtime.compute_loss(b)).backward()
        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
        assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
        opt.step();groups=q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW')
        raw=snapshot(expert)
        delta,geometry,extra_f,extra_b=q.candidate_direction(runtime,hook,expert,batches,before,raw,matrix)
        initial=q.add(before,delta)
        assert all(torch.equal(state[k],before[k]) for state in (raw,initial) for k in ('G2','G3','G4'))
        expert.load_state_dict(before)
        refs=[q.logits(runtime,hook,b,verify_base=True) for b in batches]
        repeated=q.logits(runtime,hook,batches[0]);assert all(torch.equal(a,b) for a,b in zip(refs[0],repeated))
        raw_values=evaluate(runtime,hook,expert,raw,batches,refs)
        initial_values=evaluate(runtime,hook,expert,initial,batches,refs)
        raw_gain=progress(raw_values);initial_gain=progress(initial_values)
        def state(alpha):
            if alpha==0.:return initial
            if alpha==1.:return raw
            return q.add(initial,alpha*(q.flatten(raw)-q.flatten(initial)))
        def gain(alpha):return progress(evaluate(runtime,hook,expert,state(alpha),batches,refs))
        alpha,trace=finite_math.restore(gain,initial_gain,raw_gain)
        bounded=state(alpha)
        final_values=evaluate(runtime,hook,expert,bounded,batches,refs)
        assert progress(final_values)>=.9*raw_gain
        radius=q.fm.map_norm(before,raw);actual=q.fm.map_norm(before,bounded)
        assert actual<=radius*(1+1e-6)+1e-10
        matched,matching=q.fm.match(before,raw,actual)
        matched_values=evaluate(runtime,hook,expert,matched,batches,refs)
        states=dict(RAW=raw,BOUNDED=bounded,MATCHED_RAW=matched)
        geometry.update(initial_actual_edit_gain=initial_gain,final_actual_edit_gain=progress(final_values),
            RAW_actual_edit_gain=raw_gain,restoration_alpha=alpha,restoration_trace=trace,
            restoration_evaluations=len(trace)-2,BOUNDED_map_norm=actual,
            objective_BOUNDED_final=float((matrix@(q.flatten(bounded)-q.flatten(before))).square().sum()))
        diagnostics=[]
        edit_values=dict(RAW=raw_values,BOUNDED=final_values,MATCHED_RAW=matched_values)
        for i,(ref,target) in enumerate(refs):
            # Reuse this exact FP32 reference for precision migration on identical prefixes.
            old,target16=native16[i];assert torch.equal(target,target16)
            diagnostics.append(dict(role='EDIT',index=i,group=None,kind=None,semantic_correct=None,
                baseline=q.compare(ref,ref,target),candidates={a:edit_values[a][i] for a in ARMS},
                baseline_migration=q.compare(old,ref,target)))
        groups_basis=list(dict.fromkeys(x['source_group'] for x in basis))
        for i,row in enumerate(basis):
            c.budget();b=q.protection_batch(runtime,row,task);expert.load_state_dict(before)
            ref,target=q.logits(runtime,hook,b)
            values={a:evaluate(runtime,hook,expert,s,[b],[(ref,target)])[0] for a,s in states.items()}
            diagnostics.append(dict(role='BASIS_FIT',index=i,group=groups_basis.index(row['source_group']),kind=row['answer_kind'],semantic_correct=True,baseline=q.compare(ref,ref,target),candidates=values))
        gate=progress(matched_values)>1e-8 and sum(v['lost_correct_tokens'] for v in final_values)==0
        expert.load_state_dict(before)
        assert all(torch.equal(v,expert.state_dict()[k]) for k,v in before.items())
        assert not any(v.grad is not None for v in runtime.model.parameters())
        forwards=forward_count[0]-started+5
        assert forwards==348+5*(len(trace)-2) and backwards+2+extra_b==1959
        value=dict(status='COMPLETE',expert_order=task['order'],source_gate_passed=gate,
            geometry=geometry,function_matching=matching,basis_audit=basis_audit,diagnostics=diagnostics,
            map_norms={a:q.fm.map_norm(before,s) for a,s in states.items()},RAW_groups=groups,
            forward_calls=forwards,backward_calls=1959,zero_expert_Base_checks=5,
            state_restored_exact=True,baseline_repeat_exact=True,Base_gradient=False,candidate_optimizer_steps=1,
            lock=c.digest(c.read(RUN/'private/PROBE_LOCK.json')))
        c.save(package(task),dict(states={a:{k:v.detach().cpu() for k,v in s.items()} for a,s in dict(BASE=before,**states).items()},expert_order=task['order'],lock=value['lock']))
        c.write(RUN/'private/source_results'/f"{task['order']}.json",value)
        print('SOURCE_COMPLETE',task['order'],'GATE',gate,'EDIT_RATIO',progress(final_values)/raw_gain,'ALPHA',alpha,flush=True)
    finally:
        expert.load_state_dict(before);hook.detach()


def source_worker():
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];calibration.require_memory(gpu)
    with p.lease(gpu):
        runtime,_=c.load(gpu);tasks=q.d.selected()[part::6];basis=q.split()[0]
        frozen={row['query_id']:cpu(q.protection_batch(runtime,row,tasks[0])) for row in basis}
        native={t['order']:[cpu(runtime.build_edit_batch(replace(c.record(t),question=x))) for x in [t['native']['question']]+t['fit_questions']] for t in tasks}
        count=[0];handle=runtime.model.register_forward_pre_hook(lambda *_:count.__setitem__(0,count[0]+1))
        old={t['order']:[unedited(runtime,batch_on(runtime,b)) for b in native[t['order']]] for t in tasks}
        fp32(runtime)
        q.protection_batch=lambda runtime,row,task:batch_on(runtime,frozen[row['query_id']])
        try:
            for t in tasks:source_one(runtime,t,basis,[batch_on(runtime,b) for b in native[t['order']]],old[t['order']],count)
            assert count[0]==sum(c.read(RUN/'private/source_results'/f"{t['order']}.json")['forward_calls'] for t in tasks)
        finally:
            handle.remove();c.write(RUN/'private'/f'SOURCE_COUNTS_{part}.json',dict(forwards=count[0]))
    p.done('SOURCE_WORKER_'+str(part))


def held_worker():
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['source_gate_passed']
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];calibration.require_memory(gpu)
    with p.lease(gpu):
        runtime,_=c.load(gpu);tasks=q.d.selected()[part::6];_,held=q.split()
        frozen=[cpu(q.protection_batch(runtime,row,tasks[0])) for row in held]
        count=[0];handle=runtime.model.register_forward_pre_hook(lambda *_:count.__setitem__(0,count[0]+1))
        old=[unedited(runtime,batch_on(runtime,b)) for b in frozen]
        fp32(runtime);groups=list(dict.fromkeys(r['source_group'] for r in held))
        try:
            for t in tasks:
                dest=RUN/'private/results'/f"{t['order']}.json";assert not dest.exists()
                start=count[0];saved=torch.load(package(t),map_location='cpu',weights_only=True)
                assert saved['lock']==c.digest(c.read(RUN/'private/PROBE_LOCK.json'))
                expert=p.expert(saved['states']['BASE'],t['seed'],runtime.device)
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                diagnostics=[]
                try:
                    for i,(row,b) in enumerate(zip(held,frozen)):
                        c.budget();b=batch_on(runtime,b);expert.load_state_dict(saved['states']['BASE'])
                        ref,target=q.logits(runtime,hook,b,verify_base=True)
                        if i==0:
                            repeat=q.logits(runtime,hook,b);assert torch.equal(ref,repeat[0]) and torch.equal(target,repeat[1])
                        original,target16=old[i];assert torch.equal(target,target16)
                        values={};endtoend={}
                        for a in ARMS:
                            expert.load_state_dict(saved['states'][a]);z,target2=q.logits(runtime,hook,b);assert torch.equal(target,target2)
                            values[a]=q.compare(ref,z,target);endtoend[a]=q.compare(original,z,target)
                        diagnostics.append(dict(role='HELDOUT_FIT',index=i,group=groups.index(row['source_group']),kind=row['answer_kind'],semantic_correct=row['semantic_correct'],
                            baseline=q.compare(ref,ref,target),candidates=values,baseline_migration=q.compare(original,ref,target),end_to_end=endtoend))
                    expert.load_state_dict(saved['states']['BASE'])
                    assert all(torch.equal(v.cpu(),saved['states']['BASE'][k]) for k,v in expert.state_dict().items())
                finally:hook.detach()
                assert not any(v.grad is not None for v in runtime.model.parameters())
                assert count[0]-start==481
                c.write(dest,dict(status='COMPLETE',expert_order=t['order'],diagnostics=diagnostics,
                    forward_calls=481,backward_calls=0,zero_expert_Base_checks=96,state_restored_exact=True,baseline_repeat_exact=True,Base_gradient=False,lock=saved['lock']))
                print('HELDOUT_COMPLETE',t['order'],flush=True)
        finally:
            handle.remove();c.write(RUN/'private'/f'HELD_COUNTS_{part}.json',dict(forwards=count[0],shared_FP16_Base_forwards=96))
    p.done('HELD_WORKER_'+str(part))


def freeze():
    values=[c.read(RUN/'private/source_results'/f"{t['order']}.json") for t in q.d.selected()]
    assert len(values)==8 and all(package(t).is_file() for t in q.d.selected())
    assert not list((RUN/'private/results').glob('*.json')) and not list((RUN/'private').glob('FAILURE*'))
    passed=all(v['source_gate_passed'] for v in values)
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(status='FROZEN',source_gate_passed=passed,experts=8,
        source_receipts=c.digest(values),packages=[str(package(t)) for t in q.d.selected()],epoch=time.time()))
    return passed


def plan():
    prior.plan();finite_math.selfcheck()
    qualification=c.read(BASELINE/'public/REQUALIFICATION_RESULTS.json')
    assert qualification['decision']=='ORIGINAL_QUALIFICATION_NOT_PRESERVED'
    assert qualification['panels']['BASIS']['correct']==61 and qualification['panels']['PRIMARY63']['correct']==62
    basis,held=q.split();assert len(basis)==61 and len(held)==96 and sum(r['semantic_correct'] for r in held)==63
    config=next(iter(c.read(RUN/'private/EVAL_BINDINGS.json').values()))['generation']
    assert config['max_new_tokens']==1024
    assert all(c.read(row['response_path'])['binding']['judge_input']['generation']==config for row in basis+held)
    lock=c.read(RUN/'private/PROBE_LOCK.json')
    lock.update(algorithm='FP32_FISHER_FINITE_EDIT_RESTORATION',precision='MATH_FP32_FROZEN_FP16_PREFILL',
        source_forward_minimum=2784,source_forward_maximum=3424,heldout_forward_calls=4424,
        forward_calls_minimum=7208,forward_calls_maximum=7848,backward_calls=15672,
        forward_calls=None,restoration_bisections=16,original_primary_denominator=63,FP32_primary_correct=62,
        old_qualification_failure_preserved=True,all_candidates_frozen_before_heldout=True,
        primary_prefix='ORIGINAL_FP16_GENERATED_TOKENS_FIXED_FOR_ALL_COMPARISONS',
        matching='TT_MAP_NORM_NOT_EDIT_GAIN',temporary_candidate_packages=8,
        conditional_generation_consumer_maximum=2464,conditional_generation_registered_now=False)
    c.write(RUN/'private/PROBE_LOCK.json',lock)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',selfcheck=finite_math.selfcheck(),
        **{k:v for k,v in lock.items() if k not in ('role_binding','code','split_binding')}))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('constrained.py','constrained_source',g,i) for i,g in enumerate(q.GPUS)])
    if freeze():pipeline.wait([pipeline.launch('constrained.py','constrained_held',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('constrained_report.py','constrained_report')])


if __name__=='__main__':
    try:{'constrained_plan':plan,'constrained_source':source_worker,'constrained_held':held_worker,'constrained_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),implicit_retry=False))
        raise
