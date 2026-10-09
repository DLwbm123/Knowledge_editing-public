"""Fixed paired TT cold-start optimization, source-only diagnostics and generation."""
import os
import time
import traceback
from pathlib import Path
import torch
import raw32 as short

q, c, p, RUN, old = short.q, short.c, short.p, short.RUN, short.old
PARENT = Path(os.environ['SHORT32_PARENT'])
ARMS = ('RAW', 'ADAM')
STEPS = 160
NODES = (0, 1, 8, 16, 32, 64, 96, 128, 160)


def jobs():
    return [(t, arm) for t in q.d.selected() for arm in ARMS]


def token_stats(logits, target):
    ix = torch.arange(len(target)); score = logits[ix, target]
    competitors = logits.clone(); competitors[ix, target] = -torch.inf
    margin = score - competitors.max(-1).values
    predicted = logits.argmax(-1); wrong = torch.where(predicted != target)[0]
    return dict(target_ids=target.tolist(), predicted_ids=predicted.tolist(),
        target_rank=(1+(logits > score[:, None]).sum(-1)).tolist(),
        target_margin=margin.tolist(), wrong_tokens=int(len(wrong)),
        first_wrong_index=int(wrong[0]) if len(wrong) else None,
        minimum_margin=float(margin.min()), mean_margin=float(margin.mean()))


def selfcheck():
    q.d.selfcheck()
    x=torch.tensor([[3.,2.,1.],[0.,4.,2.]],dtype=torch.float64); y=torch.tensor([0,2])
    s=token_stats(x,y)
    assert s['target_rank']==[1,2] and s['target_margin']==[1.,-2.] and s['first_wrong_index']==1
    ps=[torch.nn.Parameter(torch.tensor([1.,-2.]))]
    opt=torch.optim.Adam(ps,lr=.01)
    for _ in range(2):
        ps[0].grad=torch.tensor([.2,-.7]);before={id(v):v.detach().clone() for v in ps}
        opt.step();expected=[v.detach().clone() for v in ps]
        q.d.apply_direction(opt,before,'ADAM')
        assert all(torch.equal(v,e) for v,e in zip(ps,expected))
    j=[(i,a) for i in range(8) for a in ARMS];parts=[j[k::6] for k in range(6)]
    assert sum(map(len,parts))==16 and len(set(sum(parts,[])))==16 and all(parts)
    return dict(status='PASS',token_diagnostics=True,Adam_write_unchanged=True,RAW_original_selfcheck=True,six_disjoint_partitions=True)


def plan():
    checks=selfcheck();tasks=q.d.selected();assert len(tasks)==8
    previous=c.read(PARENT/'public/RAW32_RESULTS.json')
    assert previous['decision']=='STOP_RAW32_NO_NATIVE_REPAIR'
    assert c.read(PARENT/'public/RAW32_REVIEW_AUDIT.json')['status']=='PASS'
    rows=[]
    for t in tasks:
        diagnostics=c.read(PARENT/'private/raw_results'/f"{t['order']}.json")['diagnostics']
        for i,question in enumerate([t['native']['question']]+t['fit_questions']):
            path=PARENT/'private/outputs/RAW32'/str(t['order'])/f"edit_{t['order']}_{i}.json"
            d=c.read(path);b=c.read(d['baseline_path']);binding=d['binding']['judge_input']
            assert binding==b['binding']['judge_input'] and binding['question']==question
            assert binding['reference']==c.record(t).target==t['native']['reference']
            assert d['binding']['input']['image_sha256']==b['binding']['input']['image_sha256']
            v=diagnostics[i]
            rows.append(dict(order=t['order'],index=i,question=question,reference=binding['reference'],
                base_answer=b['R0']['raw_answer'],RAW32_answer=d['R0']['raw_answer'],diagnostics=v,path=str(path)))
    assert len(rows)==40 and all(x['diagnostics']['NLL_change']<0 for x in rows)
    c.write(RUN/'private/SUPERVISION_AUDIT.json',dict(status='PASS',rows=rows,labels_unchanged=True))
    c.write(RUN/'public/SUPERVISION_AUDIT.json',dict(status='PASS',bound_questions=40,
        training_target_equals_frozen_reference=40,all40_NLL_improved=True,
        native_remaining_wrong_tokens=[x['diagnostics']['tokens']-x['diagnostics']['correct_tokens'] for x in rows if x['index']==0],
        checkpoint_reconstruction=False,new_model_calls=0,new_Judge=0,
        limitation='Binding and answer audit; not clinical relabeling or proof of token-level generation alignment'))
    lock=dict(arms=ARMS,steps=STEPS,diagnostic_nodes=NODES,experts=8,trajectories=16,
        trainable_parameters=7168,trainable_cores=list(q.KEYS),initial_state='ORIGINAL_SEEDED_ZERO_TT88',
        optimizer_steps=2560,training_backwards=5120,source_non_generation_forwards=6092,
        new_generations=80,maximum_forwards=6092+80*1024,maximum_new_Judge=80,
        optimizer='same two-group Adam; RAW replaces direction, ADAM retains actual proposal',
        loss='.5 native + .5 cyclic FIT',clip=1.,native_updates_per_trajectory=160,FIT_updates_per_question=40,
        learning_rates=[1e-4,1e-3],seed_selection=False,checkpoint_selection=False,
        source_roles=['NATIVE','FIT'],BASIS_access=False,heldout_access=False,
        generation='MATH_FP32_FROZEN_FP16_PREFILL_GREEDY_1024',
        mechanical=dict(prefix_logits_rtol=1e-5,prefix_logits_atol=5e-4,terminal_all_native_target_prefixes=True),
        final_endpoint_only=True,automatic_protection_training=False,automatic_step_extension=False,epoch=time.time())
    c.write(RUN/'private/OPTIMIZER160_LOCK.json',lock)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**lock,selfcheck=checks))
    p.done('OPTIMIZER160_PLAN_COMPLETE')


def diagnose(runtime,hook,batches,refs):
    values=[];native=None
    for i,(b,(reference,target)) in enumerate(zip(batches,refs)):
        logits,target2=q.logits(runtime,hook,b);assert torch.equal(target,target2)
        values.append(dict(**q.compare(reference,logits,target),token_diagnostics=token_stats(logits,target)))
        if i==0:native=(logits,target)
    return values,native


def prefix_check(runtime,hook,batch,reference):
    logits,target=reference;first=int(torch.where(batch.labels[0]!=-100)[0][0])
    full=dict(batch.forward_kwargs());full.pop('labels',None);errors=[]
    with torch.inference_mode(),hook.generation_request():
        for j in range(len(target)):
            kwargs=dict(full)
            for name in ('inputs_embeds','attention_mask','position_ids'):
                if isinstance(kwargs.get(name),torch.Tensor):kwargs[name]=kwargs[name][:,:first+j]
            kwargs['use_cache']=False
            result=runtime.model(**kwargs).logits[0,-1].detach().cpu().double()
            assert torch.allclose(logits[j],result,rtol=1e-5,atol=5e-4),'Terminal native prefix parity failed'
            trace=hook.generation_trace[-1]
            assert trace['first_active_predictor']==first-1 and trace['active_predictor_count']==j+1
            assert trace['active_residual_norm']>0
            errors.append(float((logits[j]-result).abs().max()))
    return dict(status='PASS',target_prefixes=len(target),maximum_logit_difference=max(errors),
        maximum_by_prefix=errors,generation_trace=list(hook.last_generation_trace),
        teacher_target_prefixes=True,actual_free_generation_prefixes=False,use_cache=False)


def worker():
    from m3bench_repro.editors.llava_runtime import seed_everything
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    lock=c.read(RUN/'private/OPTIMIZER160_LOCK.json');totals=dict(forwards=0,backwards=0,updates=0,generations=0)
    assigned=jobs()[part::6]
    with p.lease(gpu):
        runtime,_=c.load(gpu);tasks=list({t['order']:t for t,arm in assigned}.values())
        frozen=short.prepare(runtime,tasks);old.fp32(runtime)
        handle=runtime.model.register_forward_pre_hook(lambda *_:totals.__setitem__('forwards',totals['forwards']+1))
        try:
            for t,arm in assigned:
                result_path=RUN/'private/results'/arm/f"{t['order']}.json"
                assert not result_path.exists(),'No implicit trajectory restart'
                seed_everything(t['seed']);expert=p.expert(q.d.start_state(t),t['seed'],runtime.device);base=old.snapshot(expert)
                assert sum(v.numel() for v in expert.parameters())==7168 and all(v.requires_grad for v in expert.parameters())
                assert torch.count_nonzero(expert.G1)==0
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach();start=totals['forwards']
                batches=[old.batch_on(runtime,x[1]) for x in frozen[t['order']]]
                try:
                    target_binding=[]
                    for b,(_,_,raw,expanded,binding,_) in zip(batches,frozen[t['order']]):
                        first=int(torch.where(b.labels[0]!=-100)[0][0])
                        assert first==expanded[4].shape[1]
                        assert torch.equal(b.forward_kwargs()['inputs_embeds'][:,:first].cpu(),expanded[4].float())
                        assert binding['reference']==c.record(t).target
                        target_binding.append(dict(first_target_position=first,target_ids=list(b.target_token_ids),
                            decoded_target=runtime.adapter.tokenizer.decode(b.target_token_ids,skip_special_tokens=True),reference=binding['reference']))
                    refs=[q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                    checkpoints=[dict(step=0,values=[dict(**q.compare(v,v,y),token_diagnostics=token_stats(v,y)) for v,y in refs])]
                    opt=p.optimizer(expert,runtime.model);curve=[];mechanical=None;native=None
                    for step in range(1,STEPS+1):
                        c.budget();before=old.snapshot(expert);opt.zero_grad(set_to_none=True);losses={}
                        for i in (0,1+(step-1)%4):
                            hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i])
                            assert torch.isfinite(loss);losses[str(i)]=float(loss.detach())
                            (.5*loss).backward();totals['backwards']+=1
                        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                        assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                        opt.step();groups=q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},arm)
                        totals['updates']+=1
                        curve.append(dict(step=step,pre_update_loss=losses,preclip_norm=float(norm),groups=groups,
                            core_update_norms={k:float((v-before[k]).norm()) for k,v in expert.state_dict().items()}))
                        if step==1:mechanical=short.mechanical(runtime,hook,batches[0])
                        if step in NODES:
                            values,native=diagnose(runtime,hook,batches,refs);checkpoints.append(dict(step=step,values=values))
                            c.write(RUN/'private/progress'/arm/f"{t['order']}.json",dict(step=step,curve=curve,checkpoints=checkpoints))
                            print('OPTIMIZER160_TRAIN',arm,t['order'],step,flush=True)
                    assert all(any(v['core_update_norms'][k]>0 for v in curve) for k in q.KEYS)
                    terminal=prefix_check(runtime,hook,batches[0],native)
                    source_forwards=totals['forwards']-start
                    assert source_forwards==368+len(native[1])
                    c.write(result_path,dict(status='COMPLETE',arm=arm,expert_order=t['order'],seed=t['seed'],
                        curve=curve,checkpoints=checkpoints,mechanical=mechanical,terminal_prefix_check=terminal,
                        target_binding=target_binding,edit_gain=old.progress(checkpoints[-1]['values']),
                        source_forwards=source_forwards,backwards=320,updates=160,all_four_cores_updated=True,lock=c.digest(lock)))
                    for i,(row,_,raw,expanded,b,parent_path) in enumerate(frozen[t['order']]):
                        dest=RUN/'private/outputs'/arm/str(t['order'])/(row['query_id']+'.json');assert not dest.exists()
                        before_gen=totals['forwards'];answer,trace=short.generate(runtime,hook,raw,expanded);totals['generations']+=1
                        c.write(dest,dict(arm=arm,expert_order=t['order'],index=i,role='EDIT',
                            binding=dict(input=dict(row,image_sha256=raw['image_sha256']),judge_input=b),
                            R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),
                            baseline_path=str(parent_path),generation_trace=trace,forwards=totals['forwards']-before_gen,
                            EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024,
                            lock=c.digest(lock),execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
                    expert.load_state_dict(base);assert all(torch.equal(v,base[k]) for k,v in expert.state_dict().items())
                    p.done(f'OPTIMIZER160_CONSUMED_{arm}_{t["order"]}',dict(generations=5,persistent_checkpoints=0))
                finally:hook.detach()
                assert not any(v.grad is not None for v in runtime.model.parameters())
        finally:handle.remove();c.write(RUN/'private'/f'OPTIMIZER160_COUNTS_{part}.json',totals)
    p.done('OPTIMIZER160_WORKER_'+str(part))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('optimizer160.py','optimizer160_worker',gpu,i) for i,gpu in enumerate(q.GPUS)])
    counts=[c.read(RUN/'private'/f'OPTIMIZER160_COUNTS_{i}.json') for i in range(6)]
    outputs=[c.read(x) for x in (RUN/'private/outputs').glob('*/*/*.json')]
    assert len(outputs)==80 and sum(x['updates'] for x in counts)==2560 and sum(x['backwards'] for x in counts)==5120
    assert sum(x['forwards'] for x in counts)==6092+sum(x['forwards'] for x in outputs)<=88012
    c.write(RUN/'private/OPTIMIZER160_GENERATION_COMPLETE.json',dict(status='COMPLETE',counts=counts,epoch=time.time()))
    pipeline.wait([pipeline.launch('optimizer_queue.py','optimizer160_ingest')])
    while not (RUN/'private/judge_optimizer160_astra_medium/ALL_WORKERS_COMPLETE.json').exists():time.sleep(10)
    pipeline.wait([pipeline.launch('optimizer_queue.py','optimizer160_report')])


if __name__=='__main__':
    try:{'optimizer160_plan':plan,'optimizer160_worker':worker,'optimizer160_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
