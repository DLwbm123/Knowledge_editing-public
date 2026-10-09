"""Fixed full-TT RAW32 competence gate before expensive protected training."""
import os
import time
import traceback
from pathlib import Path
from dataclasses import replace
import torch
import constrained as old

q,c,p,RUN=old.q,old.c,old.p,old.RUN
BASELINE=Path(os.environ['GENERATION_PARENT'])
STEPS=32


def point(t):return RUN/'private/raw_candidates'/f"{t['order']}.pt"


def plan():
    q.d.selfcheck()
    qualification=c.read(BASELINE/'public/GEN_RECOVERY_RESULTS.json')
    assert qualification['decision']=='EDIT_BASELINE_QUALIFIED' and qualification['panels']['NATIVE']==dict(queries=8,correct=0,wrong=8,missing=0)
    assert c.read(BASELINE/'public/GEN_RECOVERY_AUDIT.json')['status']=='PASS'
    lock=dict(steps=STEPS,experts=8,trainable_parameters=7168,trainable_cores=list(q.KEYS),
        optimizer_steps=256,training_forwards=512,training_backwards=512,source_non_generation_forwards=616,
        maximum_forwards=616+40*1024,new_generations=40,maximum_new_Judge=40,
        roles=['native','FIT'],heldout_access=False,geometry_access=False,
        initial_state='ORIGINAL_SEEDED_ZERO_TT88',optimizer='original two-group Adam step norms, raw-gradient directions',
        loss='.5 native + .5 cyclic FIT',clip=1.,all_experts_preserved=True,
        generation='MATH_FP32_FROZEN_FP16_PREFILL_GREEDY_1024',
        mechanical=dict(prefix_logits_rtol=1e-5,prefix_logits_atol=5e-4,nonzero_generation_residual_required=True),
        next_gate='all native scores present and at least one of eight originally wrong native answers repaired',
        qualification_source=c.read(BASELINE/'private/RECOVERY_SOURCE_VERSION.json'),epoch=time.time())
    c.write(RUN/'private/SHORT32_LOCK.json',lock)
    c.write(RUN/'public/RAW32_ADMISSION.json',{k:v for k,v in lock.items() if k!='qualification_source'})
    p.done('RAW32_PLAN_COMPLETE')


def prepare(runtime,tasks):
    result={};original=runtime.llava_model().prepare_inputs_labels_for_multimodal
    for t in tasks:
        values=[]
        for i,question in enumerate([t['native']['question']]+t['fit_questions']):
            row=dict(t['native'],question=question,query_id=f"edit_{t['order']}_{i}")
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),question,None)
            baseline=BASELINE/'private/outputs'/str(t['order'])/'BASE'/(row['query_id']+'.json')
            b=c.read(baseline)['binding']['judge_input']
            assert raw['input_ids'][0].tolist()==b['prompt_ids'] and raw['image_sha256']==b['image_sha256'] and runtime.generation_config==b['generation']
            batch=old.cpu(runtime.build_edit_batch(replace(c.record(t),question=question)))
            with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert expanded[4].dtype==torch.float16
            expanded=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded)
            values.append((row,batch,raw,expanded,b,baseline))
        result[t['order']]=values
    return result


def mechanical(runtime,hook,batch):
    teacher,target=q.logits(runtime,hook,batch)
    first=int(torch.where(batch.labels[0]!=-100)[0][0]);kwargs=dict(batch.forward_kwargs());kwargs.pop('labels',None)
    for name in ('inputs_embeds','attention_mask','position_ids'):
        if isinstance(kwargs.get(name),torch.Tensor):kwargs[name]=kwargs[name][:,:first]
    with torch.inference_mode(),hook.generation_request():out=runtime.model(**kwargs)
    trace=hook.last_generation_trace;assert len(trace)==1 and trace[0]['active_predictor_count']==1 and trace[0]['first_active_predictor']==first-1
    assert trace[0]['active_residual_norm']>0 and torch.isfinite(torch.tensor(trace[0]['active_residual_norm']))
    generated=out.logits[0,-1].detach().cpu().double()
    assert torch.allclose(teacher[0],generated,rtol=1e-5,atol=5e-4),'Teacher/generation prefix mismatch; do not relax tolerance'
    return dict(status='PASS',generation_trace=list(trace),logits_max_abs=float((teacher[0]-generated).abs().max()),same_prefix=True)


def generate(runtime,hook,raw,expanded):
    model=runtime.llava_model();original=model.prepare_inputs_labels_for_multimodal;used=[0]
    def prepared(*args,**kwargs):
        if args[0] is not None and torch.equal(args[0],raw['input_ids']):
            assert args[3] is None and args[4] is None;used[0]+=1;assert used[0]==1
            return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in expanded)
        return original(*args,**kwargs)
    model.prepare_inputs_labels_for_multimodal=prepared
    try:
        with torch.inference_mode(),hook.generation_request():answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
    finally:model.prepare_inputs_labels_for_multimodal=original
    assert used[0]==1 and answer.raw_token_ids
    trace=list(hook.last_generation_trace)
    assert trace and any(x['active_residual_norm']>0 for x in trace)
    return answer,trace


def worker():
    from m3bench_repro.editors.llava_runtime import seed_everything
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    lock=c.read(RUN/'private/SHORT32_LOCK.json');totals=dict(forwards=0,backwards=0,updates=0,generations=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);tasks=q.d.selected()[part::6];frozen=prepare(runtime,tasks);old.fp32(runtime)
        handle=runtime.model.register_forward_pre_hook(lambda *_:totals.__setitem__('forwards',totals['forwards']+1))
        try:
            for t in tasks:
                assert not point(t).exists(),'No implicit training restart'
                seed_everything(t['seed']);expert=p.expert(q.d.start_state(t),t['seed'],runtime.device);base=old.snapshot(expert)
                assert sum(v.numel() for v in expert.parameters())==7168 and all(v.requires_grad for v in expert.parameters()) and torch.count_nonzero(expert.G1)==0
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach();start=totals['forwards']
                batches=[old.batch_on(runtime,x[1]) for x in frozen[t['order']]]
                try:
                    refs=[q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                    opt=p.optimizer(expert,runtime.model);curve=[];check=None
                    for step in range(1,STEPS+1):
                        c.budget();before=old.snapshot(expert);opt.zero_grad(set_to_none=True)
                        for i in (0,1+(step-1)%4):
                            hook.set_teacher_routing(batches[i].labels);(.5*runtime.compute_loss(batches[i])).backward();totals['backwards']+=1
                        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                        assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                        opt.step();groups=q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW');totals['updates']+=1
                        curve.append(dict(step=step,groups=groups,core_update_norms={k:float((v-before[k]).norm()) for k,v in expert.state_dict().items()}))
                        if step==1:check=mechanical(runtime,hook,batches[0])
                        if step%8==0:print('RAW32_TRAIN',t['order'],step,flush=True)
                    assert all(any(v['core_update_norms'][k]>0 for v in curve) for k in q.KEYS)
                    state=old.snapshot(expert);diagnostics=old.evaluate(runtime,hook,expert,state,batches,refs)
                    assert totals['forwards']-start==77
                    c.save(point(t),dict(states=dict(BASE={k:v.cpu() for k,v in base.items()},RAW={k:v.cpu() for k,v in state.items()}),
                        expert_order=t['order'],seed=t['seed'],lock=c.digest(lock)))
                    c.write(RUN/'private/raw_results'/f"{t['order']}.json",dict(status='COMPLETE',expert_order=t['order'],
                        curve=curve,mechanical=check,diagnostics=diagnostics,edit_gain=old.progress(diagnostics),
                        source_forwards=77,backwards=64,updates=32,all_four_cores_updated=True,lock=c.digest(lock)))
                    for i,(row,_,raw,expanded,b,parent_path) in enumerate(frozen[t['order']]):
                        dest=RUN/'private/outputs/RAW32'/str(t['order'])/(row['query_id']+'.json');assert not dest.exists()
                        start_gen=totals['forwards'];answer,trace=generate(runtime,hook,raw,expanded);totals['generations']+=1
                        c.write(dest,dict(arm='RAW32',expert_order=t['order'],index=i,role='EDIT',
                            binding=dict(input=dict(row,image_sha256=raw['image_sha256']),judge_input=b),
                            R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),
                            baseline_path=str(parent_path),generation_trace=trace,forwards=totals['forwards']-start_gen,
                            EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024,
                            lock=c.digest(lock),execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
                    expert.load_state_dict(base);assert all(torch.equal(v,base[k]) for k,v in expert.state_dict().items())
                finally:hook.detach()
                assert not any(v.grad is not None for v in runtime.model.parameters())
        finally:handle.remove();c.write(RUN/'private'/f'RAW32_COUNTS_{part}.json',totals)
    p.done('RAW32_WORKER_'+str(part))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('raw32.py','raw32_worker',gpu,i) for i,gpu in enumerate(q.GPUS)])
    counts=[c.read(RUN/'private'/f'RAW32_COUNTS_{i}.json') for i in range(6)]
    assert sum(x['generations'] for x in counts)==40 and sum(x['updates'] for x in counts)==256 and sum(x['backwards'] for x in counts)==512
    outputs=[c.read(p) for p in (RUN/'private/outputs/RAW32').glob('*/*.json')];assert len(outputs)==40
    assert sum(x['forwards'] for x in counts)==616+sum(x['forwards'] for x in outputs)<=41576
    c.write(RUN/'private/RAW32_GENERATION_COMPLETE.json',dict(status='COMPLETE',counts=counts,epoch=time.time()))
    pipeline.wait([pipeline.launch('short_queue.py','raw32_ingest')])
    while not (RUN/'private/judge_raw32_astra_medium/ALL_WORKERS_COMPLETE.json').exists():time.sleep(10)
    pipeline.wait([pipeline.launch('short_queue.py','raw32_report')])


if __name__=='__main__':
    try:{'raw32_plan':plan,'raw32_worker':worker,'raw32_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
