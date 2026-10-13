"""Consume frozen candidates once, using the previously validated FP16 prefill."""
import os
import time
import traceback
from pathlib import Path
import torch
import constrained as local

q,c,p,RUN=local.q,local.c,local.p,local.RUN
PARENT=Path(os.environ['CONSTRAINED_PARENT'])


def jobs(task):
    held=q.split()[1]
    edit=[dict(task['native'],question=x,query_id=f"edit_{task['order']}_{i}",
               generation_role='EDIT',generation_index=i,semantic_correct=None)
          for i,x in enumerate([task['native']['question']]+task['fit_questions'])]
    return [dict(r,generation_role='HELDOUT',generation_index=i) for i,r in enumerate(held)]+edit


def candidate(task):
    return PARENT/'private/candidates'/f"{task['order']}.pt"


def plan():
    assert c.read(PARENT/'public/RESULTS.json')['decision']=='LOCAL_MECHANISM_SIGNAL'
    aliases={}
    for t in q.d.selected():
        v=torch.load(candidate(t),map_location='cpu',weights_only=True)
        assert v['lock']==c.digest(c.read(PARENT/'private/PROBE_LOCK.json'))
        assert len(jobs(t))==101
        aliases[str(t['order'])]=all(torch.equal(v['states']['RAW'][k],v['states']['MATCHED_RAW'][k]) for k in v['states']['RAW'])
    lock=dict(experts=8,arms=list(local.ARMS),heldout=96,primary=63,edit_per_expert=5,
        consumers=2464,maximum_generations=2464,actual_planned_generations=2464-101*sum(aliases.values()),
        matched_raw_alias=aliases,maximum_forwards=(2464-101*sum(aliases.values()))*1024,
        max_new_tokens=1024,backwards=0,updates=0,seed_selection=False,
        parent_freeze=c.read(PARENT/'private/CANDIDATES_FROZEN.json'),
        candidate_source=c.read(PARENT/'private/GPU_SOURCE_VERSION.json'),epoch=time.time())
    c.write(RUN/'private/GEN_LOCK.json',lock)
    c.write(RUN/'public/GEN_ADMISSION.json',{k:v for k,v in lock.items() if k not in ('parent_freeze','candidate_source')})
    p.done('GEN_PLAN_COMPLETE')


def worker():
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];local.calibration.require_memory(gpu)
    lock=c.read(RUN/'private/GEN_LOCK.json');done=0
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);model=runtime.llava_model();original=model.prepare_inputs_labels_for_multimodal
        assert not runtime.generation_config['do_sample'] and runtime.generation_config['max_new_tokens']==1024
        tasks=q.d.selected()[part::6];frozen={}
        for t in tasks:
            for row in jobs(t):
                key=row['query_id']
                if key in frozen:continue
                raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
                if row['generation_role']=='HELDOUT':
                    old=c.read(row['response_path'])['binding']['judge_input']
                    assert raw['input_ids'][0].tolist()==old['prompt_ids'] and raw['image_sha256']==old['image_sha256']
                    assert runtime.generation_config==old['generation']
                with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
                assert expanded[4].dtype==torch.float16
                frozen[key]=(raw,tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded))
        local.fp32(runtime);count=[0]
        handle=runtime.model.register_forward_pre_hook(lambda *_:count.__setitem__(0,count[0]+1))
        try:
            for t in tasks:
                saved=torch.load(candidate(t),map_location='cpu',weights_only=True)
                assert saved['lock']==c.digest(c.read(PARENT/'private/PROBE_LOCK.json'))
                expert=p.expert(saved['states']['BASE'],t['seed'],runtime.device).requires_grad_(False)
                for row in jobs(t):
                    raw,expanded=frozen[row['query_id']]
                    arms=('BASE',)+local.ARMS if row['generation_role']=='EDIT' else local.ARMS
                    for arm in arms:
                        dest=RUN/'private/outputs'/str(t['order'])/arm/(row['query_id']+'.json')
                        assert not dest.exists(),'No implicit generation retry'
                        if arm=='MATCHED_RAW' and lock['matched_raw_alias'][str(t['order'])]:
                            source=dest.parent.parent/'RAW'/dest.name;value=c.read(source)
                            value=dict(value,arm=arm,alias_of=str(source),new_generation=False)
                            c.write(dest,value);continue
                        c.budget();hook=None;used=[0]
                        def prepared(*args,**kwargs):
                            if args[0] is not None and torch.equal(args[0],raw['input_ids']):
                                assert args[3] is None and args[4] is None
                                used[0]+=1;assert used[0]==1
                                return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in expanded)
                            return original(*args,**kwargs)
                        model.prepare_inputs_labels_for_multimodal=prepared;start=count[0]
                        try:
                            if arm!='BASE':
                                expert.load_state_dict(saved['states'][arm]);hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                            with torch.inference_mode():
                                if hook:
                                    with hook.generation_request():answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                                else:answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        finally:
                            if hook:hook.detach()
                            model.prepare_inputs_labels_for_multimodal=original
                        assert used[0]==1 and answer.raw_token_ids
                        binding=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],
                            image_path=row['image_path'],prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),
                            runtime=dict(inherited_runtime=next(iter(bindings.values()))['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',frozen_FP16_prefill=True),generation=runtime.generation_config)
                        value=dict(arm=arm,expert_order=t['order'],role=row['generation_role'],index=row['generation_index'],
                            binding=dict(input=dict(row,image_sha256=raw['image_sha256']),judge_input=binding),
                            R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),
                            new_generation=True,forwards=count[0]-start,lock=c.digest(lock),execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),
                            EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024)
                        c.write(dest,value);done+=1
                    print('GENERATED',t['order'],row['generation_role'],row['generation_index'],flush=True)
                expert.load_state_dict(saved['states']['BASE'])
                assert all(torch.equal(v.cpu(),saved['states']['BASE'][k]) for k,v in expert.state_dict().items())
            assert not any(v.grad is not None for v in runtime.model.parameters())
        finally:
            model.prepare_inputs_labels_for_multimodal=original;handle.remove()
            c.write(RUN/'private'/f'GEN_COUNTS_{part}.json',dict(forwards=count[0],new_generations=done))
    p.done('GEN_WORKER_'+str(part))


def finish():
    lock=c.read(RUN/'private/GEN_LOCK.json');values=[c.read(f) for f in (RUN/'private/outputs').glob('*/*/*.json')]
    assert len(values)==2464 and sum(v['new_generation'] for v in values)==lock['actual_planned_generations']
    assert all(v['lock']==c.digest(lock) for v in values)
    counts=[c.read(RUN/'private'/f'GEN_COUNTS_{i}.json') for i in range(6)]
    assert sum(x['forwards'] for x in counts)==sum(x['forwards'] for x in values if x['new_generation'])<=lock['maximum_forwards']
    assert not list((RUN/'private').glob('FAILURE*'))
    c.write(RUN/'private/GENERATION_COMPLETE.json',dict(consumers=len(values),new_generations=lock['actual_planned_generations'],forwards=sum(x['forwards'] for x in counts),epoch=time.time()))
    # Last candidate consumer has durable outputs and complete scoring bindings.
    paths=[candidate(t) for t in q.d.selected()]
    assert all(x.is_file() and not x.is_symlink() and x.resolve().parent==(PARENT/'private/candidates').resolve() for x in paths)
    receipt=dict(paths=[str(x) for x in paths],bytes=sum(x.stat().st_size for x in paths),consumers=2464)
    c.write(RUN/'private/DELETION_PLAN.json',receipt)
    for x in paths:x.unlink()
    c.write(RUN/'public/GEN_LIFECYCLE.json',dict(status='COMPLETE',packages_deleted=8,bytes_deleted=receipt['bytes'],historical_artifacts_deleted=0,reconstruction_requires_new_authorization=True))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('generation.py','gen_worker',gpu,i) for i,gpu in enumerate(q.GPUS)])
    finish()
    pipeline.wait([pipeline.launch('generation_queue.py','ingest')])
    while not (RUN/'private/judge_generation_astra_medium/ALL_WORKERS_COMPLETE.json').exists():time.sleep(10)
    pipeline.wait([pipeline.launch('generation_report.py','report')])


if __name__=='__main__':
    try:{'gen_plan':plan,'gen_worker':worker,'gen_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
