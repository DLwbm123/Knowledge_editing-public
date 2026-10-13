"""Finite source-qualified endpoint rebuild, unseen generalization and damage audit."""
import os
import time
import traceback
from pathlib import Path
import torch
import optimizer160 as fit
q,c,p,RUN,old=fit.q,fit.c,fit.p,fit.RUN,fit.old
PARENT=Path(os.environ['OPTIMIZER160_PARENT'])


def candidate(t,arm):return RUN/'private/damage_candidates'/arm/f"{t['order']}.pt"


def rows(t):
    queries=c.read(Path(os.environ['GATE_PREVIOUS'])/'private/PAPER_QUERIES.json')['146']
    result=[]
    for event in t['events']:
        if event['task']=='T0':continue
        for key in event['all_probe_query_ids']:
            r=queries[key]
            result.append(dict(r,audit_role=event['task'],same_reference=r['reference']==t['native']['reference'],original_primary=False))
    return result


def held():return [dict(r,audit_role='HELDOUT',original_primary=r['semantic_correct'],same_reference=False) for r in q.split()[1]]


def plan():
    fit.selfcheck();tasks=q.d.selected();source=c.read(PARENT/'public/RESULTS.json')
    assert source['decision']=='BOTH_HAVE_NATIVE_REPAIR'
    assert all(source['panels'][a]['NATIVE']['correct']==8 and source['panels'][a]['FIT']['correct']==32 for a in fit.ARMS)
    training={(t['native']['image_sha256'],s) for t in tasks for s in [t['native']['question']]+t['fit_questions']}
    matrix=[dict(order=t['order'],rows=rows(t)) for t in tasks]
    values=[r for m in matrix for r in m['rows']]
    assert len(values)==len({r['query_id'] for r in values})==115
    assert not training & {(r['image_sha256'],r['question']) for r in values}
    counts={k:sum(r['audit_role']==k for r in values) for k in ('T1G','T2G','T1L','T2L')}
    assert counts==dict(T1G=32,T2G=32,T1L=49,T2L=2)
    h=held();assert len(h)==96 and sum(r['original_primary'] for r in h)==63
    assert not {r['source_group'] for r in h} & {t['native']['source_group'] for t in tasks}
    baseline=c.read(Path(os.environ['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    baseline=[r for r in baseline if r['role']=='HELDOUT'];assert len(baseline)==96 and sum(r['correct'] for r in baseline)==62
    c.write(RUN/'private/HELD_BASE.json',{r['row']['query_id']:r for r in baseline})
    c.write(RUN/'private/EVAL_MATRIX.json',matrix)
    c.write(RUN/'private/OPTIMIZER160_LOCK.json',c.read(PARENT/'private/OPTIMIZER160_LOCK.json'))
    lock=dict(experts=8,arms=fit.ARMS,steps=160,trajectories=16,training_updates=2560,training_backwards=5120,
        source_non_generation_forwards=6092,held_prefix_forwards=2112,total_non_generation_forwards=8204,
        source_generations=80,unseen_Base_generations=115,candidate_generations=1766,total_new_generations=1961,
        maximum_forwards=8204+1961*1024,maximum_new_Judge=1881,counts=counts,held=96,original_primary=63,FP32_Base_correct=62,
        matrix=c.digest(matrix),source_lock=c.digest(c.read(RUN/'private/OPTIMIZER160_LOCK.json')),
        all_candidates_frozen_before_eval=True,source_rebuild_gate='all80 exact accepted text/tokens/input binding',
        candidate_mode='forced individual owner expert',held_prefix='original FP16 Base tokens, FP32 logits for both models',
        source_optimization_unchanged=True,checkpoint_selection=False,heldout_selection=False,automatic_protection=False,
        T2L_limitation='2 queries from owner1 only',T1L_same_reference=sum(r['audit_role']=='T1L' and r['same_reference'] for r in values),
        epoch=time.time())
    c.write(RUN/'private/DAMAGE_LOCK.json',lock);c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**lock))
    p.done('DAMAGE_PLAN_COMPLETE')


def retain(t,arm,expert,base):
    path=candidate(t,arm);assert not path.exists()
    state={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
    value=dict(states=dict(BASE={k:v.cpu() for k,v in base.items()},FINAL=state),order=t['order'],arm=arm,lock=c.digest(c.read(RUN/'private/DAMAGE_LOCK.json')))
    c.save(path,value);loaded=torch.load(path,map_location='cpu',weights_only=True)
    assert all(torch.equal(state[k],loaded['states']['FINAL'][k]) for k in state)


def rebuild():fit.worker(retain)


def freeze():
    paths=list((RUN/'private/outputs').glob('*/*/*.json'));assert len(paths)==80
    for path in paths:
        a=c.read(path);b=c.read(PARENT/'private/outputs'/path.relative_to(RUN/'private/outputs'))
        assert a['binding']==b['binding'] and a['R0']==b['R0'],'Rebuild source drift; no implicit rescore or evaluation'
    counts=[c.read(RUN/'private'/f'OPTIMIZER160_COUNTS_{i}.json') for i in range(6)]
    assert sum(x['updates'] for x in counts)==2560 and sum(x['backwards'] for x in counts)==5120
    assert sum(x['forwards'] for x in counts)==6092+sum(c.read(x)['forwards'] for x in paths)
    assert all(candidate(t,a).is_file() for t,a in fit.jobs())
    assert sum(candidate(t,a).stat().st_size for t,a in fit.jobs())<=16*1024**2
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(status='PASS',source_outputs_exact=80,packages=16,counts=counts,epoch=time.time()))


def evaluate():
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['status']=='PASS'
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    assigned=fit.jobs()[part::6];lock=c.digest(c.read(RUN/'private/DAMAGE_LOCK.json'));h=held();hb=c.read(RUN/'private/HELD_BASE.json')
    counts=dict(forwards=0,generations=0,prefix_forwards=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);original=runtime.llava_model().prepare_inputs_labels_for_multimodal
        frozen={};allrows=h+[r for t,a in assigned for r in rows(t)]
        for row in allrows:
            if row['query_id'] in frozen:continue
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==row['image_sha256']
            with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert expanded[4].dtype==torch.float16
            frozen[row['query_id']]=(raw,tuple(v.detach().cpu() if isinstance(v,torch.Tensor) else v for v in expanded))
        batches=[old.cpu(q.protection_batch(runtime,r,assigned[0][0])) for r in h]
        old.fp32(runtime);batches=[old.batch_on(runtime,b) for b in batches]
        handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            references=[old.unedited(runtime,b) for b in batches];counts['prefix_forwards']+=96
            for t,arm in assigned:
                saved=torch.load(candidate(t,arm),map_location='cpu',weights_only=True);assert saved['lock']==lock
                expert=p.expert(saved['states']['FINAL'],t['seed'],runtime.device).requires_grad_(False)
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert)
                def emit(row,which,active):
                    path=RUN/'private/audit_outputs'/which/str(t['order'])/(row['query_id']+'.json');assert not path.exists()
                    c.budget();raw,expanded=frozen[row['query_id']];start=counts['forwards']
                    answer,trace=fit.short.generate(runtime,active,raw,expanded);counts['generations']+=1
                    b=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],image_path=row['image_path'],
                        prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),
                        runtime=dict(inherited_runtime=next(iter(bindings.values()))['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',frozen_FP16_prefill=True),generation=runtime.generation_config)
                    basepath=hb[row['query_id']]['path'] if row['audit_role']=='HELDOUT' else str(RUN/'private/audit_outputs/BASE'/str(t['order'])/(row['query_id']+'.json'))
                    c.write(path,dict(arm=which,expert_order=t['order'],role=row['audit_role'],original_primary=row['original_primary'],same_reference=row['same_reference'],
                        binding=dict(input=row,judge_input=b),R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),
                        baseline_path=basepath,generation_trace=trace,forwards=counts['forwards']-start,
                        EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024,lock=lock,execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
                if arm=='RAW':
                    for row in rows(t):emit(row,'BASE',None)
                hook.attach()
                try:
                    values=[]
                    for b,(ref,target) in zip(batches,references):
                        v,y=q.logits(runtime,hook,b);assert torch.equal(target,y);values.append(q.compare(ref,v,y));counts['prefix_forwards']+=1
                    c.write(RUN/'private/held_diagnostics'/arm/f"{t['order']}.json",dict(values=values,original_primary=[r['original_primary'] for r in h]))
                    for row in h+rows(t):emit(row,arm,hook)
                finally:hook.detach()
                expert.load_state_dict(saved['states']['BASE'])
                assert all(torch.equal(v.cpu(),saved['states']['BASE'][k]) for k,v in expert.state_dict().items())
                print('DAMAGE_CONSUMED',arm,t['order'],flush=True)
        finally:handle.remove();c.write(RUN/'private'/f'DAMAGE_COUNTS_{part}.json',counts)
    p.done('DAMAGE_WORKER_'+str(part))


def finish():
    lock=c.read(RUN/'private/DAMAGE_LOCK.json');paths=list((RUN/'private/audit_outputs').glob('*/*/*.json'));assert len(paths)==1881
    counts=[c.read(RUN/'private'/f'DAMAGE_COUNTS_{i}.json') for i in range(6)]
    assert sum(x['generations'] for x in counts)==1881 and sum(x['prefix_forwards'] for x in counts)==2112
    assert sum(x['forwards'] for x in counts)==2112+sum(c.read(x)['forwards'] for x in paths)
    assert all(c.read(x)['lock']==c.digest(lock) for x in paths)
    source_counts=c.read(RUN/'private/CANDIDATES_FROZEN.json')['counts']
    assert sum(x['forwards'] for x in counts+source_counts)<=lock['maximum_forwards']
    assert not list((RUN/'private').glob('FAILURE*'))
    c.write(RUN/'private/DAMAGE_GENERATION_COMPLETE.json',dict(status='COMPLETE',outputs=1881,source_outputs=80,counts=counts,epoch=time.time()))
    paths=[candidate(t,a) for t,a in fit.jobs()]
    assert all(x.is_file() and not x.is_symlink() and (RUN/'private/damage_candidates').resolve() in x.resolve().parents for x in paths)
    receipt=dict(paths=list(map(str,paths)),bytes=sum(x.stat().st_size for x in paths),last_consumers='source80+audit1881+held1536',epoch=time.time())
    c.write(RUN/'private/DELETION_PLAN.json',receipt)
    for x in paths:x.unlink()
    c.write(RUN/'public/LIFECYCLE.json',dict(status='COMPLETE',packages_deleted=16,bytes_deleted=receipt['bytes'],retained_copies=0,historical_artifacts_deleted=0,reconstruction_requires_new_training=True))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('damage.py','damage_rebuild',g,i) for i,g in enumerate(q.GPUS)])
    freeze()
    pipeline.wait([pipeline.launch('damage.py','damage_eval',g,i) for i,g in enumerate(q.GPUS)])
    finish();pipeline.wait([pipeline.launch('damage_queue.py','damage_ingest')])
    while not (RUN/'private/judge_damage_astra_medium/ALL_WORKERS_COMPLETE.json').exists():time.sleep(10)
    pipeline.wait([pipeline.launch('damage_queue.py','damage_report')])


if __name__=='__main__':
    try:{'damage_plan':plan,'damage_rebuild':rebuild,'damage_eval':evaluate,'damage_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
