"""Matched TT continuation with balanced source-answer replay; frozen routing."""
import os
import random
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import paper as p
from methods.medtrace.core import MedTraceLayerHook

c, RUN = p.c, p.RUN
PARENT = Path(os.environ['SOURCE_PARENT'])
MARGIN = Path(os.environ['BINARY_MARGIN_PARENT'])
ARMS = ('CE192', 'SOURCE_REPLAY192')
STEPS = 192


def point(task, arm):
    return RUN/'private/weights'/arm/task['anonymous_edit']/'step192.pt'


def update(runtime, hook, expert, optimizer, native, fit, replay=None):
    optimizer.zero_grad(set_to_none=True)
    values = {}
    for name, batch, weight in [('native', native, .5), ('fit', fit, .5), ('replay', replay, 1.)]:
        if batch is None:
            continue
        hook.set_teacher_routing(batch.labels)
        loss = runtime.compute_loss(batch)
        assert torch.isfinite(loss)
        (weight*loss).backward()
        values[name] = float(loss.detach())
    params = list(expert.parameters())
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    assert torch.isfinite(norm) and all(x.grad is not None and torch.isfinite(x.grad).all() for x in params)
    assert not any(x.grad is not None for x in runtime.model.parameters())
    optimizer.step()
    assert all(torch.isfinite(x).all() for x in params)
    return dict(loss=values, preclip_norm=float(norm), forwards=len(values), backwards=len(values))


def selfcheck():
    from types import SimpleNamespace
    for with_replay, expected in [(False, .26), (True, .10)]:
        e = torch.nn.Linear(1, 1, bias=False); e.weight.data.fill_(.2)
        frozen = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        runtime = SimpleNamespace(model=frozen, compute_loss=lambda b: (e.weight-b.target).square().sum())
        b = lambda x: SimpleNamespace(labels=None, target=x)
        value = update(runtime, SimpleNamespace(set_teacher_routing=lambda _: None), e,
            torch.optim.SGD(e.parameters(), lr=.1), b(0.), b(1.), b(-1.) if with_replay else None)
        assert abs(float(e.weight)-expected) < 1e-6
        assert value['forwards'] == (3 if with_replay else 2) and frozen.weight.grad is None
    return dict(status='PASS', analytic_CE_gradient=True, replay_gradient_and_clipping=True, Base_gradient=False)


def record(row, task):
    return replace(c.record(task), question=row['question'], target=row['reference'],
        image_path=Path(c.local_path(row['image_path'])), relative_image_path=row['original_image_path'],
        official_rephrase='', dataset=row['dataset'])


def prepare():
    checks = selfcheck()
    assert c.read(RUN/'public/DATA_ADMISSION.json')['status'] == 'PASS_SOURCE_ANNOTATED_DEVELOPMENT'
    tasks = p.tasks(); assert len(tasks) == 146
    old = {x['query_id']: x for x in c.read(MARGIN/'private/SELECTIONS.json') if x['arm'] == 'MARGIN_002'}
    assert set(old) == set(p.queries(146)) and len(old) == 1513
    for x in old.values():
        d = c.read(x['source_path'])
        assert c.digest(d) == x['source_output_binding']
    weights = {t['edit_id']: dict(path=str(p.initial(t)), hash=p.load_state(p.initial(t))['state_hash']) for t in tasks}
    assert next(iter(c.read(x['source_path']) for x in old.values()))['binding']['phase']['weights'] == weights
    c.write(RUN/'private/BASELINE.json', old)
    c.write(RUN/'private/REFERENCE_WEIGHTS.json', weights)
    c.write(RUN/'public/ADMISSION.json', dict(status='PASS', selfcheck=checks, edits=146,
        arms=ARMS, updates_per_expert=STEPS, new_experts=292, new_updates=292*STEPS,
        main_query_inputs=1513, source_check_inputs=96, new_outputs=3410, consumers=4923,
        router='FROZEN_MARGIN_002_ON_EXISTING_INPUTS', new_CHECK='FIXED_EXPERT_STRESS_TEST', CP_enabled=False))


def mechanical():
    with p.lease(int(os.environ['GPU'])):
        runtime,_=c.load(int(os.environ['GPU']));task=p.tasks()[0]
        batches=[runtime.build_edit_batch(record(row,task)) for row in c.read(RUN/'private/REPLAY_FIT.json')]
        assert len(batches)==192 and all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
        expert=p.expert(p.load_state(p.initial(task))['expert'],task['seed'],runtime.device)
        hook=MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
        try:
            item=update(runtime,hook,expert,p.optimizer(expert,runtime.model),runtime.build_edit_batch(c.record(task)),
                runtime.build_edit_batch(replace(c.record(task),question=task['fit_questions'][0])),batches[0])
            assert len(item['loss'])==3
        finally:hook.detach()
        peak=torch.cuda.max_memory_allocated()
        assert peak+6*1024**3 < 24*1024**3, 'Increase admission reserve before formal training'
        c.write(RUN/'public/GPU_MECHANICAL.json',dict(status='PASS',all192_training_batches=True,
            real_replay_gradient=True,Base_gradient=False,teacher_EOS=True,peak_allocated_bytes=peak,
            admission_reserve_bytes=6*1024**3,formal_weights_unchanged=True))


def fit():
    from m3bench_repro.editors.llava_runtime import seed_everything
    gpu, part = int(os.environ['GPU']), int(os.environ['PARTITION'])
    tasks = p.tasks(); rows = c.read(RUN/'private/REPLAY_FIT.json')
    assert len(rows) == STEPS and all(r['role'] == 'REPLAY_FIT' for r in rows)
    with p.lease(gpu):
        runtime, _ = c.load(gpu)
        replay_batches = [runtime.build_edit_batch(record(r, tasks[0])) for r in rows]
        assert all(b.target_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id for b in replay_batches)
        for task in tasks[part::8]:
            original = p.load_state(p.initial(task))
            native = runtime.build_edit_batch(c.record(task))
            fits = [runtime.build_edit_batch(replace(c.record(task), question=q)) for q in task['fit_questions']]
            order = list(range(4)); random.Random(task['seed']).shuffle(order)
            groups = list(range(64)); random.Random(task['seed']).shuffle(groups)
            replay_order = [3*g+k for g in groups for k in range(3)]
            assert sorted(replay_order) == list(range(STEPS))
            for arm in ARMS:
                dest = point(task, arm); receipt = dest.parent/'TRAINING.json'
                if receipt.exists():
                    assert dest.exists(); continue
                expert = p.expert(original['expert'], task['seed'], runtime.device)
                optimizer = p.optimizer(expert, runtime.model)
                seed_everything(task['seed'])
                binding = dict(edit_id=task['edit_id'], seed=task['seed'], W0=original['state_hash'],
                    arm=arm, steps=STEPS, fit_order=order, replay_order=replay_order if arm == ARMS[1] else [],
                    training_rows=c.digest(rows) if arm == ARMS[1] else None,
                    lock=c.digest(c.read(RUN/'private/REPLAY_LOCK.json')),
                    execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
                latest = dest.parent/'latest.pt'; curve=[]; start=0; began=time.time()
                if latest.exists():
                    saved=p.load_state(latest); assert saved['binding'] == binding
                    expert.load_state_dict(saved['expert']); optimizer.load_state_dict(saved['optimizer'])
                    c.restore_rng(saved); curve=saved['curve']; start=saved['step']
                hook=MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                try:
                    for step in range(start+1, STEPS+1):
                        c.budget()
                        batch=replay_batches[replay_order[step-1]] if arm == ARMS[1] else None
                        item=update(runtime,hook,expert,optimizer,native,fits[order[(step-1)%4]],batch)
                        curve.append(dict(step=step,**item))
                        if step%32 == 0:
                            c.save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=optimizer.state_dict(),
                                curve=curve,step=step,**c.rng()))
                            print('TRAIN',task['order'],arm,step,flush=True)
                    c.save(dest,dict(binding=binding,expert=expert.state_dict(),step=STEPS,state_hash=c.state_hash(expert)))
                    c.write(receipt,dict(status='COMPLETE',binding=binding,curve=curve,updates=STEPS,
                        seconds=time.time()-began,parameters=7168,Base_gradient=False))
                    latest.unlink()
                finally:
                    hook.detach()
        p.done('TRAIN_'+str(part))


def evaluate():
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION'])
    tasks=p.tasks(); indices={t['edit_id']: i for i,t in enumerate(tasks)}
    old=c.read(RUN/'private/BASELINE.json'); checks=c.read(RUN/'private/REPLAY_CHECK.json')
    jobs=[(qid,row,False) for qid,row in p.queries(146).items()]+[(r['query_id'],r,True) for r in checks]
    states={arm:[p.load_state(p.initial(t) if arm=='W0' else point(t,arm)) for t in tasks] for arm in ('W0',)+ARMS}
    weights={arm:{t['edit_id']:dict(path=str(p.initial(t) if arm=='W0' else point(t,arm)),hash=x['state_hash'])
        for t,x in zip(tasks,ss)} for arm,ss in states.items()}
    with p.lease(gpu):
        runtime,bindings=c.load(gpu)
        teachers=p.tr.teachers_for(runtime,c.read(p.BASE/'private/U_ROLES.json')['CHECK'])
        tm={'U_'+c.digest(x[4]): x for x in teachers}
        experts={arm:p.Mixture([p.expert(x['expert'],t['seed'],runtime.device).requires_grad_(False)
            for t,x in zip(tasks,ss)]) for arm,ss in states.items()}
        for qid,row,fresh in jobs[part::8]:
            c.budget(); raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            if fresh:
                row=dict(row,image_sha256=raw['image_sha256'])
                idx=row['forced_expert_index']; assert 0 <= idx < 32
                selected=tasks[idx]['edit_id']
                bind=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],
                    image_path=row['image_path'],prompt_ids=raw['input_ids'][0].tolist(),
                    attention_mask=raw['attention_mask'][0].tolist(),runtime=next(iter(bindings.values()))['runtime'],
                    generation=runtime.generation_config)
                arms=('BASE','W0')+ARMS
            else:
                origin=c.read(old[qid]['source_path']); selected=origin['effective_expert']
                idx=indices[selected] if selected else None; bind=origin['binding']['judge_input']; arms=ARMS
                assert raw['image_sha256']==bind['image_sha256'] and raw['input_ids'].tolist()==[bind['prompt_ids']]
                assert raw['attention_mask'].tolist()==[bind['attention_mask']] and runtime.generation_config==bind['generation']
            base_tokens=None
            for arm in arms:
                phase=dict(arm=arm,node=0,prefix=146,slot=0,steps=0 if arm in ('BASE','W0') else STEPS,
                    weights={} if arm=='BASE' else weights[arm],execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),
                    replay_lock=c.digest(c.read(RUN/'private/REPLAY_LOCK.json')))
                hook=None; on=arm!='BASE' and idx is not None
                if on:
                    mixture=experts[arm]; mixture.ids=[idx]; mixture.weights=[1.]
                    hook=MedTraceLayerHook(runtime.get_module(c.LAYER),mixture); hook.attach()
                began=time.time(); kl=same=None
                try:
                    with torch.inference_mode():
                        if hook:
                            with hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        else:g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    output=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
                    if arm=='BASE': base_tokens=output['raw_token_ids']
                    if fresh: same=output['raw_token_ids']==base_tokens
                    if qid in tm:
                        kwargs,labels,mask,logp,_,cache=tm[qid]
                        if hook:hook.clear_request_routing();hook.set_teacher_routing(labels)
                        with torch.no_grad():kl=float(p.tr.full_vocab_kl(runtime.model(**kwargs).logits[mask],logp))
                        same=output['raw_token_ids']==cache['tokens']
                finally:
                    if hook:hook.detach()
                d=dict(binding=dict(input=row,judge_input=bind,phase=phase,arm=arm,
                    mode='forced_source_CHECK' if fresh else 'bank_R0',prefix=146,owner_order=146),
                    R0=output,effective_expert=selected if on else None,route=dict(activated=on,logical_edit_id=selected if on else None),
                    U_KL=kl,Base_token_consistency=same,seconds=time.time()-began,
                    diagnostic_only=fresh,active_target=qid in indices,
                    original_route_preserved=not fresh,forced_expert_stress_test=fresh)
                c.write(RUN/'private/outputs'/arm/(c.digest(qid)+'.json'),d)
            print('EVAL',part,qid[:12],flush=True)
        assert not any(x.grad is not None for x in runtime.model.parameters())
    p.done('EVAL_'+str(part))


def consumers():
    old=c.read(RUN/'private/BASELINE.json'); rows=[]
    for qid in p.queries(146):
        rows.append(dict(arm='W0',mode='bank_R0',query_id=qid,path=old[qid]['source_path']))
        for arm in ARMS:
            rows.append(dict(arm=arm,mode='bank_R0',query_id=qid,path=str(RUN/'private/outputs'/arm/(c.digest(qid)+'.json'))))
    for row in c.read(RUN/'private/REPLAY_CHECK.json'):
        for arm in ('BASE','W0')+ARMS:
            rows.append(dict(arm=arm,mode='forced_source_CHECK',query_id=row['query_id'],path=str(RUN/'private/outputs'/arm/(c.digest(row['query_id'])+'.json'))))
    assert len(rows)==4923 and all(Path(r['path']).is_file() for r in rows)
    c.write(RUN/'private/CONSUMERS.json',rows)


def controller():
    import pipeline
    assert c.read(RUN/'public/ADMISSION.json')['status']=='PASS'
    assert c.read(RUN/'public/GPU_MECHANICAL.json')['status']=='PASS'
    p.progress('SOURCE_REPLAY_TRAINING')
    pipeline.wait([pipeline.launch('replay.py','replay_fit',i,i) for i in range(8)])
    p.progress('SOURCE_REPLAY_EVALUATION')
    pipeline.wait([pipeline.launch('replay.py','replay_eval',i,i) for i in range(8)])
    consumers();p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('replay_queue.py','replay_ingest')])
    removed=[]
    for folder in ('weights','teacher'):
        for path in (RUN/'private'/folder).rglob('*.pt'):
            assert not path.is_symlink() and path.resolve().is_relative_to((RUN/'private'/folder).resolve())
            removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    p.done('DELETION',dict(files=removed,all_consumers_complete=True,bindings_durable=True,historical_weights_untouched=True,rebuild='retrain'))
    p.progress('ASTRA_SCORING');root=RUN/'private/judge_replay_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget(); assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('replay_report.py','replay_report')])


if __name__=='__main__':
    try:
        {'replay_prepare':prepare,'replay_mechanical':mechanical,'replay_fit':fit,'replay_eval':evaluate,'replay_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False))
        raise
