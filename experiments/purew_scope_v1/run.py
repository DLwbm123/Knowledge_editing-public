"""Fixed native-weight experiment using the admitted runtime and data roles."""
import gc
import os
import subprocess
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import scope as data

c, old, RUN = data.c, data.old, data.RUN
GPUS = (3, 4)
ARMS = ('PROJECTED', 'UNPROJECTED')
LAYER = c.LAYER


def output(stage, owner, row):
    return RUN / 'private/outputs' / stage / str(owner) / (row['query_id'] + '.json')


def rows():
    return c.read(RUN / 'private/ROWS.json')


def plan():
    from purew_math import selfcheck
    assert c.read(data.SUP / 'public/DATA_ADMISSION.json')['status'] == 'SOURCE_EVIDENCE_GATE_PASS'
    tasks = data.tasks()
    assert [t['order'] for t in tasks] == list(range(1, 9))
    basis, held = data.q.split()
    assert len(basis) == 61 and len(held) == 96
    assert not {r['source_group'] for r in basis} & {r['source_group'] for r in held}
    allrows = []
    for t in tasks:
        source = data.source(t)
        assert len(source) == 7 and [r['audit_role'] for r in source] == ['NATIVE'] + ['FIT'] * 4 + ['GFIT'] * 2
        assert len({r['source_group'] for r in source[-2:]})==2
        allrows.extend(dict(r, owner=t['order']) for r in source + data.events(t))
    allrows.extend(dict(r, owner=0) for r in data.previous.data.held())
    assert len(allrows) == 267 and len({(r['owner'], r['query_id']) for r in allrows}) == 267
    assert not {r['source_group'] for r in allrows if r['audit_role'] in ('NATIVE','FIT','GFIT')} & {r['source_group'] for r in held}
    assert all(Path(c.local_path(r['image_path'])).is_file() for r in allrows+basis)
    import json
    schema=state_schema(torch.nn.Linear(2,1))
    assert json.loads(json.dumps(schema))==schema
    c.write(RUN / 'private/ROWS.json', allrows)
    c.write(RUN / 'private/HELD_BASE.json', c.read(data.PARENT / 'private/HELD_BASE.json'))
    lock = dict(arms=list(ARMS), GPUS=list(GPUS), layer=LAYER,
                method='native W projected SGD; final-answer CE and scope supervision; Base distribution KL',
                precision='NATIVE_FULL_FP32',native_forward=True,
                training_prefill='cached native expansion at final precision',
                full_paper_reproduction=False, trainable='one existing down_proj.weight',
                learning_rate=.01, gradient_norm_cap=5., maximum_step_norm=.05, kl_weight=1.,
                steps_per_edit=160, ordered_edits=8, trajectories=2, maximum_updates=2560,
                current_edit_loss='.25 native + .25 cyclic FIT + .5 cyclic GFIT',
                protection='cyclic 61 BASIS; KL(Base || current) on actual original Base token prefixes',
                history_replay=False, fixed_layer=True, refresh_geometry='current W at each edit boundary',
                nullspace='implicit full right nullspace via thin QR; full row-rank admission required',
                maximum_backwards=10244, mechanical_temporary_updates=2, maximum_non_generation_forwards=13000,
                maximum_generations=1019, maximum_Judge=1019, generation_max_tokens=1024,
                prefix_generations=216, base_generations=267, final_generations=534, reload_generations=2,
                all_candidates_frozen_before_event_or_held_evaluation=True,
                source_binding=c.digest(data.tasks()), anchors_binding=c.digest(c.read(data.SUP/'private/FROZEN_ANCHORS.json')),
                BASIS_binding=c.digest(basis), HELD_binding=c.digest(held),
                success_criteria=dict(native_final='8/8 correct',GFIT_final='16/16 correct; training evidence only',
                    native_or_GFIT_lost_since_insertion=0,current_Base_correct_HELD_new_damage=0,
                    untrained_generalization='no owner loses Base-correct T1G/T2G; aggregate T2G improves over Base',
                    projection_contribution='separate PROJECTED minus UNPROJECTED; ties do not establish projection benefit',
                    original63_mask_retained=True,all96_reported=True,all8_reported=True,
                    structural='same parameter keys/shapes/count; native fresh reload; no inference editor branch',
                    evidence='development pilot; not independent clinical/SOTA confirmation'),
                selection=False, automatic_tuning=False, selfcheck=selfcheck())
    c.write(RUN / 'private/LOCK.json', lock)
    c.write(RUN / 'public/ADMISSION.json', dict(status='PASS', **lock))


def counters(runtime):
    count = dict(forwards=0, backwards=0, updates=0, generations=0)
    h = runtime.model.register_forward_pre_hook(lambda *_: count.__setitem__('forwards',count['forwards']+1))
    return count, h


def emit(runtime, item, stage, counts):
    row, _, raw, expanded, binding = item
    dest = output(stage, row['owner'], row)
    assert not dest.exists(), 'No implicit generation retry'
    c.budget()
    start = counts['forwards']
    with torch.inference_mode():
        answer = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
    counts['generations'] += 1
    c.write(dest, dict(stage=stage, owner=row['owner'], role=row['audit_role'], query_id=row['query_id'],
                      binding=dict(input=row, judge_input=binding),
                      R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),
                      original_primary=row.get('original_primary',False),
                      at_cap=len(answer.raw_token_ids)>=1024, forwards=counts['forwards']-start,
                      lock=c.digest(c.read(RUN/'private/LOCK.json')),
                      execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))
    return dest


def prepare(runtime, bindings, selected, training=False):
    ts={t['order']:t for t in data.tasks()}
    items=[]
    for row in selected:
        raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
        if row.get('image_sha256'):assert raw['image_sha256']==row['image_sha256']
        with torch.no_grad():
            expanded=runtime.llava_model().prepare_inputs_labels_for_multimodal(
                raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
        assert expanded[4].dtype==torch.float32
        expanded=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded)
        binding=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],
                     image_path=row['image_path'],prompt_ids=raw['input_ids'][0].tolist(),
                     attention_mask=raw['attention_mask'][0].tolist(),generation=runtime.generation_config,
                     runtime=dict(inherited_runtime=next(iter(bindings.values()))['runtime'],
                                  actual_precision='NATIVE_FULL_FP32',native_forward=True,
                                  training_prefill='cached native expansion at final precision'))
        batch=cpu_batch(runtime.build_edit_batch(data.record(ts[row['owner'] or 1],row))) if training else None
        items.append((row,batch,raw,expanded,binding))
    return items


def cpu_batch(batch):
    kwargs={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in batch.forward_kwargs().items()}
    assert kwargs['inputs_embeds'].dtype==torch.float32
    return kwargs,batch.labels.detach().cpu(),tuple(batch.target_token_ids)


def state_schema(model):
    return [[k,list(v.shape),str(v.dtype)] for k,v in model.state_dict().items()]


def fp32_native(runtime):
    old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())


def admission(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip()
    uuid, free=line.split(', ')
    assert uuid==c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    return int(free)>=60000


def base():
    gpu=GPUS[0];c.available=admission
    with c.lease(gpu):
        runtime, bindings=c.load(gpu);counts,h=counters(runtime)
        try:
            fp32_native(runtime)
            prepared=prepare(runtime,bindings,rows())
            for item in prepared:emit(runtime,item,'BASE',counts)
            c.write(RUN/'private/BASE_COMPLETE.json',dict(outputs=267,counts=counts,
                    parameters=sum(v.numel() for v in runtime.model.parameters()),schema=state_schema(runtime.model)))
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_base_{os.getpid()}.json',counts)


def smoke():
    """One fixed same-state VLM gradient; restore both diagnostic proposals."""
    from purew_math import row_basis, step_direction, target_logits, preservation_kl
    c.available=admission
    with c.lease(GPUS[0]):
        runtime,bindings=c.load(GPUS[0]);counts,h=counters(runtime)
        try:
            selected=[r for r in rows() if r['owner']==1 and r['audit_role'] in ('NATIVE','FIT','GFIT')]
            fp32_native(runtime)
            items=prepare(runtime,bindings,selected,True)
            protection=cpu_batch(data.q.protection_batch(runtime,data.q.split()[0][0],data.tasks()[0]))
            weight=runtime.get_module(LAYER).weight
            raw,expanded=items[0][2:4]
            with torch.no_grad():
                native=runtime.llava_model().prepare_inputs_labels_for_multimodal(
                    raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert torch.equal(native[4],expanded[4].to(runtime.device)), 'native final-precision prefill drift'
            del native
            base=weight.detach().clone();q=row_basis(base)
            schema=state_schema(runtime.model)
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            pb=old.batch_on(runtime,protection)
            with torch.no_grad():reference=target_logits(runtime.model,pb)[0].detach()
            weight.requires_grad_(True)
            for i,fraction in [(0,.25),(1,.25),(5,.5)]:
                logits,target=target_logits(runtime.model,old.batch_on(runtime,items[i][1]))
                loss=fraction*F.cross_entropy(logits,target);loss.backward();counts['backwards']+=1
                del logits,loss
            logits,_=target_logits(runtime.model,pb)
            kl=preservation_kl(logits,reference);kl.backward();counts['backwards']+=1
            assert abs(float(kl.detach()))<1e-5
            del logits,kl,reference
            reports=[]
            for name,basis in [('PROJECTED',q),('UNPROJECTED',None)]:
                delta,receipt=step_direction(weight.grad,basis)
                with torch.no_grad():weight.copy_(base+delta)
                changed=int((weight!=base).sum())
                assert changed>0 and torch.isfinite(weight).all()
                reports.append(dict(arm=name,changed_elements=changed,**receipt))
                with torch.no_grad():weight.copy_(base)
                assert torch.equal(weight,base)
            weight.grad=None;weight.requires_grad_(False)
            assert schema==state_schema(runtime.model)
            assert frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            c.write(RUN/'private/MECHANICAL_COMPLETE.json',dict(status='PASS',reports=reports,
                    trainable_parameter_count=weight.numel(),weight_shape=list(weight.shape),
                    parameters=sum(p.numel() for p in runtime.model.parameters()),added_parameters=0,
                    temporary_updates=2,base_exactly_restored=True,native_prefill_exact=True,counts=counts))
            print('MECHANICAL_PASS',list(weight.shape),reports,flush=True)
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_smoke_{os.getpid()}.json',counts)


def train():
    from purew_math import row_basis, step_direction, target_logits, preservation_kl
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert c.read(RUN/'private/BASE_COMPLETE.json')['outputs']==267
    arm=ARMS[int(os.environ['PARTITION'])];gpu=GPUS[int(os.environ['PARTITION'])]
    dest=RUN/'private/candidates'/f'{arm}.pt'
    assert not dest.exists(), 'completed arm cannot restart'
    c.available=admission
    with c.lease(gpu):
        runtime,bindings=c.load(gpu);counts,h=counters(runtime)
        try:
            source_rows=[r for r in rows() if r['audit_role'] in ('NATIVE','FIT','GFIT')]
            fp32_native(runtime)
            prepared=prepare(runtime,bindings,source_rows,True)
            basis=data.q.split()[0]
            protection=[cpu_batch(data.q.protection_batch(runtime,r,data.tasks()[0])) for r in basis]
            weight=runtime.get_module(LAYER).weight
            schema=state_schema(runtime.model)
            assert schema==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            count_parameters=sum(v.numel() for v in runtime.model.parameters())
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            buffers={n:b.detach().clone() for n,b in runtime.model.named_buffers()}
            identity=id(weight)
            references=[]
            with torch.no_grad():
                for value in protection:
                    logits,_=target_logits(runtime.model,old.batch_on(runtime,value))
                    references.append(logits.detach().cpu())
            weight.requires_grad_(True)
            active=RUN/'private/active'/f'{arm}.pt'
            done=0;trace=[];pending=None
            if active.exists():
                pending=torch.load(active,map_location='cpu',weights_only=True)
                assert pending['lock']==c.digest(c.read(RUN/'private/LOCK.json'))
                done=pending['completed_edits'];trace=pending['trace']
                with torch.no_grad():weight.copy_(pending['weight'].to(weight.device))
            def verify():
                assert id(weight)==identity and state_schema(runtime.model)==schema
                assert sum(v.numel() for v in runtime.model.parameters())==count_parameters
                assert frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                assert all(torch.equal(v,buffers[n]) for n,v in runtime.model.named_buffers())
                assert [n for n,p in runtime.model.named_parameters() if p.requires_grad]==[LAYER+'.weight']
            for t in data.tasks():
                owner=t['order']
                if owner<=done:continue
                seed_everything(t['seed'])
                start_step=0
                if pending is not None and pending['active_owner']==owner:
                    start_step=pending['completed_step']
                    boundary=pending['boundary_weight'].to(weight.device)
                    # Reconstruct the exact admitted geometry rather than a current-step geometry.
                    q=row_basis(boundary) if arm=='PROJECTED' else None
                else:
                    boundary=weight.detach().clone()
                    q=row_basis(boundary) if arm=='PROJECTED' else None
                pending=None
                items=[x for x in prepared if x[0]['owner']==owner]
                assert len(items)==7
                batches=[old.batch_on(runtime,x[1]) for x in items]
                for step in range(start_step+1,161):
                    c.budget();weight.grad=None
                    losses={}
                    for i,fraction in [(0,.25),(1+(step-1)%4,.25),(5+(step-1)%2,.5)]:
                        logits,target=target_logits(runtime.model,batches[i])
                        loss=F.cross_entropy(logits,target)
                        assert torch.isfinite(loss)
                        (fraction*loss).backward();counts['backwards']+=1
                        losses[str(i)]=float(loss.detach())
                        del logits,loss
                    index=((owner-1)*160+step-1)%61
                    pb=old.batch_on(runtime,protection[index])
                    logits,_=target_logits(runtime.model,pb)
                    kl=preservation_kl(logits,references[index].to(runtime.device))
                    assert torch.isfinite(kl) and float(kl.detach())>=-1e-5
                    kl.backward();counts['backwards']+=1
                    losses['BASIS_KL']=float(kl.detach())
                    del logits,kl,pb
                    assert weight.grad is not None and all(p.grad is None for p in runtime.model.parameters() if p is not weight)
                    delta,receipt=step_direction(weight.grad,q)
                    with torch.no_grad():weight.add_(delta)
                    assert torch.isfinite(weight).all() and receipt['step_norm']<=.050001
                    counts['updates']+=1
                    trace.append(dict(owner=owner,step=step,losses=losses,**receipt))
                    weight.grad=None;del delta
                    if step%20==0:
                        verify()
                        c.save(active,dict(weight=weight.detach().cpu(),boundary_weight=boundary.detach().cpu(),
                               completed_edits=owner-1,active_owner=owner,completed_step=step,trace=trace,
                               lock=c.digest(c.read(RUN/'private/LOCK.json'))))
                        c.write(RUN/'private/progress'/f'{arm}.json',dict(owner=owner,step=step,counts=counts,receipt=receipt))
                        print('TRAIN',arm,owner,step,receipt['step_norm'],flush=True)
                verify()
                del q,boundary,batches
                # Earlier edits are tested at every prefix, but never replayed into training.
                for item in prepared:
                    r=item[0]
                    if r['owner']<=owner and r['audit_role'] in ('NATIVE','GFIT'):
                        path=output(f'{arm}_PREFIX_{owner}',r['owner'],r)
                        if path.exists():continue  # Resume preserves already completed consumers.
                        emit(runtime,item,f'{arm}_PREFIX_{owner}',counts)
                c.save(active,dict(weight=weight.detach().cpu(),boundary_weight=weight.detach().cpu(),
                       completed_edits=owner,active_owner=owner+1,completed_step=0,trace=trace,
                       lock=c.digest(c.read(RUN/'private/LOCK.json'))))
                done=owner
            weight.requires_grad_(False)
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            c.save(dest,dict(weight=weight.detach().cpu(),schema=schema,parameter_count=count_parameters,
                            trainable_parameter_count=weight.numel(),layer=LAYER,trace=trace,
                            lock=c.digest(c.read(RUN/'private/LOCK.json'))))
            c.write(RUN/'private'/f'TRAIN_COMPLETE_{arm}.json',dict(status='TRAINED_NOT_EVALUATED',
                    ordered_edits=8,trace_updates=len(trace),counts=counts,trainable_parameters=weight.numel(),
                    parameters_before=count_parameters,parameters_after=count_parameters,added_parameters=0,
                    fixed_layer=True,vision_frozen=True,noneditable_versions_verified=True,
                    training_temporary=dict(basis_bytes=weight.numel()*weight.element_size() if arm=='PROJECTED' else 0,
                    gradient_bytes=weight.numel()*weight.element_size(),boundary_bytes=weight.numel()*weight.element_size(),
                    reference_logits_bytes=sum(x.numel()*x.element_size() for x in references))))
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_train_{arm}_{os.getpid()}.json',counts)


def evaluate():
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['candidates']==2
    part=int(os.environ['PARTITION']);arm=ARMS[part];gpu=GPUS[part]
    assert 'purew_math' not in __import__('sys').modules
    c.available=admission
    with c.lease(gpu):
        runtime,bindings=c.load(gpu);counts,h=counters(runtime)
        try:
            fp32_native(runtime)
            prepared=prepare(runtime,bindings,rows())
            saved=torch.load(RUN/'private/candidates'/f'{arm}.pt',map_location='cpu',weights_only=True)
            assert state_schema(runtime.model)==saved['schema']
            assert sum(v.numel() for v in runtime.model.parameters())==saved['parameter_count']
            with torch.no_grad():runtime.get_module(saved['layer']).weight.copy_(saved['weight'].to(runtime.device))
            assert not any(p.requires_grad for p in runtime.model.parameters())
            # Native fresh-loader output must reproduce the final prefix's retained probe.
            item=next(x for x in prepared if x[0]['owner']==8 and x[0]['audit_role']=='NATIVE')
            probe=emit(runtime,item,f'{arm}_RELOAD',counts)
            assert c.read(probe)['R0']==c.read(output(f'{arm}_PREFIX_8',8,item[0]))['R0']
            for item in prepared:
                path=output(arm,item[0]['owner'],item[0])
                if not path.exists():emit(runtime,item,arm,counts)
            c.write(RUN/'private'/f'EVAL_COMPLETE_{arm}.json',dict(outputs=267,counts=counts,
                    native_fresh_reload_tokens_equal=True,added_parameters=0,editor_math_imported=False,
                    inference_structure_unchanged=True))
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_eval_{arm}_{os.getpid()}.json',counts)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('run.py','purew_smoke',GPUS[0],0)])
    assert c.read(RUN/'private/MECHANICAL_COMPLETE.json')['status']=='PASS'
    pipeline.wait([pipeline.launch('run.py','purew_base',GPUS[0],0)])
    pipeline.wait([pipeline.launch('run.py','purew_train',g,i) for i,g in enumerate(GPUS)])
    reports=[c.read(RUN/'private'/f'TRAIN_COMPLETE_{a}.json') for a in ARMS]
    assert all(x['trace_updates']==1280 and x['added_parameters']==0 for x in reports)
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=2,selection=False,epoch=time.time()))
    pipeline.wait([pipeline.launch('run.py','purew_eval',g,i) for i,g in enumerate(GPUS)])
    paths=list((RUN/'private/outputs').glob('*/*/*.json'))
    outputs=[c.read(p) for p in paths]
    assert len(outputs)==1019
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS*.json')]
    assert sum(x['updates'] for x in counts)==2560 and sum(x['backwards'] for x in counts)==10244
    assert sum(x['generations'] for x in counts)==1019
    ng=sum(x['forwards'] for x in counts)-sum(x['forwards'] for x in outputs)
    assert ng<=13000
    # Durable score material is independent of the remaining weight files.
    c.write(RUN/'private/JUDGE_REQUESTS.json',[
        dict(path=str(p),question=o['binding']['judge_input']['question'],
             reference=o['binding']['judge_input']['reference'],answer=o['R0']['raw_answer'])
        for p,o in zip(paths,outputs)])
    for arm in ARMS:
        assert c.read(RUN/'private'/f'EVAL_COMPLETE_{arm}.json')['native_fresh_reload_tokens_equal']
    c.write(RUN/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_SCORING_PENDING',counts=counts,
            outputs=1019,non_generation_forwards=ng,backwards=10244,updates=2560,mechanical_temporary_updates=2))
    # Only this run's matrices; all declared GPU/native-reload consumers have ended.
    retired=[RUN/'private'/kind/f'{a}.pt' for kind in ('active','candidates') for a in ARMS]
    assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()) for p in retired)
    c.write(RUN/'private/DELETION.json',dict(paths=list(map(str,retired)),files=4,
            bytes=sum(p.stat().st_size for p in retired),retained_copy=False,
            consumers=['ordered_edits','prefix_generation','final_generation','native_fresh_reload','durable_score_material'],
            reconstruction='rerun frozen sequence; no original checkpoint retained'))
    for p in retired:p.unlink()
    c.write(RUN/'public/PROGRESS.json',dict(status='GENERATED_SCORING_PENDING',
             training_complete=True,scoring_complete=False,publication_complete=False))


if __name__=='__main__':
    try:
        {'purew_plan':plan,'purew_smoke':smoke,'purew_base':base,'purew_train':train,
         'purew_eval':evaluate,'purew_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/f'FAILURE_{os.environ.get("ACTION")}_{os.getpid()}.json',
                dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
