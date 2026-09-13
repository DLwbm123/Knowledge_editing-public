"""Frozen C_NO_H single queue; existing V1 training, no new method or Judge."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup, record_for, routes


def training_task(t):
    if not t['roles']['U_fit'] or not t['U_fit'] or t['fit_status']!='SUPPORTED':
        raise ValueError('C_NO_H requires full U/fit support')
    return dict(canonical_edit_id=t['edit_id'],order=t['order'],seed=t['seed'],
        probes=[t['native']],U=t['U_fit'],fit_questions=t['fit_questions'])


def worker(cfg):
    setup(cfg)
    import torch
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from scripts.medtrace import stage15 as v1
    from scripts.medtrace.run_realmodel_core import load_real_runtime, LAYER
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    root=Path(cfg['run']); source=Path(cfg['source_run']); private=root/'private'
    ledger=read(source/'private/COHORT_AND_SUPPORT_LEDGER.json')
    if ledger['freeze_id']!=cfg['freeze_id'] or digest({k:v for k,v in ledger.items() if k!='freeze_id'})!=cfg['freeze_id']:
        raise ValueError('Cohort changed')
    if cfg['method']!='C_NO_H' or cfg['mode']!='single_main_T0' or cfg['seed_base']!=20260912:
        raise ValueError('Unexpected dispatch')
    if read(source/'private/migration/GPU_SMOKE.json')['status']!='PASS': raise ValueError('Mechanical preflight missing')
    by_id={t['edit_id']:t for t in ledger['tasks']}; tasks=[by_id[e] for e in ledger['main_T0']]
    if len(tasks)!=cfg['N']: raise ValueError('Queue count changed')
    for t in tasks: training_task(t)
    if shutil.disk_usage(root).free < 12*1024**3: raise OSError('Need 8GiB reserve plus 4GiB working allowance')
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    bindings=read(source/'private/BINDINGS.json')
    if runtime.generation_config!=next(iter(bindings.values()))['generation']: raise ValueError('Generation lock mismatch')
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    # Existing training's per-step budget also protects the shared disk reserve.
    old_budget=v1.budget
    def budget(run,settings):
        old_budget(run,settings)
        if shutil.disk_usage(root).free < 8*1024**3: raise OSError('8GiB reserve reached')
    v1.budget=budget
    status=root/'public/SINGLE_PROGRESS.json'; completed=0; began=time.time()
    write_json_atomic(status,dict(status='RUNNING',method='C_NO_H',N=len(tasks),completed=0,phase='INITIALIZING'))
    for t in tasks:
        directory=private/'edits'/f"e{t['order']:03d}"; directory.mkdir(parents=True,exist_ok=True)
        record=record_for(t); adapted=training_task(t)
        binding=dict(freeze_id=cfg['freeze_id'],input=t['native'],U_fit=t['U_fit'],fit=t['fit_questions'],
            seed=t['seed'],seed_base=cfg['seed_base'],runtime=cfg['runtime_lock'],generation=runtime.generation_config,
            method=cfg['method_lock'],execution_commit=cfg['code_commit'])
        identity=directory/'BINDING.json'
        if identity.exists():
            if read(identity)!=binding: raise ValueError('Resume binding mismatch')
        else: write_new(identity,binding)
        if (directory/'FAILURE.json').exists(): raise RuntimeError('Prior failure requires explicit recovery')
        if (directory/'COMPLETE.json').exists():
            if read(directory/'COMPLETE.json')['binding']!=binding: raise ValueError('Completed binding mismatch')
            completed+=1; continue
        started=time.time(); hook=None; router_editor=None
        try:
            budget(root,cfg); runtime.run_root=directory/'work'; torch.cuda.reset_peak_memory_stats()
            write_json_atomic(status,dict(status='RUNNING',method='C_NO_H',N=len(tasks),completed=completed,order=t['order'],phase='INITIALIZING'))
            # Reuse the exact frozen native Base output for initialization, not an old verdict.
            raw,batch,base_binding=v1.prepared(runtime,t['native'],record)
            b=bindings[t['native']['opaque_Base_id']]
            if (raw['input_ids'].tolist()!=[b['prompt_ids']] or raw['attention_mask'].tolist()!=[b['attention_mask']]
                or raw['image_sha256']!=b['image_sha256']): raise ValueError('Native Base input mismatch')
            cache=private/'base'/str(cfg['gpu'])/(v1.digest(base_binding)+'.json')
            base_value=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'],binding=base_binding,seconds=0)
            if cache.exists():
                if read(cache)!=base_value: raise ValueError('Base cache changed')
            else:
                cache.parent.mkdir(parents=True,exist_ok=True); write_new(cache,base_value)
            cp=v1.initialize(runtime,root,cfg,adapted,record=record,seed_base=cfg['seed_base'])
            expert=LowRankExpert(cp,t['seed'],rank=4).to(runtime.device)
            if sum(p.numel() for p in expert.parameters())!=73728: raise ValueError('Parameter count changed')
            # Lossless conversion mechanical check; not a performance qualification.
            with torch.no_grad():
                x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,14336)
                if not torch.allclose(cp.residual(x),expert.residual(x),rtol=2e-4,atol=2e-5):
                    raise ValueError('CP to freeR4 transfer mismatch')
            del cp,x,raw,batch
            expert=v1.train_steps(runtime,root,cfg,adapted,expert,'C_NO_H',record=record)
            restored=torch.load(directory/'C_NO_H/latest.pt',map_location=runtime.device,weights_only=True)
            if not all(torch.equal(v,restored['expert'][k]) for k,v in expert.state_dict().items()):
                raise ValueError('C_NO_H saved state mismatch')
            del restored
            # Reuse the already-trained BE router only, never its writer weights.
            be_dir=source/'private/single_BE'/f"e{t['order']:03d}"
            be_state=torch.load(be_dir/'state.pt',map_location='cpu',weights_only=False)
            if be_state['binding']['input']!=t['native'] or be_state['binding']['freeze_id']!=cfg['freeze_id']:
                raise ValueError('Shared Base router source mismatch')
            from m3bench_repro.editors.routing import MemoryRouter
            router_editor=BalanceEditPaperSpecEditor(runtime)
            router_editor.router=MemoryRouter.from_state(be_state['router'],device=runtime.device)
            from scripts.medtrace.run_selective_write import save
            router_path=directory/'ROUTER.pt'
            if not router_path.exists(): save(router_path,be_state['router'])
            del be_state
            hook=MedTraceLayerHook(runtime.get_module(LAYER),expert); hook.attach()
            queries=list(dict.fromkeys([t['edit_id']]+[q for event in t['events'] for q in event['all_probe_query_ids']]))
            for index,qid in enumerate(queries):
                q=ledger['queries'][qid]; b=bindings[q['opaque_Base_id']]
                out=directory/f'query_{index:03d}.json'
                qb=dict(input=q,runtime=cfg['runtime_lock'],generation=runtime.generation_config,
                    writer='C_NO_H',prefix=1,training_binding=digest(binding))
                if out.exists():
                    if read(out)['binding']!=qb: raise ValueError('Query resume mismatch')
                    continue
                budget(root,cfg); hook.clear_request_routing()
                query=replace(record,record_id='query',question=q['question'],target='',official_rephrase='',image_path=Path(q['image_path']))
                raw=runtime.adapter.prepare_inputs(query.image_path,query.question,None)
                if (raw['input_ids'].tolist()!=[b['prompt_ids']] or raw['attention_mask'].tolist()!=[b['attention_mask']]
                    or raw['image_sha256']!=b['image_sha256']): raise ValueError('Query/Base identity mismatch')
                with torch.inference_mode():
                    decision=router_editor._route(query)
                    distance=float(decision.nearest_distance); radius=float(router_editor.router.radii[0])
                    on=routes(distance,radius)
                    with hook.generation_request(): generated=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    forced=dict(raw_answer=generated.decoded_text,raw_token_ids=list(generated.raw_token_ids))
                    base=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
                    if index==0:
                        with hook.generation_request() if on['R0'] else nullcontext():
                            actual=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        if list(actual.raw_token_ids)!=(forced if on['R0'] else base)['raw_token_ids']:
                            raise ValueError('Actual R0 generation parity mismatch')
                write_new(out,dict(binding=qb,Base_cache_id=q['opaque_Base_id'],distance=distance,radius_R0=radius,
                    route_on=on,FORCED_ON=forced,R0=forced if on['R0'] else base,RC=forced if on['RC'] else base,
                    canonical_cap=1024,routing_derivation='single immutable expert with shared frozen Base BE router'))
            if any(p._version!=version or p.data_ptr()!=ptr or p.requires_grad for p,version,ptr in frozen):
                raise ValueError('Base mutated')
            write_new(directory/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',binding=binding,queries=len(queries),
                seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            completed+=1
            write_json_atomic(status,dict(status='RUNNING',method='C_NO_H',N=len(tasks),completed=completed,phase='EDIT_COMPLETE'))
            print('EDIT_COMPLETE',completed,len(tasks),flush=True)
        except Exception as error:
            write_new(directory/'FAILURE.json',dict(error=repr(error),binding=binding))
            write_json_atomic(status,dict(status='FAILED',completed=completed,N=len(tasks),error=repr(error))); raise
        finally:
            if hook: hook.detach()
            if router_editor:
                target=router_editor.target; base_module=router_editor.wrapper.base
                router_editor.reset_editor_state(); runtime.replace_module(target,base_module)
            gc.collect(); torch.cuda.empty_cache()
    write_json_atomic(status,dict(status='GENERATED_NOT_SCORED',method='C_NO_H',completed=completed,N=len(tasks),
        seconds=time.time()-began,remaining='C_NO_H sequential, BE sequential, dependent checkpoint cleanup, other methods'))


if __name__=='__main__':
    import sys
    cfg=read(Path(sys.argv[1]))
    with (Path(cfg['run'])/'private/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); worker(cfg)
