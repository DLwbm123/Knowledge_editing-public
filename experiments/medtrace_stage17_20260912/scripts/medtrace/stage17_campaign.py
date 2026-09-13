"""Finite Stage17 GPU campaign. Independent replay is not sequential retraining."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import fcntl
import gc
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup, record_for, routes


def query_ids(task):
    return list(dict.fromkeys([task['edit_id']] + [q for e in task['events'] for q in e['all_probe_query_ids']]))


def prefixes(n):
    return sorted({p for p in (1, 50, 100, n) if 0 < p <= n})


def schedule():
    return [('C_NO_H', 'sequential'), ('balancedit', 'sequential'),
            *[(m, mode) for m in ('lora', 'grace', 'belora') for mode in ('single', 'sequential')]]


def role_map(ledger):
    """Metadata-only active targets, frozen before any sequential output; never routed."""
    by_id={t['edit_id']:t for t in ledger['tasks']}
    tasks=[by_id[e] for e in ledger['main_T0']]
    result={}
    for prefix in prefixes(len(tasks)):
        targets={(t['native']['image_sha256'],t['native']['question']):t['native']['reference'] for t in tasks[:prefix]}
        result[str(prefix)]={qid:targets[(q['image_sha256'],q['question'])]
            for qid,q in ledger['queries'].items() if (q['image_sha256'],q['question']) in targets}
    return result


def check_space(root):
    if (root/'STOP').exists():
        raise RuntimeError('Explicit campaign stop')
    if shutil.disk_usage(root).free < 8*1024**3:
        raise OSError('8GiB data-disk reserve reached')


def cleanup(cfg, method, mode):
    """Delete only enumerated generated weights after all registered GPU consumers."""
    root=Path(cfg['run']); phase_root=root/'private'/f'{method}_{mode}'
    if read(phase_root/'COMPLETE.json')['status']!='GENERATED_NOT_SCORED':
        raise ValueError('Unfinished checkpoint consumer')
    receipt=phase_root/'CLEANUP.json'
    if receipt.exists(): return
    paths=[]
    if method=='balancedit':
        if read(root/'private/C_NO_H_sequential/COMPLETE.json')['status']!='GENERATED_NOT_SCORED':
            raise ValueError('C_NO_H router consumer incomplete')
        origin=Path(cfg['source_run'])/'private/single_BE'
        paths=list(origin.glob('e*/state.pt'))
    elif method=='C_NO_H':
        if not (root/'private/balancedit_sequential/COMPLETE.json').exists():
            raise ValueError('BE must complete before shared cleanup')
        origin=Path(cfg['cnoh_run'])/'private/edits'
        paths=list(origin.glob('e*/C_NO_H/latest.pt'))+list(origin.glob('e*/CP_W0/latest.pt'))
        paths+=list(origin.glob('e*/initial/*.pt'))+list(origin.glob('e*/initial/native/expert.pt'))
    else:
        origin=phase_root
        paths=list(origin.glob('e*/state.pt'))+list(origin.glob('e*/state/adapter_model.safetensors'))
    # Receipts/raw generations remain in place; methods below never consume these weights.
    for directory in phase_root.glob('e*'):
        if not (directory/'COMPLETE.json').is_file(): raise ValueError('Incomplete edit consumer')
        if not list((directory/'native').glob('*.json')): raise ValueError('Missing native output')
    validated=[]
    for p in paths:
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(origin.resolve()):
            raise ValueError('Unsafe checkpoint target')
        validated.append(dict(path=str(p.resolve()),bytes=p.stat().st_size))
    project=Path(cfg['project']).resolve()
    mount=subprocess.check_output(['findmnt','-n','-T',str(origin),'-o','TARGET'],text=True).strip()
    if mount!='/root/rivermind-data' or not origin.resolve().is_relative_to(project):
        raise ValueError('Unexpected checkpoint mount/project')
    # Own phase child has exited; require no process to hold an enumerated target open.
    targets={r['path'] for r in validated}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd) in targets: raise RuntimeError('Checkpoint still open')
                except FileNotFoundError: pass
        except FileNotFoundError: pass
        except PermissionError: raise RuntimeError('Cannot verify checkpoint open-file safety')
    plan=phase_root/'CLEANUP_PLAN.json'
    if plan.exists(): raise RuntimeError('Partial cleanup requires reviewed recovery')
    write_new(plan,validated)
    for r in validated: Path(r['path']).unlink()
    write_new(receipt,dict(files=len(validated),bytes=sum(r['bytes'] for r in validated),
        backup=False,recovery='retraining required',preserved='raw outputs/tokens/bindings/receipts/configs',
        last_consumer=f'{method}_{mode}',status='DELETED'))


def phase(cfg, method, mode):
    setup(cfg)
    import torch
    from m3bench_repro.editors.methods import (BalanceEditPaperSpecEditor, LoraPaperSpecEditor,
        LoraRuntimeConfig, GracePaperSpecEditor, BeloraPaperSpecEditor)
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    from scripts.medtrace.run_realmodel_core import load_real_runtime, LAYER
    from scripts.medtrace.run_selective_write import save
    from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert

    root=Path(cfg['run']); source=Path(cfg['source_run']); croot=Path(cfg['cnoh_run'])
    out=root/'private'/f'{method}_{mode}'; out.mkdir(parents=True,exist_ok=True)
    ledger=read(source/'private/COHORT_AND_SUPPORT_LEDGER.json')
    if digest({k:v for k,v in ledger.items() if k!='freeze_id'})!=cfg['freeze_id']:
        raise ValueError('Frozen ledger mismatch')
    lookup={t['edit_id']:t for t in ledger['tasks']}
    tasks=[lookup[e] for e in ledger['main_T0']]
    if len(tasks)!=cfg['N']: raise ValueError('Queue count mismatch')
    bindings=read(source/'private/BINDINGS.json')
    phase_binding=dict(freeze_id=cfg['freeze_id'],method=method,mode=mode,
        runtime=cfg['runtime_lock'],code_commit=cfg['code_commit'],method_lock=cfg['methods'],
        order=ledger['main_T0'],prefixes=prefixes(len(tasks)))
    marker=out/'BINDING.json'
    if marker.exists():
        if read(marker)!=phase_binding: raise ValueError('Phase binding mismatch')
    else: write_new(marker,phase_binding)
    if (out/'COMPLETE.json').exists(): return
    if (out/'FAILURE.json').exists(): raise RuntimeError('Prior phase failure requires reviewed recovery')
    check_space(root)
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    runtime.run_root=out/'work'
    if runtime.generation_config != next(iter(bindings.values()))['generation']:
        raise ValueError('Generation configuration changed')
    base_parameters=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(20260912)
    replay=method in ('C_NO_H','balancedit')
    if replay: editor=BalanceEditPaperSpecEditor(runtime)
    elif method=='lora':
        p=cfg['methods']['LoRA_Perf_v1']['profile']
        editor=LoraPaperSpecEditor(runtime,LoraRuntimeConfig(**{k:p[k] for k in
            ('profile_name','learning_rate','steps_per_edit','stop_rule','min_steps','check_interval','target_nll_threshold')}))
    else: editor={'grace':GracePaperSpecEditor,'belora':BeloraPaperSpecEditor}[method](runtime)
    actual=editor.config_lock()
    if method=='lora':
        if (actual['rank'],actual['lora_alpha'],actual['epochs_per_edit'],actual['learning_rate'])!=(16,16,80,.0005):
            raise ValueError('LoRA-Perf recipe mismatch')
    elif not replay:
        expected=cfg['methods']['baselines'][method]
        for key,value in expected.items():
            if key in actual and key!='config_sha256' and actual[key]!=value:
                raise ValueError('Frozen baseline mismatch: '+key)
    if not (out/'ACTUAL_METHOD.json').exists(): write_new(out/'ACTUAL_METHOD.json',actual)
    hook=None; bank=[]; sources={}; source_bindings={}; loaded=None; started=time.time()
    if method=='C_NO_H':
        cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device)
        expert=LowRankExpert(cp,20260912,rank=4).to(runtime.device).requires_grad_(False)
        del cp
        hook=MedTraceLayerHook(runtime.get_module(LAYER),expert); hook.attach()

    def evaluate(qid, directory, prefix, inserted):
        nonlocal loaded
        q=ledger['queries'][qid]; b=bindings[q['opaque_Base_id']]
        binding=dict(phase=phase_binding,input=q,generation=runtime.generation_config,
            prefix=prefix,inserted=list(inserted))
        dest=directory/(digest(qid)+'.json')
        if dest.exists():
            if read(dest)['binding']!=binding: raise ValueError('Output binding mismatch')
            return
        check_space(root)
        query=replace(record_for(tasks[0]),record_id='query',question=q['question'],target='',
            official_rephrase='',image_path=Path(q['image_path']))
        raw=runtime.adapter.prepare_inputs(query.image_path,query.question,None)
        if (raw['input_ids'].tolist()!=[b['prompt_ids']] or raw['attention_mask'].tolist()!=[b['attention_mask']]
                or raw['image_sha256']!=b['image_sha256']):
            raise ValueError('Realized Base input mismatch')
        with torch.inference_mode():
            if replay:
                if hook: hook.clear_request_routing()
                decision=editor._route(query)
                if decision.nearest_logical_edit_id not in inserted: raise ValueError('Future expert visible')
                on=routes(decision.nearest_distance,decision.radius)
                base=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
                if on['R0']:
                    selected=decision.logical_edit_id
                    if loaded!=selected:
                        state=torch.load(sources[selected],map_location='cpu',weights_only=False)
                        if method=='C_NO_H':
                            if (state['canonical_edit_id']!=selected or state['condition']!='C_NO_H'
                                    or state['step']!=320 or state['seed']!=lookup[selected]['seed']):
                                raise ValueError('C_NO_H expert provenance mismatch')
                            expert.load_state_dict(state['expert'])
                        else:
                            if state['binding']!=source_bindings[selected]: raise ValueError('BE expert binding mismatch')
                            editor.wrapper.clear()
                            editor.wrapper.load_exported_state(state['wrapper'])
                        loaded=selected; del state
                    context=hook.generation_request() if hook else editor._activated(selected)
                    with context: generated=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    output=dict(raw_answer=generated.decoded_text,raw_token_ids=list(generated.raw_token_ids))
                else: output=base
                modes=dict(R0=output,RC=output if on['RC'] else base)
                route=decision_as_json(decision)
            else:
                generated=editor.generate(query,use_cache=True)
                output=dict(raw_answer=generated['generation']['decoded_text'],raw_token_ids=generated['generation']['raw_token_ids'])
                modes=dict(NATIVE=output); route=generated['route']
        if any(p._version!=v or p.data_ptr()!=ptr for p,v,ptr in base_parameters):
            raise ValueError('Frozen Base mutated')
        write_new(dest,dict(binding=binding,Base_cache_id=q['opaque_Base_id'],modes=modes,route=route,
            canonical_cap=1024,status='GENERATED_NOT_SCORED'))

    try:
        for index,t in enumerate(tasks,1):
            check_space(root)
            directory=out/f'e{index:03d}'; directory.mkdir(exist_ok=True)
            if replay:
                sd=(croot/'private/edits' if method=='C_NO_H' else source/'private/single_BE')/f"e{t['order']:03d}"
                receipt=read(sd/'COMPLETE.json'); bnd=receipt['binding']
                if (receipt['status']!='GENERATED_NOT_SCORED' or bnd['freeze_id']!=cfg['freeze_id']
                        or bnd['input']!=t['native'] or bnd['runtime']!=cfg['runtime_lock']):
                    raise ValueError('Replay source binding mismatch')
                if method=='C_NO_H':
                    if (bnd['U_fit']!=t['U_fit'] or bnd['fit']!=t['fit_questions'] or bnd['seed']!=t['seed']
                            or bnd['method']!=cfg['cnoh_method']): raise ValueError('C_NO_H training mismatch')
                    router=torch.load(sd/'ROUTER.pt',map_location='cpu',weights_only=True)
                    point=sd/'C_NO_H/latest.pt'
                else:
                    if bnd['training']['native_fit']!=t['fit_questions']: raise ValueError('BE fit mismatch')
                    state=torch.load(sd/'state.pt',map_location='cpu',weights_only=False)
                    router=state['router']; del state
                    point=sd/'state.pt'
                if len(router['entries'])!=1 or router['entries'][0]['logical_edit_id']!=t['edit_id']:
                    raise ValueError('Independent router mismatch')
                bank.append(dict(router['entries'][0],label=[])); sources[t['edit_id']]=point
                source_bindings[t['edit_id']]=bnd
                editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=bank),device=runtime.device)
                bankfile=directory/'BANK.pt'
                if not bankfile.exists(): save(bankfile,editor.router.export_state())
                restored=MemoryRouter.from_state(torch.load(bankfile,map_location='cpu',weights_only=True),device=runtime.device)
                if restored.logical_ids!=ledger['main_T0'][:index] or any(not torch.equal(a,b) for a,b in zip(restored.keys,editor.router.keys)):
                    raise ValueError('Prefix state save/load mismatch')
                editor.router=restored
            else:
                checkpoint=directory/('state' if method=='lora' else 'state.pt')
                if mode=='single': editor.reset_editor_state()
                receipt=directory/'TRAINING.json'
                if receipt.exists():
                    saved=read(receipt)
                    if saved['input']!=t or saved['phase']!=phase_binding: raise ValueError('Training resume mismatch')
                    editor.load_editor_state(checkpoint)
                else:
                    if checkpoint.exists(): raise RuntimeError('Interrupted checkpoint needs reviewed recovery')
                    before=time.time(); training=editor.apply_edit(record_for(t))
                    if not training['finite_losses'] or not training['finite_gradients']: raise FloatingPointError('Nonfinite training')
                    editor.save_editor_state(checkpoint)
                    history=list(editor.edit_history)
                    probe=replace(record_for(t),record_id='query',target='',official_rephrase='')
                    before_load=editor.generate(probe) if index==1 else None
                    editor.reset_editor_state(); editor.load_editor_state(checkpoint)
                    if editor.edit_history!=history: raise ValueError('Save/load history mismatch')
                    if before_load is not None and editor.generate(probe)!=before_load:
                        raise ValueError('State save/load generation mismatch')
                    write_new(receipt,dict(input=t,phase=phase_binding,training=training,seconds=time.time()-before))
            inserted=ledger['main_T0'][:index] if mode=='sequential' else [t['edit_id']]
            native=directory/'native'; native.mkdir(exist_ok=True)
            evaluate(t['edit_id'],native,index if mode=='sequential' else 1,inserted)
            if mode=='single' or index in prefixes(len(tasks)):
                panel=directory/'panel'; panel.mkdir(exist_ok=True)
                selected=tasks[:index] if mode=='sequential' else [t]
                ids=list(dict.fromkeys(q for row in selected for q in query_ids(row)))
                for qid in ids: evaluate(qid,panel,index if mode=='sequential' else 1,inserted)
            if not (directory/'COMPLETE.json').exists():
                write_new(directory/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',index=index,phase=phase_binding))
            write_json_atomic(root/'public/PROGRESS.json',dict(status='RUNNING',method=method,mode=mode,completed=index,N=len(tasks)))
            print('COMPLETE',method,mode,index,len(tasks),flush=True)
            gc.collect(); torch.cuda.empty_cache()
        write_new(out/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',phase=phase_binding,N=len(tasks),
            seconds=time.time()-started,replay=replay,training_reused=replay,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except Exception as error:
        write_new(out/'FAILURE.json',dict(error=repr(error),phase=phase_binding)); raise
    finally:
        if hook: hook.detach()


def run(cfg):
    """One finite dependency queue, with no GPU sharing or implicit semantic retry."""
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    root=Path(cfg['run'])
    with (root/'private/campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        roles=role_map(read(Path(cfg['source_run'])/'private/COHORT_AND_SUPPORT_LEDGER.json'))
        rolefile=root/'private/PREFIX_ACTIVE_TARGETS.json'
        if rolefile.exists():
            if read(rolefile)!=roles: raise ValueError('Predeclared prefix roles changed')
        else: write_new(rolefile,roles)
        progress=Path(cfg['cnoh_run'])/'public/SINGLE_PROGRESS.json'
        while True:
            check_space(root); state=read(progress)
            if state['status']=='GENERATED_NOT_SCORED' and state['completed']==cfg['N']: break
            if state['status']!='RUNNING': raise RuntimeError('Predecessor did not finish: '+str(state))
            os.kill(cfg['predecessor_pid'],0)
            write_json_atomic(root/'public/PROGRESS.json',dict(status='WAITING_FOR_EXISTING_C_NO_H',completed=state['completed'],N=cfg['N']))
            time.sleep(60)
        # The predecessor announces generation before its final process teardown.
        with (Path(cfg['cnoh_run'])/'private/worker.lock').open('a') as previous:
            fcntl.flock(previous,fcntl.LOCK_EX)
        for method,mode in schedule():
            check_space(root)
            free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu']),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<22000: raise RuntimeError('Insufficient GPU peak margin for next phase')
            # Never mutate the active predecessor checkout or use scientific names in argv.
            env=dict(os.environ,JOB_METHOD=method,JOB_MODE=mode,JOB_ACTION='phase')
            with (root/f'{method}_{mode}.log').open('ab') as log:
                subprocess.run([cfg['python'],'-u',cfg['entry']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if method=='balancedit':
                cleanup(cfg,'balancedit','sequential'); cleanup(cfg,'C_NO_H','sequential')
            elif method not in ('C_NO_H','balancedit'): cleanup(cfg,method,mode)
        write_json_atomic(root/'public/PROGRESS.json',dict(status='GPU_GENERATED_NOT_SCORED',
            phases=schedule(),remaining='Astra packets/scoring, full report, public delivery, dependency cleanup'))


if __name__=='__main__':
    cfg=read(Path(os.environ['JOB_ROOT'])/'private/DISPATCH.json')
    try:
        if os.environ.get('JOB_ACTION')=='phase': phase(cfg,os.environ['JOB_METHOD'],os.environ['JOB_MODE'])
        else: run(cfg)
    except Exception as error:
        from m3bench_repro.editors.llava_runtime import write_json_atomic
        write_json_atomic(Path(cfg['run'])/'public/PROGRESS.json',dict(status='STOPPED',error=repr(error),
            action=os.environ.get('JOB_ACTION','campaign'),method=os.environ.get('JOB_METHOD'),mode=os.environ.get('JOB_MODE')))
        raise
