"""Two source-label smoke edits; no formal cohort, cloud Judge, or implicit expansion."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup
from scripts.medtrace.stage18_support import validate_task


def worker(cfg):
    setup(cfg)
    import torch
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.llava_runtime import EditorRecord, write_json_atomic
    from m3bench_repro.editors.routing import MemoryRouter, balanced_radius
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import LAYER,BRANCHES,assert_base_off,state_hash,teachers_for,train
    root=Path(cfg['run']); tasks=read(root/'private/TRAINING_TASKS.json')
    if tasks['scope']!='TWO_SOURCE_LABEL_SMOKES_ONLY' or digest(tasks['tasks'])!=tasks['freeze_id'] or tasks['freeze_id']!=cfg['freeze_id']: raise ValueError('Dispatch input changed')
    if [t['canonical_edit_id'] for t in tasks['tasks']]!=['core9q-1d96e59c0f64404d1cd49434','core9q-24ba2d3222c8a4ef7adedee8']:
        raise ValueError('Only the two authorized smoke edits accepted')
    if cfg['mode']!='SOURCE_LABEL_SMOKE' or cfg['branches']!=list(BRANCHES): raise ValueError('Unauthorized scope')
    if shutil.disk_usage(root).free<12*1024**3: raise OSError('Storage budget insufficient')
    old_budget=stage15.budget
    def budget(run,settings):
        old_budget(run,settings)
        if shutil.disk_usage(root).free<8*1024**3: raise OSError('Storage reserve reached')
    stage15.budget=budget
    # Source labels are checked against the actual approved author train file.
    source_rows={str(r['qid']):r for r in read(Path(cfg['source_train']))}
    for t in tasks['tasks']:
        validate_task(t)
        for row in [t['native']]+t['U_fit']+t['H_fit']+t['G_fit']:
            if not Path(row['image_path']).is_file(): raise FileNotFoundError(row['image_path'])
            if row['role']!='native':
                raw=source_rows[str(row['source_qid'])]
                if raw['question']!=row['question'] or raw['answer']!=row['reference'] or Path(row['image_path']).parent.name!=Path(raw['img_name']).parent.name:
                    raise ValueError('Own source image/question/answer mismatch')
    status=root/'public/PROGRESS.json'
    write_json_atomic(status,dict(status='RUNNING',phase='LOADING_RUNTIME',N=2,completed=0))
    print('LOADING_RUNTIME',flush=True)
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    runtime.run_root=root/'private/work'
    if runtime.generation_config!=cfg['generation_lock']: raise ValueError('Generation mismatch')
    if next(runtime.model.parameters()).dtype!=torch.float16: raise ValueError('FP16 backbone required')
    runtime.model.eval(); assert_base_off(runtime)
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    receipt=dict(generation=runtime.generation_config,torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),uuid=cfg['gpu_uuid'],code=cfg['code_commit'],runtime=cfg['runtime_lock'])
    if (root/'private/RUNTIME.json').exists():
        if read(root/'private/RUNTIME.json')!=receipt: raise ValueError('Runtime receipt changed')
    else: write_new(root/'private/RUNTIME.json',receipt)
    print('RUNTIME_READY',flush=True); completed=0
    for task in tasks['tasks']:
        order=task['order']; directory=root/'private/edits'/f'e{order:03d}'; directory.mkdir(parents=True,exist_ok=True)
        binding=dict(task=task,code=cfg['code_commit'],runtime=cfg['runtime_lock'],generation=runtime.generation_config)
        if (directory/'BINDING.json').exists():
            if read(directory/'BINDING.json')!=binding: raise ValueError('Initialization binding mismatch')
        else: write_new(directory/'BINDING.json',binding)
        if (directory/'FAILURE.json').exists(): raise ValueError('Failure requires explicit versioned recovery')
        if (directory/'COMPLETE.json').exists(): completed+=1; continue
        began=time.time(); n=task['native']
        record=EditorRecord(task['canonical_edit_id'],n['dataset'],n['question'],n['reference'],task['fit_questions'][0],Path(n['image_path']),n['image_path'],order,'VERIFIED_SOURCE_ANSWER','NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')
        adapted=dict(canonical_edit_id=task['canonical_edit_id'],order=order,seed=task['seed'],probes=[n],U=task['U_fit'],fit_questions=task['fit_questions'])
        try:
            write_json_atomic(status,dict(status='RUNNING',phase='SHARED_NATIVE_A2_W0',order=order,completed=completed,N=2))
            print('INITIALIZE',order,flush=True)
            cp=stage15.initialize(runtime,root,cfg,adapted,record=record,seed_base=20260912)
            template=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device)
            if sum(p.numel() for p in template.parameters())!=73728: raise ValueError('Writer parameter count')
            with torch.no_grad():
                x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,14336)
                if not torch.allclose(cp.residual(x),template.residual(x),rtol=2e-4,atol=2e-5): raise ValueError('CP transfer mismatch')
            del cp,x
            initial=copy.deepcopy(template.state_dict()); w0=state_hash(template)
            save(directory/'SHARED_W0.pt',dict(expert=initial,W0=w0,binding=binding))
            teachers=teachers_for(runtime,root,cfg,task,record)
            assert_base_off(runtime)
            target=runtime.target_lock['balancedit']['targets'][0]
            def key_for(r):
                assert_base_off(runtime)
                with torch.inference_mode(): return runtime.extract_layer_input_key(runtime.build_question_batch(r),module_path=target,pooling='mean')
            key=key_for(record); positive=key_for(replace(record,question=task['fit_questions'][0]))
            black=runtime.make_black_image(record,directory/'black'); negative=key_for(replace(record,image_path=black))
            router=MemoryRouter('euclidean'); router.add(task['canonical_edit_id'],key,balanced_radius(key,positive,negative,alpha=.2,distance='euclidean'))
            save(directory/'ROUTER.pt',router.export_state())
            restored=MemoryRouter.from_state(torch.load(directory/'ROUTER.pt',weights_only=True),device=runtime.device)
            if asdict(router.route(key))!=asdict(restored.route(key)): raise ValueError('Router save/load mismatch')
            rows=[n]+task['H_fit']+task['G_fit']+task['U_fit']; prepared=[]
            for row in rows:
                base,raw,_,_=stage15.base_output(runtime,root,row,record)
                q=replace(record,question=row['question'],target='',official_rephrase='',image_path=Path(row['image_path']))
                prepared.append((row,base,raw,key_for(q)))
            # Real-model zero-residual ON must reproduce the fresh Base tokens.
            with torch.no_grad(): template.B.zero_()
            zero_hook=MedTraceLayerHook(runtime.get_module(LAYER),template); zero_hook.attach()
            try:
                _,base,raw,_=prepared[0]
                if stage15.generate(runtime,raw,base['binding'],zero_hook)['raw_token_ids']!=base['raw_token_ids']: raise ValueError('Zero residual parity failure')
            finally: zero_hook.detach()
            template.load_state_dict(initial)
            diagnostics=[]
            for branch in BRANCHES:
                template.load_state_dict(initial)
                expert=train(runtime,root,cfg,task,template,branch,record,teachers,w0)
                hook=MedTraceLayerHook(runtime.get_module(LAYER),expert); hook.attach()
                try:
                    for row,base,raw,qkey in prepared:
                        hook.clear_request_routing(); assert_base_off(runtime)
                        query=replace(record,question=row['question'],target='',official_rephrase='',image_path=Path(row['image_path']))
                        if not torch.equal(key_for(query),qkey): raise ValueError('Base routing feature changed under writer')
                        route=asdict(router.route(qkey)); forced=stage15.generate(runtime,raw,base['binding'],hook)
                        if row['role']=='native':
                            off=stage15.generate(runtime,raw,base['binding'])
                            if off['raw_token_ids']!=base['raw_token_ids']: raise ValueError('OFF did not recover Base')
                            actual=stage15.generate(runtime,raw,base['binding'],hook if route['activated'] else None)
                            if actual['raw_token_ids']!=(forced if route['activated'] else base)['raw_token_ids']: raise ValueError('R0 actual generation mismatch')
                        diagnostics.append(dict(branch=branch,role=row['role'],source=row,Base=base,FORCED_OWN=forced,route=route,
                            R0=forced if route['activated'] else base,RC=forced if route['nearest_distance']<=route['radius']*.7696741135364367 else base,
                            interpretation='SOURCE_LABEL_TRAINING_DIAGNOSTIC_NOT_INDEPENDENT_EVALUATION'))
                finally: hook.detach()
            if any(p._version!=v or p.data_ptr()!=ptr or p.requires_grad for p,v,ptr in frozen): raise ValueError('Frozen Base mutated')
            write_new(directory/'DIAGNOSTICS.json',diagnostics)
            write_new(directory/'COMPLETE.json',dict(status='SMOKE_PASS_NOT_SCORED',branches=list(BRANCHES),W0=w0,CP_transfer=True,zero_residual=True,Base_OFF=True,
                Base_route_isolation=True,save_load=True,seconds=time.time()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),independent_H_eval=0))
            completed+=1; del template,initial,teachers,prepared,diagnostics; gc.collect(); torch.cuda.empty_cache()
            print('EDIT_COMPLETE',completed,2,flush=True)
        except Exception as error:
            write_new(directory/'FAILURE.json',dict(error=repr(error),code=cfg['code_commit'])); raise
    write_json_atomic(status,dict(status='SMOKE_PASS_NOT_SCORED',completed=completed,N=2,formal_N=0))
    write_new(root/'public/SMOKE_RESULT.json',dict(status='PASS',N=2,branches=list(BRANCHES),independent_H_eval=0,formal_results=False))
    write_new(root/'public/GPU3_FOLLOWUP.json',dict(status='BLOCKED_DATA_CONTRACT',requested_gpu=3,compute_authorized=True,
        reasons=['No independent H_eval','No source-isolated DEV16','No fresh scored Base mask; cloud Judge budget not granted'],automatic_formal_start=False))


if __name__=='__main__':
    cfg=read(Path(os.environ['JOB_CONFIG']))
    root=Path(cfg['run'])
    with (root/'private/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: worker(cfg)
        except Exception as error:
            from m3bench_repro.editors.llava_runtime import write_json_atomic
            write_json_atomic(root/'public/PROGRESS.json',dict(status='FAILED',error=repr(error)))
            raise
