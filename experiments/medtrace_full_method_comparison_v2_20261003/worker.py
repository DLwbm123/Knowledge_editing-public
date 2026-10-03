"""Original146 available-H dispatch using unchanged Stage15/18 writer updates."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback

RUN = Path(os.environ['RUN_ROOT'])
SOURCE = RUN/'private/source'
sys.path.insert(0, str(SOURCE))
LAYER = 'model.layers.21.mlp.down_proj'
GPUS = {4:'GPU-fb8f2a01-3910-41f5-39f3-9ce03b7cb7dd',5:'GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca'}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n'); tmp.replace(p)


@contextmanager
def locked_ledger():
    with (RUN/'RESOURCE_LEDGER.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); ledger = read(RUN/'RESOURCE_LEDGER.json')
        yield ledger
        write(RUN/'RESOURCE_LEDGER.json', ledger)


def budget(*_):
    manifest = read(RUN/'RUN_MANIFEST.json')
    ledger = read(RUN/'RESOURCE_LEDGER.json')
    used = ledger['gpu_seconds_used'] + sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None)
    if (RUN/'STOP').exists() or time.time() >= manifest['deadline_epoch'] or used >= manifest['GPU_seconds_limit']:
        raise TimeoutError('Persisted stop/deadline/cumulative GPU budget; retain partial state')
    if shutil.disk_usage(RUN).free < 8*1024**3:
        raise OSError('8GiB data-disk reserve reached')


@contextmanager
def lease(gpu):
    budget()
    actual = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert actual[0] == GPUS[gpu] and int(actual[1]) >= 40000, 'GPU UUID/free-memory admission failed'
    with locked_ledger() as ledger:
        assert not any(s['gpu_uuid'] == GPUS[gpu] and s.get('ended_epoch') is None for s in ledger['gpu_sessions'])
        entry = dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_uuid=GPUS[gpu],started_epoch=time.time(),action=os.environ['ACTION'])
        ledger['gpu_sessions'].append(entry)
    try:
        yield
    finally:
        with locked_ledger() as ledger:
            s = next(s for s in ledger['gpu_sessions'] if s['pid'] == os.getpid() and s['started_epoch'] == entry['started_epoch'])
            s['ended_epoch'] = time.time(); s['resident_seconds'] = s['ended_epoch']-s['started_epoch']; ledger['gpu_seconds_used'] += s['resident_seconds']


def local_row(row):
    from admit import relocate
    return dict(row,image_path=relocate(row['image_path']))


def record_for(t):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    n = local_row(t['native'])
    return EditorRecord(t['edit_id'],n['dataset'],n['question'],n['reference'],t['fit_questions'][0],Path(n['image_path']),n['original_image_path'],t['order'],'VERIFIED_SOURCE_ANSWER','NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')


def load(gpu):
    assert read(RUN/'private/CPU_ADMISSION.json')['status'] == 'PASS'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=GPUS[gpu],M3BENCH_LLAVA_SOURCE=str(RUN/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(RUN/'private/official_llava'),M3BENCH_MODEL_PATH='/data/bmw/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b',M3BENCH_VISION_PATH='/data/bmw/hugging_cache/openai/clip-vit-large-patch14-336',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TMPDIR=str(RUN/'private/tmp'))
    (RUN/'private/tmp').mkdir(exist_ok=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    assert commit == read(RUN/'private/SOURCE_COMMIT.json')['commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=RUN/'private/cpu_gate'))
    bindings = read(RUN/'private/legacy_stage17/BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    from scripts.medtrace import stage15
    stage15.budget = budget  # Same writer; resource guard replaces only old campaign deadline.
    runtime.run_root = RUN/'private/work'/str(gpu); runtime.run_root.mkdir(parents=True,exist_ok=True)
    return runtime, bindings


def check_input(runtime, row, bindings):
    n = local_row(row); b = bindings[row['opaque_Base_id']]
    raw = runtime.adapter.prepare_inputs(Path(n['image_path']),n['question'],None)
    assert raw['input_ids'].tolist() == [b['prompt_ids']] and raw['attention_mask'].tolist() == [b['attention_mask']] and raw['image_sha256'] == b['image_sha256'], 'Accepted Base input binding mismatch'
    return raw,b


def build_router(runtime, t, record):
    import torch
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor, balanced_radius, record_seed
    from m3bench_repro.editors.llava_runtime import seed_everything
    editor = BalanceEditPaperSpecEditor(runtime)
    try:
        seed_everything(record_seed(record.record_id,'balancedit'))
        with torch.no_grad():
            key,positive,negative,_ = editor._anchors(record,runtime.run_root/'inputs/black_images')
            radius = balanced_radius(key,positive,negative,alpha=.2,distance='euclidean')
        editor.router.add(t['edit_id'],key,radius,())
        return editor.router.export_state()
    finally:
        base,target = editor.wrapper.base,editor.target
        editor.reset_editor_state(); runtime.replace_module(target,base)


def continuation(runtime, t, expert, record, cfg, H):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import optimizer_for
    from m3bench_repro.editors.llava_runtime import seed_everything
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    seed_everything(t['seed']); expert.requires_grad_(True)
    optimizer = optimizer_for(expert,runtime.model)
    directory = RUN/'private/edits'/f"e{t['order']:03d}"/'AVAILABLE_H'; point = directory/'latest.pt'
    task = dict(t,U_fit=[local_row(u) for u in t['U_fit']],H_fit=H)
    for u in task['U_fit']:
        base,*_=cf.stage15.base_output(runtime,RUN,u,record)
        assert 0<len(base['raw_token_ids'])<=128 and base['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id, 'U teacher requires actual EOS within cap128; no truncation'
    teachers = cf.teachers_for(runtime,RUN,cfg,task,record)
    batches = [runtime.build_edit_batch(record)] + [runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    hb = [cf.source_batch(runtime,record,h) for h in H]
    eos = runtime.adapter.tokenizer.eos_token_id
    assert len(batches) == 5 and all(eos in b.target_token_ids for b in batches+hb)
    assert all(len(b.target_token_ids) <= 128 for b in hb), 'H actual EOS answer cap exceeded; no truncation'
    fit = list(range(1,5)); random.Random(t['seed']).shuffle(fit)
    order = cf.extra_schedule(task) if H else []
    binding = dict(input=t['native'],U=t['U_fit'],H=H,seed=t['seed'],source=cfg['code_commit'],runtime=cfg['runtime_lock'],W0=cf.state_hash(expert),fit_order=fit,H_order=order,steps=320,branch='C_FACT' if H else 'C_NO_H')
    curve=[]; step0=0
    if point.exists(): step0,curve=cf.resume(point,binding,expert,optimizer)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert); hook.attach()
    try:
        for step in range(step0+1,321):
            budget()
            extra=(hb[order[step-1]],H[order[step-1]]) if H else None
            item=cf.update(runtime,hook,expert,optimizer,batches[0],batches[fit[(step-1)%4]],teachers[(step-1)%len(teachers)],extra)
            item.update(step=step,fit_index=fit[(step-1)%4],H_index=order[step-1] if H else None); curve.append(item)
            if step%20 == 0:
                save(point,dict(binding=binding,expert=expert.state_dict(),optimizer=optimizer.state_dict(),step=step,curve=curve,**cf.rng_state()))
                write(directory/'PROGRESS.json',dict(step=step,N=320,branch=binding['branch'])); print('UPDATE',t['order'],step,flush=True)
        state=torch.load(point,map_location=runtime.device,weights_only=True)
        assert all(torch.equal(v,state['expert'][k]) for k,v in expert.state_dict().items())
        assert not H or any(c['terms']['extra']['weighted_gradient_norm']>0 for c in curve), 'H gradient did not contribute'
        write(directory/'TRAINING.json',dict(status='COMPLETE',steps=320,branch=binding['branch'],binding=binding,curve=curve))
    finally:
        hook.detach()
    expert.requires_grad_(False)
    return point


def query_ids(t):
    return list(dict.fromkeys([t['edit_id']]+[q for e in t['events'] for q in e['all_probe_query_ids']]))


def evaluate(runtime, ledger, bindings, ids, bank, points, directory, prefix, phase):
    import torch
    from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    from scripts.medtrace.stage17_prepare import digest
    cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device); expert=LowRankExpert(cp,20260912,rank=4).to(runtime.device).requires_grad_(False); del cp
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert); hook.attach()
    editor=BalanceEditPaperSpecEditor(runtime); editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=bank),device=runtime.device)
    loaded=None
    try:
        for qid in ids:
            budget(); q=ledger['queries'][qid]; raw,b=check_input(runtime,q,bindings)
            dest=directory/(digest(qid)+'.json'); binding=dict(input=q,prefix=prefix,phase=phase,inserted=[x['logical_edit_id'] for x in bank],generation=runtime.generation_config)
            if dest.exists():
                assert read(dest)['binding'] == binding; continue
            query=replace(record_for(next(t for t in ledger['tasks'] if t['edit_id']==ledger['main_T0'][0])),record_id='query',question=q['question'],target='',official_rephrase='',image_path=Path(local_row(q)['image_path']))
            hook.clear_request_routing()
            with torch.inference_mode():
                decision=editor._route(query); on=decision.nearest_distance <= decision.radius
                if on:
                    selected=decision.logical_edit_id; assert selected in points
                    if loaded != selected:
                        state=torch.load(points[selected],map_location='cpu',weights_only=True);assert state['step']==320 and state['binding']==read(points[selected].parent/'TRAINING.json')['binding'];expert.load_state_dict(state['expert']);loaded=selected
                    with hook.generation_request(): result=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    output=dict(raw_answer=result.decoded_text,raw_token_ids=list(result.raw_token_ids))
                else:
                    output=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
            write(dest,dict(binding=binding,Base_cache_id=q['opaque_Base_id'],R0=output,route=decision_as_json(decision),status='GENERATED_NOT_SCORED'))
    finally:
        hook.detach(); target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)


def train_one(runtime, ledger, bindings, t, cfg, mechanical=False):
    import torch
    from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace import stage15
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.llava_runtime import seed_everything
    directory=RUN/'private/edits'/f"e{t['order']:03d}";directory.mkdir(parents=True,exist_ok=True)
    H=read(RUN/'private/H_AVAILABLE.json').get(t['edit_id'],[])
    binding=dict(freeze_id=ledger['freeze_id'],input=t['native'],U_fit=t['U_fit'],fit=t['fit_questions'],seed=t['seed'],H=H,code=cfg['code_commit'],runtime=cfg['runtime_lock'],generation=runtime.generation_config,arm='MedTRACE_AVAILABLE_H_R0')
    if (directory/'COMPLETE.json').exists():
        assert read(directory/'COMPLETE.json')['binding'] == binding; return
    assert not (directory/'FAILURE.json').exists(), 'Prior failure requires root-cause recovery'
    frozen=[(p,p._version,p.data_ptr(),p.requires_grad) for p in runtime.model.parameters()]
    original_hooks=set(runtime.get_module(LAYER)._forward_hooks)
    record=record_for(t);runtime.run_root=directory/'work';runtime.run_root.mkdir(exist_ok=True)
    started=time.time();write(directory/'BINDING.json',binding)
    generated_bytes=sum(p.stat().st_size for p in (RUN/'private/edits').glob('**/*.pt'))+sum(p.stat().st_size for p in (RUN/'private/teacher').glob('*.pt'))
    assert generated_bytes < 64*1024**3, 'Frozen generated checkpoint cap64GiB reached'
    try:
        raw,b=check_input(runtime,t['native'],bindings)
        if mechanical:
            rng=torch.get_rng_state(); crng=torch.cuda.get_rng_state(); prng=random.getstate()
            zero=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),t['seed'],rank=4).to(runtime.device).requires_grad_(False)
            with torch.no_grad(): zero.B.zero_()
            hook=MedTraceLayerHook(runtime.get_module(LAYER),zero);hook.attach()
            try:
                with torch.inference_mode():
                    off=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    with hook.generation_request(): z=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                assert list(off.raw_token_ids) == list(z.raw_token_ids) == b['output']['raw_generated_token_ids'], 'Actual Base/zero/accepted native token parity failed'
            finally: hook.detach()
            del zero;torch.set_rng_state(rng);torch.cuda.set_rng_state(crng);random.setstate(prng)
        adapted=dict(canonical_edit_id=t['edit_id'],order=t['order'],seed=t['seed'],probes=[local_row(t['native'])],U=[local_row(u) for u in t['U_fit']],fit_questions=t['fit_questions'])
        cp=stage15.initialize(runtime,RUN,cfg,adapted,record=record,seed_base=20260912)
        expert=LowRankExpert(cp,t['seed'],rank=4).to(runtime.device)
        assert sum(p.numel() for p in expert.parameters()) == 73728
        with torch.no_grad():
            x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,14336);assert torch.allclose(cp.residual(x),expert.residual(x),rtol=2e-4,atol=2e-5)
        if mechanical:
            with torch.inference_mode():
                outputs=[]
                for w in (cp,expert):
                    h=MedTraceLayerHook(runtime.get_module(LAYER),w);h.attach()
                    try:
                        with h.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        outputs.append(list(g.raw_token_ids))
                    finally:h.detach()
                assert outputs[0]==outputs[1], 'Actual CP/freeR4 token transfer failed'
        del cp,x
        point=continuation(runtime,t,expert,record,cfg,H);del expert
        router=build_router(runtime,t,record);router['entries'][0]['label']=[];save(directory/'ROUTER.pt',router)
        evaluate(runtime,ledger,bindings,query_ids(t),router['entries'],{t['edit_id']:point},directory/'single',1,binding)
        assert all(p._version==v and p.data_ptr()==ptr and p.requires_grad==req==False for p,v,ptr,req in frozen), 'Frozen Base changed'
        assert set(runtime.get_module(LAYER)._forward_hooks)==original_hooks, 'Hook lifecycle leak'
        write(directory/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',binding=binding,mechanical=mechanical,source_branch='C_FACT' if H else 'C_NO_H',seconds=time.time()-started,queries=len(query_ids(t))))
        print('EDIT_COMPLETE',t['order'],flush=True)
    except Exception as exc:
        write(directory/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),binding=binding));raise
    finally:
        gc.collect();torch.cuda.empty_cache()


def main():
    gpu=int(os.environ['GPU']);assert gpu in GPUS
    with lease(gpu):
        runtime,bindings=load(gpu)
        ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');lookup={t['edit_id']:t for t in ledger['tasks']};tasks=[lookup[e] for e in ledger['main_T0']]
        cfg=dict(code_commit=read(RUN/'private/SOURCE_COMMIT.json')['commit'],runtime_lock=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'),campaign_epoch=1791036070.2599673,train_seconds=72*3600)
        action=os.environ['ACTION']
        if action=='mechanical':
            selected=[t for t in tasks if t['edit_id'] in read(RUN/'private/H_AVAILABLE.json')][:2]
            assert [t['order'] for t in selected] == [31,35]
            for t in selected:train_one(runtime,ledger,bindings,t,cfg,mechanical=True)
            write(RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',orders=[31,35],actual_H_gradient=True,Base_zero_accepted_token_parity=True,CP_freeR4_transfer=True,writer_save_load=True,Base_hooks_unchanged=True,GPU_uuid=GPUS[gpu],reuse_in_original146_single=True))
        elif action=='single':
            assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
            partition=int(os.environ['PARTITION'])
            for t in tasks[partition::2]:train_one(runtime,ledger,bindings,t,cfg)
            write(RUN/f'private/SINGLE_PART_{partition}_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',N=len(tasks[partition::2])))
        elif action=='sequential':
            assert all((RUN/'private/edits'/f"e{t['order']:03d}"/'COMPLETE.json').exists() for t in tasks)
            import torch
            bank=[];points={};phase=dict(arm='MedTRACE_AVAILABLE_H_R0',freeze=ledger['freeze_id'],source=cfg['code_commit'],order=ledger['main_T0'],prefixes=[1,50,100,146])
            for i,t in enumerate(tasks,1):
                d=RUN/'private/edits'/f"e{t['order']:03d}";r=torch.load(d/'ROUTER.pt',map_location='cpu',weights_only=True);assert len(r['entries'])==1 and r['entries'][0]['logical_edit_id']==t['edit_id'];bank+=r['entries'];points[t['edit_id']]=d/'AVAILABLE_H/latest.pt'
                evaluate(runtime,ledger,bindings,[t['edit_id']],bank,points,RUN/f'private/sequential/e{i:03d}/native',i,phase)
                if i in [1,50,100,146]:
                    ids=list(dict.fromkeys(q for row in tasks[:i] for q in query_ids(row)));evaluate(runtime,ledger,bindings,ids,bank,points,RUN/f'private/sequential/e{i:03d}/panel',i,phase)
                write(RUN/'public/PROGRESS.json',dict(status='SEQUENTIAL_RUNNING',completed=i,N=146))
            write(RUN/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',N=146,remaining='common new Judge epoch, full metrics, public delivery'))
        else:
            raise ValueError('Unknown action')


if __name__=='__main__':
    def stopped(*_):
        raise InterruptedError('Own worker stop requested; preserve last accepted checkpoint')
    signal.signal(signal.SIGTERM,stopped)
    try:
        main()
    except Exception as exc:
        write(RUN/'private'/('WORKER_FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('GPU','unknown')+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid(),epoch=time.time()))
        raise
