"""Qualified existing-data DEV: three shared-W0 branches, BE, and true prefix replay."""
from dataclasses import asdict, replace
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_smoke import worker as train_c


def prefixes(n):
    return sorted({p for p in (1,10,25,50,100,n) if 0<p<=n})


def run(cfg):
    root=Path(cfg['run']); packet=read(root/'private/DEV_EVALUATION.json')
    gate=read(root/'private/DEV_QUALIFICATION.json')
    if digest(packet)!=gate['evaluation_binding'] or cfg['mode']!='EXPLORATORY_DEV_PILOT':
        raise ValueError('DEV evaluation binding changed')
    tasks=read(root/'private/TRAINING_TASKS.json')['tasks']
    if [p['candidate_id'] for p in packet['candidate_packages']]!=[t['canonical_edit_id'] for t in tasks]:
        raise ValueError('Training/evaluation order mismatch')
    runtime=train_c(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.llava_runtime import EditorRecord,write_json_atomic
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace.stage18_cfact import LAYER,BRANCHES,assert_base_off
    from scripts.medtrace import stage15
    baseline=read(root/'private/QUALIFICATION_BASE_OUTPUTS.json')
    base_by_query={r['query_id']:r for r in baseline['records']}
    if digest(baseline)!=gate['base_binding']:raise ValueError('Qualified Base changed')
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    status=root/'public/PROGRESS.json'; target=runtime.target_lock['balancedit']['targets'][0]
    def check():
        stage15.budget(root,cfg)
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('Storage reserve')
        if any(p._version!=v or p.data_ptr()!=ptr or p.requires_grad for p,v,ptr in frozen):raise ValueError('Frozen backbone changed')
    def record(t):
        n=t['native']
        return EditorRecord(t['canonical_edit_id'],n['dataset'],n['question'],n['reference'],t['fit_questions'][0],Path(n['image_path']),n['image_path'],t['order'],'VERIFIED_SOURCE_ANSWER','NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')
    def panel(packages):
        rows={}
        for p in packages:
            for row in [p['training']['native']]+p['evaluation']:
                rows.setdefault(digest([row['image_sha256'],row['question']]),row)
        return rows
    records={t['canonical_edit_id']:record(t) for t in tasks}
    task_by_id={t['canonical_edit_id']:t for t in tasks}
    query_record=next(iter(records.values()))
    cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device)
    expert=LowRankExpert(cp,20260912,rank=4).to(runtime.device).requires_grad_(False);del cp
    routers={t['canonical_edit_id']:torch.load(root/'private/edits'/f"e{t['order']:03d}"/'ROUTER.pt',map_location='cpu',weights_only=True) for t in tasks}
    directories={t['canonical_edit_id']:root/'private/edits'/f"e{t['order']:03d}" for t in tasks}
    outputs=[]
    def evaluate(method,mode,index,inserted,rows,router,hook=None,editor=None,forced=False):
        for qid,row in rows.items():
            check()
            if hook:hook.clear_request_routing()
            if editor:editor.wrapper.set_active(None)
            if editor:
                if editor.wrapper.active_logical_id is not None or any(p.requires_grad for p in runtime.model.parameters()):
                    raise ValueError('BE Base path is not inactive and frozen')
            else:assert_base_off(runtime)
            query=replace(query_record,record_id='query',question=row['question'],target='',official_rephrase='',image_path=Path(row['image_path']))
            base,raw,_,_=stage15.base_output(runtime,root,row,query)
            qualified=base_by_query[qid]['output']
            if base['binding']!=qualified['binding'] or base['raw_token_ids']!=qualified['raw_token_ids']:
                raise ValueError('Runtime Base drifted from qualification')
            with torch.inference_mode():
                key=runtime.extract_layer_input_key(runtime.build_question_batch(query),module_path=target,pooling='mean')
                decision=router.route(key)
                if decision.nearest_logical_edit_id not in inserted:raise ValueError('Future expert visible')
                selected=decision.nearest_logical_edit_id
                actual=base;own=None
                if decision.activated or forced:
                    if hook:
                        state=torch.load(directories[selected]/method/'latest.pt',map_location='cpu',weights_only=False)
                        if (state['binding']['branch']!=method or state['binding']['task']!=task_by_id[selected]
                                or state['binding']['code']!=cfg['code_commit'] or state['step']!=320):raise ValueError('Writer binding mismatch')
                        expert.load_state_dict(state['expert']);del state
                        own=stage15.generate(runtime,raw,base['binding'],hook)
                    else:
                        state=torch.load(directories[selected]/'BE.pt',map_location='cpu',weights_only=False)
                        if state['task']!=task_by_id[selected] or state['code']!=cfg['code_commit']:raise ValueError('BE source changed')
                        editor.wrapper.clear();editor.wrapper.load_exported_state(state['wrapper']);del state
                        with editor._activated(selected):own=stage15.generate(runtime,raw,base['binding'])
                    if decision.activated:actual=own
                rc=actual if decision.nearest_distance<=decision.radius*.7696741135364367 else base
            outputs.append(dict(method=method,mode=mode,prefix=index,inserted=list(inserted),query_id=qid,
                source=row,route=asdict(decision),Base=base,R0=actual,RC=rc,FORCED_OWN=own if forced else None,
                interpretation='EXPLORATORY_SOURCE_LABEL_DEV_PATIENT_UNKNOWN',code=cfg['code_commit']))
    def replay(method,hook=None,editor=None):
        entries=[]
        for index,p in enumerate(packet['candidate_packages'],1):
            t=p['training'];edit=t['canonical_edit_id'];entry=routers[edit]['entries'][0]
            if entry['logical_edit_id']!=edit:raise ValueError('Router identity mismatch')
            single=MemoryRouter.from_state(dict(distance='euclidean',entries=[entry]),device=runtime.device)
            evaluate(method,'single',1,[edit],panel([p]),single,hook,editor,forced=hook is not None)
            entries.append(dict(entry,label=[]));router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=runtime.device)
            inserted=[t['canonical_edit_id'] for t in tasks[:index]]
            save(root/'private'/f'{method}_prefix_{index:03d}.pt',router.export_state())
            restored=MemoryRouter.from_state(torch.load(root/'private'/f'{method}_prefix_{index:03d}.pt',map_location='cpu',weights_only=True),device=runtime.device)
            if restored.logical_ids!=inserted or any(not torch.equal(a,b) for a,b in zip(router.keys,restored.keys)):raise ValueError('Bank save/load mismatch')
            rows=panel(packet['candidate_packages'][:index]) if index in prefixes(len(tasks)) else panel([dict(p,evaluation=[])])
            evaluate(method,'sequential',index,inserted,rows,restored,hook,editor)
            write_json_atomic(status,dict(status='RUNNING',phase='PILOT_GENERATION',method=method,completed=index,N=len(tasks)))
        write_new(root/'private'/f'{method}_EVALUATION.json',dict(records=[r for r in outputs if r['method']==method],status='GENERATED_NOT_SCORED'))
        print('METHOD_GENERATED',method,flush=True)
    for method in BRANCHES:
        hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach()
        try:replay(method,hook=hook)
        finally:hook.detach()
    del expert;gc.collect();torch.cuda.empty_cache()
    editor=BalanceEditPaperSpecEditor(runtime)
    lock=editor.config_lock()
    if lock['steps_per_edit']!=50 or lock['learning_rate']!=.01 or lock['alpha']!=.2:raise ValueError('BE recipe changed')
    write_new(root/'private/BE_METHOD.json',lock)
    for index,t in enumerate(tasks,1):
        check();edit=t['canonical_edit_id'];editor.reset_editor_state()
        point=directories[edit]/'BE.pt'
        if point.exists():raise ValueError('Existing BE state requires explicit recovery')
        write_json_atomic(status,dict(status='RUNNING',phase='BE_TRAINING',completed=index-1,N=len(tasks)))
        training=editor.apply_edit(records[edit])
        if not training['finite_losses'] or not training['finite_gradients']:raise FloatingPointError('BE nonfinite')
        editor.router.labels=[()]*len(editor.router)
        state=dict(task=t,wrapper=editor.wrapper.export_state(),router=editor.router.export_state(),training=training,code=cfg['code_commit'])
        save(point,state);restored=torch.load(point,map_location='cpu',weights_only=False)
        for key,weights in state['wrapper']['edits'].items():
            if not all(torch.equal(w,restored['wrapper']['edits'][key][name]) for name,w in weights.items()):raise ValueError('BE save/load mismatch')
        if not torch.equal(routers[edit]['entries'][0]['key'],state['router']['entries'][0]['key']):raise ValueError('C/BE Base router mismatch')
        if routers[edit]['entries'][0]['radius']!=state['router']['entries'][0]['radius']:raise ValueError('C/BE radius mismatch')
        del state,restored;editor.reset_editor_state();gc.collect();torch.cuda.empty_cache()
    replay('BalancEdit',editor=editor)
    editor.reset_editor_state();check()
    write_new(root/'public/DEV_RESULT.json',dict(status='FOUR_METHODS_GENERATED_NOT_SCORED',N=len(tasks),methods=list(BRANCHES)+['BalancEdit'],
        modes=['single','sequential'],prefixes=prefixes(len(tasks)),output_records=len(outputs),formal_results=False,
        checkpoints='Preserved until result completeness and registered consumers are checked; no historical state deleted'))
    write_json_atomic(status,dict(status='FOUR_METHODS_GENERATED_NOT_SCORED',N=len(tasks),completed=len(tasks)))


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG']);root=Path(cfg['run'])
    with (root/'private/campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(cfg)
        except Exception as error:
            write_new(root/'public/FAILURE.json',dict(error=repr(error),code=cfg['code_commit']))
            raise
