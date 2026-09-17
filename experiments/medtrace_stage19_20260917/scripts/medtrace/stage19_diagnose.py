"""Bounded forced FACT expert matrix; read-only weights, GPU3 only, no retraining."""
import argparse
from dataclasses import replace
import fcntl
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup


def run(cfg):
    if cfg['gpu']!='3' or cfg['gpu_uuid']!='GPU-43e3d478-7979-ea29-8130-64a467b48a5c':raise ValueError('Only physical GPU3 authorized')
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.llava_runtime import EditorRecord,write_json_atomic
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from scripts.medtrace.stage18_cfact import LAYER,assert_base_off
    from scripts.medtrace import stage15
    root=Path(cfg['run']);plan=read(root/'private/DIAGNOSTIC_PLAN.json');old=Path(plan['previous_run'])
    if digest({k:v for k,v in plan.items() if k!='freeze_id'})!=plan['freeze_id'] or cfg['freeze_id']!=plan['freeze_id']:raise ValueError('Plan changed')
    bank=torch.load(old/'private/FINAL_FACT_BANK.pt',map_location='cpu',weights_only=False)
    tasks=read(old/'private/TRAINING_TASKS.json')['tasks'];byid={t['canonical_edit_id']:t for t in tasks}
    entries={e['binding']['task']['canonical_edit_id']:e for e in bank['entries']}
    if list(entries)!=plan['experts'] or bank['code']!=plan['previous_code'] or len(plan['matrix'])!=128:raise ValueError('Retained bank scope changed')
    for e,t in zip(bank['entries'],tasks):
        if e['binding']['task']!=t or e['binding']['branch']!='C_FACT' or e['binding']['code']!=bank['code']:raise ValueError('Expert binding changed')
    original={r['query_id']:r for r in read(old/'private/QUALIFICATION_BASE_OUTPUTS.json')['records']}
    print('LOADING_RUNTIME',flush=True)
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])));runtime.run_root=root/'private/work'
    runtime.model.eval();assert_base_off(runtime)
    if runtime.generation_config!=cfg['generation_lock'] or next(runtime.model.parameters()).dtype!=torch.float16:raise ValueError('Runtime changed')
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=[e['router']['entries'][0] for e in bank['entries']]),device=runtime.device)
    native=tasks[0]['native'];record=EditorRecord('query',native['dataset'],'','','',Path(native['image_path']),native['image_path'],0,'DIAGNOSTIC','NO_GOLD_ROUTING')
    prepared={};base_checks=[];began=time.time()
    for q,row in plan['queries'].items():
        stage15.budget(root,cfg);assert_base_off(runtime)
        out,raw,_,_=stage15.base_output(runtime,root,row,record)
        expected=original[q]['output']
        if out['binding']!=expected['binding'] or out['raw_token_ids']!=expected['raw_token_ids']:raise ValueError('Base token/input drift')
        query=replace(record,question=row['question'],image_path=Path(row['image_path']))
        with torch.inference_mode():key=runtime.extract_layer_input_key(runtime.build_question_batch(query),module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
        prepared[q]=(out,raw,key);base_checks.append(dict(query_id=q,output=out))
    cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device);expert=LowRankExpert(cp,20260912,rank=4).to(runtime.device).requires_grad_(False);del cp
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach();checks=[];outputs=[]
    def forced(q,e):
        hook.clear_request_routing();assert_base_off(runtime);expert.load_state_dict(entries[e]['expert'])
        out,raw,_=prepared[q]
        return stage15.generate(runtime,raw,out['binding'],hook)
    try:
        for item in plan['reused'][:2]:
            out=forced(item['query_id'],item['expert'])
            if out['raw_token_ids']!=item['output']['raw_token_ids'] or out['binding']!=item['output']['binding']:raise ValueError('Restored expert differs from original output')
            checks.append(dict(query_id=item['query_id'],expert=item['expert'],output=out))
        q=plan['reused'][0]['query_id'];out,raw,_=prepared[q];assert_base_off(runtime)
        if stage15.generate(runtime,raw,out['binding'])['raw_token_ids']!=out['raw_token_ids']:raise ValueError('OFF did not restore Base')
        for i,item in enumerate(plan['new'],1):
            stage15.budget(root,cfg);q=item['query_id'];e=item['expert'];out=forced(q,e)
            route=router.route(prepared[q][2])
            outputs.append(dict(item,output=out,layer_id=21,expert_binding=digest(entries[e]['binding']),bank_code=bank['code'],
                code=cfg['code_commit'],plan=plan['freeze_id'],natural_final_selected=route.nearest_logical_edit_id,
                interpretation='FORCED_EXPERT_DIAGNOSTIC_NOT_DEPLOYMENT'))
            write_json_atomic(root/'public/PROGRESS.json',dict(status='RUNNING',new_combinations=i,N=len(plan['new'])))
            print('DIAGNOSTIC',i,len(plan['new']),flush=True)
    finally:hook.detach()
    if any(p.requires_grad or p._version!=v or p.data_ptr()!=ptr for p,v,ptr in frozen):raise ValueError('Frozen Base changed')
    write_new(root/'private/NEW_MATRIX_OUTPUTS.json',dict(records=outputs,plan=plan['freeze_id']))
    write_new(root/'private/RESTORE_CHECKS.json',dict(Base=base_checks,forced=checks,OFF=True))
    write_new(root/'public/DIAGNOSTIC_RESULT.json',dict(status='GENERATED_NOT_SCORED',counts=plan['counts'],actual_new=len(outputs),restore=True,
        Base_OFF=True,seconds_after_load=time.time()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),code=cfg['code_commit'],new_training=False))
    print('COMPLETE',len(outputs),flush=True)


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG']);root=Path(cfg['run'])
    with (root/'private/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(cfg)
        except Exception as e:
            write_new(root/'public/FAILURE.json',dict(error=repr(e)));raise
