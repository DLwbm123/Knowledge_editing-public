"""Retain every final expert, then conditionally evaluate the actual bank."""
import copy
import os
from pathlib import Path
import subprocess
import time
import traceback
import continuous_protocol as f
import balance_controller as shared
RUN=Path(os.environ['RUN_ROOT']);shared.f=f
read,write=shared.read,shared.write

def hold_arm(single,arm):
    folder=single/'private/continuous_retained'/arm;folder.mkdir(parents=True,exist_ok=True)
    for owner in range(1,147):
        target=folder/f'{owner}.pt';source=single/'private/candidates'/arm/f'{owner}.pt';result=single/'private/results'/arm/f'{owner}.json'
        if target.exists():continue
        if source.exists() and result.exists():
            d=read(result);assert d['status']=='COMPLETE' and d['save_load_exact'] and len(d['curve'])==540
            assert not source.is_symlink();os.link(source,target)
            assert source.stat().st_ino==target.stat().st_ino and source.stat().st_dev==target.stat().st_dev
    paths=list(folder.glob('*.pt'));size=sum(p.stat().st_size for p in paths);assert size<=32*1024**2
    receipt=dict(status='COMPLETE' if len(paths)==146 else 'COLLECTING',count=len(paths),bytes=size,paths=[str(p) for p in paths],last_consumer='conditional146 actual bank',hardlinks=True,epoch=time.time())
    write(single/'private'/f'CONTINUOUS_RETENTION_{arm}.json',receipt);write(RUN/'private'/f'CONTINUOUS_RETENTION_{arm}.json',receipt)
    deletion=single/'private'/f'DELETION_{arm}.json'
    if deletion.exists():
        assert len(paths)==146,'single retired before all continuation links were retained'
        d=read(deletion)
        if d.get('no_retained_copy'):
            d.update(no_retained_copy=False,retained_hardlinks=str(folder),lifecycle_amendment='CONTINUOUS_AMENDMENT_ZH.md');write(deletion,d)
    return receipt

def retire(single,arm,reason):
    receipt=read(RUN/'private'/f'CONTINUOUS_RETENTION_{arm}.json');folder=(single/'private/continuous_retained'/arm).resolve();paths=[Path(p) for p in receipt['paths']]
    assert len(paths)==146 and all(p.is_file() and not p.is_symlink() and p.resolve().parent==folder for p in paths)
    d=dict(status='PENDING',paths=[str(p) for p in paths],count=146,bytes=sum(p.stat().st_size for p in paths),reason=reason,no_retained_copy=True,rebuild_requires_training=True)
    write(RUN/'private'/f'CONTINUOUS_DELETION_{arm}.json',d)
    for p in paths:p.unlink()
    d.update(status='DELETED',epoch=time.time());write(RUN/'private'/f'CONTINUOUS_DELETION_{arm}.json',d)

def prepare(single,selected):
    import contract
    import common as c
    gate=read(RUN/'private/CONTINUOUS_ADMISSION.json');parent=Path(gate['parent'])
    ledger=read(parent/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    prior={k:ledger[k] for k in ('gpu_seconds_used','Judge_attempts')};manifest=read(parent/'RUN_MANIFEST.json')
    assert prior['gpu_seconds_used']+30000+1488.5244522094727<manifest['GPU_seconds_limit']
    assert prior['Judge_attempts']+4538<=manifest['Judge_limit'],'Preserve weights: project Judge headroom insufficient'
    manifest.update(stage=f.EPOCH,candidate='M13P',starting_epoch=time.time(),GPU_seconds_limit=prior['gpu_seconds_used']+30000,Judge_limit=prior['Judge_attempts']+4538,physical_GPUs=[0,1,3,4])
    write(RUN/'RUN_MANIFEST.json',manifest);write(RUN/'STAGE_CAP.json',dict(GPU_seconds_limit=manifest['GPU_seconds_limit']))
    write(RUN/'RESOURCE_LEDGER.json',dict(prior,gpu_sessions=[]));write(RUN/'private/INHERITED_COST.json',prior)
    plan=read(RUN/'private/EVAL_PLAN146.json')
    for row in plan:
        if row['method']!='BASE':row['method']=selected
    write(RUN/'private/EVAL_PLAN146.json',plan);write(RUN/'private/CONTINUOUS_NEW_PLAN.json',[v for v in plan if v['mode'] in ('bank','insertion')])
    expected={contract.consumer_id(v):v for v in read(RUN/'private/EVAL_PLAN146.json') if v['mode'] in ('base','single')};seen=set()
    for p in (single/'private/outputs').glob('*/*/*.json'):
        d=read(p)
        if d['method'] not in (selected,'BASE'):continue
        assert d['consumer_id'] in expected and d['consumer_id'] not in seen;seen.add(d['consumer_id'])
        d.update(alias_of=str(p),execution=read(RUN/'private/GPU_SOURCE_VERSION.json'),lock=c.digest(read(RUN/'private/SCOPE_LOCK.json')))
        write(RUN/'private/outputs'/p.relative_to(single/'private/outputs'),d)
    assert len(seen)==5049
    gate.update(status='PASS',conditional_gate_passed=True,selected_method=selected,source_commit=read(RUN/'private/GPU_SOURCE_VERSION.json')['commit'],epoch=time.time())
    for n in ('CONTINUOUS_ADMISSION.json','BALANCE_ADMISSION1.json','CM_ADMISSION1.json'):write(RUN/'private'/n,gate)
    write(RUN/'private/CONTINUOUS_FROZEN.json',dict(candidates=146,steps=540,methods=[selected],retained_experts=True))

def main():
    a=read(RUN/'private/CONTINUOUS_ADMISSION.json');single=Path(a['single_run'])
    assert read(RUN/'private/CONTINUOUS_CPU_CHECK.json')['status']=='PASS'
    assert read(RUN/'private/CONTINUOUS_SELECTION_CPU_CHECK.json')['status']=='PASS'
    with (RUN/'private'/('CONTINUOUS_FINISH_RECOVERY_CONTROLLER_ONCE' if os.environ.get('CURRICULUM_FINISH_ONLY')=='1' else 'CONTINUOUS_RECOVERY_CONTROLLER_ONCE' if os.environ.get('CURRICULUM_RECOVERY_CONTROLLER')=='1' else 'CONTINUOUS_FOUR_GPU_CONTROLLER_ONCE')).open('x'):pass
    while True:
        receipts={arm:hold_arm(single,arm) for arm in ('C','M1','M13P')}
        assert not list((single/'private').glob('FAILURE_curriculum*.json')),'Single failed; preserve retained artifacts'
        assert time.time()<read(single/'RUN_MANIFEST.json')['deadline_epoch']
        if (single/'private/CURRICULUM_COMPLETE.json').exists():break
        time.sleep(30)
    from continuous_selection import evaluate
    selection,details=evaluate(single)
    write(RUN/'private/CONTINUOUS_SELECTION.json',selection);write(RUN/'private/CONTINUOUS_SELECTION_DETAILS.json',details)
    write(RUN/'public/CONTINUOUS_SELECTION.json',selection)
    if not selection['all_scoring_complete']:raise RuntimeError('Single scoring incomplete; preserve all candidates')
    assert all(d['count']==146 for d in receipts.values())
    selected=selection['selected_method']
    if selected is None:
        a.update(status='NOT_ADMITTED',conditional_gate_passed=False,epoch=time.time());write(RUN/'private/CONTINUOUS_ADMISSION.json',a)
        for arm in ('C','M1','M13P'):retire(single,arm,'No candidate satisfied the fixed single gate')
        write(RUN/'public/CONTINUOUS_NOT_ADMITTED.json',dict(status='NOT_ADMITTED',selection=selection))
        return
    prepare(single,selected)
    for arm in ('C','M1','M13P'):
        if arm!=selected:retire(single,arm,'Not selected by the fixed three-candidate rule; single consumers complete')
    f.admit(RUN)
    while True:
        f.project_budget(RUN)
        free=subprocess.check_output(['nvidia-smi','-i','0,1,3,4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if all(int(v)>=45*1024 for v in free.split()):break
        time.sleep(30)
    jobs=[shared.launch('continuous_runtime.py','m13_146_bank',p,g) for p,g in enumerate((0,1,3,4))]
    assert [j.wait() for j in jobs]==[0,0,0,0]
    counts={k:0 for k in f.CAPS}
    for p in range(4):
        for k,v in read(RUN/'private'/f'CONTINUOUS_WORKER_COMPLETE_{p}.json')['counts'].items():
            if k in counts:counts[k]+=v
    assert all(counts[k]<=cap for k,cap in f.CAPS.items()) and counts['generation_attempts']==4538
    from common import digest
    import contract
    expected={contract.consumer_id(v) for v in read(RUN/'private/CONTINUOUS_NEW_PLAN.json')};seen=set()
    for path in (RUN/'private/outputs').glob(f'{selected}_*/*/*.json'):
        d=read(path)
        if d['mode']=='single':continue
        assert d['lock']==digest(read(RUN/'private/SCOPE_LOCK.json')) and d['execution']==read(RUN/'private/GPU_SOURCE_VERSION.json')
        assert d['consumer_id'] in expected and d['consumer_id'] not in seen;seen.add(d['consumer_id'])
    assert seen==expected and len(list((RUN/'private/outputs').glob('*/*/*.json')))==9587
    write(RUN/'private/CONTINUOUS_GENERATION_COMPLETE.json',dict(status='COMPLETE',counts=counts,epoch=time.time()));shared.merge('GPU');retire(single,selected,'All actual bank consumers and durable output bindings complete')
    root=RUN/'private'/f.QUEUE
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        f.project_budget(RUN);assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        os.environ['QUEUE_REQUEST']='{"action":"status"}'
        assert shared.launch('continuous_queue.py','continuous_queue',0,-1).wait()==0
        time.sleep(20)
    shared.merge('Judge');assert shared.launch('continuous_report.py','continuous_report',0,-1).wait()==0
    write(RUN/'private/CONTINUOUS_COMPLETE.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))
if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_continuous_controller.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        if read(RUN/'private/CONTINUOUS_ADMISSION.json')['status']=='PASS':
            for phase in ('GPU','Judge'):
                try:shared.merge(phase)
                except BaseException as error:write(RUN/'private'/f'CONTINUOUS_COST_ERROR_{phase}.json',dict(error=repr(error)))
        write(RUN/'private/CONTINUOUS_CONTROLLER_EXIT.json',dict(exit_code=code,epoch=time.time()))
