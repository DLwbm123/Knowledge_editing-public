"""Background, dependency-ordered three-arm chain with bounded lifecycle."""
import os
import time
import traceback
from pathlib import Path
import curriculum_protocol as f
import balance_controller as shared
RUN=Path(os.environ['RUN_ROOT']);shared.f=f
read,write=shared.read,shared.write

def stage(action,arm):
    os.environ['CURRICULUM_ARM']=arm
    devices=(0,1,3,4) if action=='curriculum_generate_M13P' else (0,1)
    os.environ['GENERATION_PARTITIONS']=str(len(devices))
    jobs=[shared.launch('curriculum_runtime.py',action,p,g) for p,g in enumerate(devices)]
    codes=[j.wait() for j in jobs];assert codes==[0]*len(devices),(action,codes)

def retire(arm):
    from common import digest
    import contract
    plan=read(RUN/'private'/f'PLAN_{arm}.json');expected={contract.consumer_id(v) for v in plan};actual=set()
    for path in (RUN/'private/outputs'/f'{arm}_single_p1').glob('*/*.json'):
        d=read(path);assert d['method']==arm and d['lock']==digest(read(RUN/'private/SCOPE_LOCK.json')) and d['execution']==read(RUN/'private/GPU_SOURCE_VERSION.json')
        assert d['consumer_id'] in expected and d['consumer_id'] not in actual;actual.add(d['consumer_id'])
    assert actual==expected and len(actual)==2956
    paths=[RUN/'private/candidates'/arm/f'{i}.pt' for i in range(1,147)]
    assert all(p.is_file() and not p.is_symlink() and p.resolve().parent==(RUN/'private/candidates'/arm).resolve() for p in paths)
    d=dict(status='PENDING',paths=[str(p) for p in paths],count=len(paths),bytes=sum(p.stat().st_size for p in paths),last_consumer='single generation; dependent step norms retained as JSON',no_retained_copy=True,rebuild_requires_training=True)
    write(RUN/'private'/f'DELETION_{arm}.json',d)
    for p in paths:p.unlink()
    d.update(status='DELETED',epoch=time.time());write(RUN/'private'/f'DELETION_{arm}.json',d)

def main():
    f.admit(RUN);assert read(RUN/'private/CURRICULUM_CPU_CHECK.json')['status']=='PASS'
    adopt=os.environ.get('CURRICULUM_ADOPT_TRAINING')=='1'
    if adopt:
        # Adopt the two already-running training workers; never replay their work.
        with (RUN/'private/CURRICULUM_FOUR_GPU_CONTROLLER_ONCE').open('x'):pass
        for part in (0,1):
            start=read(RUN/'private'/f'START_curriculum_train_M13P_{part}.json')
            exit_path=RUN/'private'/f'EXIT_curriculum_train_M13P_{part}.json'
            while not exit_path.exists():
                stat=Path(f'/proc/{start["pid"]}/stat').read_text()
                assert stat.split(') ')[1].split()[19]==str(start['start_ticks'])
                f.project_budget(RUN);time.sleep(10)
            assert read(exit_path)['exit_code']==0
            assert read(RUN/'private'/f'COMPLETE_curriculum_train_M13P_{part}.json')['status']=='COMPLETE'
    else:
        with (RUN/'private/CURRICULUM_CONTROLLER_ONCE').open('x'):pass
    counts={k:0 for k in f.CAPS}
    for arm in f.METHODS:
        if not adopt:stage('curriculum_train_'+arm,arm)
        elif arm!='M13P':
            assert read(RUN/'private'/f'DELETION_{arm}.json')['status']=='DELETED'
        for i in range(1,147):
            d=read(RUN/'private/results'/arm/f'{i}.json');assert d['status']=='COMPLETE' and d['save_load_exact'] and len(d['curve'])==540
        if not adopt or arm=='M13P':
            write(RUN/'private'/f'FROZEN_{arm}.json',dict(candidates=146,steps=540,methods=[arm],epoch=time.time()))
            stage('curriculum_generate_'+arm,arm);retire(arm)
        for mode in ('train','generate'):
            for p in range(4 if mode=='generate' and arm=='M13P' else 2):
                for k,v in read(RUN/'private'/f'COMPLETE_curriculum_{mode}_{arm}_{p}.json')['counts'].items():
                    if k in counts:counts[k]+=v
        assert all(counts[k]<=cap for k,cap in f.CAPS.items())
    assert counts['updates']==236520 and counts['backwards']==411724 and counts['operator_backwards']==78840 and counts['generation_attempts']==8868
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==f.CONSUMERS
    write(RUN/'private/CURRICULUM_GENERATION_COMPLETE.json',dict(status='COMPLETE',counts=counts,epoch=time.time()));shared.merge('GPU')
    root=RUN/'private'/f.QUEUE
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        f.project_budget(RUN);assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        os.environ['QUEUE_REQUEST']='{"action":"status"}'
        assert shared.launch('curriculum_queue.py','curriculum_queue',0,-1).wait()==0
        time.sleep(20)
    shared.merge('Judge');assert shared.launch('curriculum_report.py','curriculum_report',0,-1).wait()==0
    write(RUN/'private/CURRICULUM_COMPLETE.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))
if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_curriculum_controller.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        for phase in ('GPU','Judge'):
            try:shared.merge(phase)
            except BaseException as error:write(RUN/'private'/f'COST_TRANSFER_ERROR_{phase}.json',dict(error=repr(error)))
        write(RUN/'private/CURRICULUM_CONTROLLER_EXIT.json',dict(exit_code=code,epoch=time.time()))
