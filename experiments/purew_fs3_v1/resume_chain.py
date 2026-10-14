"""Continue the original installation after verified first-source boundary; no old phase replay."""
import os
import time
import traceback
from pathlib import Path
import fcntl
from base_controller import launch
from controller import provide
import common as c
R=c.RUN


def stage(action,arm,role,owner=0):
    while not __import__('base_run').available(5):c.budget();time.sleep(30)
    os.environ.update(SOURCE_ROLE=role,SOURCE_OWNER=str(owner),CUDA_VISIBLE_DEVICES='5')
    child=launch(action,arm,'source_stage.py')
    assert child.wait()==0,'Retain stage failure; no automatic replay'


def main():
    with (Path(os.environ['TMPDIR'])/'resume_chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=c.read(R/'private/RECOVERY_CHECKPOINT_VALIDATION.json');assert proof['status']=='PASS' and proof['owner']==1 and proof['slot']==160
        assert c.read(R/'public/REAL_VLM_ALL_ADMITTED.json')['status']=='PASS'
        assert not list((R/'private/failures').glob('*.json'))
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];rows=c.read(R/'private/operator/source.json')['rows']
        for arm,owners in (('F',(2,3)),('S',(1,2,3))):
            ctrl=R/'private/control'/arm;window=R/'private/window'/arm
            for owner in owners:
                packet=window/'current.json'
                if packet.exists():assert c.read(packet)['task']['order']==owner
                else:provide(rows[owner-1],tasks[owner-1],owner,window)
                c.write(R/'public/PROGRESS.json',dict(status='WAITING_SERIAL_GPU_ADMISSION',arm=arm,owner=owner,minimum_free_MiB=45*1024,epoch=time.time()))
                stage('serial_learn_'+arm+'_'+str(owner),arm,'learn',owner)
                stage('serial_evaluate_'+arm+'_'+str(owner),arm,'evaluate',owner)
                c.write(R/'private/units'/arm/(str(owner)+'.json'),dict(status='COMPLETE',slots=160,completed_slots=owner*160))
                for p in window.iterdir():assert not p.is_symlink();p.unlink()
                for name in ('EVALUATE.json','native.pt'):(ctrl/name).unlink()
                c.write(ctrl/(str(owner)+'.RETIRED.json'),dict(owner=owner,current_case_buffers_retired=True))
            stage('serial_reload_'+arm,arm,'reload')
            assert (R/'private'/('FINAL_CONSUMED_'+arm+'.json')).exists()
        stage('serial_reference_M','M','reference')
        assert all(x.get('ended_epoch') for x in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
        c.write(R/'private/TRAINING_GENERATION_COMPLETE.json',dict(status='COMPLETE',formal_attempt_slots=960))
        c.write(R/'private/FINAL_CONSUMED.json',dict(status='COMPLETE',arms=['F','S','M_REFERENCE']))
        c.write(R/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',expected_consumers=483))
        c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time(),serial_recovery=True))


if __name__=='__main__':
    try:main()
    except BaseException:c.write(R/'private/failures/resume_controller.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
