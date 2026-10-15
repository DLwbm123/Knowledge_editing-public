"""Two bounded three-source trajectories, followed by readonly M-reference evaluation."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from base_controller import launch
import common as c
R = c.RUN


def provide(row,task,owner,window):
    assert row['question']==task['native']['question'] and row['answer']==task['native']['reference']
    assert row['original_fits']==task['fit_questions']
    target=window/('current'+Path(row['image_file']).suffix);shutil.copyfile(row['image_file'],target)
    source=dict(order=owner,edit_id=task['edit_id'],seed=task['seed'],fit_questions=task['fit_questions'],
        native={k:task['native'][k] for k in ('dataset','question','reference','original_image_path')})
    source['native']['image_path']=str(target)
    c.write(window/'current.json',dict(task=source,gate=False,boxes=[],image_size=row['image_size']))


def main():
    with (Path(os.environ['TMPDIR'])/'chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:3]
        rows = c.read(R/'private/operator/source.json')['rows'][:3]
        for arm in ('F','S'):
            c.budget()
            u,free=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            assert u==c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5'] and int(free)>=45*1024
            ctrl=R/'private/control'/arm;window=R/'private/window'/arm
            assert not ctrl.exists();ctrl.mkdir(parents=True);window.mkdir(parents=True)
            provide(rows[0],tasks[0],1,window)
            learner=launch('admit_learn_'+arm,arm,'learner.py')
            evaluator=launch('admit_evaluate_'+arm,arm,'evaluator.py')
            assert learner.wait()==evaluator.wait()==0
            assert c.read(R/'public'/('ADMISSION_'+arm+'.json'))['status']=='PASS'
            for path in window.iterdir():assert not path.is_symlink();path.unlink()
        c.write(R/'public/REAL_VLM_ALL_ADMITTED.json',dict(status='PASS',arms=['F','S'],formal_slots=0))
        for arm in ('F', 'S'):
            while True:
                c.budget()
                u, free = subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                assert u == c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5']
                if int(free) >= 45*1024: break
                time.sleep(30)
            ctrl = R/'private/control'/arm; window = R/'private/window'/arm
            assert c.read(R/'public'/('ADMISSION_'+arm+'.json'))['status']=='PASS'
            learner = launch('learn_'+arm, arm, 'learner.py')
            evaluator = launch('evaluate_'+arm, arm, 'evaluator.py')
            for owner, (row, task) in enumerate(zip(rows, tasks), 1):
                need = ctrl/'NEED_SOURCE.json'
                while not need.exists() or c.read(need)['owner'] != owner:
                    c.budget(); assert learner.poll() is None and evaluator.poll() is None; time.sleep(.2)
                provide(row,task,owner,window)
                while not (ctrl/(str(owner)+'.EVAL_DONE.json')).exists():
                    c.budget(); assert learner.poll() is None and evaluator.poll() is None; time.sleep(.2)
                while not (R/'private/units'/arm/(str(owner)+'.json')).exists():
                    c.budget(); assert learner.poll() is None; time.sleep(.1)
                for path in window.iterdir(): assert not path.is_symlink(); path.unlink()
                (ctrl/'EVALUATE.json').unlink(); (ctrl/'native.pt').unlink()
                c.write(ctrl/(str(owner)+'.RETIRED.json'), dict(owner=owner, current_case_buffers_retired=True))
            assert learner.wait() == evaluator.wait() == 0
            fresh = launch('reload_'+arm, arm, 'evaluator.py'); assert fresh.wait() == 0
            assert (R/'private'/('FINAL_CONSUMED_'+arm+'.json')).exists()
        reference = launch('reference_M','M','evaluator.py'); assert reference.wait() == 0
        assert all(s.get('ended_epoch') for s in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
        c.write(R/'private/TRAINING_GENERATION_COMPLETE.json', dict(status='COMPLETE', formal_attempt_slots=960))
        c.write(R/'private/FINAL_CONSUMED.json', dict(status='COMPLETE', arms=['F','S','M_REFERENCE']))
        c.write(R/'private/GENERATION_COMPLETE.json', dict(status='GENERATED_NOT_SCORED', expected_consumers=483))
        c.write(R/'private/EXIT_controller.json', dict(exit_code=0, epoch=time.time()))
        c.write(R/'public/PROGRESS.json', dict(status='GENERATION_COMPLETE_SCORING_PENDING', epoch=time.time()))


if __name__ == '__main__':
    try: main()
    except BaseException:
        c.write(R/'private/failures/controller.json', dict(traceback=traceback.format_exc(), epoch=time.time())); raise
