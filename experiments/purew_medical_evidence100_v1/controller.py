"""One bounded M -> M-S -> M-E chain, source broker and isolated evaluator."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
import common as c
R=c.RUN


def launch(action,arm,file):
    assert not (R/'private'/('START_'+action+'.json')).exists(),'Review attempted stage; never replay blindly'
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env=dict(os.environ,ACTION=action,ARM=arm,TASK_FILE=str(R/'private/tools'/file))
    child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs'/(action+'.log')).open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
    assert not any(x in argv for x in ('wangbomin','Knowledge_editing','purew'))
    c.write(R/'private'/('START_'+action+'.json'),dict(pid=child.pid,entry=entry,argv=argv,
        start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
    return child


def main():
    assert c.read(R/'public/CPU_VALIDATION.json')['status']=='PASS'
    with (Path(os.environ['TMPDIR'])/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:100]
        rows=c.read(R/'private/operator/source.json')['rows'];reviews=c.read(R/'private/operator/review.json')['rows']
        assert len(rows)==len(reviews)==len(tasks)==100
        for arm in ('M','M-S','M-E'):
            while True:
                c.budget();u,free=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                assert u==c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5']
                if int(free)>=45*1024:break
                time.sleep(30)
            ctrl=R/'private/control'/arm;window=R/'private/window'/arm
            if ctrl.exists():
                assert arm=='M'
                if (R/'private/CHECKPOINT_LAYOUT_RECOVERY.json').exists():
                    assert c.read(R/'private/CHECKPOINT_LAYOUT_RECOVERY.json')['status']=='VERIFIED_ONE_REJECTED_SLOT_AFTER_BOUNDARY'
                else:
                    reviewed=c.read(R/'private/ENGINEERING_PRETRAINING_RESTART.json')
                    assert reviewed['status']=='VERIFIED_ZERO_NATIVE_WRITES' and not list((R/'private/units'/arm).glob('*.json'))
            ctrl.mkdir(parents=True,exist_ok=True);window.mkdir(parents=True,exist_ok=True)
            learner=launch('learn_'+arm,arm,'learner.py');evaluator=launch('evaluate_'+arm,arm,'evaluator.py')
            start_owner=2 if arm=='M' and (R/'private/CHECKPOINT_LAYOUT_RECOVERY.json').exists() else 1
            for owner in range(start_owner,101):
                need=ctrl/'NEED_SOURCE.json'
                while not need.exists() or c.read(need)['owner']!=owner:
                    c.budget();assert learner.poll() is None and evaluator.poll() is None,'Preserve stage failure';time.sleep(.2)
                row=rows[owner-1];review=reviews[owner-1];task=tasks[owner-1]
                assert row['question']==task['native']['question'] and row['answer']==task['native']['reference']
                assert row['original_fits']==task['fit_questions']
                target=window/('current'+Path(row['image_file']).suffix)
                shutil.copyfile(row['image_file'],target)
                source=dict(order=owner,edit_id=task['edit_id'],seed=task['seed'],fit_questions=task['fit_questions'],
                    native={k:task['native'][k] for k in ('dataset','question','reference','original_image_path')})
                source['native']['image_path']=str(target)
                gate=review['region_status']=='ELIGIBLE_COARSE_BOX'
                boxes=[row['boxes'][i]['xywh'] for i in review['selected_box_indices']] if gate else []
                c.write(window/'current.json',dict(task=source,gate=gate,boxes=boxes,image_size=row['image_size']))
                while not (ctrl/(str(owner)+'.EVAL_DONE.json')).exists():
                    c.budget();assert learner.poll() is None and evaluator.poll() is None,'Preserve stage failure';time.sleep(.2)
                while not (R/'private/units'/arm/(str(owner)+'.json')).exists():time.sleep(.1)
                for p in window.iterdir():assert not p.is_symlink();p.unlink()
                (ctrl/'EVALUATE.json').unlink();(ctrl/'native.pt').unlink()
                c.write(ctrl/(str(owner)+'.RETIRED.json'),dict(owner=owner,current_case_buffers_retired=True,old_samples_retained_by_learner=0))
            assert learner.wait()==0 and evaluator.wait()==0
            assert (R/'private'/('TRAINING_COMPLETE_'+arm+'.json')).exists() and (R/'private'/('EVALUATION_COMPLETE_'+arm+'.json')).exists()
            reload=launch('reload_'+arm,arm,'evaluator.py');assert reload.wait()==0
            assert (R/'private'/('FINAL_CONSUMED_'+arm+'.json')).exists()
        assert not any(not s.get('ended_epoch') for s in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
        c.write(R/'private/TRAINING_GENERATION_COMPLETE.json',dict(status='COMPLETE',formal_attempt_slots=48000))
        c.write(R/'private/FINAL_CONSUMED.json',dict(status='COMPLETE',arms=['M','M-S','M-E']))
        c.write(R/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',expected_consumers=6543,epoch=time.time()))
        c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
        c.write(R/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_SCORING_PENDING',epoch=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
