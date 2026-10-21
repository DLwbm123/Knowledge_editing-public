"""One revision2 X/Y chain, serial GPU stages and no evaluation feedback."""
import os,time,traceback
from pathlib import Path
import shutil
import common as c
import runtime
from base_controller import launch
R=c.RUN

def provide(row,task,owner,arm,review=None):
    window=R/'private/window'/arm;window.mkdir(parents=True,exist_ok=True)
    assert row['question']==task['native']['question'] and row['answer']==task['native']['reference']
    assert row['original_fits']==task['fit_questions']
    for p in window.iterdir():assert not p.is_symlink();p.unlink()
    target=window/('current'+Path(row['image_file']).suffix);shutil.copyfile(row['image_file'],target)
    source=dict(order=task['order'],edit_id=task['edit_id'],seed=task['seed'],fit_questions=task['fit_questions'],
        native={k:task['native'][k] for k in ('dataset','question','reference','original_image_path')})
    source['native']['image_path']=str(target)
    gate=review is not None;boxes=[row['boxes'][i]['xywh'] for i in review['selected_box_indices']] if gate else []
    c.write(window/'current.json',dict(quick_owner=owner,task=source,gate=gate,boxes=boxes,image_size=row['image_size']))


def wait_available():
    while not runtime.available(4):c.budget();time.sleep(10)


def stage(action,arm,owner,phase=None):
    os.environ['OWNER']=str(owner)
    if phase:os.environ['EVAL_PHASE']=phase
    wait_available();child=launch(action,arm,'worker.py');assert child.wait()==0,'Retain stage failure, never replay'


def retire(arm):
    for p in (R/'private/window'/arm).iterdir():assert not p.is_symlink();p.unlink()


def main():
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];source=c.read(R/'private/operator/source.json')['rows'];reviews=c.read(R/'private/operator/review.json')['rows']
    for arm in ('X','Y'):
        stage('eval_base_'+arm,arm,0,'base');assert c.read(R/'public'/('BASE_IDENTITY_'+arm+'.json'))['status']=='PASS'
        for owner in (1,2,3):
            review=reviews[owner-1]
            provide(source[owner-1],tasks[owner-1],owner,arm,review if review['region_status']=='ELIGIBLE_COARSE_BOX' else None)
            stage('learn_'+arm+'_'+str(owner),arm,owner)
            retire(arm);stage('eval_'+arm+'_'+str(owner),arm,owner,'owner')
            c.write(R/'private/units'/arm/(str(owner)+'.json'),dict(status='COMPLETE',slots=20,outputs_durable=True))
        stage('eval_reload_'+arm,arm,3,'reload')
    l=c.read(R/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in l['gpu_sessions'])
    assert sum(c.read(p).get('attempt_slots',0) for p in (R/'private/attempts').glob('*.json'))==120
    c.write(R/'private/FINAL_CONSUMED.json',dict(status='PASS',arms=['X','Y'],retained_native_master_and_statistics=True))
    c.write(R/'private/GENERATION_COMPLETE.json',dict(status='COMPLETE',consumers=322,epoch=time.time()))
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
    c.write(R/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_WAIT_SCORING',epoch=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException:c.write(R/'private/failures/controller.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
