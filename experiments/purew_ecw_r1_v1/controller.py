"""R source1 only; independent admission, output and exact reload barriers."""
import os,time,traceback,shutil,signal
from pathlib import Path
import common as c
import runtime
from base_controller import launch
R=c.RUN

def provide(row,task,owner,arm,review=None):
    window=R/'private/window'/arm;window.mkdir(parents=True,exist_ok=True)
    assert row['question']==task['native']['question'] and row['answer']==task['native']['reference']
    assert row['original_fits']==task['fit_questions']
    retire(arm)
    target=window/('current'+Path(row['image_file']).suffix);shutil.copyfile(row['image_file'],target)
    source=dict(order=task['order'],edit_id=task['edit_id'],seed=task['seed'],fit_questions=task['fit_questions'],
        native={k:task['native'][k] for k in ('dataset','question','reference','original_image_path')})
    source['native']['image_path']=str(target)
    gate=review is not None;boxes=[row['boxes'][i]['xywh'] for i in review['selected_box_indices']] if gate else []
    c.write(window/'current.json',dict(quick_owner=owner,task=source,gate=gate,boxes=boxes,image_size=row['image_size']))

def wait_available(action):
    while not runtime.available(2):
        c.write(R/'public/PROGRESS.json',dict(status='WAITING_GPU2_ADMISSION',next_stage=action,required_free_MiB=46080,epoch=time.time()))
        time.sleep(30)

def stage(action,arm,owner,phase=None):
    os.environ['OWNER']=str(owner)
    if phase:os.environ['EVAL_PHASE']=phase
    wait_available(action);child=launch(action,arm,'worker.py');assert child.wait()==0,'Retain stage failure; no replay'

def retire(arm):
    folder=R/'private/window'/arm;folder.mkdir(parents=True,exist_ok=True)
    for p in folder.iterdir():assert not p.is_symlink();p.unlink()

def main():
    assert c.read(R/'public/CPU_VALIDATION.json')['status']=='PASS'
    assert c.read(R/'public/BASE_PARAMETER_REUSE.json')['status']=='PASS'
    assert c.read(R/'private/FROZEN_PROTOCOL.json')['active_arms']==['R']
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    row=c.read(R/'private/operator/source.json')['rows'][0]
    review=c.read(R/'private/operator/review.json')['rows'][0]
    assert review['region_status']=='ELIGIBLE_COARSE_BOX'
    stage('eval_base_R','R',0,'base')
    provide(row,tasks[0],1,'R',review);stage('admit_R','R',1);retire('R')
    assert c.read(R/'public/REAL_VLM_ADMISSION_R.json')['status']=='PASS'
    c.write(R/'public/REAL_VLM_ADMISSION_ALL.json',dict(status='PASS',arms=1,temporary_native_writes=1,formal_slots=0,physical_GPU=2))
    provide(row,tasks[0],1,'R',review);stage('learn_R_1','R',1);retire('R')
    stage('eval_R_1','R',1,'owner')
    stopped=(R/'private/control/R/SCIENTIFIC_STOP.json').exists()
    terminal=dict(prefix=1,status='SCIENTIFIC_STOP' if stopped else 'COMPLETE',consumers=87)
    c.write(R/'private/units/R/1.json',dict(status=terminal['status'],slots=20,outputs_durable=True))
    c.write(R/'private/ARM_TERMINALS.json',{'R':terminal})
    stage('eval_reload_R','R',1,'reload')
    assert c.read(R/'private/FINAL_CONSUMED_R.json')['status']=='PASS'
    assert all(x.get('ended_epoch') for x in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    slots=sum(c.read(p).get('attempt_slots',0) for p in (R/'private/attempts').glob('*.json'));assert slots==20
    # Only this run's consumed statistics retire; final native and all history remain.
    p=R/'private/active/R/state.pt'
    c.write(R/'private/control/R/STATISTICS_RETIRED.json',dict(reason='All source1 outputs and exact fresh reload consumed',bytes=p.stat().st_size));p.unlink()
    if not stopped:c.write(R/'private/TRAINING_GENERATION_COMPLETE.json',dict(status='COMPLETE',formal_slots=20,scope='R_SOURCE1_ONLY'))
    c.write(R/'private/ACTUAL_TRAJECTORIES_TERMINAL.json',dict(status=terminal['status'],formal_slots=20,arms={'R':terminal}))
    c.write(R/'private/FINAL_CONSUMED.json',dict(status='PASS',arms=['R']))
    c.write(R/'private/GENERATION_COMPLETE.json',dict(status='ALL_ACTUAL_GENERATION_TERMINAL',new_consumers=87,planned_maximum=87,full_protocol_complete=not stopped,epoch=time.time()))
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
    c.write(R/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_WAIT_SCORING',epoch=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
