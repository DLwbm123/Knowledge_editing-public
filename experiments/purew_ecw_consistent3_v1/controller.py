"""One activation, GPU2 finite admission wait, serial calibration/admission/N/J/NJ."""
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

def calibration():
    pool=c.read('/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run/private/BASIS_BUFFER.json')
    locators=c.read(R/'private/operator/calibration_locators.json');assert len(locators)==8
    chosen=[pool[x['pool_index']] for x in locators]
    bound=[{k:row[k] for k in ('dataset','source_qid','source_group','question','reference','image_sha256')} for row in chosen]
    assert c.digest(bound)==c.read(R/'private/FROZEN_PROTOCOL.json')['calibration']['binding']
    assert len({row['source_group'] for row in chosen})==8
    del bound,chosen
    wait_available('calibration');child=launch('calibration','CAL','calibrator.py');ctrl=R/'private/control/CAL'
    for index,locator in enumerate(locators,1):
        path=ctrl/'NEED_SOURCE.json'
        while not path.exists() or c.read(path)['index']!=index:
            c.budget();assert child.poll() is None;time.sleep(.2)
        row=pool[locator['pool_index']]
        assert row['source_file']==locator['source_file'] and row['source_qid']==locator['source_qid']
        window=R/'private/window/CAL';retire('CAL');dest=window/('current'+Path(row['image_path']).suffix);shutil.copyfile(row['image_path'],dest)
        old=c.read(row['response_path']);safe=dict(row,image_path=str(dest));safe.pop('response_path',None)
        c.write(window/'current.json',dict(index=index,row=safe,expected=old['binding']['judge_input'],historical_native=old['R0']))
        while not (ctrl/(str(index)+'.DONE.json')).exists():
            c.budget();assert child.poll() is None;time.sleep(.2)
        retire('CAL');c.write(ctrl/(str(index)+'.RETIRED.json'),dict(source_buffers_retired=True))
    assert child.wait()==0 and c.read(R/'public/BASE_CALIBRATION.json')['status']=='PASS'

def main():
    assert c.read(R/'public/CPU_VALIDATION.json')['status']=='PASS'
    calibration()
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];source=c.read(R/'private/operator/source.json')['rows'];reviews=c.read(R/'private/operator/review.json')['rows']
    for arm in ('N','J','NJ'):
        stage('eval_base_'+arm,arm,0,'base')
        review=reviews[0];provide(source[0],tasks[0],1,arm,review)
        stage('admit_'+arm,arm,1);retire(arm)
    c.write(R/'public/REAL_VLM_ADMISSION_ALL.json',dict(status='PASS',arms=3,temporary_native_writes=3,formal_slots=0,physical_GPU=2))
    terminals={}
    for arm in ('N','J','NJ'):
        for owner in (1,2,3):
            review=reviews[owner-1]
            provide(source[owner-1],tasks[owner-1],owner,arm,review if review['region_status']=='ELIGIBLE_COARSE_BOX' else None)
            stage('learn_'+arm+'_'+str(owner),arm,owner)
            retire(arm);stage('eval_'+arm+'_'+str(owner),arm,owner,'owner')
            c.write(R/'private/units'/arm/(str(owner)+'.json'),dict(status='SCIENTIFIC_STOP' if (R/'private/control'/arm/'SCIENTIFIC_STOP.json').exists() else 'COMPLETE',slots=20,outputs_durable=True))
            if (R/'private/control'/arm/'SCIENTIFIC_STOP.json').exists():break
        terminals[arm]=dict(prefix=owner,status='COMPLETE' if owner==3 else 'SCIENTIFIC_STOP',consumers=161 if owner==3 else 87)
        c.write(R/'private/ARM_TERMINALS.json',terminals)
        stage('eval_reload_'+arm,arm,owner,'reload')
        assert c.read(R/'private'/('FINAL_CONSUMED_'+arm+'.json'))['status']=='PASS'
        # Consumed, generated-only statistics retire after exact reload to leave atomic save space for D.
        p=R/'private/active'/arm/'state.pt'
        c.write(R/'private/control'/arm/'STATISTICS_RETIRED.json',dict(reason='Trajectory and native reload complete; no subsequent learner consumer',bytes=p.stat().st_size))
        p.unlink()
    l=c.read(R/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in l['gpu_sessions'])
    slots=sum(c.read(p).get('attempt_slots',0) for p in (R/'private/attempts').glob('*.json'))
    assert slots==sum(t['prefix']*20 for t in terminals.values())<=180
    full=all(t['status']=='COMPLETE' for t in terminals.values())
    if full:c.write(R/'private/TRAINING_GENERATION_COMPLETE.json',dict(status='COMPLETE',formal_slots=slots))
    c.write(R/'private/ACTUAL_TRAJECTORIES_TERMINAL.json',dict(status='COMPLETE' if full else 'PARTIAL_SCIENTIFIC_STOP',formal_slots=slots,arms=terminals))
    c.write(R/'private/FINAL_CONSUMED.json',dict(status='PASS',arms=list(terminals)))
    c.write(R/'private/GENERATION_COMPLETE.json',dict(status='ALL_ACTUAL_GENERATION_TERMINAL',new_consumers=sum(t['consumers'] for t in terminals.values()),planned_maximum=483,full_protocol_complete=full,epoch=time.time()))
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
    c.write(R/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_WAIT_SCORING',epoch=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
