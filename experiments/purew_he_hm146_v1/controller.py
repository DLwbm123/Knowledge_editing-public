"""Finite paired schedule. No startup side effect before explicit execution checks."""
import os,shutil,time,traceback
from pathlib import Path
from configuration import ARMS,N,ELIGIBLE,schedule,require_execution


def run_plan(dispatch):
    for stage in schedule():dispatch(stage)


def main():
    import common as c
    import runtime
    from base_controller import launch
    from PIL import Image
    R=c.RUN
    require_execution(c.read(R/'public/PROTOCOL.json'),c.read(R/'public/EXECUTION_AUTHORIZATION.json'),c.read(R/'public/HARDWARE_AUTHORIZATION.json'))
    assert c.read(R/'public/CPU_VALIDATION.json')['status']=='PASS'
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    rows={x['order']:x for x in c.read(R/'private/operator/source.json')['rows']}
    reviews={x['order']:x for x in c.read(R/'private/operator/review.json')['rows']}
    assert len(tasks)==N and [t['order'] for t in tasks]==list(range(1,N+1))
    terminals={}
    def retire_window(arm):
        folder=R/'private/window'/arm;folder.mkdir(parents=True,exist_ok=True)
        for p in folder.iterdir():assert not p.is_symlink() and p.is_file();p.unlink()
    def provide(owner,arm):
        task=tasks[owner-1];row=rows.get(owner);retire_window(arm)
        window=R/'private/window'/arm
        source_image=Path(row['image_file']) if row else Path(c.local_path(task['native']['image_path']))
        if row:
            assert row['question']==task['native']['question'] and row['answer']==task['native']['reference']
            assert row['original_fits']==task['fit_questions']
        target=window/('current'+source_image.suffix);shutil.copyfile(source_image,target)
        source=dict(order=owner,edit_id=task['edit_id'],seed=task['seed'],fit_questions=task['fit_questions'],
            native={k:task['native'][k] for k in ('dataset','question','reference','original_image_path')})
        source['native']['image_path']=str(target)
        boxes=[]
        if owner in ELIGIBLE:
            assert reviews[owner]['region_status']=='ELIGIBLE_COARSE_BOX'
            boxes=[row['boxes'][i]['xywh'] for i in reviews[owner]['selected_box_indices']]
        with Image.open(target) as im:size=list(im.size)
        c.write(window/'current.json',dict(quick_owner=owner,task=source,gate=owner in ELIGIBLE,boxes=boxes,image_size=size))
    def dispatch(stage):
        arm=stage['arm'];owner=stage['owner'];action=stage['action']
        if action=='retire':
            assert c.read(R/'private'/('FINAL_CONSUMED_'+arm+'.json'))['status']=='PASS'
            assert all(x.get('ended_epoch') for x in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
            state=R/'private/active'/arm/'state.pt';assert not state.is_symlink()
            c.write(R/'private/control'/arm/'STATISTICS_RETIRED.json',dict(bytes=state.stat().st_size,reason='146 panels and exact reload durable'))
            state.unlink();retire_window(arm)
            terminals[arm]=dict(prefix=N,slots=N*20,consumers=4123,status='COMPLETE_WITH_SCIENTIFIC_FAILURES_RETAINED')
            return
        if action in ('learn','admit'):provide(owner,arm)
        os.environ['OWNER']=str(owner)
        if action not in ('learn','admit'):os.environ['EVAL_PHASE']=action
        while not runtime.available(runtime.GPU):c.budget();time.sleep(30)
        label=('eval_'+action if action not in ('learn','admit') else action)+'_'+arm+'_'+str(owner)
        child=launch(label,arm,'worker.py');assert child.wait()==0,'Retain failure; no replay'
        if action=='insertion':
            assert len(list((R/'private/steps'/arm/str(owner)).glob('*.json')))==20
            retire_window(arm)
    try:run_plan(dispatch)
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
        raise
    assert set(terminals)==set(ARMS)
    c.write(R/'private/ARM_TERMINALS.json',terminals)
    c.write(R/'private/FINAL_CONSUMED.json',dict(status='PASS'))
    c.write(R/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCIENTIFIC_SUCCESS',new_consumers=8246))
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0))

if __name__=='__main__':main()
