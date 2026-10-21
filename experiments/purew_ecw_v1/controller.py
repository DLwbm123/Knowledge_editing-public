"""One bounded quick chain on GPU4; no scientific Judge or full ECW experiment."""
import os
from pathlib import Path
import shutil
import time
import traceback
from collections import Counter
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


def report():
    ledger=c.read(R/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    records=[]
    for owner in (1,2,3):
        events=[c.read(p) for p in (R/'private/steps/ECW'/str(owner)).glob('*.json')]
        events.sort(key=lambda e:e['slot']);reasons=Counter()
        for event in events:
            if not event['accepted']:
                for reason in set(x for candidate in event['candidates'] for x in candidate['reasons']):reasons[reason]+=1
        accepted=sum(e['accepted'] for e in events)
        blocked=sum(not e['accepted'] and any('NATIVE_RESIDUAL_ENVELOPE' in z['reasons'] and z['coordinate_risk']<=.01 for z in e['candidates']) for e in events)
        last=events[-1] if events else None
        records.append(dict(source_position=owner,slots=len(events),accepted=accepted,rejected=len(events)-accepted,
            zero_native_fraction=(len(events)-accepted)/len(events) if events else None,rejected_slot_flags=dict(reasons),
            quantization_bound_blocked_slots=blocked,first_source_CE=events[0]['source_CE'] if events else None,
            last_source_CE=last['source_CE'] if last else None,last_native_risk_bound=last['native_risk_bound'] if last else None,
            last_residual_energy=last.get('residual_energy',last.get('current_residual_energy')) if last else None,
            freezing_flag=bool(events and blocked/len(events)>=.95 and last['source_CE']>.05)))
    counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ['forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts']}
    failures=list((R/'private/failures').glob('*.json'))
    medical=c.read(R/'public/MEDICAL_MECHANICAL.json') if (R/'public/MEDICAL_MECHANICAL.json').exists() else None
    complete=(R/'private/EXIT_controller.json').exists() and not failures
    result=dict(status='QUICK_MECHANICAL_COMPLETE' if complete else 'QUICK_PARTIAL_NOT_COMPLETE',
        diagnostic_only=True,scientific_success=False,full_ECW_authorized=False,source_diagnostics=records,
        real_VLM_admission=c.read(R/'public/REAL_VLM_ADMISSION.json') if (R/'public/REAL_VLM_ADMISSION.json').exists() else None,
        medical_mechanical=medical,native_reload=c.read(R/'public/NATIVE_RELOAD.json') if (R/'public/NATIVE_RELOAD.json').exists() else None,
        actual_counts=counts,GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts'],failures=len(failures),
        T2G='UNATTEMPTED_NO_JUDGE',T1L_T2L_damage='UNKNOWN_NOT_PASS',permanent_missing_not_substituted=True,
        original146='PAUSED',original_M='PAUSED',OL='WAITING',FS='CONTINUES_GPU6')
    c.write(R/'public/QUICK_RESULTS.json',result)
    text='# ECW GPU4短程实现验证\n\n'+result['status']+'。仅机械与训练源诊断；没有进行科学评分，保护损伤未知、不通过硬门槛，不能声称解决遗忘。\n\n'
    text+='原前三项各20槽，独立医学资格测试只用首个既有合格粗框源。拒绝不补槽；全部预算和失败保留。\n\n'
    text+='|源位置|槽|接受|拒绝|零更新比例|末端源CE|量化界阻塞槽|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in records:text+='|'+ '|'.join(str(row[k]) for k in ['source_position','slots','accepted','rejected','zero_native_fraction','last_source_CE','quantization_bound_blocked_slots'])+'|\n'
    text+='\n详见QUICK_RESULTS.json。量化残差Frobenius界存在严格过保守反例，不是无冻结保证。当前统计仅界定冻结采样线性代理；临床正确性、真实非线性KL和隐私未获证明。F/S继续GPU6，原146/M暂停，O/L等待。\n'
    (R/'public/QUICK_REPORT_ZH.md').write_text(text)


def main():
    wait_available();os.environ['EVAL_PHASE']='base'
    child=launch('eval_base','ECW_EVAL','worker.py');assert child.wait()==0
    assert c.read(R/'public/BASE_IDENTITY.json')['status']=='PASS'
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];source=c.read(R/'private/operator/source.json')['rows']
    provide(source[0],tasks[0],1,'ECW');wait_available();child=launch('learn','ECW','worker.py')
    for owner in (1,2,3):
        done=R/'private/control/ECW'/(str(owner)+'.DONE.json')
        while not done.exists():c.budget();assert child.poll() is None,'Retain learner failure';time.sleep(.2)
        if owner<3:provide(source[owner],tasks[owner],owner+1,'ECW')
    assert child.wait()==0
    for p in (R/'private/window/ECW').iterdir():p.unlink()
    for phase in ('final','reload'):
        wait_available();os.environ['EVAL_PHASE']=phase;child=launch('eval_'+phase,'ECW_EVAL','worker.py');assert child.wait()==0
    m=c.read(R/'private/operator/medical.json');provide(m['row'],m['task'],1,'ECW_MEDICAL',m['review'])
    wait_available();child=launch('medical','ECW_MEDICAL','worker.py');assert child.wait()==0
    for p in (R/'private/window/ECW_MEDICAL').iterdir():p.unlink()
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
    report()


if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(R/'private/failures/controller.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        try:report()
        except BaseException:c.write(R/'private/REPORT_FAILURE.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        raise
