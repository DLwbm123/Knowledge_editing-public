"""Frozen 24-edit interaction check using existing weights and baseline outputs."""
import os
import time
import traceback
from pathlib import Path
import research as base
import pipeline

common,worker,train,RUN,BASE=base.common,base.worker,base.train,base.RUN,base.BASE
ARMS=('CE_ONLY','CE_U_MULTI','COMBO24_CE_FIT5','COMBO24_CEU_FIT5')


def tasks():return base.selected()


def point(t,arm):return BASE/'private/edits'/t['anonymous_edit']/'s0'/arm/'step160.pt'


def prepare():
    import torch
    common.budget();ts=tasks();assert len(ts)==24
    roles=common.read(BASE/'private/U_ROLES.json');rows=[];weights={}
    for arm in ARMS[:2]:
        weights[arm]={}
        for t in ts:
            s=torch.load(point(t,arm),map_location='cpu',weights_only=True);task=s['binding']['task']
            assert s['step']==160 and s['binding']['steps']==160 and task['seed']==t['seed']
            assert task['native']==t['native'] and task['fit_questions']==t['fit_questions']
            assert task['U_fit']==(roles['FIT'] if arm=='CE_U_MULTI' else [])
            weights[arm][t['edit_id']]=dict(path=str(point(t,arm)),hash=s['state_hash'])
        paths=sorted((BASE/'private/outputs/bank'/ts[-1]['anonymous_edit']/'s0'/arm/'n160/p24/R0').glob('*.json'))
        assert len(paths)==283
        for p in (paths[0],paths[-1]):
            d=common.read(p);phase=d['binding']['phase']
            assert (phase['prefix'],phase['node'],phase['slot'])==(24,160,0)
            assert list(phase['weights'])==[t['edit_id'] for t in ts]
            assert all(Path(phase['weights'][t['edit_id']]['path'])==point(t,arm) for t in ts)
        rows.extend(map(str,paths))
    common.write(RUN/'private/COMBO24_BASELINE_PATHS.json',rows)
    common.write(RUN/'private/COMBO24_WEIGHT_BINDINGS.json',weights)
    common.write(RUN/'public/COMBO24_ADMISSION.json',dict(status='PASS',edits=24,seed_repeats=1,weights_reused=48,
        baseline_outputs_reused=566,new_output_consumers=566,training_required=False,original_deadline_retained=True))


def generate():
    gpu=part=int(os.environ['PARTITION']);assert int(os.environ['GPU'])==gpu and part in (0,1)
    base.configure('COMBO24_LOCK.json');ts=tasks();lookup={t['edit_id']:t for t in base.retro.queue()['tasks']}
    with common.lease(gpu):
        runtime,bindings=common.load(gpu)
        teachers=train.teachers_for(runtime,common.read(BASE/'private/U_ROLES.json')['CHECK'])
        base.install_fitkeys([lookup[t['edit_id']] for t in ts])
        points={t['edit_id']:point(t,ARMS[part]) for t in ts};bank=[]
        for t in ts:bank+=worker.router(runtime,t)
        worker.evaluate(runtime,bindings,ts[-1],0,ARMS[2+part],160,points,bank,ts,mode='bank',prefix=24,teacher_rows=teachers)
    common.write(RUN/'private'/('COMBO24_DONE_'+str(part)+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))


def controller():
    assert common.read(RUN/'public/COMBO24_ADMISSION.json')['status']=='PASS'
    common.write(RUN/'public/COMBO24_PROGRESS.json',dict(status='GENERATING_FITKEY_COMBINATIONS',complete=False))
    pipeline.wait([pipeline.launch('combo24.py','combo24_generate',i,i) for i in (0,1)])
    common.write(RUN/'private/COMBO24_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',epoch=time.time()))
    pipeline.wait([pipeline.launch('combo24_queue.py','combo24_ingest')])
    common.write(RUN/'public/COMBO24_PROGRESS.json',dict(status='ASTRA_SCORING',complete=False))
    root=RUN/'private/judge_combo24_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        common.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    pipeline.wait([pipeline.launch('combo24_report.py','combo24_report')])


if __name__=='__main__':
    try:{'combo24_prepare':prepare,'combo24_generate':generate,'combo24_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        common.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
