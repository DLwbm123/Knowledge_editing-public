"""One preregistered weaker-U coefficient on the complete exposed 24-edit bank."""
import functools
import os
import time
import traceback
import combo24 as base

common,worker,train,RUN,BASE=base.common,base.worker,base.train,base.RUN,base.BASE
ARMS=('CE_ONLY','CE_U_MULTI','WEAK24_CEU001')
tasks=base.tasks


def point(t):return RUN/'private/weak24'/t['anonymous_edit']/'step160.pt'


def selfcheck():
    import torch
    from types import SimpleNamespace
    def run(weight=None):
        e=torch.nn.Linear(1,2,bias=False);e.weight.data.copy_(torch.tensor([[.2],[-.1]]))
        frozen=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
        class Model:
            def __call__(self,**kw):return SimpleNamespace(logits=e(torch.ones(1,1)).reshape(1,1,2)*frozen)
        runtime=SimpleNamespace(model=Model(),compute_loss=lambda batch:e.weight.square().sum())
        hook=SimpleNamespace(set_teacher_routing=lambda labels:None)
        batch=SimpleNamespace(labels=None,target_token_ids=[1]);mask=torch.tensor([[True]])
        teacher=({},None,mask,torch.tensor([[.5,.5]]).log(),{'source_group':'toy'},None)
        result=train.update(runtime,hook,e,torch.optim.SGD(e.parameters(),lr=0),batch,batch,teacher,**({} if weight is None else {'u_weight':weight}))
        assert frozen.grad is None
        return result
    default,strong,weak=run(),run(.01),run(.001)
    assert default==strong
    assert strong['terms']['native']==weak['terms']['native'] and strong['terms']['fit']==weak['terms']['fit']
    assert abs(strong['terms']['U']['weighted']/weak['terms']['U']['weighted']-10)<1e-6
    assert abs(strong['U_gradient_norm']/weak['U_gradient_norm']-1)<.002
    assert strong['terms']['U']['unweighted']==weak['terms']['U']['unweighted']
    for bad in (0.,-1.,float('nan'),float('inf')):
        try:run(bad)
        except AssertionError:pass
        else:raise AssertionError('invalid coefficient accepted')
    common.write(RUN/'public/WEAK24_CPU_TEST.json',dict(status='PASS',default_parity=True,weighted_KL_ratio=10,CE_unchanged=True,unweighted_KL_unchanged=True,Base_gradient=False,invalid_coefficients_rejected=True))


def prepare():
    import torch
    common.budget();selfcheck();ts=tasks();assert len(ts)==24
    assert common.read(RUN/'public/COMBO24_ADMISSION.json')['status']=='PASS'
    for t in ts:
        w0=torch.load(worker.w0_path(t,0),map_location='cpu',weights_only=True)
        strong=torch.load(base.point(t,ARMS[1]),map_location='cpu',weights_only=True)
        assert strong['binding']['W0']==common.state_hash(train.clone(w0['expert'],t['seed'],'cpu'))
        assert strong['binding']['task']['seed']==t['seed'] and strong['binding']['steps']==160
    common.write(RUN/'public/WEAK24_ADMISSION.json',dict(status='PASS',edits=24,seed_repeats=1,old_W0_reused=24,baseline_outputs_reused=566,new_output_consumers=283,new_training_updates=24*160,U_weight=.001,original_deadline_retained=True))


def fit():
    import torch
    part,gpu=int(os.environ['PARTITION']),int(os.environ['GPU']);assert part==gpu and part in (0,1)
    base.base.configure('WEAK24_LOCK.json')
    train.update=functools.partial(train.update,u_weight=.001)
    with common.lease(gpu):
        runtime,_=common.load(gpu)
        teachers=train.teachers_for(runtime,common.read(BASE/'private/U_ROLES.json')['FIT'])
        for t in tasks()[part::2]:
            initial=torch.load(worker.w0_path(t,0),map_location='cpu',weights_only=True)['expert']
            task=worker.train_task(t,0,[x[4] for x in teachers])
            task['bindings'].update(research_lock=common.digest(common.read(RUN/'private/WEAK24_LOCK.json')),U_weight=.001)
            train.continuation(runtime,task,common.record(t),initial,point(t).parent,teachers,160,[160])
            print('WEAK_TRAIN_COMPLETE',t['order'],flush=True)
    common.write(RUN/'private'/('WEAK24_TRAIN_'+str(part)+'.json'),dict(status='COMPLETE',epoch=time.time()))


def generate():
    import torch
    base.base.configure('WEAK24_LOCK.json');ts=tasks();gpu=int(os.environ['GPU'])
    weights=common.read(RUN/'private/COMBO24_WEIGHT_BINDINGS.json');weights[ARMS[2]]={}
    for t in ts:
        s=torch.load(point(t),map_location='cpu',weights_only=True)
        assert s['step']==160 and s['binding']['task']['bindings']['U_weight']==.001
        weights[ARMS[2]][t['edit_id']]=dict(path=str(point(t)),hash=s['state_hash'])
    common.write(RUN/'private/WEAK24_WEIGHT_BINDINGS.json',weights)
    with common.lease(gpu):
        runtime,bindings=common.load(gpu)
        teachers=train.teachers_for(runtime,common.read(BASE/'private/U_ROLES.json')['CHECK'])
        bank=[]
        for t in ts:bank+=worker.router(runtime,t)
        worker.evaluate(runtime,bindings,ts[-1],0,ARMS[2],160,{t['edit_id']:point(t) for t in ts},bank,ts,mode='bank',prefix=24,teacher_rows=teachers)
    common.write(RUN/'private/WEAK24_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',epoch=time.time()))


def controller():
    p=base.pipeline
    assert common.read(RUN/'public/WEAK24_ADMISSION.json')['status']=='PASS'
    common.write(RUN/'public/WEAK24_PROGRESS.json',dict(status='TRAINING',complete=False))
    p.wait([p.launch('weak24.py','weak24_fit',i,i) for i in (0,1)])
    common.write(RUN/'public/WEAK24_PROGRESS.json',dict(status='GENERATING',complete=False))
    p.wait([p.launch('weak24.py','weak24_generate',0,0)])
    p.wait([p.launch('weak24_queue.py','weak24_ingest')])
    paths=[point(t) for t in tasks()];assert len(paths)==24 and all(x.is_file() and not x.is_symlink() and x.resolve().is_relative_to((RUN/'private/weak24').resolve()) for x in paths)
    removed=[dict(path=str(x),bytes=x.stat().st_size) for x in paths]
    for x in paths:x.unlink()
    common.write(RUN/'private/WEAK24_DELETION.json',dict(files=removed,all_registered_GPU_consumers_complete=True,bindings_durable=True,rebuild='retrain',epoch=time.time()))
    common.write(RUN/'public/WEAK24_PROGRESS.json',dict(status='ASTRA_SCORING',complete=False))
    root=RUN/'private/judge_weak24_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        common.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    p.wait([p.launch('weak24_report.py','weak24_report')])


if __name__=='__main__':
    try:{'weak24_prepare':prepare,'weak24_fit':fit,'weak24_generate':generate,'weak24_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        common.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
