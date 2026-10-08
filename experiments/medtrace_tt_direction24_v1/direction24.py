"""One fixed 24-expert replication; reuse pilot8 and train only the next16."""
import os
import sqlite3
import time
import traceback
from pathlib import Path
import torch
import direction as d
import initialwrite as iw

p,c,RUN=d.p,d.c,d.RUN
PARENT=Path(os.environ['DIRECTION_PARENT'])
ARMS=d.ARMS
LABELS=('W0',)+ARMS
PRIMARY=d.PRIMARY
TRAIN_GPUS=d.TRAIN_GPUS
EVAL_GPUS=d.EVAL_GPUS


def selected():return p.tasks()[:24]
def new_tasks():return p.tasks()[8:24]
def point(t,arm):
    root=PARENT if t['edit_id'] in {x['edit_id'] for x in p.tasks()[:8]} else RUN
    return root/'private/weights'/t['anonymous_edit']/arm/'FINAL.pt'
def staged(t):return RUN/'private/staged'/str(t['order'])/'FINAL.pt'
def start_state(t):return p.load_state(staged(t))['expert']

# Reuse the frozen pilot's math, loss, evaluation and binding checks verbatim.
d.selected=selected;d.point=point;d.start_state=start_state


def plan():
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    previous=c.read(PARENT/'public/RESULTS.json')
    assert previous['decision']=='INCONCLUSIVE_POSITIVE_POINT_ONLY'
    assert previous['edit_deltas']['T2G']['delta_bounds_pp'][0]>0
    checks=d.selfcheck();iw.selfcheck()
    db=sqlite3.connect('file:'+str(PARENT/'private/judge_direction_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    old={(a,m,q):path for a,m,q,path in db.execute('SELECT method,mode,query_id,path FROM consumer')};db.close()
    routes=c.read(RUN/'private/ROUTES.json');first8={t['edit_id'] for t in p.tasks()[:8]};new16={t['edit_id'] for t in new_tasks()};chosen=first8|new16
    assert len(first8)==8 and len(new16)==16 and not first8&new16
    consumers=[];jobs=[];counts={}
    def add(arm,mode,row,expert,path=None):
        dest=str(path or d.output(arm,row['query_id']))
        consumers.append(dict(arm=arm,mode=mode,query_id=row['query_id'],path=dest,produced_arm=arm))
        if path is None:jobs.append(dict(arm=arm,mode=mode,row=row,expert=expert,path=dest))
    def candidate(mode,row,expert):
        label=PRIMARY+('_NAT' if mode=='natural_source_CHECK' else '')
        baseline='W0_NAT' if mode=='natural_source_CHECK' else 'W0'
        path=None if expert in new16 else old[label if expert in first8 else baseline,mode,row['query_id']]
        add(label,mode,row,expert,path)
    for q,row in p.queries(146).items():
        expert=routes[q]['effective_expert'];add('W0','bank_R0',row,expert,old['W0','bank_R0',q]);candidate('bank_R0',row,expert)
    source_rows=c.read(RUN/'private/SOURCE_CHECK_ROWS.json')
    for row in source_rows:
        q=row['query_id']
        for mode,expert,suffix in [('forced_source_CHECK',p.tasks()[row['forced_expert_index']]['edit_id'],''),('natural_source_CHECK',routes[q]['effective_expert'],'_NAT')]:
            for label in ('BASE','W0'):add(label+suffix,mode,row,expert if label=='W0' else None,old[label+suffix,mode,q])
            candidate(mode,row,expert)
    for name,ids in [('pilot8',first8),('new16',new16),('all24',chosen)]:
        counts[name]=dict(main=sum(routes[q]['effective_expert'] in ids for q in p.queries(146)),forced=sum(p.tasks()[q['forced_expert_index']]['edit_id'] in ids for q in source_rows),natural=sum(routes[q['query_id']]['effective_expert'] in ids for q in source_rows))
    assert len(consumers)==3602 and len(jobs)==sum(counts['new16'].values()) and len({x['path'] for x in jobs})==len(jobs)
    assert all(Path(x['path']).is_file() for x in consumers if not Path(x['path']).is_relative_to(RUN/'private/outputs'))
    assert all(point(t,PRIMARY).is_file() for t in p.tasks()[:8])
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json');unique={j['row']['query_id']:j['row'] for j in jobs}
    assert all(d.scoped.base_tokens(row,bindings) for row in unique.values())
    lock=dict(arms=list(ARMS),primary=PRIMARY,experts=24,new_experts=16,reused_experts=8,selected_edits=[t['edit_id'] for t in selected()],new_edits=[t['edit_id'] for t in new_tasks()],pilot_edits=[t['edit_id'] for t in p.tasks()[:8]],support_counts=counts,
        phases=[['native',140],['A2',80],['original_tail_RAW',320]],optimizer_steps=8960,training_forwards=15680,training_backwards=15680,diagnostic_forwards=160,endpoint_rebuild_extra_steps=320,
        new_generations=len(jobs),Judge_upper_bound=len(jobs),consumers=len(consumers),training_roles=['native','FIT'],replay=False,training_GPUs=list(TRAIN_GPUS),inference_GPUs=list(EVAL_GPUS),parameters=7168,final_TT=16,
        require_24_T2G_edit_CI_positive=True,require_new16_positive_point_and_no_harm=True,require_no_increase_generation_caps=True)
    c.write(RUN/'private/DIRECTION_LOCK.json',lock);c.write(RUN/'private/JOBS.json',jobs);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k not in ('selected_edits','new_edits','pilot_edits')},selfcheck=checks,Base_lookup_unique_inputs=len(unique)))
    p.done('PLAN_COMPLETE')


def prepare(runtime,t):
    dest=staged(t)
    if dest.exists():return
    expert=p.tr.TT4(t['seed'],8,8).to(runtime.device)
    curve=iw.phases(runtime,t,'STAGED_220',expert,[('native',140),('A2',80)])
    c.save(dest,dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},state_hash=c.state_hash(expert)))
    c.write(dest.parent/'PREPARATION.json',dict(status='COMPLETE',steps=220,forwards=300,backwards=300,curve=curve))
    resume=iw.point(t,'STAGED_220').parent/'resume_native_A2.pt'
    d.consume_resume(resume)


def mechanical():
    t=new_tasks()[0]
    with p.lease(TRAIN_GPUS[0]):
        runtime,_=c.load(TRAIN_GPUS[0]);prepare(runtime,t)
        expert,audit,resume=d.fit(runtime,t,'ADAM');original=p.load_state(p.initial(t))['expert']
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in expert.state_dict().items()),'Original540 endpoint differs'
        c.write(RUN/'private/ENDPOINT_PARITY.json',dict(status='PASS',expert_order=t['order'],exact_all_cores=True,reconstruction_steps=540,extra_steps=320,extra_forwards=640,extra_backwards=640,audit=audit))
        d.consume_resume(resume)
    p.done('MECHANICAL_COMPLETE')


def train():
    assert (RUN/'private/MECHANICAL_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=TRAIN_GPUS[part]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in new_tasks()[part::2]:
            dest=point(t,PRIMARY)
            if not (dest.parent/'TRAINING.json').exists():
                prepare(runtime,t);expert,audit,resume=d.fit(runtime,t,'RAW')
                c.save(dest,dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},binding=audit['binding'],state_hash=c.state_hash(expert)))
                saved=p.load_state(dest);assert all(torch.equal(saved['expert'][k],v.detach().cpu()) for k,v in expert.state_dict().items())
                audit.update(save_load_exact=True,parameters=7168,final_state_hash=c.state_hash(expert));c.write(dest.parent/'TRAINING.json',audit)
                d.consume_resume(resume);del expert
            # This generated220 state has no remaining training consumer.
            if staged(t).exists():d.consume_resume(staged(t))
    p.done('TRAIN_'+str(part))


def controller():
    import pipeline
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    if not (RUN/'private/MECHANICAL_COMPLETE.json').exists():pipeline.wait([pipeline.launch('direction24.py','direction24_mechanical',TRAIN_GPUS[0],0)])
    pipeline.wait([pipeline.launch('direction24.py','direction24_train',g,i) for i,g in enumerate(TRAIN_GPUS) if not (RUN/'private'/('TRAIN_'+str(i)+'.json')).exists()])
    pipeline.wait([pipeline.launch('direction24.py','direction24_eval',g,i) for i,g in enumerate(EVAL_GPUS) if not (RUN/'private'/('EVAL_'+str(i)+'.json')).exists()])
    assert all(Path(x['path']).exists() for x in c.read(RUN/'private/CONSUMERS.json'));p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('direction24_queue.py','direction24_ingest')])
    root=RUN/'private/judge_direction24_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('direction24_report.py','direction24_report')])


if __name__=='__main__':
    try:
        {'direction24_plan':plan,'direction24_mechanical':mechanical,'direction24_train':train,'direction24_eval':d.evaluate,'direction24_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
