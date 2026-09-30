"""CPU self-check for fixed cardinalities, CHECK isolation and exact admission."""
import os,json,tempfile
from pathlib import Path
def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);os.environ['RUN_ROOT']=tmp
        import phases
        phases.write=lambda p,d:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(d)))
        phases.references=lambda phase:[];phases.read_scores=lambda root:{}
        phases.write(root/'private/TASKS_R2_LOCKED.json',dict(tasks=[dict(order=o,evaluation=[dict(task='T0')]) for o in range(1,49)]))
        phases.write(root/'RESOURCE_LEDGER.json',dict(judge_submission_attempt_items_limit=None))
        phases.write(root/'private/judge_sol/JUDGE_MISSING_LOCK.json',dict(keys=[]));phases.write(root/'QUEUE.json',[])
        phases.enqueue('PILOT');q=phases.read(root/'QUEUE.json');assert len(q)==8 and sum(len(j['methods']) for j in q)==48
        assert all(j['check'] and not j['formal'] and j['mode']=='train' for j in q)
        phases.write(root/'public/RHO_SELECTION.json',dict(selected=2));phases.enqueue('DEV');q=phases.read(root/'QUEUE.json')
        train=[j for j in q if j['phase']=='DEV' and j['mode']=='train'];assert len(train)==16 and sum(len(j['methods']) for j in train)==48
        assert all(j['methods']==['CAP_2','EGP_2','EGP_A_2'] for j in train)
        phases.write(root/'public/JOINT_GAIN_DECISION.json',dict(REG_methods=['EGP_2','EGP_A_2']));phases.enqueue('REG');q=phases.read(root/'QUEUE.json')
        assert sum(len(j['methods']) for j in q if j['phase']=='REG' and j['mode']=='train')==48
        import controller
        controller.enqueue=lambda phase:(_ for _ in ()).throw(AssertionError('Duplicate admission'))
        assert controller.restore()==('REG',[])
        q[-1].update(status='RUNNING',pid=99999999);phases.write(root/'QUEUE.json',q)
        try:controller.restore()
        except AssertionError as error:assert 'diagnosed repair' in str(error)
        else:raise AssertionError('Dead job silently retried')
        phases.curves=lambda phase:(_ for _ in ()).throw(AssertionError('Frozen rho recomputed'))
        assert phases.select()
        phases.write(root/'public/JOINT_GAIN_DECISION.json',dict(REG_allowed=False))
        assert not phases.decide()
        # A reference with absent qualification must never overwrite the frozen cohort.
        h=[dict(arm='H',mode='EXPOSED_REGRESSION',input_id=str(i),frozen_base_correct=i<35) for i in range(47)]
        sp=[{k:v for k,v in dict(r,arm='SP').items() if k!='frozen_base_correct'} for r in h]
        phases.references=lambda phase:h+sp
        phases.write(root/'QUEUE.json',[dict(id='DEV-0',phase='DEV')])
        phases.write(root/'jobs/DEV-0/CONSUMERS.json',[dict(r,arm='CAP_2') for r in h])
        actual=phases.candidates('DEV')
        assert len(actual)==47 and sum(r['frozen_base_correct'] is True for r in actual)==35
    from policy import joint
    scores={};candidate=[];baseline=[]
    def row(mode,prefix,task,i,key):return dict(mode=mode,prefix=prefix,task=task,edit='e'+str(i),input_id=str(i),judge_key=key,source_group='s',frozen_base_correct=True)
    for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]:
        for task in ['T0','T1G','T2G','T1L','T2L','T2L_PRESSURE']:
            for i in range(5):
                a=f'a-{mode}-{prefix}-{task}-{i}';b=f'b-{mode}-{prefix}-{task}-{i}';candidate.append(row(mode,prefix,task,i,a));baseline.append(row(mode,prefix,task,i,b))
                scores[a]=True;scores[b]=task!='T2L_PRESSURE'
    for prefix in [4,8,12,24]:
        for i in range(35):
            key=f'z-{prefix}-{i}';candidate.append(row('EXPOSED_REGRESSION',prefix,'EXPOSED_REGRESSION',i,key));baseline.append(row('EXPOSED_REGRESSION',prefix,'EXPOSED_REGRESSION',i,key))
    refs={m:baseline for m in ['H','S','A0','E_orig','SP']};mm={'CAP_2':dict(global_clip_fraction=.01)}
    assert joint({'CAP_2':candidate},refs,scores,mm)['REG_methods']==['CAP_2']
    mm['CAP_2']['global_clip_fraction']=.20;d=joint({'CAP_2':candidate},refs,scores,mm)
    assert d['candidates']['CAP_2']['status']=='PERFORMANCE_PASS_MECHANISM_UNRESOLVED' and not d['REG_allowed']
    mm['CAP_2']['global_clip_fraction']=.01
    scores['a-sequential-24-T2G-0']=scores['a-sequential-24-T2G-1']=False
    assert not joint({'CAP_2':candidate},refs,scores,mm)['REG_allowed']
    print('PASS: fixed 48+48 counts, CHECK-only pilot, shared rho, REG48 ceiling, exact gates, restart and frozen selections')
if __name__=='__main__':main()
