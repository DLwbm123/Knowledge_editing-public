"""Dry-run complete-block counts without GPU, Judge, or remote writes."""
import os,tempfile,json
from pathlib import Path
os.environ.setdefault('RUN_ROOT',tempfile.gettempdir())
import phases

def main():
    from reporting import paired
    r=lambda key,source,judge:dict(edit='panel',task='H',input_id=key,source_group=source,judge_key=judge)
    a=[r('1','s1','a'),r('2','s2','b')];b=[r('1','s1','c'),r('2','s2','d')];scores=dict(a=True,b=True,c=True,d=False)
    assert 'CI95_complete_pairs' not in paired(a,b,scores,'edit')
    assert 'CI95_complete_pairs' in paired(a,b,scores,'source_group')
    with tempfile.TemporaryDirectory() as tmp:
        phases.ROOT=Path(tmp);db={}
        tasks=[dict(order=i,evaluation=[{}]*10) for i in range(1,49)]
        db['private/TASKS_R2_LOCKED.json']=dict(tasks=tasks);db['QUEUE.json']=[]
        db['RESOURCE_LEDGER.json']=dict(judge_submission_attempt_items_limit=None)
        db['public/BETA_GAMMA_SELECTION.json']=dict(beta=dict(selected=.1),gamma=dict(selected=.5))
        phases.read=lambda p:db[str(p.relative_to(phases.ROOT))]
        phases.write=lambda p,d:db.__setitem__(str(p.relative_to(phases.ROOT)),d)
        phases.references=lambda phase:[dict(judge_key='r',base_judge_key='b')]
        phases.read_scores=lambda root:{}
        phases.enqueue('PILOT');q=db['QUEUE.json'];assert sum(len(j['methods']) for j in q if j['mode']=='train')==40
        assert len([j for j in q if j['mode']=='check'])==3
        assert all(j['prefixes']==[4,8] for j in q if j['mode']=='bank')
        phases.enqueue('DEV');q=[j for j in db['QUEUE.json'] if j['phase']=='DEV']
        assert sum(len(j['methods']) for j in q if j['mode']=='train')==48
        assert len([j for j in q if j.get('holdout')])==4 and len([j for j in q if j['mode']=='single'])==1
        phases.enqueue('REG');q=[j for j in db['QUEUE.json'] if j['phase']=='REG']
        assert sum(len(j['methods']) for j in q if j['mode']=='train')==72
        assert all(set(j['methods'])=={'P_01','SP_01','L'} for j in q if j['mode']=='train')
    print('PASS: pilot40 / remainingDEV48 / conditionalREG72, full comparison arms and prefixes, separate scalar/no-training path')
if __name__=='__main__':main()
