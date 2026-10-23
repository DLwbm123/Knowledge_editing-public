"""Dependency-ordered score, train and free-answer generation with independent guard."""
import os
import time
import traceback
import v81_control as c


def controller():
    with (c.ROOT/'CONTROLLER_ONCE').open('x'):pass
    c.stage('teacher')
    scores=[c.read(c.ROOT/'scores'/f'{owner}.json') for owner in range(1,9)]
    assert sum(x['sequence_scores'] for x in scores)==480
    assert all(x['status']=='PASS' and len(x['records'])==5 for x in scores)
    c.write(c.ROOT/'SUPERVISION_FROZEN.json',dict(status='PASS',owners=8,inputs=40,candidates_per_input=3,
        scores=480,review_before_scores=True,selection=False,time=time.time()))
    c.stage('train')
    assert len(list((c.ROOT/'training').glob('*/*.json')))==32
    c.write(c.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=32,steps=160,time=time.time(),selection=False))
    c.stage('evaluate')
    assert len(list((c.ROOT/'outputs').glob('*/*/*.json')))==620
    totals={k:0 for k in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts']}
    for f in c.ROOT.glob('EXIT_*.json'):
        d=c.read(f);assert d['exit_code']==0
        for k,v in d['counts'].items():totals[k]+=v
    assert totals['updates']==5120 and totals['backwards']==25600
    assert all(totals[k]<=v for k,v in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts'].items())
    c.write(c.ROOT/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=620,counts=totals,time=time.time()))


if __name__=='__main__':
    try:(c.guard if os.environ['CONTROL_ACTION']=='guard' else controller)()
    except BaseException as e:
        c.write(c.ROOT/f'FAILURE_{os.environ["CONTROL_ACTION"]}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        if os.environ['CONTROL_ACTION']=='controller':c.write(c.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
