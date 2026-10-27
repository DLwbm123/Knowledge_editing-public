"""Four-GPU full-training campaign, all endpoints frozen before evaluation."""
import os
import time
import traceback
import v81_control as c


def stage(phase):
    jobs=[c.launch(phase,p,g) for p,g in enumerate(c.read(c.ROOT/'GPUS.json'))]
    codes=[]
    for child in jobs:
        codes.append(child.wait())
        d=c.read(c.ROOT/'RESOURCE_TARGETS.json')
        next(t for t in d['targets'] if t['pid']==child.pid)['ended_epoch']=time.time()
        c.write(c.ROOT/'RESOURCE_TARGETS.json',d)
    assert codes==[0]*4,(phase,codes)
    assert all(c.read(c.ROOT/f'COMPLETE_{phase}_{p}.json')['status']=='PASS' for p in range(4))


def controller():
    with (c.ROOT/'CONTROLLER_ONCE').open('x'):pass
    stage('train')
    assert len(list((c.ROOT/'candidates').glob('*/*.pt')))==40
    c.write(c.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=40,time=time.time(),no_selection=True))
    stage('evaluate')
    assert len(list((c.ROOT/'outputs').glob('*/*/*.json')))==775
    totals={k:0 for k in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts']}
    for f in c.ROOT.glob('EXIT_*.json'):
        d=c.read(f);assert d['exit_code']==0
        for k,v in d['counts'].items():totals[k]+=v
    assert totals['evaluation_generations']==775
    assert all(totals[k]<=v for k,v in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts'].items())
    c.write(c.ROOT/'MAIN_COMPLETE.json',dict(status='FULL_METHOD_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=775,counts=totals,time=time.time()))
    from review import evaluation_pack
    evaluation_pack(c.ROOT)


if __name__=='__main__':
    try:(c.guard if os.environ['CONTROL_ACTION']=='guard' else controller)()
    except BaseException as exc:
        c.write(c.ROOT/f'FAILURE_{os.environ["CONTROL_ACTION"]}.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        if os.environ['CONTROL_ACTION']=='controller' and not (c.ROOT/'STOP').exists():c.write(c.ROOT/'STOP',dict(reason='CONTROLLER_FAILURE',time=time.time()))
        raise
    finally:
        if os.environ['CONTROL_ACTION']=='controller':c.write(c.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
