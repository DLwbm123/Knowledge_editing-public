"""Bounded two-worker mechanism panel with the existing independent resource guard."""
import os
import time
import traceback
import v81_control as c


def controller():
    with (c.ROOT/'CONTROLLER_ONCE').open('x'):pass
    c.stage('geometry')
    cells=list((c.ROOT/'cells').glob('*.json'))
    assert len(cells)==16
    counts={k:0 for k in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts']}
    for f in c.ROOT.glob('EXIT_geometry_*.json'):
        d=c.read(f);assert d['exit_code']==0
        for k,v in d['counts'].items():counts[k]+=v
    assert counts['updates']==32 and counts['backwards']==80
    c.write(c.ROOT/'MAIN_COMPLETE.json',dict(status='SOURCE_MECHANISM_PANEL_COMPLETE_ANALYSIS_PENDING',
        cells=16,counts=counts,teacher_calls=0,heldout_calls=0,time=time.time()))


if __name__=='__main__':
    try:(c.guard if os.environ['CONTROL_ACTION']=='guard' else controller)()
    except BaseException as exc:
        c.write(c.ROOT/f'FAILURE_{os.environ["CONTROL_ACTION"]}.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        if os.environ['CONTROL_ACTION']=='controller' and not (c.ROOT/'STOP').exists():
            c.write(c.ROOT/'STOP',dict(reason='CONTROLLER_FAILURE',time=time.time()))
        raise
    finally:
        if os.environ['CONTROL_ACTION']=='controller':c.write(c.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
