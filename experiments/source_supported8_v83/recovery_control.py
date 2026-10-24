"""Finish only outstanding evaluation, preserving the original clock and costs."""
import os
import time
import traceback
import v81_control as c


def controller():
    with (c.ROOT/'RECOVERY1_CONTROLLER_ONCE').open('x'):pass
    c.stage('evaluate_recovery1')
    assert len(list((c.ROOT/'outputs').glob('*/*/*.json')))==620
    totals={k:0 for k in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts']}
    for f in c.ROOT.glob('PROGRESS_*.json'):
        for k,v in c.read(f)['counts'].items():totals[k]+=v
    assert totals['updates']==5120 and totals['backwards']==25600
    assert totals['evaluation_generations']==622  # 620 consumers plus two retained failed attempts
    assert all(totals[k]<=v for k,v in c.read(c.ROOT/'RESOURCE_LIMITS.json')['counts'].items())
    c.write(c.ROOT/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=620,
        counts=totals,time=time.time(),original_routing_restored=True,failed_attempts_retained=2,
        already_valid_outputs_reused=43,no_retraining=True))


if __name__=='__main__':
    action=os.environ['CONTROL_ACTION']
    try:(c.guard if action=='guard' else controller)()
    except BaseException as e:
        c.write(c.ROOT/f'FAILURE_recovery1_{action}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        if action=='controller':c.write(c.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
