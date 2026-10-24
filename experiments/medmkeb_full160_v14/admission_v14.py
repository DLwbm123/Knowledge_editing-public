"""The next registered gradient pass validates the previous replay transition."""
import math

def replay_metrics(actual, reference):
    assert len(actual['support_nll'])==len(reference['support_nll'])==54
    a=[actual['source_loss']]+actual['support_nll'];b=[reference['source_loss']]+reference['support_nll']
    assert all(math.isfinite(x) for x in a+b)
    difference=max(abs(x-y) for x,y in zip(a,b));assert difference<=1e-6,('REPLAY_NLL_DRIFT',difference)
    return difference

def source_resume(saved, complete):
    assert saved['scalar']['step']==31 and len(saved['curve'])==31 and complete['updates']==31
    assert len(saved['scalar']['second_moments'])==2 and all(math.isfinite(x) and x>=0 for x in saved['scalar']['second_moments'])
    return complete['final_source_loss']

def selfcheck():
    m=dict(source_loss=1.,support_nll=[1.,2.]*27);assert replay_metrics(m,m)==0
    for bad in [dict(m,source_loss=1.0001),dict(m,support_nll=[1.]*53)]:
        try:replay_metrics(bad,m)
        except AssertionError:pass
        else:raise AssertionError('Missing/drifted replay admitted')
    assert source_resume(dict(scalar=dict(step=31,second_moments=[0.,1.]),curve=[{}]*31),dict(updates=31,final_source_loss=.5))==.5
    return dict(status='PASS',full55_replay_metrics=True,missing_drift_rejected=True,optimizer_step_bound=True,no_model_calls=True)
