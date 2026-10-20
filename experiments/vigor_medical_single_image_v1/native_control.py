"""Frozen finite-response rules, shared by both direction mechanisms."""
import math
SCALES=(1.,.5,.25)

def admission(cost,linear,cap,bias):
    return all(math.isfinite(x) for x in [cost,*linear,bias]) and 0<cost<=1.01*cap and linear[0]<-1e-5 and linear[1]>1e-5 and abs(bias+linear[2])<=.01

def response(before,after,margin_before,margin_after,linear):
    gain=before-after;mg=margin_after-margin_before
    rn=gain/(-linear[0]) if linear[0]<0 else None
    rm=mg/linear[1] if linear[1]>0 else None
    valid=rn is not None and rm is not None and all(math.isfinite(v) for v in [gain,mg,rn,rm])
    return dict(NLL_gain=gain,margin_gain=mg,rho_NLL=rn,rho_margin=rm,passed=valid and gain>1e-5 and mg>1e-5 and min(rn,rm)>=.1)

def next_radius(used,result):
    if not result['passed']:raise ValueError('Cannot update radius from a rejection')
    rho=min(result['rho_NLL'],result['rho_margin'])
    return used*.5 if rho<.25 else min(1.,used*2) if rho>.75 else used

def run_checks():
    assert admission(1,[-1,1,0],1,0)
    assert not admission(1,[1,1,0],1,0)
    assert not admission(1,[-1,1,.02],1,0)
    z=response(2,1.5,0,.5,[-1,1,0]);assert z['passed'] and next_radius(.5,z)==.5
    assert not response(2,2.1,0,.5,[-1,1,0])['passed']
    assert next_radius(.25,response(2,1,0,1,[-1,1,0]))==.5
    return dict(status='PASS',native_admission=True,actual_response_acceptance=True,bounded_radius_update=True,no_radius_from_rejection=True)
