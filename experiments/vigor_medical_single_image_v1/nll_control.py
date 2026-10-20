"""NLL response controller; margin measured but never used for acceptance/radius."""
import math
SCALES=(1.,.5,.25)
def admission(cost,linear,cap,bias):
 return all(math.isfinite(x) for x in [cost,*linear,bias]) and 0<cost<=1.01*cap and linear[0]<-1e-5 and abs(bias+linear[2])<=.01
def response(before,after,margin_before,margin_after,linear):
 gain=before-after;mg=margin_after-margin_before;rn=gain/(-linear[0]) if linear[0]<0 else None
 rm=mg/linear[1] if linear[1]!=0 else None
 valid=rn is not None and all(math.isfinite(v) for v in [gain,mg,rn])
 return dict(NLL_gain=gain,margin_gain=mg,rho_NLL=rn,rho_margin=rm,margin_is_diagnostic_only=True,passed=valid and gain>1e-5 and rn>=.1)
def next_radius(used,result):
 if not result['passed']:raise ValueError('Rejected trial cannot set radius')
 rho=result['rho_NLL'];return used*.5 if rho<.25 else min(1.,used*2) if rho>.75 else used
def run_checks():
 assert admission(1,[-1,-1,0],1,0)
 assert not admission(1,[-1,1,.02],1,0)
 assert not admission(1,[1,1,0],1,0)
 z=response(2,1.5,0,-2,[-1,-1,0]);assert z['passed'] and next_radius(.5,z)==.5
 assert not response(2,2.1,0,2,[-1,1,0])['passed']
 return dict(status='PASS',margin_decline_allowed=True,NLL_decline_required=True,null_and_cost_required=True,radius_NLL_only=True)
