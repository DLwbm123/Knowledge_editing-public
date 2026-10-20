"""Closure checks for fixed diagnostic attribution; no model or candidate."""
from .signed_response_readonly import parts
from .responseSignedBalance_cpu_check import run as old_checks

def run():
    result=old_checks()
    z=parts(1.,.93,.94,1.01,.95,-.1)
    assert abs(z['closure_error'])<1e-12 and abs(z['R']-.03)<1e-12
    full=parts(1.,1.04,1.04,1.01,1.06,-.1)
    assert full['Q']==0 and abs(full['closure_error'])<1e-12
    result['fixed_response_scalar_closures']='PASS'
    return result
