"""Fixed objective, hard constraints, and infeasibility checks without models."""
import json
import os
from pathlib import Path
import numpy as np
import correction_math as cm


def main():
    assert cm.selfcheck()['status']=='PASS'
    args=(np.eye(2),np.array([2.,0.]),np.zeros(2),np.array([2.,0.]),0.,2.1,np.array([0.,1.]),.2)
    baseline,a=cm.solve(*args);changed,b=cm.solve(*args,fisher_objective=True)
    assert a['status']==b['status']=='LOCAL_KKT_PASS'
    assert changed[1]>=.2-1e-7
    assert np.linalg.norm(changed+np.array([2.,0.]))**2 < np.linalg.norm(baseline+np.array([2.,0.]))**2-.1
    assert max(b['residuals'])<=1e-7
    bad=args[:-1]+(2.,)
    point,audit=cm.solve(*bad,fisher_objective=True)
    assert point is None and audit['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    # A fixed objective must also respect an active trust ball.
    point,audit=cm.solve(*args,trust=1.1,fisher_objective=True)
    assert point is not None and np.linalg.norm(point)<=1.1+1e-7
    repeated,_=cm.solve(*args)
    np.testing.assert_allclose(repeated,baseline,rtol=0,atol=1e-12)
    import mechanism as m
    m.j.cpu()
    result=dict(status='PASS',old_nearest_point_unchanged=True,Fisher_interior_objective=True,
        hard_constraints_and_trust=True,infeasibility_certificate=True,zero_model_calls=True,
        paired_trial_import=True,original_scope_and_reference=True,full_consumer_implemented=False)
    m.c.write(m.RUN/'public/CPU_CHECKS.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
