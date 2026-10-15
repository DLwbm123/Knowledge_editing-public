"""Algebra checks for the stated local lemma; this is not a model experiment."""
import json
import math
from pathlib import Path


def directional_case(regularization):
    theta, source, penalty_grad, metric = (1.,0.),(-3.,.5),(2.,0.),(.4,.7)
    total=[g+regularization*q for g,q in zip(source,penalty_grad)]
    clip=min(1.,1./math.sqrt(sum(v*v for v in total)))
    direction=[-clip*d*v for d,v in zip(metric,total)]
    derivative=sum(q*v for q,v in zip(penalty_grad,direction))
    formula=-clip*(sum(q*d*g for q,d,g in zip(penalty_grad,metric,source))+
                   regularization*sum(q*d*q for q,d in zip(penalty_grad,metric)))
    assert math.isclose(derivative,formula,abs_tol=1e-14)
    h=1e-6
    numerical=((theta[0]+h*direction[0])**2-(theta[0]-h*direction[0])**2)/(2*h)
    assert math.isclose(numerical,derivative,abs_tol=1e-9)
    return dict(regularization=regularization,penalty_directional_derivative=derivative)


if __name__=='__main__':
    weak,strong=directional_case(.1),directional_case(2.)
    assert weak['penalty_directional_derivative']>0 and strong['penalty_directional_derivative']<0
    # A zero quadratic gradient does not prevent a positive second-order penalty.
    h=1e-3;assert 2*0.==0. and (0.+h)**2>0
    result=dict(status='PASS',weak_regularization=weak,strong_regularization=strong,
                zero_origin_first_order_derivative=0.,zero_origin_second_order_penalty=h*h,
                scope='fixed-metric local algebra before C step matching',model_calls=0,GPU_leases=0,Judge=0)
    path=Path(__file__).resolve().parents[2]/'reports/medtrace_m13_146_preparation_v1_20261014/THEORY_DIRECTION_CHECKS.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Local lemma algebra PASS; no model calls')
