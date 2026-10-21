"""Frozen scale ordering, actual-NLL admission and no secant rescue of rejection."""
from .bshrink_ops import SCALES,FLOOR,conditions,action
from .bdecompose_cpu_check import run as guards


def run():
    guards();assert SCALES==(.5,.25)
    margin0=-5.8671875
    def row(gain,margin,alpha,trust=True):return dict(main_gate=dict(gates=dict(trust=trust)),main_conditions=conditions(gain,margin,margin0,alpha))
    assert action(row(.01,margin0+.1,.5),0)=='NEXT_FROZEN_SCALE'
    assert action(row(.01,margin0+.1,.25),1)=='STOP_NO_ELIGIBLE_SCALE'
    assert action(row(.1,margin0,.5),0)=='NEXT_FROZEN_SCALE'
    assert action(row(.1,margin0+.1,.5),0)=='PRESELECT'
    assert action(row(.1,margin0+.1,.5,False),0)=='STOP_TRUST'
    assert conditions(.5*FLOOR,margin0+.1,margin0,.5)['NLL_pass']
    assert not conditions(.5*FLOOR-2e-5,margin0+.1,margin0,.5)['NLL_pass']
    # A first main-eligible scale closes search even if later postconditions fail.
    visited=[];preselected=None
    for i,alpha in enumerate(SCALES):
        visited.append(alpha)
        if action(row(.1,margin0+.1,alpha),i)=='PRESELECT':preselected=alpha;break
    assert visited==[.5] and preselected==.5
    return dict(status='PASS',frozen_two_scales=True,actual_NLL_floor=True,margin_required=True,trust_failure_stops=True,first_eligible_closes_search=True,prior_weight_guard_and_restore_checks=True)
