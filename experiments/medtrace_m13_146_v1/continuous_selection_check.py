"""Selection edge cases: no protection exceptions and one deterministic winner."""
import copy
from continuous_selection import choose

def selfcheck():
    gates=dict.fromkeys(('complete_no_missing','FIT','T1G','T1L','T2L','damage'),True)
    candidates={m:dict(T2G=80.,new_damage=2,protection_gate=copy.deepcopy(gates)) for m in ('C','M1','M13P')}
    assert choose(candidates) is None
    candidates['C']['T2G']=95.;candidates['C']['protection_gate']['T2L']=False
    candidates['M1']['T2G']=82.;candidates['M13P']['T2G']=81.;assert choose(candidates)=='M1'
    candidates['M13P']['T2G']=82.;assert choose(candidates)=='M13P'
    candidates['M1']['new_damage']=1;assert choose(candidates)=='M1'
    for key in gates:
        broken=copy.deepcopy(candidates)
        broken['M1']['protection_gate'][key]=False;broken['M13P']['protection_gate'][key]=False
        assert choose(broken) is None
    return dict(status='PASS',strict_threshold=True,protection_required_for_all_arms=True,highest_eligible_T2G=True,damage_tie_break=True,method_tie_break=True)
if __name__=='__main__':print(selfcheck())
