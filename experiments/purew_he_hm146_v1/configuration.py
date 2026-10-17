"""Frozen preparation contract: execution remains separately unauthorized."""
import math

VERSION='HE_HM146_DIAGNOSTIC_V1_20261016'
ARMS=('HM','HE')
PREFIXES=(1,50,100,146)
ELIGIBLE=(1,22,25,26,30,33,46,79,89)
N=146
SLOTS=20
MAX_ROWS=16+4*N+18*len(ELIGIBLE)
PER_ARM=dict(attempt_slots=2920,committed_updates=2921,backwards=43800,
             forwards=2190000,generation_attempts=4128,candidate_checks=23360,
             active_set_checks=0,new_Judge=4123,GPU_seconds=64*3600)
RECIPE='ORIGINAL_BASE_H095_PLUS_ARM_E101_SEQUENTIAL146_V1'


def validate_protocol(p):
    assert p['version']==VERSION and tuple(p['arms'])==ARMS
    assert p['orders']==list(range(1,N+1)) and p['slots_per_edit']==SLOTS
    assert tuple(p['eligible_orders'])==ELIGIBLE
    assert p['budgets']['per_arm']==PER_ARM
    assert p['budgets']['global']=={k:2*v for k,v in PER_ARM.items()}
    assert p['budgets']['wall_hours']==168 and p['memory']['maximum_rows']==MAX_ROWS
    assert p['objective']==dict(H_alpha=.95,HE_E_weight=1,HM_E_weight=0,E_target_ratio=1.01,
        conditional_KL_weight=0,prefix_KL_weight=0,removed_KL_weight=0,CE_candidate_guard=False)
    assert p['policy']['scientific_failure']=='RECORD_AND_CONTINUE'
    assert p['policy']['mechanical_failure']=='STOP_NO_REPLAY'
    return p


def require_execution(p,authorization,hardware):
    validate_protocol(p)
    assert authorization.get('version')==VERSION
    assert all(authorization.get(k) is True for k in ('execution','install','model_calls','Judge')),'PREPARATION_ONLY'
    assert hardware.get('allocation_authorized') is True
    assert hardware.get('memory_admission')=='PASS'
    assert hardware.get('source_token_span_admission')=='PASS'
    assert hardware.get('production_integration_admission')=='PASS'
    assert type(hardware.get('physical_GPU')) is int
    assert str(hardware.get('GPU_UUID','')).startswith('GPU-')


def source_gate(ce,margins,eligible):
    fit=math.isfinite(ce) and ce<.1
    evidence=(len(margins)==10 and all(math.isfinite(v) and v>0 for v in margins)) if eligible else None
    return dict(CE_pass=fit,evidence_pass=evidence,qualified=fit and (evidence if eligible else True),
                failed_scientific_gate=not fit or (eligible and not evidence),continue_queue=True)


def check_counts(own,total):
    for k,cap in PER_ARM.items():
        a=own.get(k,0);b=total.get(k,0)
        assert math.isfinite(a) and math.isfinite(b) and 0<=a<=cap and 0<=b<=2*cap,(k,a,b,cap)


def schedule():
    """No outcome-dependent selection or automatic launch."""
    for arm in ARMS:
        yield dict(arm=arm,action='base',owner=0)
        yield dict(arm=arm,action='admit',owner=1)
        for owner in range(1,N+1):
            yield dict(arm=arm,action='learn',owner=owner)
            yield dict(arm=arm,action='insertion',owner=owner)
            if owner in PREFIXES:
                yield dict(arm=arm,action='bank',owner=owner)
                yield dict(arm=arm,action='protection',owner=owner)
        yield dict(arm=arm,action='reload',owner=N)
        yield dict(arm=arm,action='retire',owner=N)
