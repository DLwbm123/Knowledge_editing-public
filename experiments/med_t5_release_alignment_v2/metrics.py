"""Released M3Bench T5: dataset-level follow-up accuracy difference.

The public release has a task specification, but no T5 builder/evaluator or
author experiment manifest. This implementation follows that specification;
it does not claim to recover the paper table's unpublished scoring pipeline.
"""
from collections import Counter


def bounds(values):
    assert all(v is None or type(v) is bool for v in values)
    n = len(values)
    return dict(n=n, correct_known=sum(v is True for v in values),
        unknown=sum(v is None for v in values),
        lower_pct=100*sum(v is True for v in values)/n if n else None,
        upper_pct=100*sum(v is not False for v in values)/n if n else None)


def admitted(events):
    # Follow-up correctness is an outcome/baseline measurement, not eligibility.
    # Missing prior judgments remain unresolved, never silently converted false.
    return [e for e in events if e['prior_core'] is False]


def temporal(events, after):
    assert len({e['event_id'] for e in events}) == len(events)
    before = [e['followup_core'] for e in events]
    post = [after.get(e['owner']) for e in events]
    pre, result = bounds(before), bounds(post)
    delta = dict(n=len(events), lower_pp=result['lower_pct']-pre['upper_pct'] if events else None,
        upper_pp=result['upper_pct']-pre['lower_pct'] if events else None,
        unknown_before=pre['unknown'], unknown_after=result['unknown'])
    delta.update(lower=delta['lower_pp']/100 if events else None,
        upper=delta['upper_pp']/100 if events else None,units='lower/upper are fractions; lower_pp/upper_pp are percentage points')
    retention = [v for e,v in zip(events,post) if e['followup_core'] is True]
    fix = [v for e,v in zip(events,post) if e['followup_core'] is False]
    return dict(primary='delta_ACC_T2', ACC_before_T2=pre, ACC_after_T2=result, delta_ACC=delta,
        retention_on_known_pre_correct=bounds(retention), fix_on_known_pre_wrong=bounds(fix),
        pre_correct_membership_unresolved=sum(v is None for v in before),
        by_presence_code={code:temporal([e for e in events if e['presence_code']==code],after)
            for code in ('11','01','10')} if len({e['presence_code'] for e in events})>1 else {})


def evaluate(events, consumers, decisions):
    panels = {}
    yes={e['owner']:e['presence_code'][1]=='1' for e in events}
    no={owner:not value for owner,value in yes.items()}
    majority_accuracy=100*sum(yes.values())/len(events)
    for mode in ('single','bank'):
        rows = [r for r in consumers if r['mode']==mode]
        by_owner = {r['owner']:decisions.get(r['key'],{}).get('core_correct') for r in rows if r['role']=='T5'}
        assert set(by_owner)=={e['owner'] for e in events}, 'Do not drop missing or incorrect follow-ups'
        future=temporal(events,by_owner)
        future['delta_advantage_over_always_Yes_pp'] = dict(
            lower_pp=future['ACC_after_T2']['lower_pct']-majority_accuracy,
            upper_pp=future['ACC_after_T2']['upper_pct']-majority_accuracy,
            explanation='Shared Base term cancels; equivalent to post-accuracy difference on the identical cohort')
        panels[mode] = dict(T0=bounds([decisions.get(r['key'],{}).get('core_correct') for r in rows if r['role']=='T0']),
            T5=future)
    return dict(primary='delta_ACC_T2',aggregation='dataset-level mean over all prior-incorrect admitted pairs',
        panels=panels,unknowns='Conservative unknown-label bounds, not confidence intervals',
        analytical_controls=dict(unchanged_Base=dict(ACC_T2=bounds([e['followup_core'] for e in events]),delta_ACC_pp=0),
            always_Yes=temporal(events,yes),always_No=temporal(events,no),
            provenance='Computed from frozen binary references; no new model/Judge calls'),
        correctness='Unchanged CORE LLM contract; a permitted LLM option in the public release',
        exact_author_paper_reproduction=False)


def selfcheck():
    e=[dict(event_id='a',owner=1,prior_core=False,followup_core=False,presence_code='11'),
       dict(event_id='b',owner=2,prior_core=False,followup_core=True,presence_code='10'),
       dict(event_id='c',owner=3,prior_core=False,followup_core=None,presence_code='10'),
       dict(event_id='d',owner=4,prior_core=True,followup_core=False,presence_code='01'),
       dict(event_id='e',owner=5,prior_core=None,followup_core=False,presence_code='01')]
    assert [x['owner'] for x in admitted(e)]==[1,2,3]
    # No-change known predictions must have zero delta; old all-edit 1-flip
    # would instead award 100% even when both future answers remain incorrect.
    pair=e[:2];v=temporal(pair,{1:False,2:True})
    assert v['delta_ACC']['lower_pp']==v['delta_ACC']['upper_pp']==0
    v=temporal(pair,{1:True,2:False})
    assert v['ACC_after_T2']['lower_pct']==50 and v['delta_ACC']['lower_pp']==0
    v=temporal([e[2]],{3:False})
    assert v['delta_ACC']['lower_pp']==-100 and v['delta_ACC']['upper_pp']==0
    v=temporal([e[2]],{})
    assert v['ACC_after_T2']['n']==1 and v['ACC_after_T2']['unknown']==1
    return dict(status='PASS',gates=['prior-only eligibility','unknown follow-up retained',
        'unchanged model delta zero','gain and damage cancel','unknown paired baseline bounds','missing prediction denominator retained'])
