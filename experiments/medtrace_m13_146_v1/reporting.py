"""Pure report calculations; no scoring, model selection, or execution admission."""
import collections
import random
import contract


def summarize(owners):
    nonempty=[xs for xs in owners.values() if xs];n=sum(map(len,nonempty))
    missing=sum(v is None for xs in nonempty for v in xs)
    lower=[sum(v==1 for v in xs)/len(xs) for xs in nonempty]
    upper=[sum(v is None or v==1 for v in xs)/len(xs) for xs in nonempty]
    return dict(observations=n,edit_units=len(nonempty),missing=missing,
        known_correct=sum(v==1 for xs in nonempty for v in xs),
        macro=100*sum(lower)/len(lower) if n and not missing else None,
        micro=100*sum(v==1 for xs in nonempty for v in xs)/n if n and not missing else None,
        macro_bounds=[100*sum(lower)/len(lower),100*sum(upper)/len(upper)] if n else [None,None])


def preserved(reference,candidate):
    assert reference.keys()==candidate.keys()
    return all(len(reference[k])==len(candidate[k]) and all(v is not None for v in reference[k]+candidate[k])
        and sum(candidate[k])>=sum(reference[k]) for k in reference if reference[k])


def interval(values,groups=None):
    if not values or any(v is None for v in values):return None
    packages=[[v] for v in values]
    if groups is not None:
        assert len(groups)==len(values)
        buckets=collections.defaultdict(list)
        for group,value in zip(groups,values):buckets[group].append(value)
        packages=list(buckets.values())
    rng=random.Random(20261013);samples=[]
    for _ in range(5000):
        draw=[v for _ in packages for v in rng.choice(packages)]
        samples.append(100*sum(draw)/len(draw))
    samples.sort();return [samples[125],samples[4874]]


def damage_preserved(reference,candidate):
    assert reference.keys()==candidate.keys()
    return all(reference[k]['fresh_Base_correct']==candidate[k]['fresh_Base_correct']
        and reference[k]['missing']==candidate[k]['missing']==0
        and candidate[k]['new_damage']<=reference[k]['new_damage'] for k in reference)


def success(gates):
    required={'complete','T2L_better_than_M1','T2L_edit_CI_positive','T2L_source_CI_positive',
        'T2L_historical_floor','FIT_per_owner','T2G_per_owner','prefix_FIT_T2G',
        'Base_correct_no_new_damage','M3_damage_per_owner'}
    return set(gates)==required and all(v is True for v in gates.values())


def selfcheck():
    assert summarize({1:[1],2:[0,0,0]})['macro']==50
    assert summarize({1:[1],2:[0,0,0]})['micro']==25
    assert summarize({1:[1,None]})['macro_bounds']==[50,100]
    assert not preserved({1:[1],2:[0]},{1:[0],2:[1]})
    assert not preserved({1:[1]},{1:[None]})
    left={1:dict(fresh_Base_correct=4,missing=0,new_damage=0),2:dict(fresh_Base_correct=4,missing=0,new_damage=1)}
    right={1:dict(fresh_Base_correct=4,missing=0,new_damage=1),2:dict(fresh_Base_correct=4,missing=0,new_damage=0)}
    assert not damage_preserved(left,right)
    assert interval([0.,0.])[0]==0 and interval([.1,.1],['a','b'])[0]>0
    good={k:True for k in ('complete','T2L_better_than_M1','T2L_edit_CI_positive','T2L_source_CI_positive',
        'T2L_historical_floor','FIT_per_owner','T2G_per_owner','prefix_FIT_T2G',
        'Base_correct_no_new_damage','M3_damage_per_owner')}
    assert success(good) and not success({})
    for k in good:assert not success(dict(good,**{k:False}))
    return dict(status='PASS',macro_not_micro=True,missing_bounds=True,
        average_does_not_hide_edit_or_damage_regression=True,each_failed_gate_blocks_success=True)
