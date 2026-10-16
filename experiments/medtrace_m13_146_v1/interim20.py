"""Fixed first-20 single-expert look; no training or full-bank replacement."""
import collections
import contract

METHOD_CONSUMERS={'M13':371,'FISHER_MIN':371,'BASE':335}
CONSUMERS=1077
EPOCH='MEDTRACE_NATIVE146_INTERIM20_20261015_V1'
QUEUE='judge_interim20_astra_medium'


def plan_subset(full):
    single=[r for r in full if r['mode']=='single' and 1<=r['owner']<=20]
    ids={r['query']['query_id'] for r in single}
    rows=single+[r for r in full if r['method']=='BASE' and r['query']['query_id'] in ids]
    assert collections.Counter(r['method'] for r in rows)==METHOD_CONSUMERS
    assert len(rows)==len({contract.consumer_id(r) for r in rows})==CONSUMERS
    assert not any(r['mode'] not in ('base','single') for r in rows)
    return rows


def frozen(parent,read):
    for owner in range(1,21):
        for arm in ('M13','FISHER_MIN'):
            result=read(parent/'private/results'/arm/f'{owner}.json')
            assert result['status']=='COMPLETE' and len(result['curve'])==160 and result['save_load_exact']
            path=parent/'private/candidates'/arm/f'{owner}.pt'
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(parent.resolve())
    return True
