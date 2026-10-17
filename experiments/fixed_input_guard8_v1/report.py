"""Anonymous aggregate report from durable, terminal Judge consumers only."""
import json
import os
from pathlib import Path
import sqlite3


def rate(values):
    n=len(values); correct=sum(v==1 for v in values);missing=sum(v is None for v in values)
    return dict(correct=correct,total=n,missing=missing,lower=correct/n if n else None,upper=(correct+missing)/n if n else None)


def paired(before,after):
    assert len(before)==len(after)
    known=[i for i,v in enumerate(before) if v==1]
    return dict(before=rate(before),after=rate(after),retention_among_known_insert_correct=rate([after[i] for i in known]),
                correct_to_wrong=sum(a==1 and b==0 for a,b in zip(before,after)),
                wrong_to_correct=sum(a==0 and b==1 for a,b in zip(before,after)),
                pairs_with_unknown=sum(a is None or b is None for a,b in zip(before,after)))


def main():
    r=Path(os.environ['RUN_ROOT']);read=lambda p:json.loads(p.read_text())
    arms=('P',) if read(r/'private/config.json').get('source6_diagnostic',False) else ('U','P')
    consumers=382*len(arms)
    db=sqlite3.connect('file:'+str(r/'private/judge_fixed8/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone(),'Scoring not terminal'
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==consumers
    tasks=read(r/'private/BENCHMARK146_QUEUE.json')['tasks'][:8];protect=read(r/'private/protection_ids.json')
    rows=db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key').fetchall()
    data={(x['method'],x['mode'],x['prefix'],x['query_id']):x['correct'] for x in rows};assert len(data)==consumers
    def get(a,mode,prefix,ids):return [data[a,mode,prefix,q] for q in ids]
    def ids(t,group):return [t['edit_id']] if group=='T0' else [q for e in t['events'] if e['task']==group for q in e['all_probe_query_ids']]
    result=dict(status='COMPLETE_WITH_MISSING' if any(x['correct'] is None for x in rows) else 'COMPLETE',
                consumers=consumers,consumer_status=dict(db.execute('SELECT p.status,count(*) FROM consumer c JOIN payload p ON c.payload_key=p.key GROUP BY p.status')),
                payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),ingest=read(r/'private/judge_fixed8/INGEST.json'),arms={})
    for a in arms:
        assert read(r/'private'/('FINAL_CONSUMED_'+a+'.json'))['status']=='PASS'
        summaries=[read(r/'private/fixed8'/a/'sources'/f'{i}.json') for i in range(1,9)]
        steps=[read(p) for p in (r/'private/fixed8'/a/'steps').rglob('*.json')];assert len(steps)==160
        arm=dict(formal_slots=len(steps),formal_accepted=sum(x['accepted'] for x in steps),qualified=sum(x['qualified'] for x in summaries),
                 sources=[{k:s[k] for k in ('owner','CE','qualified','protection_bytes')} for s in summaries],reload='PASS',prefixes={},old1to7={},per_source=[])
        for prefix in (0,4,8):
            cohort=tasks if prefix==0 else tasks[:prefix];mode='base' if prefix==0 else 'prefix'
            arm['prefixes'][str(prefix)]={g:rate(get(a,mode,prefix,[q for t in cohort for q in ids(t,g)])) for g in ('T0','T1G','T2G')}
            values=get(a,mode,prefix,protect);arm['prefixes'][str(prefix)]['protection']=dict(rate(values),known_wrong=sum(x==0 for x in values))
        for g in ('T0','T2G'):
            before=[v for t in tasks[:7] for v in get(a,'insertion',t['order'],ids(t,g))];after=get(a,'prefix',8,[q for t in tasks[:7] for q in ids(t,g)])
            arm['old1to7'][g]=paired(before,after)
        arm['old1to7']['T1G_final']=rate(get(a,'prefix',8,[q for t in tasks[:7] for q in ids(t,'T1G')]))
        for t in tasks:
            arm['per_source'].append(dict(order=t['order'],final={g:rate(get(a,'prefix',8,ids(t,g))) for g in ('T0','T1G','T2G')},insertion={g:rate(get(a,'insertion',t['order'],ids(t,g))) for g in ('T0','T2G')}))
        arm['evaluation_counts']=read(r/'private/fixed8'/a/'evaluation-budget.json')['counts']
        result['arms'][a]=arm
    result['common_insert_correct_old1to7']={}
    for g in ('T0','T2G'):
        common=[q for t in tasks[:7] for q in ids(t,g) if all(data[a,'insertion',t['order'],q]==1 for a in arms)]
        result['common_insert_correct_old1to7'][g]={a:rate(get(a,'prefix',8,common)) for a in arms}
    if len(arms)==1:result.pop('common_insert_correct_old1to7')
    resources=read(r/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in resources['gpu_sessions'])
    result['cost']=dict(GPU_seconds=sum(s['ended_epoch']-s['started_epoch'] for s in resources['gpu_sessions']),new_Judge_payload_attempts=resources['Judge_attempts'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if os.environ.get('SELF_CHECK'):
        assert rate([1,0,None])==dict(correct=1,total=3,missing=1,lower=1/3,upper=2/3)
        p=paired([1,1,0,None],[1,0,1,1]);assert p['correct_to_wrong']==1 and p['wrong_to_correct']==1 and p['retention_among_known_insert_correct']['lower']==.5
        assert rate([])['lower'] is None
        print('PASS: missing bounds, paired retention, zero denominator')
    else:main()
