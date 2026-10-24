"""Fixed-prefix probability-distribution diagnostic."""
import json
import statistics
from pathlib import Path
from protocol import brief_row, swapped_rows, gate_ready

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
ARMS=('BASE','BASE_SHORT','COVER','COVER_SHORT')

def rows(root):
    tasks=read(root/'DATA_ADMISSION.json')['tasks'];out=[]
    for i,t in enumerate(tasks):
        for arm in ARMS:
            short=arm.endswith('_SHORT')
            expected=swapped_rows(t,tasks,short) if arm.startswith('SWAP_') else [brief_row(q) if short else q for q in t['eval_rows']]
            for q,e in zip(t['eval_rows'],expected):
                d=read(root/str(i%2)/t['key']/'outputs'/arm/(q['query_id']+'.json'))
                assert all(d['binding'][k]==e[k] for k in ('question','reference','image_path'))
                out.append(dict(case=t['key'],arm=arm,q=q,tokens=d['raw_token_ids'],answer=d['raw_answer'],eos=d['EOS'],cap=d['at_cap']))
    assert len(out)==124 and len({(x['arm'],x['q']['query_id']) for x in out})==124
    return out

def validate(root):
    assert gate_ready(root)
    for part,n in (('0',60),('1',64)):
        assert read(root/part/'EXIT.json')['code']==0
        for f in ('REAL_ADMISSION.json','SCORING_ADMISSION.json','FOUR_SOURCE_GRADIENT_ADMISSION.json'):
            assert read(root/part/f)['status']=='PASS'
        d=read(root/part/'MODEL_COMPLETE.json');assert d['status']=='PASS' and d['counts']['updates']==320 and d['counts']['backwards']<=1300 and d['counts']['generations']==n
        release=read(root/part/'TRAINING_RELEASE.json');assert release['base_original_and_short_exact']==62
        assert release['time']>=max(read(root/p/'ORIGINAL_PARITY.json')['time'] for p in ('0','1'))
    tasks=read(root/'DATA_ADMISSION.json')['tasks']
    for i,t in enumerate(tasks):
        f=root/str(i%2)/t['key'];d=read(f/'COVER/TRAIN_COMPLETE.json')
        assert d['steps']==160 and d['parameters']==7168 and d['source_rows']==4 and d['heldout_training_rows']==0 and d['save_load_exact']
        assert read(f/'COVER/REPLAY_PARITY.json')['outputs']==2*len(t['eval_rows'])
        assert read(f/'LIFECYCLE.json')['status']=='RETIRED'
    rr=rows(root);controls={(x['arm'],x['q']['query_id']):x for x in read(root/'CONTROL_OUTPUTS.json')}
    for x in rr:
        if x['arm'] in ('COVER','COVER_SHORT'):
            old=controls[(x['arm'],x['q']['query_id'])]
            assert x['tokens']==old['tokens'] and x['answer']==old['answer'] and x['eos']==old['eos'] and x['cap']==old['at_cap']
    records=metrics(root)
    assert len(records)==62
    return dict(status='PASS',outputs=124,base_replay=62,cover_replay=62,distribution_queries=62,no_swapped_correctness_grading=True)

def metrics(root):
    tasks=read(root/'DATA_ADMISSION.json')['tasks'];out=[]
    for i,t in enumerate(tasks):
        folder=root/str(i%2)/t['key']/'COVER'
        receipt=read(folder/'DISTRIBUTION_COMPLETE.json')
        assert receipt['status']=='PASS' and receipt['queries']==2*len(t['eval_rows']) and receipt['parameter_updates']==0 and receipt['state_exact_after_measurement']
        for suffix in ('','_SHORT'):
            original=[brief_row(q) if suffix else q for q in t['eval_rows']];foreign=swapped_rows(t,tasks,bool(suffix))
            for q,left,right in zip(t['eval_rows'],original,foreign):
                d=read(folder/('DISTRIBUTIONS'+suffix)/(q['query_id']+'.json'))
                assert d['same_reference_prefix'] and d['parameter_updates_during_measurement']==0
                for name,e in [('original_binding',left),('foreign_binding',right)]:
                    assert all(d[name][k]==e[k] for k in ('question','reference','image_path'))
                assert d['original_binding']['image_sha256']!=d['foreign_binding']['image_sha256']
                assert d['original_binding']['prompt_ids']==d['foreign_binding']['prompt_ids']
                assert d['base']['tokens']==d['cover']['tokens']>0
                out.append(dict(case=t['key'],context=suffix or 'ORIGINAL',q=q,base=d['base'],cover=d['cover']))
    reference={(x['case'],x['context'],x['q']['query_id']):x for x in read(root/'ADAPT_REFERENCE_METRICS.json')}
    assert len(reference)==62
    for x in out:
        old=reference[(x['case'],x['context'],x['q']['query_id'])]
        assert x['q']==old['q'] and x['base']==old['base']
        x['adapt_reference']=old['adapt']
    return out

def summarize(root):
    import math
    admission=validate(root);rr=metrics(root)
    fields=('tv_mean','js_nats_mean','argmax_changed_fraction')
    def summary(rows):
        cases=sorted({x['case'] for x in rows})
        means={a:{m:statistics.mean(x[a][m] for x in rows) for m in fields} for a in ('base','cover','adapt_reference')}
        case_means={c:{a:{m:statistics.mean(x[a][m] for x in rows if x['case']==c) for m in fields} for a in ('base','cover','adapt_reference')} for c in cases}
        equal_case={a:{m:statistics.mean(case_means[c][a][m] for c in cases) for m in fields} for a in ('base','cover','adapt_reference')}
        assert all(math.isfinite(v) for a in means.values() for v in a.values())
        return dict(n=len(rows),cases=len(cases),query_means=means,equal_case_means=equal_case,equal_case_cover_minus_base={m:equal_case['cover'][m]-equal_case['base'][m] for m in fields},equal_case_adapt_minus_cover={m:equal_case['adapt_reference'][m]-equal_case['cover'][m] for m in fields},queries_cover_tv_lower=sum(x['cover']['tv_mean']<x['base']['tv_mean'] for x in rows),queries_cover_tv_higher=sum(x['cover']['tv_mean']>x['base']['tv_mean'] for x in rows),per_case=case_means)
    result=dict(status='COMPLETE',admission=admission,accuracy_test=False,grounding_confirmed=False,limited_improvement_gate=False,contexts={})
    for context in ('ORIGINAL','_SHORT'):
        selected=[x for x in rr if x['context']==context]
        groups=sorted({x['q']['eval_group'] for x in selected});roles=sorted({x['q']['relation_role'] for x in selected})
        result['contexts'][context]=dict(all=summary(selected),groups={g:summary([x for x in selected if x['q']['eval_group']==g]) for g in groups},roles={g:summary([x for x in selected if x['q']['relation_role']==g]) for g in roles})
    write(root/'RESULTS.json',result);return result

if __name__=='__main__':
    import sys
    op,path=sys.argv[1:];print(json.dumps(globals()[op](Path(path)),indent=2))
