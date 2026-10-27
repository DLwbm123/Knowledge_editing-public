"""Position-specific three-arm sensitivity with exact prior replay barriers."""
import json,statistics
from pathlib import Path
from protocol import brief_row,swapped_rows,gate_ready

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def validate(root):
    assert gate_ready(root)
    tasks=read(root/'DATA_ADMISSION.json')['tasks'];controls={(x['arm'],x['q']['query_id']):x for x in read(root/'CONTROL_OUTPUTS.json')}
    reference={(x['case'],x['context'],x['q']['query_id']):x for x in read(root/'REFERENCE_METRICS.json')};assert len(reference)==62
    for part,n in (('0',90),('1',96)):
        assert read(root/part/'EXIT.json')['code']==0
        for name in ('REAL_ADMISSION.json','SCORING_ADMISSION.json','FOUR_SOURCE_GRADIENT_ADMISSION.json'):assert read(root/part/name)['status']=='PASS'
        d=read(root/part/'MODEL_COMPLETE.json');assert d['status']=='PASS' and d['counts']['updates']==640 and d['counts']['generations']==n and d['counts']['backwards']<=2600
        assert read(root/part/'TRAINING_RELEASE.json')['time']>=max(read(root/p/'ORIGINAL_PARITY.json')['time'] for p in ('0','1'))
    result=[]
    for i,t in enumerate(tasks):
        folder=root/str(i%2)/t['key']
        for arm in ('COVER','ADAPT'):
            d=read(folder/arm/'TRAIN_COMPLETE.json');assert d['steps']==160 and d['parameters']==7168 and d['save_load_exact'] and d['paired_initial_exact'] and d['arm']==arm
            assert read(folder/arm/'REPLAY_PARITY.json')['outputs']==2*len(t['eval_rows'])
            r=read(folder/arm/'DISTRIBUTION_COMPLETE.json');assert r['queries']==2*len(t['eval_rows']) and r['state_exact_after_measurement'] and r['parameter_updates']==0
            assert read(folder/arm/'LIFECYCLE.json')['status']=='RETIRED'
        for suffix in ('','_SHORT'):
            left=[brief_row(q) if suffix else q for q in t['eval_rows']];right=swapped_rows(t,tasks,bool(suffix))
            for q,l,r in zip(t['eval_rows'],left,right):
                old=reference[(t['key'],suffix or 'ORIGINAL',q['query_id'])];assert old['q']==q
                joined={}
                for arm in ('COVER','ADAPT'):
                    output=read(folder/'outputs'/(arm+suffix)/(q['query_id']+'.json'));control=controls[(arm+suffix,q['query_id'])]
                    assert output['raw_token_ids']==control['tokens'] and output['raw_answer']==control['answer'] and output['EOS']==control['eos'] and output['at_cap']==control['at_cap']
                    d=read(folder/arm/('DISTRIBUTIONS'+suffix)/(q['query_id']+'.json'))
                    assert d['same_reference_prefix'] and d['parameter_updates_during_measurement']==0
                    for field,e in [('original_binding',l),('foreign_binding',r)]:assert all(d[field][k]==e[k] for k in ('question','reference','image_path'))
                    assert d['original_binding']['prompt_ids']==d['foreign_binding']['prompt_ids'] and d['original_binding']['image_sha256']!=d['foreign_binding']['image_sha256']
                    assert d['base']['all']==old['base'] and d['edited']['all']==old[arm.lower()]
                    if 'base' in joined:assert joined['base']==d['base']
                    joined['base']=d['base'];joined[arm.lower()]=d['edited']
                result.append(dict(case=t['key'],context=suffix or 'ORIGINAL',q=q,**joined))
    assert len(result)==62
    return result

def summarize(root):
    rr=validate(root);fields=('tv_mean','js_nats_mean','argmax_changed_fraction')
    def summary(rows,position):
        cases=sorted({x['case'] for x in rows});arms=('base','cover','adapt')
        means={c:{a:{m:statistics.mean(x[a][position][m] for x in rows if x['case']==c) for m in fields} for a in arms} for c in cases}
        equal={a:{m:statistics.mean(means[c][a][m] for c in cases) for m in fields} for a in arms}
        return dict(n=len(rows),cases=len(cases),equal_case_means=equal,adapt_minus_cover={m:equal['adapt'][m]-equal['cover'][m] for m in fields},cover_minus_base={m:equal['cover'][m]-equal['base'][m] for m in fields},adapt_minus_base={m:equal['adapt'][m]-equal['base'][m] for m in fields},per_case=means)
    result=dict(status='COMPLETE',replay_outputs=186,measurement_records=124,paired_queries=62,prior_all_position_metrics_exact=True,accuracy_test=False,grounding_confirmed=False,limited_improvement_gate=False,contexts={})
    for c in ('ORIGINAL','_SHORT'):
        selected=[x for x in rr if x['context']==c]
        result['contexts'][c]={pos:dict(all=summary(selected,pos),groups={g:summary([x for x in selected if x['q']['eval_group']==g],pos) for g in sorted({x['q']['eval_group'] for x in selected})},roles={g:summary([x for x in selected if x['q']['relation_role']==g],pos) for g in sorted({x['q']['relation_role'] for x in selected})}) for pos in ('first','continuation','all')}
    write(root/'RESULTS.json',result);return result

if __name__=='__main__':
    import sys
    op,path=sys.argv[1:];r=globals()[op](Path(path));print(json.dumps(dict(status='PASS',paired_queries=len(r)) if op=='validate' else r,indent=2))
