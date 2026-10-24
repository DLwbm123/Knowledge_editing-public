"""Exploratory influence/style readout of frozen V48; no model calls or rescoring."""
import json,os,re,signal,statistics,time
from pathlib import Path

def read(p):return json.loads(p.read_text())
def article(text):
    words=re.findall(r'[A-Za-z]+',text.lower())
    return bool(words) and words[0] in ('a','an','the')

def run(base):
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s analysis limit')));signal.alarm(60)
    root=base/'.private/m3bench_firstpos_v48_20261022/completed';tasks=read(root/'DATA_ADMISSION.json')['tasks']
    assert len(tasks)==4
    rows=[];style={}
    for suffix in ('','_SHORT'):
        style[suffix or 'ORIGINAL']={a:dict(n=0,leading_article=0) for a in ('BASE','COVER','ADAPT')}
        for i,t in enumerate(tasks):
            for q in t['eval_rows']:
                a=read(root/str(i%2)/t['key']/'ADAPT'/('DISTRIBUTIONS'+suffix)/(q['query_id']+'.json'))
                c=read(root/str(i%2)/t['key']/'COVER'/('DISTRIBUTIONS'+suffix)/(q['query_id']+'.json'))
                assert a['base']==c['base']
                rows.append(dict(case=t['key'],context=suffix or 'ORIGINAL',reference_leading_article=article(q['reference']),base=a['base'],cover=c['edited'],adapt=a['edited']))
                for arm in ('BASE','COVER','ADAPT'):
                    out=read(root/str(i%2)/t['key']/'outputs'/(arm+suffix)/(q['query_id']+'.json'))
                    style[suffix or 'ORIGINAL'][arm]['n']+=1
                    style[suffix or 'ORIGINAL'][arm]['leading_article']+=article(out['raw_answer'])
    def influence(selected,pos,left,right):
        cases=sorted({x['case'] for x in selected})
        delta={c:statistics.mean(x[left][pos]['tv_mean']-x[right][pos]['tv_mean'] for x in selected if x['case']==c) for c in cases}
        return dict(cases=len(cases),per_case_delta=delta,mean_delta=statistics.mean(delta.values()),positive_cases=sum(v>0 for v in delta.values()),negative_cases=sum(v<0 for v in delta.values()),leave_one_case_out={c:statistics.mean(v for k,v in delta.items() if k!=c) for c in cases} if len(cases)>1 else {})
    contrasts=(('adapt','base'),('cover','base'),('adapt','cover'))
    result=dict(status='COMPLETE',posthoc=True,new_model_calls=0,new_grading=0,independent_confirmation=False,contexts={},output_leading_article=style)
    for context in ('ORIGINAL','_SHORT'):
        selected=[x for x in rows if x['context']==context]
        result['contexts'][context]={pos:{a+'_minus_'+b:influence(selected,pos,a,b) for a,b in contrasts} for pos in ('first','continuation','all')}
        result['contexts'][context]['reference_prefix_strata']={name:dict(n=len(sub),contrasts={a+'_minus_'+b:influence(sub,'first',a,b) for a,b in contrasts}) for value,name in ((True,'leading_article'),(False,'other')) if (sub:=[x for x in selected if x['reference_leading_article']==value])}
    result['elapsed_seconds']=time.time()-start
    out=base/'reports/m3bench_influence_v49_20261022';payload=json.dumps(result,indent=2)+'\n'
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(payload.encode())<25000000
    (out/'RESULTS.json').write_text(payload);signal.alarm(0)
    return result

if __name__=='__main__':
    r=run(Path(os.environ['ANALYSIS_ROOT']))
    for c,v in r['contexts'].items():print(c,{p:{k:dict(mean=d['mean_delta'],positive=d['positive_cases'],negative=d['negative_cases'],loo=d['leave_one_case_out']) for k,d in v[p].items()} for p in ('first','continuation')})
    print(r['output_leading_article'])
