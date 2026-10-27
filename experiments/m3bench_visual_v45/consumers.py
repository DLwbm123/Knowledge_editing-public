"""Fixed image-substitution readout. No accuracy labels for donor images."""
import json
import statistics
from pathlib import Path
from protocol import brief_row, swapped_rows, gate_ready

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
ARMS=('BASE','BASE_SHORT','ADAPT','ADAPT_SHORT','SWAP_BASE','SWAP_BASE_SHORT','SWAP_ADAPT','SWAP_ADAPT_SHORT')

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
    assert len(out)==248 and len({(x['arm'],x['q']['query_id']) for x in out})==248
    return out

def validate(root):
    assert gate_ready(root)
    for part,n in (('0',120),('1',128)):
        assert read(root/part/'EXIT.json')['code']==0
        for f in ('REAL_ADMISSION.json','SCORING_ADMISSION.json','FOUR_SOURCE_GRADIENT_ADMISSION.json'):
            assert read(root/part/f)['status']=='PASS'
        d=read(root/part/'MODEL_COMPLETE.json');assert d['status']=='PASS' and d['counts']['updates']==320 and d['counts']['backwards']<=1300 and d['counts']['generations']==n
        release=read(root/part/'TRAINING_RELEASE.json');assert release['base_original_and_short_exact']==62
        assert release['time']>=max(read(root/p/'ORIGINAL_PARITY.json')['time'] for p in ('0','1'))
    tasks=read(root/'DATA_ADMISSION.json')['tasks']
    for i,t in enumerate(tasks):
        f=root/str(i%2)/t['key'];d=read(f/'ADAPT/TRAIN_COMPLETE.json')
        assert d['steps']==160 and d['parameters']==7168 and d['source_rows']==4 and d['heldout_training_rows']==0 and d['save_load_exact']
        assert read(f/'ADAPT/REPLAY_PARITY.json')['outputs']==2*len(t['eval_rows'])
        assert read(f/'LIFECYCLE.json')['status']=='RETIRED'
    rr=rows(root);controls={(x['arm'],x['q']['query_id']):x for x in read(root/'CONTROL_OUTPUTS.json')}
    for x in rr:
        if x['arm'] in ('ADAPT','ADAPT_SHORT'):
            old=controls[(x['arm'],x['q']['query_id'])]
            assert x['tokens']==old['tokens'] and x['answer']==old['answer'] and x['eos']==old['eos'] and x['cap']==old['at_cap']
    return dict(status='PASS',outputs=248,base_replay=62,adapt_replay=62,swapped_outputs=124,no_swapped_correctness_grading=True)

def summarize(root):
    admission=validate(root);rr=rows(root)
    arms={a:{x['q']['query_id']:x for x in rr if x['arm']==a} for a in ARMS}
    def stats(ids,suffix):
        unchanged={a:{q:arms[a+suffix][q]['tokens']==arms['SWAP_'+a+suffix][q]['tokens'] for q in ids} for a in ('BASE','ADAPT')}
        return dict(n=len(ids),unchanged_tokens={a:sum(v.values()) for a,v in unchanged.items()},changed_tokens={a:len(ids)-sum(v.values()) for a,v in unchanged.items()},both_unchanged=sum(unchanged['BASE'][q] and unchanged['ADAPT'][q] for q in ids),base_changed_adapt_unchanged=sum(not unchanged['BASE'][q] and unchanged['ADAPT'][q] for q in ids),base_unchanged_adapt_changed=sum(unchanged['BASE'][q] and not unchanged['ADAPT'][q] for q in ids),both_changed=sum(not unchanged['BASE'][q] and not unchanged['ADAPT'][q] for q in ids))
    result=dict(status='COMPLETE',admission=admission,independent_confirmation=False,grounding_confirmed=False,accuracy_test=False,limited_improvement_gate=False,contexts={})
    for suffix in ('','_SHORT'):
        ref=arms['BASE'];groups={g:[q for q,x in ref.items() if x['q']['eval_group']==g] for g in sorted({x['q']['eval_group'] for x in ref.values()})}
        roles={g:[q for q,x in ref.items() if x['q']['relation_role']==g] for g in sorted({x['q']['relation_role'] for x in ref.values()})}
        cases={g:[q for q,x in ref.items() if x['case']==g] for g in sorted({x['case'] for x in ref.values()})}
        result['contexts'][suffix or 'ORIGINAL']=dict(all=stats(list(ref),suffix),groups={g:stats(ids,suffix) for g,ids in groups.items()},roles={g:stats(ids,suffix) for g,ids in roles.items()},cases={g:stats(ids,suffix) for g,ids in cases.items()})
    result['stability']={a:dict(empty=sum(not x['answer'].strip() for x in v.values()),eos=sum(x['eos'] for x in v.values()),cap=sum(x['cap'] for x in v.values()),mean_tokens=statistics.mean(len(x['tokens']) for x in v.values())) for a,v in arms.items()}
    write(root/'RESULTS.json',result);return result

if __name__=='__main__':
    import sys
    op,path=sys.argv[1:];print(json.dumps(globals()[op](Path(path)),indent=2))
