"""Private blinded grading and anonymous paired relation summaries."""
import json,random,sys
from pathlib import Path
from collections import Counter
LABELS={'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def rows(root):
    out=[]
    for task in read(root/'DATA_ADMISSION.json')['tasks']:
        for q in task['eval_rows']:
            for arm in ('BASE','EXPR_FORWARD','EXPR_REVERSE'):
                src=root/'1'/task['key']/'outputs'/arm/(q['query_id']+'.json')
                d=read(src);assert d['binding']['question']==q['question'] and d['binding']['reference']==q['reference']
                assert isinstance(d['raw_token_ids'],list) and isinstance(d['raw_answer'],str)
                out.append(dict(case=task['key'],arm=arm,q=q,answer=d['raw_answer'],tokens=d['raw_token_ids'],EOS=d['EOS'],at_cap=d['at_cap']))
    assert len(out)==180
    return out

def validate(root):
    assert read(root/'1/EXIT.json')['code']==0
    rr=rows(root)
    assert read(root/'1/MODEL_COMPLETE.json')['status']=='PASS'
    for t in read(root/'DATA_ADMISSION.json')['tasks']:
        assert read(root/'1'/t['key']/'LIFECYCLE.json')['status']=='RETIRED'
        assert read(root/'1'/t['key']/'REPLAY_ADMISSION.json')['consumers']==20
        for arm in ('EXPR_FORWARD','EXPR_REVERSE'):
            d=read(root/'1'/t['key']/arm/'TRAIN_COMPLETE.json');assert d['steps']==160 and d['parameters']==7168 and d['save_load_exact'] and d['reconstruction160']==('PASS' if arm=='EXPR_FORWARD' else 'NOT_APPLICABLE_NEW_ORDER')
    return dict(status='PASS',consumers=len(rr),method_outputs=120,base_outputs=60)

def blind(root):
    validate(root);rr=rows(root);unique={}
    for r in rr:
        key=(r['q']['query_id'],r['answer'])
        unique.setdefault(key,r)
    keys=list(unique);random.Random(310021).shuffle(keys);packet=[];mapping={};reuse={}
    prior={}
    for i,k in enumerate(keys):
        code=f'Q{i+1:03d}';r=unique[k];mapping[code]=[n for n,x in enumerate(rr) if (x['q']['query_id'],x['answer'])==k]
        packet.append(dict(code=code,question=r['q']['question'],reference=r['q']['reference'],answer=r['answer']))
        if any(rr[n]['arm']=='BASE' for n in mapping[code]) and r['q']['query_id'] in prior:reuse[code]=prior[r['q']['query_id']]
    assert not (root/'BLIND_PACKET.json').exists()
    write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping);write(root/'BLIND_REUSE.json',reuse)
    return dict(unique=len(packet),exact_base_reuse=len(reuse),new=len(packet)-len(reuse))

def summarize(root):
    validate(root);rr=rows(root);mp=read(root/'BLIND_MAPPING.json');vv={**read(root/'BLIND_REUSE.json'),**read(root/'BLIND_VERDICTS.json')}
    assert set(vv)==set(mp) and all(v.get('label') in LABELS and v.get('reason') for v in vv.values())
    for code,inds in mp.items():
        for n in inds:rr[n]['label']=vv[code]['label']
    eligible={p['case'] for p in read(root/'BASE_OPPORTUNITY.json')['pairs'] if p['opportunity']}
    result=dict(status='COMPLETE_FIXED_REFERENCE_REVIEW',eligible_pairs=len(eligible),parameters_per_edit=7168,fp32_parameter_bytes=28672,independent_confirmation=False,clinical_adjudication=False,arms={},paired={})
    for arm in ('BASE','EXPR_FORWARD','EXPR_REVERSE'):
        ar=[r for r in rr if r['arm']==arm];entry={}
        for role in ('UPDATE','PRESERVE'):
            subset=[r for r in ar if r['q']['relation_role']==role];entry[role]=dict(n=len(subset),labels=dict(Counter(r['label'] for r in subset)))
        entry['stability']=dict(cap=sum(r['at_cap'] for r in ar),empty_decoded=sum(not r['answer'].strip() for r in ar),eos=sum(r['EOS'] for r in ar),tokens=sum(len(r['tokens']) for r in ar))
        entry['strata']={}
        base={r['q']['query_id']:r for r in rr if r['arm']=='BASE'}
        for name,subset in [(g+'_'+s,[r for r in ar if r['q']['eval_group']==g and (s=='all' or r['case'] in eligible)]) for g in ('SOURCE','OLD_PARAPHRASE','V29_REPLAY') for s in ('all','source_qualified')]:
            u=[r for r in subset if r['q']['relation_role']=='UPDATE'];p=[r for r in subset if r['q']['relation_role']=='PRESERVE']
            wrong=[r for r in u if base[r['q']['query_id']]['label']!='CORRECT'];correct=[r for r in p if base[r['q']['query_id']]['label']=='CORRECT']
            pairs={}
            for r in subset:pairs.setdefault((r['case'],r['q']['pair_id']),[]).append(r)
            assert all(len(v)==2 for v in pairs.values())
            entry['strata'][name]=dict(update_correct=sum(r['label']=='CORRECT' for r in u),update_n=len(u),preserve_correct=sum(r['label']=='CORRECT' for r in p),preserve_n=len(p),update_base_nonC=len(wrong),update_repairs=sum(r['label']=='CORRECT' for r in wrong),preserve_base_C=len(correct),preserve_loss_nonC=sum(r['label']!='CORRECT' for r in correct),preserve_loss_I=sum(r['label']=='INCORRECT' for r in correct),both_correct=sum(all(r['label']=='CORRECT' for r in v) for v in pairs.values()),pair_n=len(pairs))
            st=entry['strata'][name]
            st['labels']=dict(Counter(r['label'] for r in subset))
            st['UPDATE_labels']=dict(Counter(r['label'] for r in u))
            st['PRESERVE_labels']=dict(Counter(r['label'] for r in p))
            st['stability']=dict(cap=sum(r['at_cap'] for r in subset),empty_decoded=sum(not r['answer'].strip() for r in subset),eos=sum(r['EOS'] for r in subset),tokens=sum(len(r['tokens']) for r in subset))
        # Exact normalized cross-relation reference copy is diagnostic, not a correctness rule.
        copy=0
        for r in ar:
            t=next(t for t in read(root/'DATA_ADMISSION.json')['tasks'] if t['key']==r['case'])
            other=next(q['reference'] for q in t['train_rows'] if q['relation_role']!=r['q']['relation_role'])
            norm=lambda s:' '.join(s.casefold().split()).strip(' .')
            copy+=norm(r['answer'])==norm(other)
        entry['cross_relation_exact_copy']=copy;result['arms'][arm]=entry
    for group,role,stratum in [(g,r,s) for g in ('SOURCE','OLD_PARAPHRASE','V29_REPLAY') for r in ('UPDATE','PRESERVE') for s in ('all','source_qualified')]:
        a={r['q']['query_id']:r for r in rr if r['arm']=='EXPR_FORWARD' and r['q']['relation_role']==role and r['q']['eval_group']==group and (stratum=='all' or r['case'] in eligible)};b={r['q']['query_id']:r for r in rr if r['arm']=='EXPR_REVERSE' and r['q']['relation_role']==role and r['q']['eval_group']==group and (stratum=='all' or r['case'] in eligible)}
        result['paired'][group+'_'+role+'_'+stratum]=dict(repairs=sum(a[k]['label']!='CORRECT' and b[k]['label']=='CORRECT' for k in a),losses=sum(a[k]['label']=='CORRECT' and b[k]['label']!='CORRECT' for k in a),transitions=dict(Counter(a[k]['label']+'->'+b[k]['label'] for k in a)))
    write(root/'RESULTS.json',result);return result
if __name__=='__main__':
    op,root=sys.argv[1:];print(json.dumps(globals()[op](Path(root))))
