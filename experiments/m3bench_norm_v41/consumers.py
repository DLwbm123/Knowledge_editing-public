"""Ten-condition readout: four fresh arms and six immutable historical controls."""
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from protocol import brief_row, gate_ready, source_rows

NEW_ARMS=('BASE','BASE_SHORT','NORM','NORM_SHORT')
ARMS=(*NEW_ARMS,'ADAPT','ADAPT_SHORT','COVER','COVER_SHORT','JOINT','JOINT_SHORT')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def rows(root):
    config=read(root/'DATA_ADMISSION.json');result=[]
    for i,t in enumerate(config['tasks']):
        source_rows(t)
        for q in t['eval_rows']:
            for arm in NEW_ARMS:
                d=read(root/str(i%2)/t['key']/'outputs'/arm/(q['query_id']+'.json'))
                expected=brief_row(q) if arm.endswith('_SHORT') else q
                assert all(d['binding'][k]==expected[k] for k in ('question','reference','image_path'))
                result.append(dict(case=t['key'],arm=arm,q=q,answer=d['raw_answer'],tokens=d['raw_token_ids'],eos=d['EOS'],at_cap=d['at_cap']))
    controls=read(root/'CONTROL_OUTPUTS.json');assert len(controls)==186
    index={q['query_id']:q for t in config['tasks'] for q in t['eval_rows']}
    for r in controls:
        assert r['arm'] in ('ADAPT','ADAPT_SHORT','COVER','COVER_SHORT','JOINT','JOINT_SHORT') and r['q']==index[r['q']['query_id']]
    result+=controls
    assert len(result)==310 and len({(r['arm'],r['q']['query_id']) for r in result})==310
    return result

def validate(root):
    assert gate_ready(root)
    for part,gen in [('0',60),('1',64)]:
        assert read(root/part/'EXIT.json')['code']==0
        for receipt in ('REAL_ADMISSION.json','SCORING_ADMISSION.json','FOUR_SOURCE_GRADIENT_ADMISSION.json'):
            assert read(root/part/receipt)['status']=='PASS'
        g=read(root/part/'FOUR_SOURCE_GRADIENT_ADMISSION.json')
        assert g['objective']=='unit_source_directions_raw_aggregate_norm' and len(g['weights'])==4
        d=read(root/part/'MODEL_COMPLETE.json');assert d['status']=='PASS' and d['counts']['updates']==320 and d['counts']['generations']==gen and d['counts']['backwards']<=1300
        release=read(root/part/'TRAINING_RELEASE.json');assert release['base_original_and_short_exact']==62
        assert release['time']>=max(read(root/p/'ORIGINAL_PARITY.json')['time'] for p in ('0','1'))
    rr=rows(root)
    for i,t in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
        f=root/str(i%2)/t['key'];d=read(f/'NORM/TRAIN_COMPLETE.json')
        assert d['steps']==160 and d['parameters']==7168 and d['source_rows']==4 and d['heldout_training_rows']==0 and d['save_load_exact']
        assert read(f/'LIFECYCLE.json')['status']=='RETIRED'
    return dict(status='PASS',fresh_outputs=124,base_parity_outputs=62,new_method_outputs=62,reused_control_outputs=186,total_bindings=len(rr))

def blind(root):
    validate(root);rr=rows(root)
    keys=list(dict.fromkeys((r['q']['query_id'],r['answer']) for r in rr if r['arm'].startswith('NORM')));random.Random(410022).shuffle(keys)
    packet=[];mapping={}
    for i,key in enumerate(keys):
        code=f'Q{i+1:03d}';indices=[j for j,r in enumerate(rr) if r['arm'].startswith('NORM') and (r['q']['query_id'],r['answer'])==key];r=rr[indices[0]]
        mapping[code]=indices;packet.append(dict(code=code,question=r['q']['question'],reference=r['q']['reference'],answer=r['answer']))
    assert not (root/'BLIND_PACKET.json').exists()
    write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping)
    return dict(unique=len(packet),new_method_prior_verdict_reuse=0)

def summarize(root):
    validate(root);assert (root/'VERDICTS_FROZEN.json').exists();rr=rows(root)
    mapping=read(root/'BLIND_MAPPING.json');verdicts=read(root/'BLIND_VERDICTS.json');assert set(mapping)==set(verdicts)
    for code,ii in mapping.items():
        v=verdicts[code];assert v['label'] in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') and v['reason']
        for i in ii:rr[i]['label']=v['label']
    prior=read(root/'PRIOR_LABELS.json')
    for r in rr:
        if not r['arm'].startswith('NORM'):
            p=prior[r['arm']][r['q']['query_id']];assert p['tokens']==r['tokens'] and p['answer']==r['answer'];r['label']=p['label']
    arms={a:{r['q']['query_id']:r for r in rr if r['arm']==a} for a in ARMS};tasks=read(root/'DATA_ADMISSION.json')['tasks']
    qualified={t['key'] for t in tasks if arms['BASE'][t['train_rows'][0]['query_id']]['label']!='CORRECT' and arms['BASE'][t['train_rows'][1]['query_id']]['label']=='CORRECT'}
    result=dict(status='COMPLETE',qualified_cases=len(qualified),independent_confirmation=False,clinical_expert_review=False,groups={})
    for group in ('SOURCE','HELDOUT','LOCALITY_CLINICAL','LOCALITY_TECHNICAL'):
        ids=[q for q,r in arms['BASE'].items() if r['q']['eval_group']==group]
        entry=dict(n=len(ids),labels={a:dict(Counter(arms[a][q]['label'] for q in ids)) for a in ARMS},paired={})
        for a,b in [('ADAPT','NORM'),('ADAPT_SHORT','NORM_SHORT'),('JOINT','NORM'),('JOINT_SHORT','NORM_SHORT'),('COVER','NORM'),('COVER_SHORT','NORM_SHORT'),('NORM','NORM_SHORT')]:
            entry['paired'][a+'_TO_'+b]=dict(repairs=sum(arms[a][q]['label']!='CORRECT' and arms[b][q]['label']=='CORRECT' for q in ids),losses=sum(arms[a][q]['label']=='CORRECT' and arms[b][q]['label']!='CORRECT' for q in ids),support=sum(arms[a][q]['label']=='CORRECT' for q in ids))
        entry['base_C_damage']={a:dict(support=sum(arms['BASE'][q]['label']=='CORRECT' for q in ids),lost=sum(arms['BASE'][q]['label']=='CORRECT' and arms[a][q]['label']!='CORRECT' for q in ids)) for a in ('NORM','NORM_SHORT')}
        entry['roles']={role:{a:dict(n=sum(arms[a][q]['q']['relation_role']==role for q in ids),correct=sum(arms[a][q]['q']['relation_role']==role and arms[a][q]['label']=='CORRECT' for q in ids)) for a in ARMS} for role in sorted({arms['BASE'][q]['q']['relation_role'] for q in ids})}
        qi=[q for q in ids if arms['BASE'][q]['case'] in qualified];entry['qualified']={a:dict(n=len(qi),correct=sum(arms[a][q]['label']=='CORRECT' for q in qi)) for a in ARMS};result['groups'][group]=entry
    result['all_four_heldout_correct']={a:sum(all(r['label']=='CORRECT' for r in arms[a].values() if r['case']==t['key'] and r['q']['eval_group']=='HELDOUT') for t in tasks) for a in ARMS}
    result['heldout_both_contexts_correct']={a:sum(arms[a][q]['label']=='CORRECT' and arms[a+'_SHORT'][q]['label']=='CORRECT' for q,r in arms[a].items() if r['q']['eval_group']=='HELDOUT') for a in ('JOINT','COVER','ADAPT','NORM')}
    result['stability']={a:dict(empty=sum(not r['answer'].strip() for r in arms[a].values()),cap=sum(r['at_cap'] for r in arms[a].values()),eos=sum(r['eos'] for r in arms[a].values()),mean_tokens=statistics.mean(len(r['tokens']) for r in arms[a].values())) for a in ARMS}
    checks=[]
    for suffix in ('','_SHORT'):
        for control in ('JOINT','COVER','ADAPT'):
            key=control+suffix+'_TO_NORM'+suffix
            for group in ('SOURCE','HELDOUT','LOCALITY_CLINICAL'):checks.append(result['groups'][group]['paired'][key]['losses']==0)
    result['limited_improvement_gate']=all(checks) and result['heldout_both_contexts_correct']['NORM']>result['heldout_both_contexts_correct']['ADAPT']
    write(root/'RESULTS.json',result);return result

if __name__=='__main__':
    import sys
    op,path=sys.argv[1:];print(json.dumps(globals()[op](Path(path)),indent=2))
