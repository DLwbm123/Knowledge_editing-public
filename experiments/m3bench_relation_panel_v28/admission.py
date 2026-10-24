"""Frozen Base opportunity gate; never infer correctness from binary false."""
import json
from pathlib import Path
LABELS={'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'}

def evaluate(tasks,verdicts):
    expected={r['query_id'] for t in tasks for r in t['train_rows']}
    assert len(tasks)==6 and len(expected)==12 and set(verdicts)==expected
    assert all(v.get('label') in LABELS and v.get('reason') for v in verdicts.values())
    pairs=[]
    for t in tasks:
        roles={r['relation_role']:r['query_id'] for r in t['train_rows']}
        assert set(roles)=={'UPDATE','PRESERVE'}
        u,p=(verdicts[roles[k]]['label'] for k in ('UPDATE','PRESERVE'))
        pairs.append(dict(case=t['key'],update=u,preserve=p,opportunity=u!='CORRECT' and p=='CORRECT'))
    passed=any(p['opportunity'] for p in pairs)
    return dict(status='PASS' if passed else 'STOP_NO_REPAIR_PRESERVE_OPPORTUNITY',cases=6,source_rows=12,
                eligible_pairs=sum(p['opportunity'] for p in pairs),pairs=pairs,
                all_cases_retained=True,clinical_adjudication=False,independent_confirmation=False)

def selfcheck():
    ts=[dict(key=str(i),train_rows=[dict(query_id=f'{i}u',relation_role='UPDATE'),dict(query_id=f'{i}p',relation_role='PRESERVE')]) for i in range(6)]
    vv={r['query_id']:dict(label='INCORRECT',reason='fixture') for t in ts for r in t['train_rows']}
    assert evaluate(ts,vv)['status']=='STOP_NO_REPAIR_PRESERVE_OPPORTUNITY'
    vv['0p']['label']='CORRECT';assert evaluate(ts,vv)['eligible_pairs']==1
    for bad in ({},dict(vv,**{'0p':dict(label=False,reason='binary')})):
        try:evaluate(ts,bad)
        except AssertionError:pass
        else:raise AssertionError('invalid labels passed')
    return dict(status='PASS',missing_and_binary_rejected=True,zero_pair_rejected=True,one_pair_passes=True)

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);check=selfcheck()
    result=evaluate(json.loads((root/'DATA_ADMISSION.json').read_text())['tasks'],json.loads((root/'BASE_VERDICTS.json').read_text()))
    result['gate_selfcheck']=check
    (root/'BASE_OPPORTUNITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
