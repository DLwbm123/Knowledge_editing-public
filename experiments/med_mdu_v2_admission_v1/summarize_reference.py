"""Verify both reference shards and compute the reusable pilot anchors."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata


def norm(text):
    text=unicodedata.normalize('NFKC',text).casefold().strip()
    text=re.sub(r'^(?:answer|答案)\s*[:：]\s*','',text)
    text=re.sub(r'\s*([,;，；])\s*',r'\1',text)
    return ' '.join(text.strip(' \t\n.!?。！？').split())


def median(values):
    data=list(values)
    assert data
    return statistics.median(data)


def main(run,manifest_path,inventory_path):
    entries={r['owner']:r for r in json.loads(manifest_path.read_text())['entries']}
    assert len(entries)==30
    sources=json.loads(inventory_path.read_text())['sources'].values()
    targets={r['edit_id']:r['source_rows'][0]['reference'] for r in sources}
    assert len(targets)==150
    all_rows=[]
    done=[]
    for lane in (0,1):
        note=json.loads((run/f'done{lane}.json').read_text())
        assert note['events']==120 and note['training']==note['judge']==0
        assert note['counts']['generation']==90 and note['counts']['routes']==45
        records=[json.loads(x) for x in (run/f'metrics{lane}.jsonl').open()]
        assert len(records)==120 and sorted(v['n'] for v in records)==list(range(120))
        all_rows+=records
        done.append(note)
    by_role=defaultdict(list)
    fit=defaultdict(list)
    for row in all_rows:
        by_role[row['role']].append(row)
        if row['role']=='fit':
            fit[row['owner']].append(row['reference_prefix']['MDU']['projection'])
    assert dict(Counter(r['role'] for r in all_rows))=={
        'native':30,'fit':120,'holdout_modal':30,'holdout_context':30,'holdout_independent':30}
    assert fit.keys()==entries.keys() and all(len(x)==4 for x in fit.values())
    fit_lock={o:median(v) for o,v in fit.items()}
    roles=('holdout_modal','holdout_context','holdout_independent')
    group={}
    scoring=[]
    for name in roles:
        records=by_role[name]
        result={'events':len(records),'activated':sum(r['route']['activated'] for r in records),
          'owner_route':sum(r['route']['logical_edit_id']==entries[r['owner']]['source']['edit_id']
                            and r['route']['activated'] for r in records)}
        for arm in ('MED','MDU'):
            result[arm]={'owner_readout_ratio':median(r['reference_prefix'][arm]['projection']/fit_lock[r['owner']]
                                                      for r in records),
              'owner_target_exact':sum(norm(r['bank_generation'][arm]['answer'])==
                  norm(entries[r['owner']]['source']['source_rows'][0]['reference']) for r in records),
              'active_target_exact':sum(bool(r['route']['activated']) and
                  norm(r['bank_generation'][arm]['answer'])==norm(targets[r['route']['logical_edit_id']])
                  for r in records),
              'cap':sum(r['bank_generation'][arm]['at_cap'] for r in records)}
            if name=='holdout_independent':
                for prefix in ('reference_prefix','base_prefix'):
                    result[arm][prefix+'_bank_kl']=median(r[prefix].get('bank_'+arm,r[prefix][arm])['kl']
                                                           for r in records)
        for r in records:
            e=entries[r['owner']]
            if name=='holdout_independent':
                q=e['negative']['holdout'][0]['question']
            else:
                candidates=[v['question'] for v in e['views'] if v['role']==name]
                assert len(candidates)==1
                q=candidates[0]
            scoring.append({'owner':r['owner'],'role':name,'question':q,
                   'reference':r['reference'],'route':r['route'],'answers':r['bank_generation']})
        group[name]=result
    margins={}
    for arm in ('MED','MDU'):
        pos=by_role['holdout_modal']+by_role['holdout_context']
        r=median(x['reference_prefix'][arm]['projection']/fit_lock[x['owner']] for x in pos)
        margins[arm]=r-group['holdout_independent'][arm]['owner_readout_ratio']
    negative_details=[]
    for r in by_role['holdout_independent']:
        e=entries[r['owner']]
        raw=e['negative']['holdout'][0]['base_continuation']['model_answer_raw']
        target=e['source']['source_rows'][0]['reference']
        filler=set('a an the to of is are was were be this that and or in on it its for'.split())
        tok=lambda s:set(re.findall(r'\w+',norm(s))) - filler
        a,b=tok(raw),tok(target)
        overlap=len(a & b)/len(a | b) if a | b else 0
        negative_details.append({
            'owner_readout':r['reference_prefix']['MDU']['projection']/fit_lock[r['owner']],
            'reference_bank_delta':r['reference_prefix'].get('bank_MDU',r['reference_prefix']['MDU'])['kl']
                -r['reference_prefix'].get('bank_MED',r['reference_prefix']['MED'])['kl'],
            'base_bank_delta':r['base_prefix'].get('bank_MDU',r['base_prefix']['MDU'])['kl']
                -r['base_prefix'].get('bank_MED',r['base_prefix']['MED'])['kl'],
            'lexical_jaccard':overlap,
        })
    for name,val in [('fit_denominators.private.json',fit_lock),('unscored_core_panel.private.json',scoring),
                     ('locality_regression_input.private.json',negative_details)]:
        with (run/name).open('x') as stream:
            json.dump(val,stream,ensure_ascii=False,allow_nan=False)
    result={'status':'REFERENCE_GPU_COMPLETE_CORE_PENDING',
        'events':len(all_rows),'teacher_forwards':sum(x['counts']['teacher'] for x in done),
        'generations':sum(x['counts']['generation'] for x in done),'groups':group,
        'owner_readout_gap':margins,'judge':0,'training':0,'independent_human_review':0}
    with (run/'summary.json').open('x') as out:json.dump(result,out,ensure_ascii=False,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
