"""Classify existing errors by text/token identity only; no new semantic labels."""
import os
import re
import json
from pathlib import Path
from collections import Counter,defaultdict


def normalized(text):return re.sub(r'\W+',' ',text.casefold()).strip()


def main():
    run=Path(os.environ['RUN_ROOT']);parent=Path(os.environ['DAMAGE_PARENT'])
    read=lambda p:json.loads(p.read_text())
    assert read(parent/'public/REVIEW_AUDIT.json')['status']=='PASS'
    scores={v['path']:v['correct'] for v in read(parent/'private/DAMAGE_SEMANTIC.json')}
    hb=read(parent/'private/HELD_BASE.json');scores.update({v['path']:v['correct'] for v in hb.values()})
    tasks=read(parent/'private/BENCHMARK146_QUEUE.json')['tasks'][:8];records=[];summaries={};sets={}
    assert normalized('YES!')==normalized(' yes ')
    for arm in ('RAW','ADAM'):
        summary=Counter();per=defaultdict(list)
        for path in sorted((parent/'private/audit_outputs'/arm).glob('*/*.json')):
            d=read(path)
            if d['role']!='HELDOUT' or scores[d['baseline_path']]!=1 or scores[str(path)]!=0:continue
            base=read(Path(d['baseline_path']));t=tasks[d['expert_order']-1]
            prefix=0
            for x,y in zip(d['R0']['raw_token_ids'],base['R0']['raw_token_ids']):
                if x!=y:break
                prefix+=1
            target=normalized(t['native']['reference']);answer=normalized(d['R0']['raw_answer'])
            flags=dict(target_text_equal=answer==target,target_phrase_present=bool(target) and (' '+target+' ') in (' '+answer+' '),diverges_first_token=prefix==0)
            query=d['binding']['input']['query_id'];per[query].append(d['expert_order']);summary.update({k:int(v) for k,v in flags.items()});summary['damage']+=1
            records.append(dict(arm=arm,owner=d['expert_order'],query_id=query,first_divergence_token=prefix,**flags))
        summaries[arm]=dict(summary,distinct_questions=len(per),affected_multiple_owners=sum(len(v)>1 for v in per.values()),owner_multiplicity_histogram=dict(Counter(map(len,per.values()))))
        sets[arm]={(x['owner'],x['query_id']) for x in records if x['arm']==arm}
    assert summaries['RAW']['damage']==18 and summaries['ADAM']['damage']==26 and sets['RAW']<=sets['ADAM']
    result=dict(status='PASS',summary=summaries,RAW_damage_subset_ADAM=True,model_calls=0,new_Judge=0,
        limitation='Normalized text and token positions only; not clinical error labels or proof of causal target intrusion',training_geometry_uses_these_errors=False)
    for path,value in [(run/'private/ATTRIBUTION_ROWS.json',records),(run/'public/ATTRIBUTION.json',result)]:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
