"""Outcome-independent compact source-bank scope and image-group calibration split."""
import json,re,collections
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def group(row):
    path=str(row['image_path']);m=re.search(r'xmlab\d+',path)
    if m:return 'SLAKE:'+m[0]
    m=re.search(r'synpic\d+',path,re.I)
    return 'VQA-RAD:'+m[0].lower() if m else 'IMAGE:'+row['image_sha256']

def freeze(root):
    assert not (root/'ROUTE_PLAN_FROZEN.json').exists()
    config=read(root/'COMPACT_DATA_ADMISSION.json');tasks=config['tasks'];assert len(tasks)==10
    queries={};relations={};reuse={}
    for task in tasks:
        part=(task['order']-1)%5
        for row in task['eval_rows']:
            q=row['query_id']
            if q in queries:assert all(queries[q][k]==row[k] for k in ('image_sha256','question','reference'))
            queries[q]=row
            yes=bool(set(row['roles'])&{'T0','T1G','T2G','REPHRASE_DIAGNOSTIC'});no=bool(set(row['roles'])&{'T1L','T2L'})
            relations[task['key']+'|'+q]='YES' if yes and not no else 'NO' if no and not yes else 'UNKNOWN'
            reuse[task['key']+'|'+q]=str(Path(str(part))/task['key']/'outputs/SOURCE'/(q+'.json'))
    assert len(queries)==180 and len(reuse)==216
    groups=sorted({group(row) for row in queries.values()});dev={g for i,g in enumerate(groups) if i%4==1}
    roles={q:'DEV' if group(row) in dev else 'HELDOUT_DEVELOPMENT' for q,row in queries.items()}
    allpairs={t['key']+'|'+q for t in tasks for q in queries};assert len(allpairs)==1800
    for pair in allpairs:relations.setdefault(pair,'UNKNOWN')
    source_ids={t['source']['query_id'] for t in tasks};assert source_ids<=queries.keys()
    result=dict(status='FROZEN_BEFORE_NEW_BANK_MODEL_CALLS',tasks=[dict(key=t['key'],order=t['order'],source=t['source']) for t in tasks],queries=queries,query_groups={q:group(r) for q,r in queries.items()},query_roles=roles,native_ids=sorted(source_ids),relations=relations,existing_source_outputs=reuse,missing_pairs=sorted(allpairs-reuse.keys()),counts=dict(experts=10,queries=180,cells=1800,reused=216,new_answers=1584,groups=len(groups),DEV=sum(x=='DEV' for x in roles.values()),HELDOUT_DEVELOPMENT=sum(x!='DEV' for x in roles.values()),known_relation_pairs=sum(x!='UNKNOWN' for x in relations.values())),source_group_rule='SLAKE xmlab/VQA-RAD synpic including derived variants, else exact image digest',independent_confirmation=False)
    assert not ({g for q,g in result['query_groups'].items() if roles[q]=='DEV'} & {g for q,g in result['query_groups'].items() if roles[q]!='DEV'})
    (root/'ROUTE_PLAN_FROZEN.json').write_text(json.dumps(result,indent=2)+'\n');return result['counts']
if __name__=='__main__':
    import sys
    print(freeze(Path(sys.argv[1])))
