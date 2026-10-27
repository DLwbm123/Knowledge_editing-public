"""Freeze the existing benchmark ledger and a disclosed, shared support split."""
import json,re,collections
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def group(row):
    name=str(row['image_path'])
    m=re.search(r'(?:SLAKE/(?:imgs/)?|t1g_images/SLAKE/)(xmlab\d+)',name)
    return 'SLAKE:'+m[1] if m else 'IMAGE:'+row['image_sha256']

def freeze(root):
    assert not (root/'DATA_ADMISSION.json').exists()
    tasks=read(root/'ORIGINAL146_TASKS.json');bindings=read(root/'EVAL_BINDINGS.json');mask=read(root/'ORIGINAL_BASE_MASK146.json');plan=read(root/'EVAL_PLAN146.json')
    assert len(tasks)==146 and [t['order'] for t in tasks]==list(range(1,147))
    queries={v['query_id']:v for v in bindings.values()};native_groups={group(t['native']) for t in tasks};support=[];seen=set()
    fields=('query_id','question','reference','image_path','image_sha256')
    for row in sorted(queries.values(),key=lambda x:x['query_id']):
        g=group(row)
        if not mask.get(row['query_id']) or g in native_groups or g in seen:continue
        seen.add(g);v={k:row[k] for k in fields};ref=v['reference'].strip().rstrip('.')
        wrong={'yes':'No','no':'Yes'}.get(ref.lower(),'This is not '+ref)
        support.append(dict(row=v,wrong_choice=wrong,source_group=g))
        if len(support)==8:break
    assert len(support)==8
    removed={q for q,r in queries.items() if group(r) in seen};config=[];consumers=[]
    # Original full single-event ledger contains all generalized and locality rows.
    singles=[p for p in plan if p['method']=='C' and p['mode']=='single']
    for t in tasks:
        qmap={p['query']['query_id']:p['query'] for p in singles if p['owner']==t['order'] and p['query']['query_id'] not in removed}
        native={k:t['native'][k] for k in fields};qmap[native['query_id']]=native
        rows=[]
        for qid,row in sorted(qmap.items()):
            roles=[e['task'] for e in t['events'] if qid in e['all_probe_query_ids']]
            if not roles:
                assert row.get('audit_role')=='FIT' and qid.startswith('edit_')
                roles=['REPHRASE_DIAGNOSTIC']
            rows.append(dict(row,roles=roles));consumers.extend(dict(owner=t['order'],query=qid,role=r) for r in roles)
        task=dict(key=f'E{t["order"]:03}',order=t['order'],source=native,eval_rows=rows)
        config.append(task)
    evaluation={r['query_id']:r for t in config for r in t['eval_rows']}
    assert not removed.intersection(evaluation)
    partitions=[]
    for part in range(5):
        assigned=config[part::5];base_queries=[q for i,q in enumerate(sorted(evaluation)) if i%5==part]
        outputs=7*sum(len(t['eval_rows']) for t in assigned)
        assert outputs+len(base_queries)<=6000
        partitions.append(dict(partition=part,edit_count=len(assigned),method_answers=outputs,base_answers=len(base_queries)))
    write(root/'SUPPORT_FROZEN.json',support);write(root/'QUERY_ROWS.json',evaluation);write(root/'EVAL_CONSUMERS.json',consumers)
    result=dict(status='PASS',tasks=config,partitions=partitions,removed_support_queries=len(removed),support_groups=len(seen),unique_queries=len(evaluation),method_answers=7*sum(len(t['eval_rows']) for t in config),task_consumers=len(consumers),source='Existing development benchmark, new disclosed support split',historical_base_reuse_for_current_damage=False)
    assert result['method_answers']+len(evaluation)<=30000
    write(root/'DATA_ADMISSION.json',result)
    return {k:v for k,v in result.items() if k!='tasks'}
if __name__=='__main__':
    import sys
    print(freeze(Path(sys.argv[1])))
