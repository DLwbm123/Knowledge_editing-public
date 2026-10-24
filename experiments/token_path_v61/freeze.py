"""Restore original evaluation consumers; only each original native edit is training."""
import json,collections
from pathlib import Path

def freeze(parent,out):
    assert not (out/'DATA_ADMISSION.json').exists()
    original=json.loads((parent/'ORIGINAL146_TASKS.json').read_text())[:10]
    plan=json.loads((parent/'EVAL_PLAN146.json').read_text())
    fields=('query_id','question','reference','image_path','image_sha256')
    tasks=[];consumers=[]
    for t in original:
        source={k:t['native'][k] for k in fields}
        query={p['query']['query_id']:p['query'] for p in plan if p['method']=='C' and p['mode']=='single' and p['owner']==t['order']}
        query[source['query_id']]=source;rows=[]
        for qid,row in sorted(query.items()):
            roles=[e['task'] for e in t['events'] if qid in e['all_probe_query_ids']]
            if not roles:
                assert row.get('audit_role')=='FIT' and qid.startswith('edit_');roles=['REPHRASE_DIAGNOSTIC']
            rows.append(dict(row,roles=roles))
            consumers.extend(dict(owner=t['order'],query=qid,role=role) for role in roles)
        assert len([r for r in rows if 'T0' in r['roles']])==1
        tasks.append(dict(key=f'E{t["order"]:03}',order=t['order'],source=source,eval_rows=rows))
    assert [t['order'] for t in tasks]==list(range(1,11))
    queries={r['query_id']:r for t in tasks for r in t['eval_rows']}
    source_keys={t['source']['query_id'] for t in tasks}
    data=dict(status='PASS',tasks=tasks,train_fields=['image','question','target_answer'],train_rows_per_edit=1,auxiliary_training_rows=0,all_original_consumers_restored=True,development_only=True)
    out.mkdir(parents=True,exist_ok=True)
    for n,obj in [('DATA_ADMISSION.json',data),('QUERY_ROWS.json',queries),('EVAL_CONSUMERS.json',consumers)]:
        (out/n).write_text(json.dumps(obj,indent=2)+'\n')
    return dict(edits=10,unique_eval_inputs=len(queries),per_arm_generations=sum(len(t['eval_rows']) for t in tasks),consumers=dict(collections.Counter(x['role'] for x in consumers)),training_source_ids=len(source_keys),new_support=0,source_only=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('parent',type=Path);a.add_argument('out',type=Path);v=a.parse_args();print(json.dumps(freeze(v.parent,v.out),indent=2))
