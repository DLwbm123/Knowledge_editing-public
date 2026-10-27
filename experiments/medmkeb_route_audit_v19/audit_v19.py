"""Read-only task identities and existing inference-mode inventory; no model work."""
import collections,json,sqlite3
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def relation(source,query):
    return ('same_image' if source['image_sha256']==query['image_sha256'] else 'different_image')+'__'+('same_question' if source['question']==query['question'] else 'different_question')
def audit(v18,out):
    env=read(v18/'PARENT_ENV.json');old=Path(env['PAPER_PARENT']);tasks=read(old/'private/BENCHMARK146_QUEUE.json')['tasks'];queries=read(old/'private/EVAL_LEDGER.json')['queries'];counts=collections.defaultdict(collections.Counter);exceptions=[]
    for task in tasks:
        for event in task['events']:
            for qid in event['all_probe_query_ids']:
                label=relation(task['native'],queries[qid]);counts[event['task']][label]+=1
                if event['task']=='T1L' and not label.endswith('__same_question'):exceptions.append(dict(owner=task['edit_id'],query=qid,relation=label))
    # One queue inventory: prior accepted reports retain score/recovery identities.
    with sqlite3.connect('file:'+str(old/'private/judge_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        modes=[dict(method=m,mode=mode,consumers=n) for m,mode,n in db.execute('select method,mode,count(*) from consumer group by method,mode')]
    current=read(v18/'DATA_ADMISSION.json')['tasks'];mapping=[]
    for task in current:
        bs={kind:read(read(v18/'BASE_REUSE.json')[task[kind]['query_id']])['binding'] for kind in ('source','cross','rephrase')}
        assert relation(bs['source'],bs['cross'])=='different_image__same_question'
        assert relation(bs['source'],bs['rephrase'])=='same_image__different_question'
        mapping.append(dict(task=task['key'],cross=relation(bs['source'],bs['cross']),rephrase=relation(bs['source'],bs['rephrase'])))
    result=dict(status='PASS_DEFINITIONS_AUDITED',historical_edits=len(tasks),historical_relations={k:dict(v) for k,v in counts.items()},V18_relations=mapping,existing_modes=modes,legacy_single_R0_is_forced=False,new_model_calls=0,new_pixels=0,T1L_text_mismatch=len(exceptions),official_T1G='different image / same question',official_T2G='same image / rephrased question',past_T2G_image_generalization_prose_incorrect=True,historical_scores_unchanged=True)
    write(out/'DEFINITION_AUDIT_PRIVATE.json',dict(result,exceptions=exceptions));return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--v18',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.v18,a.out)))

def inventory_be(v18,out):
    tasks=read(v18/'DATA_ADMISSION.json')['tasks'];base=read(v18/'BASE_REUSE.json');rows=[]
    for g,task in enumerate(tasks):
        folder=v18/str(g)/task['key'];qs=read(v18/'EVAL_ROWS.json')+[task[k] for k in ('source','cross','rephrase')]
        for arm in ('BALANCEDIT','BALANCEDIT_CE'):
            for q in qs:
                qid=q['query_id'];route=read(folder/'routing'/arm/(qid+'.json'));selected=route['route']['logical_edit_id'];assert selected in (task['key'],None)
                expected='FORCED' if selected==task['key'] else 'BASE';assert route['natural_output_reuses']==expected
                path=folder/'outputs'/arm/(qid+'.json') if expected=='FORCED' else Path(base[qid]);answer=read(path);reference=read(base[qid])['binding']
                for k in ('image_sha256','question','reference','prompt_ids','attention_mask','runtime','generation'):assert answer['binding'][k]==reference[k]
                rows.append(dict(task=task['key'],arm=arm,query_id=qid,route=route['route'],reuse=expected,output_path=str(path)))
    assert len(rows)==100;write(out/'BE_SINGLE_REUSE_INDEX.json',rows)
    return dict(consumers=100,new_generations=0,scope='SINGLE_GATED_NOT_TWO_EXPERT_BANK',reuse=dict(collections.Counter(x['reuse'] for x in rows)))
