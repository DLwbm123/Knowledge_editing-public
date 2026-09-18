"""Stage21 exact semantic reuse and isolated frozen Judge; independent 800 cap."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest,PROTOCOL
from scripts.medtrace.stage18_score import score_key
from scripts.medtrace.stage19_fasttrack_budget import write

def judge(root,records,name,category='main'):
    from scripts.medtrace.stage18_base_judge import run
    root=Path(root);p=root/'private';path=p/'QUALIFIED_SCORE_CACHE.json'
    original=read(path if path.exists() else p/'EXISTING_SCORE_CACHE.json')
    if original['protocol']!=PROTOCOL:raise ValueError('Semantic protocol changed')
    cache=original['scores'];receipt=p/(name+'_SCORE_RECEIPT.json')
    binding=digest(records)
    if receipt.exists():
        if read(receipt)['consumer_binding']!=binding:raise ValueError('Score consumers changed')
        return cache
    unique={score_key(r['source'],r['output']):dict(query_id=score_key(r['source'],r['output']),source=r['source'],output=dict(raw_answer=r['output']['raw_answer'])) for r in records}
    novel=[r for k,r in unique.items() if k not in cache]
    ledgerpath=root/'public/BUDGET_LEDGER.json';ledger=read(ledgerpath)
    key=category+'_items_dispatched';limit=ledger[category+'_limit']
    if ledger.get(key,0)+len(novel)>limit or ledger['new_judgment_items_dispatched']+len(novel)>800:raise ValueError('Category or total judgment budget exceeded')
    ledger['exact_reuse_count']+=len(unique)-len(novel);write(ledgerpath,ledger)
    write_new(p/(name+'_SCORE_CONSUMERS.json'),dict(binding=binding,rows=[dict(arm=r.get('arm'),prefix=r.get('prefix'),mode=r.get('mode','Base'),query_id=r['query_id'],score_key=score_key(r['source'],r['output'])) for r in records]))
    for offset in range(0,len(novel),60):
        chunk=novel[offset:offset+60];label=name+f'_{offset//60:03d}'
        source=dict(scope='STAGE21_'+label,records=chunk);inputpath=p/(label+'_JUDGE_INPUT.json');bundle=p/(label+'_JUDGE')
        write_new(inputpath,source)
        ledger=read(ledgerpath);ledger['new_judgment_items_dispatched']+=len(chunk);ledger[key]=ledger.get(key,0)+len(chunk)
        item=dict(name=label,category=category,items=len(chunk),input_binding=digest(source),status='DISPATCHED',actual_cost=None,cost_status='Provider currency billing unavailable; never treated as zero')
        ledger['judgment_batches'].append(item);write(ledgerpath,ledger)
        run(dict(base_outputs=str(inputpath),bundle=str(bundle),N_queries=len(chunk),authorization='USER_AUTHORIZED_ASTRA_DEV_REVIEW',cli='/Applications/ChatGPT.app/Contents/Resources/codex'))
        verdict=read(bundle/'operator/VERDICTS.json')
        if verdict['protocol']!=PROTOCOL or verdict['source_binding']!=digest(source):raise ValueError('Judge lineage changed')
        by={r['opaque_query_id']:r['is_correct'] for r in verdict['decisions']}
        if set(by)!={digest(r) for r in chunk} or any(type(x) is not bool for x in by.values()):raise ValueError('Judge coverage invalid')
        for row in chunk:cache[row['query_id']]=by[digest(row)]
        evidence=read(bundle/'operator/execution_evidence/b000.json');item.update(status='COMPLETE',usage=evidence.get('usage'),execution_binding=digest(evidence))
        write(path,dict(scores=cache,protocol=PROTOCOL));write(ledgerpath,ledger)
    write(path,dict(scores=cache,protocol=PROTOCOL))
    write_new(receipt,dict(consumer_binding=binding,new=len(novel),reused=len(unique)-len(novel),category=category))
    return cache

