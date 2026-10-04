"""Score the completed eight-edit pilot with the existing fixed Qwen worker."""
import json
from collections import defaultdict
from pathlib import Path
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest, PROMPT
from experiments.alphaedit_stage17.qwen_remaining import MODEL, REVISION

PROTOCOL = 'DIRECTW_RECURSIVE8_QWEN32_20261004_V1'


def worker():
    # Spawn must not try reopening the synthetic filename of a neutral stdin entry.
    import __main__
    from experiments.alphaedit_stage17 import qwen_remaining
    if getattr(__main__, '__file__', None) == '<stdin>':
        __main__.__file__ = None
    qwen_remaining.PROTOCOL = PROTOCOL
    qwen_remaining.worker()


def prepare(root):
    root=Path(root);c=read(root/'CONFIG.private.json')
    assert read(root/'COMPLETE.json')['state']=='GENERATED_NOT_SEMANTICALLY_SCORED'
    score=root/'scoring';score.mkdir()
    records={};mapping=[];summary={}
    for arm in ['base',*c['arms']]:
        folder=root/arm if arm=='base' else root/arm/'final'
        files=sorted(folder.glob('*.private.json'))
        assert len(files)==len(c['queries'])
        if arm!='base':
            summary[arm]=dict(completion=read(root/arm/'COMPLETE.json'),
                deletion=read(root/arm/'DELETION.json'),
                edits=[read(p) for p in sorted((root/arm).glob('*.RECEIPT.json'))])
            assert len(summary[arm]['edits'])==8
            files+=sorted((root/arm).glob('*.TARGET.private.json'))
        for path in files:
            row=read(path);q=c['queries'][row['query_id']]
            binding=dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],answer=row['raw_answer'],protocol=PROTOCOL)
            oid=digest(binding)
            records[oid]=dict(opaque_query_id=oid,question=q['question'],gold_answer=q['reference'],raw_base_answer=row['raw_answer'])
            mapping.append(dict(arm=arm,phase='insertion' if '.TARGET.' in path.name else 'final',query_id=q['query_id'],opaque_query_id=oid,source=str(path.relative_to(root))))
    assert len(mapping)==3*123+2*8
    lock=dict(protocol=PROTOCOL,model=MODEL,revision=REVISION,prompt=PROMPT,concurrency=32,
              thinking=False,temperature=0,seed=0,quantization='awq_marlin',dtype='half',
              max_output_tokens=256,max_num_batched_tokens=4096,prefix_caching=False,
              enforce_eager=True,gpu_memory_utilization=.75,semantic_retries=0,
              eligibility='original frozen Base masks; active first-eight targets excluded from locality',
              deduplication='identical query, reference and candidate text within this pilot only')
    lock['config_sha256']=digest(lock)
    write_new(score/'LOCK.json',lock);write_new(score/'MAPPING.private.json',mapping)
    write_new(score/'GENERATION_SUMMARY.json',summary)
    write_new(score/'CONFIG.private.json',dict(records=list(records.values()),prompt=PROMPT,lock=lock,
        model_path='/data/bmw/hugging_cache/Qwen3-32B-AWQ/'+REVISION,snapshot=REVISION,
        gpu_uuid='GPU-ffc224df-661c-5aff-e221-ec361bb4e5d6',chunk_size=256))
    return dict(occurrences=len(mapping),unique_judge_records=len(records),protocol=PROTOCOL)


def report(root):
    root=Path(root);c=read(root/'CONFIG.private.json');score=root/'scoring'
    status=read(score/'STATUS.json');controller=read(score/'CONTROLLER_RESULT.json')
    assert status['state']=='COMPLETE' and controller['exit_code']==0
    verdicts=[v for p in sorted((score/'chunks').glob('*.VERDICTS.json')) for v in read(p)]
    values={v['opaque_query_id']:v['is_correct'] for v in verdicts}
    inputs=read(score/'CONFIG.private.json')['records']
    assert len(values)==len(verdicts)==len(inputs)==status['completed_records']
    assert set(values)=={r['opaque_query_id'] for r in inputs}
    assert all(v['protocol']==PROTOCOL and v['model_revision']==REVISION and type(v['is_correct']) is bool for v in verdicts)
    mapping=read(score/'MAPPING.private.json')
    lock=read(score/'LOCK.json')
    assert digest({k:v for k,v in lock.items() if k!='config_sha256'})==lock['config_sha256']
    assert len(mapping)==385 and len({(r['arm'],r['phase'],r['query_id']) for r in mapping})==385
    for r in mapping:
        q=c['queries'][r['query_id']];out=read(root/r['source'])
        assert out['query_id']==r['query_id']
        assert digest(dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],
                           answer=out['raw_answer'],protocol=PROTOCOL))==r['opaque_query_id']
    lookup={(r['arm'],r['phase'],r['query_id']):values[r['opaque_query_id']] for r in mapping}
    active={t['edit_id'] for t in c['tasks']}
    assert len(active)==8 and all(c['active_targets']['50'][q]==c['queries'][q]['reference'] for q in active)
    panels=[]
    for arm in ['base',*c['arms']]:
        grouped=defaultdict(list)
        for task in c['tasks']:
            for event in task['events']:
                name=event['task'];locality=name.endswith('L')
                grouped.setdefault(name, [])
                for qid in event['all_probe_query_ids']:
                    if locality and qid in active:continue
                    if bool(c['Base_correctness'][qid]) != locality:continue
                    grouped[name].append((task['edit_id'],lookup[arm,'final',qid]))
        for name,rows in sorted(grouped.items()):
            edits=defaultdict(list)
            for eid,value in rows:edits[eid].append(int(value))
            macro=sum(sum(v)/len(v) for v in edits.values())/len(edits) if edits else None
            panels.append(dict(arm=arm,task=name,metric='retention' if name.endswith('L') else 'fix',
                correct=sum(v for _,v in rows),denominator=len(rows),edits=len(edits),macro=macro))
    generation=read(score/'GENERATION_SUMMARY.json')
    tokens={arm:{read(p)['query_id']:read(p)['raw_token_ids'] for p in (root/arm/'final' if arm!='base' else root/arm).glob('*.private.json')} for arm in ['base',*c['arms']]}
    text_changes={arm:sum(tokens[arm][q]!=tokens['base'][q] for q in c['queries']) for arm in c['arms']}
    between=sum(tokens['recursive'][q]!=tokens['static'][q] for q in c['queries'])
    result=dict(run_id=c['run_id'],status='COMPLETE_SEMANTICALLY_SCORED',generation=generation,
        generation_seconds=read(root/'STATUS.json')['updated_epoch']-c['started_epoch'],
        judge_lock=read(score/'LOCK.json'),judge_execution=status,judged_occurrences=len(mapping),
        base_label_changes=sum(lookup['base','final',q]!=c['Base_correctness'][q] for q in c['queries']),
        final_correct_all_queries={arm:sum(lookup[arm,'final',q] for q in c['queries']) for arm in ['base',*c['arms']]},
        source_groups=len({t['source_group'] for t in c['tasks']}),
        panels=panels,final_token_changes_vs_base=text_changes,final_token_differences_between_arms=between,
        insertion_target_correct={arm:sum(lookup[arm,'insertion',q] for q in active) for arm in c['arms']},
        limits=['Eight developmental edits in one fixed order; no confirmatory independence claim',
                'Original Astra-based eligibility retained; new outputs uniformly Qwen-scored',
                'Source-answer agreement is not clinical validation or visual-grounding proof',
                'Single language layer and frozen vision are a declared M-ORE adaptation'])
    (score/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
