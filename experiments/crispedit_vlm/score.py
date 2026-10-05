"""Uniform, frozen Qwen scoring and deidentified reporting for this comparison."""
import json
from collections import defaultdict
from pathlib import Path
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest,PROMPT
from experiments.alphaedit_stage17.qwen_remaining import MODEL,REVISION

PROTOCOL='CRISPEDIT_VLM8_QWEN32_20261005_V1'


def worker():
    import __main__
    from experiments.alphaedit_stage17 import qwen_remaining
    if getattr(__main__,'__file__',None)=='<stdin>':__main__.__file__=None
    qwen_remaining.PROTOCOL=PROTOCOL;qwen_remaining.worker()


def prepare(root):
    root=Path(root);c=read(root/'CONFIG.private.json');score=root/'scoring'
    assert read(root/'COMPLETE.json')['state']=='GENERATED_NOT_SCORED'
    if score.exists():raise RuntimeError('Scoring already prepared; inspect existing state before recovery')
    score.mkdir();records={};mapping=[]
    for arm in ['base',*c['arms']]:
        folder=root/'base' if arm=='base' else root/arm/'final'
        files=sorted(folder.glob('*.private.json'));assert len(files)==len(c['queries'])
        if arm!='base':files+=sorted((root/arm).glob('*.TARGET.private.json'))
        for path in files:
            row=read(path);q=c['queries'][row['query_id']]
            binding=dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],answer=row['raw_answer'],protocol=PROTOCOL)
            oid=digest(binding)
            records[oid]=dict(opaque_query_id=oid,question=q['question'],gold_answer=q['reference'],raw_base_answer=row['raw_answer'])
            mapping.append(dict(arm=arm,phase='insertion' if '.TARGET.' in path.name else 'final',query_id=q['query_id'],opaque_query_id=oid,source=str(path.relative_to(root))))
    assert len(mapping)==3*len(c['queries'])+2*len(c['tasks'])
    lock=dict(protocol=PROTOCOL,model=MODEL,revision=REVISION,prompt=PROMPT,concurrency=32,
        thinking=False,temperature=0,seed=0,quantization='awq_marlin',dtype='half',max_output_tokens=256,
        max_num_batched_tokens=4096,enforce_eager=True,gpu_memory_utilization=.75,prefix_caching=False,
        eligibility='original Base masks; current edit targets excluded from locality; separate fixed 64-query holdout',
        deduplication='identical query/reference/answer within this run only')
    lock['config_sha256']=digest(lock)
    write_new(score/'LOCK.json',lock);write_new(score/'MAPPING.private.json',mapping)
    write_new(score/'CONFIG.private.json',dict(records=list(records.values()),prompt=PROMPT,lock=lock,
        model_path=c['judge_model_path'],snapshot=REVISION,gpu_uuid=c['gpu_uuid'],chunk_size=256))
    return dict(occurrences=len(mapping),unique_records=len(records))


def report(root):
    root=Path(root);c=read(root/'CONFIG.private.json');s=root/'scoring'
    status=read(s/'STATUS.json');assert status['state']=='COMPLETE'
    assert read(s/'CONTROLLER_RESULT.json')['exit_code']==0
    lock=read(s/'LOCK.json');assert digest({k:v for k,v in lock.items() if k!='config_sha256'})==lock['config_sha256']
    rows=[v for p in sorted((s/'chunks').glob('*.VERDICTS.json')) for v in read(p)]
    values={v['opaque_query_id']:v['is_correct'] for v in rows};records=read(s/'CONFIG.private.json')['records']
    assert len(values)==len(rows)==len(records)==status['completed_records']
    assert set(values)=={r['opaque_query_id'] for r in records}
    assert all(v['protocol']==PROTOCOL and v['model_revision']==REVISION and type(v['is_correct']) is bool for v in rows)
    mapping=read(s/'MAPPING.private.json');lookup={}
    assert len(mapping)==3*len(c['queries'])+2*len(c['tasks'])
    for row in mapping:
        q=c['queries'][row['query_id']];out=read(root/row['source'])
        assert digest(dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],answer=out['raw_answer'],protocol=PROTOCOL))==row['opaque_query_id']
        key=(row['arm'],row['phase'],row['query_id']);assert key not in lookup
        lookup[key]=values[row['opaque_query_id']]
    active={t['edit_id'] for t in c['tasks']};panels=[];mechanics={}
    for arm in ['base',*c['arms']]:
        grouped=defaultdict(list)
        for task in c['tasks']:
            for event in task['events']:
                name=event['task'];locality=name.endswith('L');grouped.setdefault(name,[])
                for qid in event['all_probe_query_ids']:
                    if locality and qid in active:continue
                    if bool(c['Base_correctness'][qid])!=locality:continue
                    grouped[name].append((task['edit_id'],lookup[arm,'final',qid]))
        for name,entries in sorted(grouped.items()):
            edits=defaultdict(list)
            for eid,value in entries:edits[eid].append(value)
            panels.append(dict(arm=arm,task=name,correct=sum(v for _,v in entries),denominator=len(entries),
                edits=len(edits),macro=sum(sum(v)/len(v) for v in edits.values())/len(edits) if edits else None))
        panels.append(dict(arm=arm,task='INDEPENDENT_HOLDOUT',correct=sum(lookup[arm,'final',q] for q in c['holdout_ids']),
            denominator=len(c['holdout_ids']),edits=None,macro=None))
        if arm!='base':
            receipts=[read(p) for p in sorted((root/arm).glob('*.RECEIPT.json'))];assert len(receipts)==len(c['tasks'])
            mechanics[arm]=dict(completion=read(root/arm/'COMPLETE.json'),deletion=read(root/arm/'DELETION.json'),
                receipts=receipts,text_holdout=read(root/arm/'TEXT_LOSS.json'),
                insertion_correct=sum(lookup[arm,'insertion',q] for q in active),
                lost_insertions=sum(lookup[arm,'insertion',q] and not lookup[arm,'final',q] for q in active),
                newly_broken_holdout=sum(lookup['base','final',q] and not lookup[arm,'final',q] for q in c['holdout_ids']))
    tokens={arm:{read(p)['query_id']:read(p)['raw_token_ids'] for p in (root/'base' if arm=='base' else root/arm/'final').glob('*.private.json')} for arm in ['base',*c['arms']]}
    result=dict(run_id=c['run_id'],status='COMPLETE_SCORED_PUBLICATION_PENDING',upstream_commit=c['upstream_commit'],
        source_commit=c['source_commit'],panels=panels,mechanics=mechanics,judge_lock=lock,judge_execution=status,
        mapped_occurrences=len(mapping),unique_judged_records=len(rows),queries=len(c['queries']),
        base_label_changes=sum(lookup['base','final',q]!=c['Base_correctness'][q] for q in c['queries']),
        current_base_correct_holdout=sum(lookup['base','final',q] for q in c['holdout_ids']),
        token_changes_vs_base={arm:sum(tokens[arm][q]!=tokens['base'][q] for q in c['queries']) for arm in c['arms']},
        base_text_holdout=read(root/'base/TEXT_LOSS.json'),capability_statistics=read(root/'CAPABILITY.json'),
        generation_seconds=read(root/'STATUS.json')['updated_epoch']-c['started_epoch'],
        limits=['Medical VLM adaptation, not a reproduction of original text-LLM benchmark numbers',
                'Eight developmental edits, one fixed order, no hyperparameter search',
                'Original Base eligibility is Astra-derived; new outputs uniformly Qwen-scored',
                'Upstream empirical CE-gradient covariance path, not an exact GGN estimator',
                'Independent holdout is disjoint by recorded source group, not guaranteed patient independence'])
    (root/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# CrispEdit native medical-VLM comparison','',
          'Completed eight sequential edits per arm. Both arms edit language layers 19–23, use 25 steps at most per edit and lr=5e-4, and retain the same native FP16 deployment. CrispEdit uses the pinned official ProjectedAdam with online K-FAC statistics; Adam is the matched unprojected control. Vision stays frozen. This is a declared VLM adaptation.','',
          '| Arm | Task | Correct / eligible | Edit macro |','|---|---|---:|---:|']
    for p in panels:
        macro='NA' if p['macro'] is None else f"{100*p['macro']:.2f}%"
        text.append(f"| {p['arm']} | {p['task']} | {p['correct']}/{p['denominator']} | {macro} |")
    text+=['','Zero-denominator locality panels are unsupported, not evidence of preservation. The separate 64-query holdout is selected before editing from originally correct, source-disjoint queries and never used for optimization or curvature statistics.',
           '',f"Current Base holdout correctness: {result['current_base_correct_holdout']}/64. Fresh Base label changes across all queries: {result['base_label_changes']}. Exact-token changes versus Base: {result['token_changes_vs_base']}.",
           '',f"Generation/editing wall time: {result['generation_seconds']:.2f} seconds. Qwen scored {len(rows)} unique inputs covering {len(mapping)} occurrences; no prior-run verdict is reused.",'']
    for arm,v in mechanics.items():
        text.append(f"{arm}: insertion targets {v['insertion_correct']}/8; successful insertions lost at final evaluation {v['lost_insertions']}; newly broken current-Base-correct holdout answers {v['newly_broken_holdout']}; text holdout NLL {v['text_holdout']['nll']:.6f} (Base {result['base_text_holdout']['nll']:.6f}). Fresh native reload: {v['completion']}. Temporary-state deletion: {v['deletion']}.")
    text+=['','## Limitations','',*['- '+x for x in result['limits']],
           '', 'Raw medical queries, answers, tokens, mappings and checkpoints are excluded from publication. See PROTOCOL.md for precision, data, optimizer and sequential-history adaptations; RESULTS.json contains complete deidentified per-step receipts. No follow-on tuning or additional experiment is authorized by these results.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    return dict(status=result['status'],panels=panels,unique_judged_records=len(rows))
