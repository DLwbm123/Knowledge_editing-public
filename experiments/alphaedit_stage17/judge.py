"""One authorized Astra queue, then frozen-cohort aggregation and publication."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from scripts.medtrace.astra_judge_bundle import read, schema, write_new
from scripts.medtrace.stage17_prepare import digest, lines, PROMPT, PROTOCOL
from scripts.medtrace.stage17_judge import run as run_judge, execution_path
from scripts.medtrace.stage17_report import metric, interval

METHODS=('C_NO_H','balancedit','belora','lora')


def panel_values(c, lookup, methods):
    panels=[]; macros=[]
    groups={t['edit_id']:t['source_group'] for t in c['tasks']}
    for method in methods:
        for mode in ('single','sequential'):
            for prefix in ([1] if mode=='single' else c['prefixes']):
                tasks=c['tasks'] if mode=='single' else c['tasks'][:prefix]
                rows=[]
                for task in tasks:
                    for event in task['events']:
                        for qid in event['all_probe_query_ids']:
                            rows.append(dict(edit=task['edit_id'],task=event['task'],base=c['Base_correctness'][qid],
                                correct=lookup[method,mode,prefix,task['edit_id'] if mode=='single' else None,qid],
                                active=mode=='sequential' and qid in c['active_targets'][str(prefix)]))
                for task_name in sorted({r['task'] for r in rows}):
                    allrows=[r for r in rows if r['task']==task_name]
                    selected=[r for r in allrows if not(task_name.endswith('L') and r['active'])]
                    primary,values=metric(selected,'correct',lambda r:r['base'] if task_name.endswith('L') else not r['base'])
                    panels.append(dict(method=method,mode=mode,prefix=prefix,task=task_name,route='NATIVE' if method not in ('C_NO_H','balancedit') else 'R0',
                        primary_metric='Retention' if task_name.endswith('L') else 'Fix',primary=primary,
                        post_accuracy=metric(selected,'correct')[0],active_locality_exclusions=len(allrows)-len(selected)))
                    macros.append(dict(method=method,mode=mode,prefix=prefix,task=task_name,values=values))
    return panels,macros,groups


def historical(c, campaign, student, expected):
    """Validate old accepted scores/bindings, then reproduce published point estimates."""
    op=campaign/'operator';lock=read(op/'JUDGE_LOCK.json')
    if read(execution_path(op))['status']!='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED':raise ValueError('Historical queue not complete')
    bounds=read(op/'BINDINGS.json');verdicts=lines(op/'VERDICTS_ASTRA.jsonl')
    reused=read(op/'REUSED_PRIORITY.json')
    bounds.update(reused['bindings']);verdicts+=reused['verdicts']
    scores={v['opaque_query_id']:v['is_correct'] for v in verdicts}
    if set(scores)!=set(bounds) or len(scores)!=len(verdicts):raise ValueError('Historical coverage mismatch')
    for oid,b in bounds.items():
        if digest(b)!=oid or b['judge']!=lock or type(scores[oid]) is not bool:raise ValueError('Historical binding invalid')
    lookup={}
    for r in read(op/'MODE_MAPPING.json'):
        if r['method'] not in METHODS or r['panel']!='panel' or r['role']!='original':continue
        if r['route']!=('R0' if r['method'] in ('C_NO_H','balancedit') else 'NATIVE'):continue
        value=c['Base_correctness'][r['query_id']] if r['source']=='Base' else scores[r['opaque_query_id']]
        key=(r['method'],r['mode'],r['prefix'],r['edit'] if r['mode']=='single' else None,r['query_id'])
        if key in lookup and lookup[key]!=value:raise ValueError('Conflicting old output mapping')
        lookup[key]=value
    so=student/'operator';sb=read(so/'BINDINGS.json');sv=lines(so/'VERDICTS_ASTRA.jsonl')
    if read(so/'JUDGE_LOCK.json')!=lock or read(execution_path(so))['status']!='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED':raise ValueError('Historical BE Judge mismatch')
    ss={v['opaque_query_id']:v['is_correct'] for v in sv}
    if set(ss)!=set(sb) or len(ss)!=len(sv):raise ValueError('Historical BE coverage mismatch')
    for oid,b in sb.items():
        if digest(b)!=oid or b['judge']!=lock or type(ss[oid]) is not bool:raise ValueError('Historical BE binding invalid')
    for r in read(so/'MODE_MAPPING.json'):
        ref=r['modes']['R0'];qid=r['query_id']
        lookup['balancedit','single',1,r['edit_id'],qid]=c['Base_correctness'][qid] if ref['source']=='Base' else ss[ref['opaque_query_id']]
    panels,macros,_=panel_values(c,lookup,METHODS)
    previous={(p['method'],p['mode'],p['prefix'],p['task']):p for p in expected['panels'] if p['method'] in METHODS and p['route'] in ('R0','NATIVE')}
    for p in panels:
        old=previous[p['method'],p['mode'],p['prefix'],p['task']]
        for field in ('numerator','probes','edits','micro','macro'):
            a,b=p['primary'][field],old['primary'][field]
            if a!=b and (a is None or b is None or abs(a-b)>1e-12):raise ValueError('Historical point estimates not reproduced')
    return dict(panels=[previous[p['method'],p['mode'],p['prefix'],p['task']] for p in panels],macros=macros,
                validation='Accepted bound verdicts reproduce all published primary denominators and point estimates')


def prepare(bundle,base_bundle,campaign,student,repository):
    source=bundle/'source';c=read(source/'CONFIG.private.json')
    if read(source/'RESULT.json')['state']!='GENERATED_NOT_SCORED':raise ValueError('GPU generation not complete')
    base=base_bundle/'operator';lock=read(base/'JUDGE_LOCK.json');bb=read(base/'BINDINGS.json')
    baseline=historical(c,campaign,student,read(repository/'reports/alphaedit_stage17_20261003/HISTORICAL_BASELINES.json'))
    op=bundle/'operator';op.mkdir();visible_dir=bundle/'judge_only';visible_dir.mkdir()
    bindings={};mapping=[];config_id=digest(c)
    def add(qid,out,method,mode,prefix,index,role,reference):
        q=c['queries'][qid];old=bb[q['opaque_Base_id']]
        if (q['question'],q['reference'],q['image_sha256'],q['expected_prompt_ids'])!=(old['question'],old['reference'],old['image_sha256'],old['prompt_ids']):
            raise ValueError('Original question/image/prompt binding mismatch')
        if not isinstance(out['raw_answer'],str) or any(type(t) is not int or t<0 for t in out['raw_token_ids']) or len(out['raw_token_ids'])>1024:raise ValueError('Invalid continuation')
        full=dict(query_id=qid,question=q['question'],reference=reference,image_sha256=q['image_sha256'],prompt_ids=q['expected_prompt_ids'],
            output=dict(model_answer_raw=out['raw_answer'],raw_generated_token_ids=out['raw_token_ids']),judge=lock,
            generation=c['generation'],run=c['run_id'],configuration_binding=config_id,source_commit=c['source_commit'],
            method=method,mode=mode,prefix=prefix,edit_index=index,cohort=c['freeze_id'])
        oid=digest(full);bindings[oid]=full
        mapping.append(dict(method=method,mode=mode,prefix=prefix,index=index,query_id=qid,role=role,opaque_query_id=oid))
    for arm in c['arms']:
        for phase in ('single','sequential'):
            directory=source/arm/phase
            if read(directory/'COMPLETE.json')['completed']!=146:raise ValueError('Incomplete arm')
            for index,task in enumerate(c['tasks'],1):
                out=read(directory/f'{index:03d}.OUTPUT.private.json')
                expected=set(task['query_ids']) if phase=='single' else ({q for t in c['tasks'][:index] for q in t['query_ids']} if index in c['prefixes'] else {task['edit_id']})
                if out['edit_id']!=task['edit_id'] or set(out['outputs'])!=expected:raise ValueError('Realized panel coverage mismatch')
                for qid,value in out['outputs'].items():
                    prefix=1 if phase=='single' else index
                    add(qid,value,arm,phase,prefix,index,'original',c['queries'][qid]['reference'])
                    active=c['active_targets'].get(str(index),{}) if phase=='sequential' else {}
                    if qid in active:
                        add(qid,value,arm,phase,prefix,index,'active',active[qid])
    base_rows=lines(source/'statistics/BASE_OUTPUTS.private.jsonl')
    if len(base_rows)!=len(c['queries']) or {r['query_id'] for r in base_rows}!=set(c['queries']):raise ValueError('New Base coverage mismatch')
    for row in base_rows:add(row['query_id'],row,'Base_new','base',0,0,'original',c['queries'][row['query_id']]['reference'])
    if len(bindings)>c['maximum_new_judge_keys']:raise ValueError('Frozen Judge key cap exceeded')
    write_new(op/'BINDINGS.json',bindings);write_new(op/'MODE_MAPPING.json',mapping)
    write_new(op/'JUDGE_LOCK.json',lock);write_new(op/'HISTORICAL_VALIDATED.private.json',baseline)
    visible=[dict(opaque_query_id=k,question=v['question'],gold_answer=v['reference'],raw_base_answer=v['output']['model_answer_raw']) for k,v in bindings.items()]
    batches=[]
    for start in range(0,len(visible),50):
        name=f'batch_{start//50+1:03d}';batch=dict(batch_id=name,records=visible[start:start+50])
        write_new(visible_dir/f'{name}.input.json',batch);write_new(visible_dir/f'{name}.schema.json',schema(batch))
        (visible_dir/f'{name}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
        batches.append(dict(batch_id=name,count=len(batch['records'])))
    manifest=dict(protocol=PROTOCOL,config_sha256=lock['config_sha256'],records=len(visible),batches=batches,freeze_id=c['freeze_id'],N=146,
                  state='READY',scope='New AlphaEdit and new Base only; historical four-method verdicts reused after validation')
    write_new(op/'MANIFEST.json',manifest)
    return manifest


def report(bundle,repository,hybrid_operator=None):
    op=bundle/'operator';c=read(bundle/'source/CONFIG.private.json')
    if hybrid_operator is None:
        if read(execution_path(op))['status']!='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED':raise ValueError('New Judge incomplete')
        v=lines(op/'VERDICTS_ASTRA.jsonl')
    else:
        hybrid=read(hybrid_operator/'EXECUTION_RECORD.json')
        if hybrid['status']!='COMPLETE_HYBRID_FORMAT_AND_COVERAGE_VALIDATED':raise ValueError('Hybrid Judge incomplete')
        v=lines(hybrid_operator/'VERDICTS_HYBRID.jsonl')
    b=read(op/'BINDINGS.json');scores={r['opaque_query_id']:r['is_correct'] for r in v}
    if len(scores)!=len(v) or set(scores)!=set(b):raise ValueError('New verdict coverage mismatch')
    labels={r['opaque_query_id']:r['judge_model'] for r in v}
    lookup={};label_lookup={};native={};base={};base_labels={}
    for r in read(op/'MODE_MAPPING.json'):
        if r['role']!='original':continue
        score=scores[r['opaque_query_id']];qid=r['query_id'];method=r['method'];mode=r['mode'];prefix=r['prefix']
        if method=='Base_new':base[qid]=score;base_labels[qid]=labels[r['opaque_query_id']];continue
        eid=c['tasks'][r['index']-1]['edit_id']
        if mode=='single' or prefix in c['prefixes']:
            key=(method,mode,prefix,eid if mode=='single' else None,qid)
            lookup[key]=score;label_lookup[key]=labels[r['opaque_query_id']]
        if qid==eid:native[method,mode,r['index']]=score
    panels,macros,groups=panel_values(c,lookup,c['arms'])
    for p,m in zip(panels,macros):p['primary'].update(interval(m['values'],groups));p['precision']='float16'
    old=read(op/'HISTORICAL_VALIDATED.private.json');panels+=old['panels'];macros+=old['macros']
    if hybrid_operator is not None:
        for p in panels:
            coverage={};tasks=c['tasks'] if p['mode']=='single' else c['tasks'][:p['prefix']]
            for task in tasks:
                for event in task['events']:
                    if event['task']!=p['task']:continue
                    for qid in event['all_probe_query_ids']:
                        locality=p['task'].endswith('L')
                        if locality and p['mode']=='sequential' and qid in c['active_targets'][str(p['prefix'])]:continue
                        if not (c['Base_correctness'][qid] if locality else not c['Base_correctness'][qid]):continue
                        label=label_lookup.get((p['method'],p['mode'],p['prefix'],task['edit_id'] if p['mode']=='single' else None,qid),'gpt-6-astra')
                        coverage[label]=coverage.get(label,0)+1
            if sum(coverage.values())!=p['primary']['probes']:raise ValueError('Panel Judge coverage does not match primary denominator')
            p['primary_judge_coverage']=coverage
    values={(m['method'],m['mode'],m['prefix'],m['task']):m['values'] for m in macros};paired=[]
    for arm in c['arms']:
        for m in [m for m in macros if m['method']==arm]:
            for reference in METHODS:
                ref=values[reference,m['mode'],m['prefix'],m['task']];new=m['values'];common=set(ref)&set(new)
                delta={e:new[e]-ref[e] for e in common}
                paired.append(dict(contrast=arm+' minus '+reference,mode=m['mode'],prefix=m['prefix'],task=m['task'],
                    edits=len(delta),macro_delta=sum(delta.values())/len(delta) if delta else None,
                    precision_matched=not(reference=='lora' and m['mode']=='sequential'),**interval(delta,groups)))
    trajectories=[]
    for arm in c['arms']:
        counts={f'{a}_to_{b}':0 for a in (0,1) for b in (0,1)}
        for i,eid in enumerate(c['main_T0'],1):counts[f'{int(native[arm,"sequential",i])}_to_{int(lookup[arm,"sequential",146,None,eid])}']+=1
        trajectories.append(dict(method=arm,counts=counts))
    result=dict(status='SCORED_COMPARISON_COMPLETE_PUBLICATION_PENDING',N=146,panels=panels,paired=paired,insertion_to_final=trajectories,
        judge=dict(model='gpt-6-astra',reasoning='high',new_records=len(v),immutable_snapshot=None,semantic_retries=0),
        base_audit=dict(queries=len(base),new_correct=sum(base.values()),semantic_label_changes=sum(base[q]!=c['Base_correctness'][q] for q in base),eligibility_masks='Original accepted Base masks retained'),
        limitations=['C_NO_H has no H; not full C_FACTH','Historical immutable model snapshot unavailable; new and old verdict epochs differ',
            'Historical hardware/runtime differ','LoRA single FP16 and sequence BF16; original FP16 sequence failed edit17 step4',
            'One fixed edit order; no order robustness or full patient independence','AlphaEdit VLM adaptation with bounded text-only projection statistics',
            'Source-answer agreement is not clinical validation'])
    if hybrid_operator is not None:
        result['status']='HYBRID_EXPLORATORY_COMPARISON_COMPLETE_PUBLICATION_PENDING'
        result['judge']=hybrid['judge_summary']
        result['base_audit']['judge_coverage']={label:list(base_labels.values()).count(label) for label in sorted(set(base_labels.values()))}
        result['limitations'][:0]=[
            'User-authorized mid-queue judge switch: accepted Astra prefix retained; only the remaining inputs were judged by Qwen3-32B-AWQ',
            'Judge assignment follows queue order, not randomization; method/task comparisons may be confounded by grader changes',
            'Historical baselines and Base eligibility masks remain Astra-based; hybrid scores are exploratory, not a uniform-judge confirmatory comparison',
            'Paired confidence intervals quantify edit variation only and do not account for differences between judges']
    dest=repository/'reports/alphaedit_stage17_20261003'
    stem='HYBRID_SEMANTIC' if hybrid_operator is not None else 'SEMANTIC'
    write_new(dest/(stem+'_RESULTS.json'),result)
    with (dest/(stem+'_RESULTS.csv')).open('x') as stream:
        fields=['method','mode','prefix','task','primary_metric','numerator','probes','edits','micro','macro'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for p in panels:writer.writerow({**{k:p[k] for k in fields[:5]},**{k:p['primary'][k] for k in fields[5:]}})
    text=['# Frozen 146-edit semantic comparison','',
          'Astra/high judged all new AlphaEdit and Base outputs. Historical four-method bound verdicts reproduce the published point estimates and are reused. C_NO_H is the no-H version. Values below are edit-macro percentages; all denominators and confidence intervals are in SEMANTIC_RESULTS.json.','']
    if hybrid_operator is not None:
        text=['# Exploratory 146-edit comparison with mixed judges','',
              'The user requested switching the remaining queue from Astra/high to Qwen3-32B-AWQ at concurrency 32. Accepted Astra scores are retained; the remaining inputs are scored once by Qwen. Per-record provenance is retained privately, and primary-panel judge counts are in HYBRID_SEMANTIC_RESULTS.json. Historical baselines and Base eligibility masks remain Astra-based. Differences below may reflect both the editing method and the judge; this is not a uniform-judge ranking.','',
              'New-record judge coverage: '+json.dumps(hybrid['judge_summary']['record_counts'])+'. Values are edit-macro percentages; C_NO_H is the no-H version.','']
    for mode in ('single','sequential'):
        text+=['## '+mode+(' (prefix 146)' if mode=='sequential' else ''),'','| Method | T0 Fix | T1G Fix | T2G Fix | T2L retention |','|---|---:|---:|---:|---:|']
        for method in (*c['arms'],*METHODS):
            row=[]
            for task in ('T0','T1G','T2G','T2L'):
                p=next((p for p in panels if p['method']==method and p['mode']==mode and p['prefix']==(1 if mode=='single' else 146) and p['task']==task),None)
                value=p['primary']['macro'] if p else None;row.append('NA' if value is None else f'{100*value:.2f}')
            text.append('| '+method+' | '+' | '.join(row)+' |')
        text.append('')
    text+=['## Limits','',*['- '+s for s in result['limitations']],'',
        'Do not rank solely by target Fix: generalization, locality, paired uncertainty and supported denominators are required. Generation costs and adverse outputs are recorded separately in GENERATION_REPORT.md.']
    (dest/(stem+'_REPORT.md')).write_text('\n'.join(text)+'\n')
    return [stem+'_RESULTS.json',stem+'_RESULTS.csv',stem+'_REPORT.md']


def publish(repository,names):
    env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    def git(*args):return subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897',*args],cwd=repository,env=env,text=True,timeout=120).strip()
    allowed={'reports/alphaedit_stage17_20261003/'+name for name in names}
    changed={line[3:] for line in git('status','--porcelain').splitlines()}
    if changed!=allowed or git('branch','--show-current')!='research/directw-evidence-v1' or git('remote','get-url','origin')!='https://github.com/DLwbm123/Knowledge_editing-public.git':raise RuntimeError('Publication boundary changed; preserve report without touching unrelated work')
    git('add',*sorted(allowed));git('commit','-m','Report frozen AlphaEdit 146-edit semantic comparison');commit=git('rev-parse','HEAD');git('push','origin','research/directw-evidence-v1')
    if git('ls-remote','origin','refs/heads/research/directw-evidence-v1').split()[0]!=commit:raise RuntimeError('Remote SHA not verified')
    report_name=next(name for name in names if name.endswith('_REPORT.md'))
    url='https://github.com/DLwbm123/Knowledge_editing-public/blob/'+commit+'/reports/alphaedit_stage17_20261003/'+report_name
    subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','30','--output',os.devnull,url],env=env,check=True)
    return dict(commit=commit,url=url)


def run(config):
    bundle=Path(config['bundle']);repository=Path(config['repository']);state=bundle/'FOLLOWER.json'
    def status(value,**kwargs):state.write_text(json.dumps(dict(status=value,updated_at=datetime.now(timezone.utc).isoformat(),**kwargs),indent=2)+'\n')
    try:
        approval=read(repository/'reports/alphaedit_stage17_20261003/SCORING_AUTHORIZATION.json')
        if approval['decision']!='APPROVED_BY_USER' or read(bundle/'operator/MANIFEST.json')['records']>approval['maximum_new_judge_keys']:
            raise ValueError('Cloud scoring authorization or record budget invalid')
        status('ASTRA_SCORING');run_judge(config)
        status('REPORTING');names=report(bundle,repository)
        status('PUBLISHING');receipt=publish(repository,names)
        status('COMPARISON_PUBLISHED',**receipt)
    except Exception as error:
        status('STOPPED_NO_AUTOMATIC_RETRY',error=repr(error));raise


if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
