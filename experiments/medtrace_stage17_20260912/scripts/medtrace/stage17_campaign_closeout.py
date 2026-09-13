"""Finite local follower: collect small outputs, one Astra queue, report and publish.

GPU training never depends on this local process remaining alive. Transport/Judge
failures stop here without restarting completed computation or accepted judging.
"""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from scripts.medtrace.astra_judge_bundle import read, write_new, schema
from scripts.medtrace.stage17_prepare import digest, lines, PROMPT, PROTOCOL
from scripts.medtrace.stage17_freeze import accepted
from scripts.medtrace.stage17_campaign import prefixes, schedule, query_ids, role_map
from scripts.medtrace.stage17_report import metric, interval


def prepare(bundle, base_bundle, be_bundle):
    source=bundle/'source'
    # Relocated runtime ledger is the authority for output bindings.
    cfg=read(source/'campaign/private/DISPATCH.json')
    ledger=read(source/'COHORT_AND_SUPPORT_LEDGER.json')
    if digest({k:v for k,v in ledger.items() if k!='freeze_id'})!=cfg['freeze_id']:
        raise ValueError('Cohort mismatch')
    lock=read(base_bundle/'operator/JUDGE_LOCK.json')
    bb=read(base_bundle/'operator/BINDINGS.json')
    c0=accepted(bb,lines(base_bundle/'operator/VERDICTS_ASTRA.jsonl'))
    if c0!=ledger['Base_correctness']: raise ValueError('Base mask mismatch')
    tasks={t['edit_id']:t for t in ledger['tasks']}; bindings={}; mapping=[]
    roles=role_map(ledger)
    if read(source/'campaign/private/PREFIX_ACTIVE_TARGETS.json')!=roles:
        raise ValueError('Prefix role map mismatch')

    def add(output, method, mode, prefix, edit, panel, qid, values, off):
        q=ledger['queries'][qid]; b=bb[q['opaque_Base_id']]
        if output['binding']['input']!=q or output['Base_cache_id']!=q['opaque_Base_id']:
            raise ValueError('Output/query binding mismatch')
        if output['binding']['generation']!=b['generation']: raise ValueError('Generation mismatch')
        if (q['question'],q['reference'],q['image_sha256'])!=(b['question'],b['reference'],b['image_sha256']):
            raise ValueError('Base/realized query identity mismatch')
        refs={'original':q['reference']}
        if mode=='sequential' and str(prefix) in roles and qid in roles[str(prefix)]:
            refs['active']=roles[str(prefix)][qid]
        for route,value in values.items():
            if (set(value)!={'raw_answer','raw_token_ids'} or not isinstance(value['raw_answer'],str)
                    or not isinstance(value['raw_token_ids'],list) or len(value['raw_token_ids'])>1024
                    or any(type(t) is not int or t<0 for t in value['raw_token_ids'])):
                raise ValueError('Raw output contract mismatch')
            for role,reference in refs.items():
                if route in off and reference==q['reference']:
                    if value!=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids']):
                        raise ValueError('OFF is not exact bound Base output')
                    oid=q['opaque_Base_id']; origin='Base'
                else:
                    full=dict(query_id=qid,question=b['question'],reference=reference,
                        image_path=b['image_path'],image_sha256=b['image_sha256'],prompt_ids=b['prompt_ids'],
                        attention_mask=b['attention_mask'],output=dict(model_answer_raw=value['raw_answer'],
                        raw_generated_token_ids=value['raw_token_ids']),runtime=cfg['runtime_lock'],
                        generation=b['generation'],training=output['binding'],judge=lock,
                        source_lineage=b['source_lineage'],Base_cache_id=q['opaque_Base_id'])
                    oid=digest(full); origin='student'; bindings[oid]=full
                mapping.append(dict(method=method,mode=mode,prefix=prefix,edit=edit,panel=panel,
                    query_id=qid,route=route,role=role,source=origin,opaque_query_id=oid,
                    agreement=' '.join(value['raw_answer'].lower().split())==' '.join(b['output']['model_answer_raw'].lower().split())))

    for eid in ledger['main_T0']:
        t=tasks[eid]; directory=source/'cnoh/private/edits'/f"e{t['order']:03d}"
        receipt=read(directory/'COMPLETE.json')
        if receipt['status']!='GENERATED_NOT_SCORED' or receipt['binding']['input']!=t['native']:
            raise ValueError('C_NO_H incomplete')
        ids=query_ids(t)
        if receipt['queries']!=len(ids): raise ValueError('C_NO_H coverage mismatch')
        for i,qid in enumerate(ids):
            o=read(directory/f'query_{i:03d}.json')
            if o['binding']!=dict(input=ledger['queries'][qid],runtime=cfg['runtime_lock'],
                    generation=receipt['binding']['generation'],writer='C_NO_H',prefix=1,
                    training_binding=digest(receipt['binding'])):
                raise ValueError('C_NO_H output/training binding mismatch')
            add(o,'C_NO_H','single',1,eid,'panel',qid,{m:o[m] for m in ('R0','RC','FORCED_ON')},
                {m for m,on in o['route_on'].items() if not on})
    for method,mode in schedule():
        root=source/'campaign/private'/f'{method}_{mode}'
        receipt=read(root/'COMPLETE.json')
        if receipt['status']!='GENERATED_NOT_SCORED' or receipt['N']!=cfg['N']:
            raise ValueError('Phase incomplete')
        for index,eid in enumerate(ledger['main_T0'],1):
            directory=root/f'e{index:03d}'
            groups={'native':[eid]}
            if mode=='single' or index in prefixes(cfg['N']):
                es=ledger['main_T0'][:index] if mode=='sequential' else [eid]
                groups['panel']=list(dict.fromkeys(q for e in es for q in query_ids(tasks[e])))
            for panel,ids in groups.items():
                actual=list((directory/panel).glob('*.json'))
                if len(actual)!=len(ids): raise ValueError('Generated query count mismatch')
                for qid in ids:
                    o=read(directory/panel/(digest(qid)+'.json')); off=set()
                    expected_inserted=ledger['main_T0'][:index] if mode=='sequential' else [eid]
                    if (o['binding']['phase']!=receipt['phase'] or o['binding']['inserted']!=expected_inserted
                            or o['binding']['prefix']!=(index if mode=='sequential' else 1)):
                        raise ValueError('Sequential prefix/training binding mismatch')
                    if method in ('C_NO_H','balancedit'):
                        route=o['route']
                        if not route['activated']: off.add('R0')
                        if route['nearest_distance']>route['radius']*.7696741135364367: off.add('RC')
                    add(o,method,mode,index if mode=='sequential' else 1,eid,panel,qid,o['modes'],off)
    operator=bundle/'operator'; operator.mkdir(); (bundle/'judge_only').mkdir()
    write_new(operator/'JUDGE_LOCK.json',lock); write_new(operator/'BINDINGS.json',bindings)
    write_new(operator/'MODE_MAPPING.json',mapping)
    visible=[dict(opaque_query_id=k,question=v['question'],gold_answer=v['reference'],
        raw_base_answer=v['output']['model_answer_raw']) for k,v in bindings.items()]
    batches=[]
    for start in range(0,len(visible),50):
        name=f'batch_{start//50+1:03d}'; batch=dict(batch_id=name,records=visible[start:start+50])
        write_new(bundle/'judge_only'/f'{name}.input.json',batch)
        write_new(bundle/'judge_only'/f'{name}.schema.json',schema(batch))
        (bundle/'judge_only'/f'{name}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
        batches.append(dict(batch_id=name,count=len(batch['records'])))
    write_new(operator/'MANIFEST.json',dict(protocol=PROTOCOL,config_sha256=lock['config_sha256'],
        records=len(visible),batches=batches,freeze_id=cfg['freeze_id'],status='READY',
        scope='remaining main T0 cohort single and sequential; BE single accepted separately',N=cfg['N']))


def report(bundle,base_bundle,be_bundle):
    ledger=read(bundle/'source/COHORT_AND_SUPPORT_LEDGER.json'); c0=ledger['Base_correctness']
    tasks={t['edit_id']:t for t in ledger['tasks']}; groups={e:tasks[e]['native']['source_group'] for e in ledger['main_T0']}
    op=bundle/'operator'
    if read(op/'EXECUTION_RECORD.json')['status']!='COMPLETE_FORMAT_AND_COVERAGE_VALIDATED':
        raise ValueError('Judge incomplete')
    verdicts=lines(op/'VERDICTS_ASTRA.jsonl'); bounds=read(op/'BINDINGS.json')
    if len(verdicts)!=len(bounds) or {v['opaque_query_id'] for v in verdicts}!=set(bounds): raise ValueError('Judge coverage mismatch')
    score={v['opaque_query_id']:v['is_correct'] for v in verdicts}
    if any(type(v) is not bool for v in score.values()): raise ValueError('Invalid verdict')
    records=read(op/'MODE_MAPPING.json'); lookup={}
    for r in records:
        r['correct']=c0[r['query_id']] if r['source']=='Base' else score[r['opaque_query_id']]
        key=(r['method'],r['mode'],r['prefix'],r['edit'] if r['mode']=='single' else None,r['panel'],r['query_id'],r['route'],r['role'])
        if key in lookup and lookup[key]!=r: raise ValueError('Conflicting duplicate output')
        lookup[key]=r
    # Existing BE single scoring is reused from its accepted fixed queue, never rejudged.
    bescores={v['opaque_query_id']:v['is_correct'] for v in lines(be_bundle/'operator/VERDICTS_ASTRA.jsonl')}
    for r in read(be_bundle/'operator/MODE_MAPPING.json'):
        for route,ref in r['modes'].items():
            lookup['balancedit','single',1,r['edit_id'],'panel',r['query_id'],route,'original']=dict(
                correct=c0[r['query_id']] if ref['source']=='Base' else bescores[ref['opaque_query_id']],agreement=None)
    rows=[]; roles=role_map(ledger)
    for method in ('C_NO_H','balancedit','lora','grace','belora'):
        for mode in ('single','sequential'):
            for prefix in ([1] if mode=='single' else prefixes(len(ledger['main_T0']))):
                ids=ledger['main_T0'] if mode=='single' else ledger['main_T0'][:prefix]
                routes=('R0','RC','FORCED_ON') if method in ('C_NO_H','balancedit') and mode=='single' else ('R0','RC') if method in ('C_NO_H','balancedit') else ('NATIVE',)
                for eid in ids:
                    for event in tasks[eid]['events']:
                        for qid in event['all_probe_query_ids']:
                            active=mode=='sequential' and qid in roles[str(prefix)]
                            for route in routes:
                                key=(method,mode,prefix,eid if mode=='single' else None,'panel',qid,route,'original')
                                value=lookup[key]
                                rows.append(dict(method=method,mode=mode,prefix=prefix,edit=eid,task=event['task'],route=route,
                                    base=c0[qid],correct=value['correct'],agreement=value['agreement'],active_target=active))
    panels=[]; macro={}
    keys=sorted({(r['method'],r['mode'],r['prefix'],r['task'],r['route']) for r in rows})
    for method,mode,prefix,task,route in keys:
        allrows=[r for r in rows if (r['method'],r['mode'],r['prefix'],r['task'],r['route'])==(method,mode,prefix,task,route)]
        selected=[r for r in allrows if not (task.endswith('L') and r['active_target'])]
        primary,values=metric(selected,'correct',lambda r:r['base'] if task.endswith('L') else not r['base'])
        primary.update(interval(values,groups)); macro[method,mode,prefix,task]=values if route in ('R0','NATIVE') else macro.get((method,mode,prefix,task),{})
        panels.append(dict(method=method,mode=mode,prefix=prefix,task=task,route=route,
            primary_metric='Retention' if task.endswith('L') else 'Fix',primary=primary,
            post_accuracy=metric(selected,'correct')[0],active_locality_exclusions=len(allrows)-len(selected),
            c2w=None if not task.endswith('L') or primary['micro'] is None else 1-primary['micro']))
    paired=[]
    for method,mode,prefix,task in sorted(macro):
        if method=='C_NO_H': continue
        ref=macro.get(('C_NO_H',mode,prefix,task),{}); values=macro[method,mode,prefix,task]
        common=set(ref)&set(values); delta={e:ref[e]-values[e] for e in common}
        paired.append(dict(contrast='C_NO_H minus '+method,mode=mode,prefix=prefix,task=task,
            edits=len(common),macro_delta=sum(delta.values())/len(delta) if delta else None,**interval(delta,groups)))
    destination=bundle/'public'; destination.mkdir()
    trajectories=[]
    for method in ('C_NO_H','balancedit','lora','grace','belora'):
        route='R0' if method in ('C_NO_H','balancedit') else 'NATIVE'
        counts=defaultdict(int)
        for index,eid in enumerate(ledger['main_T0'],1):
            before=lookup[method,'sequential',index,None,'native',eid,route,'original']['correct']
            after=lookup[method,'sequential',len(ledger['main_T0']),None,'panel',eid,route,'original']['correct']
            counts[f'{int(before)}_to_{int(after)}']+=1
        trajectories.append(dict(method=method,counts=dict(counts)))
    costs={f'{m}_{mode}':read(bundle/'source/campaign/private'/f'{m}_{mode}'/'COMPLETE.json') for m,mode in schedule()}
    # Only numeric counters and elapsed costs are public; private phase bindings stay private.
    costs={k:{f:v for f,v in r.items() if f in ('N','seconds','replay','training_reused','peak_allocated_bytes','peak_reserved_bytes')} for k,r in costs.items()}
    result=dict(status='MAIN_T0_COHORT_REPORTED_NOT_YET_PUBLISHED',N=len(ledger['main_T0']),panels=panels,paired=paired,
        insertion_to_final=trajectories,costs=costs,
        judge=dict(model='gpt-6-astra',reasoning='high',new_records=len(verdicts),immutable_snapshot=None),
        scope='frozen main T0 queue with attached supported tasks; one fixed sequence order',
        unsupported='C_FACT/C_EXTRA main cohort lacks H/G; other task-specific support and clinical review remain separate',
        limitations=['not paper exact','not a complete ten-task benchmark','source group sensitivity is not full patient independence',
            'active-target locality excluded per predeclared map; original-target outputs and active-reference verdicts retained privately',
            'no order robustness claim','historical Stage15/16 labels unchanged'])
    write_new(destination/'CAMPAIGN_RESULTS.json',result)
    with (destination/'RESULTS.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','mode','prefix','task','route','primary_metric','numerator','probes','edits','micro','macro'])
        writer.writeheader()
        for p in panels: writer.writerow({**{k:p[k] for k in ('method','mode','prefix','task','route','primary_metric')},
            **{k:p['primary'][k] for k in ('numerator','probes','edits','micro','macro')}})
    (destination/'GPT_PRO_REVIEW.md').write_text('# Stage17 frozen main-cohort campaign\n\n'
        'Completed N=146 main-cohort single/sequential panels for C_NO_H, BalancEdit, LoRA-Perf-v1, GRACE and BELoRA. '
        'C_NO_H/BE sequential are independent-checkpoint insertion replays with new full-bank generation; other sequential methods update their native state. '
        'See CAMPAIGN_RESULTS.json and RESULTS.csv for counts, uncertainty and paired comparisons. '
        'This is not a full ten-task or paper-exact result. Unsupported H/G and additional task-specific cohorts are not filled with zeros. '
        'Existing BE single and Base Astra verdicts were reused; historical Stage15/16 and Qwen results were not mixed.\n')


def publish(cfg,bundle):
    release=Path(cfg['public_checkout'])
    def git(*args):
        return subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897',
            '-c','http.version=HTTP/1.1',*args],cwd=release,text=True,timeout=120).strip()
    if git('status','--porcelain'): raise RuntimeError('Public checkout is dirty; do not overwrite concurrent changes')
    if git('branch','--show-current')!='main' or git('remote','get-url','origin')!='https://github.com/DLwbm123/Knowledge_editing-public.git':
        raise ValueError('Public branch/remote mismatch')
    subprocess.run(['gh','auth','status'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    destination=release/'reports/medtrace_stage17_20260912/campaign_closeout'
    if destination.exists(): raise FileExistsError('Public report already exists; inspect before retry')
    destination.mkdir()
    for name in ('CAMPAIGN_RESULTS.json','RESULTS.csv','GPT_PRO_REVIEW.md'):
        shutil.copyfile(bundle/'public'/name,destination/name)
    git('add',str(destination)); git('commit','-m','Report frozen Stage17 main-cohort single and sequential campaign')
    commit=git('rev-parse','HEAD'); git('push','origin','main')
    if git('ls-remote','origin','refs/heads/main').split()[0]!=commit: raise RuntimeError('Remote commit not verified')
    url='https://github.com/DLwbm123/Knowledge_editing-public/blob/'+commit+'/reports/medtrace_stage17_20260912/campaign_closeout/GPT_PRO_REVIEW.md'
    with urllib.request.urlopen(url,timeout=30) as response:
        if response.status!=200: raise RuntimeError('Anonymous public access not verified')
    write_new(bundle/'public/PUBLICATION_RECEIPT.json',dict(status='PUBLISHED',commit=commit,branch='main',url=url))
    return url


def run(cfg):
    bundle=Path(cfg['bundle']); status=bundle/'FOLLOWER.json'
    def state(**values): status.write_text(json.dumps(values,indent=2)+'\n')
    state(status='WAITING_FOR_GPU_CAMPAIGN')
    try:
        ssh=['ssh','-S',cfg['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=15','-p','30270',cfg['host']]
        while True:
            response=subprocess.check_output([*ssh,'cat '+cfg['remote_campaign']+'/public/PROGRESS.json'],text=True,timeout=30)
            progress=json.loads(response)
            if progress['status']=='GPU_GENERATED_NOT_SCORED': break
            if progress['status'] not in ('RUNNING','WAITING_FOR_EXISTING_C_NO_H'):
                raise RuntimeError('GPU campaign stopped: '+str(progress))
            time.sleep(60)
        state(status='COLLECTING_SMALL_OUTPUTS')
        source=bundle/'source'; source.mkdir()
        transport=' '.join(ssh[:-1])
        for remote,name in ((cfg['remote_campaign'],'campaign'),(cfg['remote_cnoh'],'cnoh')):
            subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--exclude=*',
                '-e',transport,cfg['host']+':'+remote+'/',str(source/name)+'/'],check=True)
        subprocess.run([*ssh,'cat '+cfg['remote_source']+'/private/COHORT_AND_SUPPORT_LEDGER.json'],
            stdout=(source/'COHORT_AND_SUPPORT_LEDGER.json').open('wb'),check=True)
        base=Path(cfg['base_bundle']); be=Path(cfg['be_bundle'])
        prepare(bundle,base,be); state(status='ASTRA_SCORING')
        from scripts.medtrace.stage17_judge import run as judge
        judge(dict(bundle=str(bundle),repository=cfg['repository'],cli=cfg['cli'],explicit_proxy=True))
        state(status='REPORTING'); report(bundle,base,be)
        try:
            url=publish(cfg,bundle); state(status='MAIN_COHORT_PUBLISHED',url=url)
        except Exception as error:
            state(status='REPORTED_PUBLICATION_PENDING',report=str(bundle/'public'),error=repr(error))
    except Exception as error:
        state(status='STOPPED_NO_COMPUTE_RETRY',error=repr(error)); raise


if __name__=='__main__': run(read(os.environ['JOB_CONFIG']))
