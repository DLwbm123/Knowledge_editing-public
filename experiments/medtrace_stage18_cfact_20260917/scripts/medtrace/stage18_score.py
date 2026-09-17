"""Receipt-bound DEV scoring; reuse exact frozen Base decisions, judge new texts once."""
from pathlib import Path
import math
from statistics import mean

from scripts.medtrace.astra_judge_bundle import read, write_new, validate
from scripts.medtrace.stage17_prepare import digest, PROTOCOL
from scripts.medtrace.stage18_dev import prefixes

METHODS=('C_NO_H','C_EXTRA','C_FACT','BalancEdit')
MODES=('Base','R0','RC','FORCED_OWN')


def query_id(row):return digest([row['image_sha256'],row['question']])


def score_key(row,out):
    return digest([row['image_sha256'],row['question'],row['reference'],out['raw_answer']])


def load_run(run):
    cfg=read(run/'private/DISPATCH.json');gate=read(run/'private/DEV_QUALIFICATION.json')
    training=read(run/'private/TRAINING_TASKS.json');packet=read(run/'private/DEV_EVALUATION.json')
    base=read(run/'private/QUALIFICATION_BASE_OUTPUTS.json');done=read(run/'public/DEV_RESULT.json')
    if (read(run/'public/EXIT.json')['exit_code']!=0 or done['status']!='FOUR_METHODS_GENERATED_NOT_SCORED'
            or digest(training['tasks'])!=training['freeze_id'] or cfg['freeze_id']!=training['freeze_id']
            or gate['training_freeze']!=training['freeze_id'] or digest(packet)!=gate['evaluation_binding']
            or digest(base)!=gate['base_binding']):raise ValueError('Incomplete or changed run')
    tasks=training['tasks'];packages=packet['candidate_packages'];ids=[t['canonical_edit_id'] for t in tasks]
    if len(tasks)!=16 or [p['candidate_id'] for p in packages]!=ids or ids!=sorted(ids):raise ValueError('Cohort/order changed')
    base_by_id={r['query_id']:r for r in base['records']};outputs=[];diagnostics=[];curves=[]
    panel=lambda ps:{query_id(r):r for p in ps for r in [p['training']['native']]+p['evaluation']}
    for method in METHODS:
        rows=read(run/'private'/f'{method}_EVALUATION.json')['records'];expected={}
        for i,p in enumerate(packages,1):
            for q,r in panel([p]).items():expected['single',1,(ids[i-1],),q]=r
            pp=packages[:i] if i in prefixes(len(tasks)) else [dict(p,evaluation=[])]
            for q,r in panel(pp).items():expected['sequential',i,tuple(ids[:i]),q]=r
        actual={}
        for r in rows:
            k=(r['mode'],r['prefix'],tuple(r['inserted']),r['query_id'])
            if k in actual or k not in expected:raise ValueError('Duplicate/unexpected evaluation')
            actual[k]=r
            if r['method']!=method or r['code']!=cfg['code_commit'] or r['source']!=expected[k]:raise ValueError('Output binding mismatch')
            b=base_by_id[r['query_id']]['output'];route=r['route']
            if r['Base']['binding']!=b['binding'] or r['Base']['raw_token_ids']!=b['raw_token_ids']:raise ValueError('Base drift')
            if route['nearest_logical_edit_id'] not in r['inserted'] or route['activated']!=(route['nearest_distance']<=route['radius']):raise ValueError('Invalid route')
            for mode in MODES:
                o=r.get(mode)
                required=mode!='FORCED_OWN' or method!='BalancEdit' and r['mode']=='single'
                if required and o is None:raise ValueError('Missing realized output')
                if o is not None and (o['binding']!=b['binding'] or not isinstance(o['raw_answer'],str) or not isinstance(o['raw_token_ids'],list)):raise ValueError('Output/input changed')
            if not route['activated'] and r['R0']['raw_token_ids']!=b['raw_token_ids']:raise ValueError('Rejected route is not Base')
            rc=r['R0'] if route['nearest_distance']<=route['radius']*.7696741135364367 else r['Base']
            if r['RC']['raw_token_ids']!=rc['raw_token_ids']:raise ValueError('RC substitution mismatch')
        if set(actual)!=set(expected):raise ValueError('Evaluation panel incomplete')
        outputs+=rows
    for task in tasks:
        d=run/'private/edits'/f"e{task['order']:03d}";receipt=read(d/'COMPLETE.json');reference=None;branches={}
        if any(receipt.get(k) is not True for k in ('CP_transfer','zero_residual','Base_OFF','Base_route_isolation','save_load')):raise ValueError('Missing mechanical proof')
        if read(d/'BINDING.json')!=dict(task=task,code=cfg['code_commit'],runtime=cfg['runtime_lock'],generation=cfg['generation_lock']):raise ValueError('Training binding changed')
        for method in METHODS[:3]:
            v=read(d/method/'TRAINING.json');curve=v['curve'];branches[method]=v
            if v['steps']!=320 or [x['step'] for x in curve]!=list(range(1,321)) or v['W0']!=receipt['W0']:raise ValueError('Training steps/W0 changed')
            sig=[(x['fit_index'],*[x['terms'][k]['sample'] for k in ('native','fit','U')]) for x in curve]
            if reference is not None and sig!=reference:raise ValueError('Branch schedule changed')
            reference=sig
            for x in curve:
                weights=dict(native=.5,fit=.5,U=.01,**({} if method=='C_NO_H' else dict(extra=1.)))
                if set(x['terms'])!=set(weights):raise ValueError('Wrong loss terms')
                for k,w in weights.items():
                    term=x['terms'][k]
                    if term['weight']!=w or not math.isfinite(term['unweighted']) or not math.isclose(term['weighted'],w*term['unweighted'],abs_tol=1e-9):raise ValueError('Invalid weighted loss')
                    if not math.isfinite(term['weighted_gradient_norm']) or term['tokens']<=0 or k=='extra' and term['weighted_gradient_norm']<=0:raise ValueError('Missing effective supervision')
            curves.append(v)
        if [x['extra_index'] for x in branches['C_FACT']['curve']]!=[x['extra_index'] for x in branches['C_EXTRA']['curve']]:raise ValueError('H/G slot mismatch')
        ds=read(d/'DIAGNOSTICS.json')
        if len(ds)!=12:raise ValueError('Diagnostic coverage')
        diagnostics.extend(dict(x,edit=task['canonical_edit_id']) for x in ds)
    if done['output_records']!=len(outputs):raise ValueError('Output count changed')
    return cfg,packages,base,outputs,diagnostics,curves


def prepare(run,qualification):
    cfg,packages,base,outputs,diagnostics,curves=load_run(run)
    old=read(qualification/'BASE_JUDGE_V2/operator/VERDICTS.json')
    if old['source_binding']!=digest(base) or old['protocol']!=PROTOCOL:raise ValueError('Base Judge lineage changed')
    batch=dict(batch_id='reuse',records=[dict(opaque_query_id=digest(r)) for r in base['records']])
    scores={v['opaque_query_id']:v['is_correct'] for v in validate(batch,dict(batch_id='reuse',decisions=old['decisions']))}
    cached={score_key(r['source'],r['output']):dict(correct=scores[digest(r)],base_opaque_id=digest(r)) for r in base['records']}
    novel={};uses=0
    for r in outputs+diagnostics:
        for mode in MODES:
            o=r.get(mode)
            if o is None:continue
            key=score_key(r['source'],o);uses+=1
            if key not in cached:novel.setdefault(key,dict(query_id=key,source=r['source'],output=dict(raw_answer=o['raw_answer'])))
    source=dict(scope='DEV_STUDENT_AND_TRAINING_DIAGNOSTICS',records=[novel[k] for k in sorted(novel)],
        run_code=cfg['code_commit'],training_freeze=cfg['freeze_id'])
    write_new(run/'private/STUDENT_JUDGE_INPUTS.json',source)
    write_new(run/'private/SCORE_CACHE.json',dict(Base_source_binding=digest(base),Base_verdict_binding=digest(old),cached=cached,
        new_source_binding=digest(source),new_texts=len(novel),output_uses=uses,
        reuse='Exact image identity, question, reference and answer text only; all decisions use frozen Astra protocol'))
    acceptance=dict(status='GPU_OUTPUT_AND_C_TRAINING_VALIDATED',N=len(packages),C_branches=len(curves),
        output_records=len(outputs),diagnostic_records=len(diagnostics),H_nonzero_updates=16*320,G_nonzero_updates=16*320,
        steps=sum(v['steps'] for v in curves),new_Judge_texts=len(novel),code=cfg['code_commit'])
    write_new(run/'public/ACCEPTANCE.json',acceptance)
    return acceptance


def aggregate(rows):
    from scripts.medtrace.stage17_contract import correctness
    result={}
    for role in ('native','H_eval','U_eval'):
        selected=[r for r in rows if r['role']==role]
        panel={name:correctness(selected,name) for name in ('postacc','retention','fix')}
        panel['Base_postacc']=correctness([dict(r,post_correct=r['base_correct']) for r in selected],'postacc')
        panel['output_agreement']=correctness([dict(r,post_correct=r['agreement']) for r in selected],'postacc')
        panel['route_activation']=correctness([dict(r,post_correct=r['on']) for r in selected],'postacc')
        panel['unique_QA']=len({r['query_id'] for r in selected});panel['unique_images']=len({r['image'] for r in selected})
        retention=panel['retention'];panel['c2w_micro']=None if retention['probe_micro'] is None else 1-retention['probe_micro']
        result[role]=panel
    result['PairCorrect']=correctness([r for r in rows if r['role']=='H_eval'],'paircorrect')
    return result


def report(run):
    from scripts.medtrace.stage17_report import interval
    cfg,packages,base,outputs,diagnostics,curves=load_run(run)
    cache=read(run/'private/SCORE_CACHE.json');source=read(run/'private/STUDENT_JUDGE_INPUTS.json')
    be=read(run/'private/BE_TRAINING_RECEIPTS.json')
    if ([r['task'] for r in be]!=[p['candidate_id'] for p in packages]
            or any(not r['training']['finite_losses'] or not r['training']['finite_gradients'] or r['training']['steps']!=50 for r in be)
            or cache['Base_source_binding']!=digest(base)):raise ValueError('BE/Base evidence changed')
    verdicts=read(run/'STUDENT_JUDGE/operator/VERDICTS.json');evidence=read(run/'STUDENT_JUDGE/operator/execution_evidence/b000.json')
    if (digest(source)!=cache['new_source_binding'] or verdicts['source_binding']!=digest(source)
            or verdicts['protocol']!=PROTOCOL or evidence['status']!='FORMAT_VALID' or evidence['exit_code']!=0
            or evidence['tool_event_types'] or not all(evidence['isolation_checks'].values())):raise ValueError('Student Judge incomplete/changed')
    batch=dict(batch_id='scores',records=[dict(opaque_query_id=digest(r)) for r in source['records']])
    decisions={v['opaque_query_id']:v['is_correct'] for v in validate(batch,dict(batch_id='scores',decisions=verdicts['decisions']))}
    scores={k:v['correct'] for k,v in cache['cached'].items()}
    for r in source['records']:
        k=score_key(r['source'],r['output'])
        if k in scores:raise ValueError('Previously judged text was judged again')
        scores[k]=decisions[digest(r)]
    score=lambda row,o:scores[score_key(row,o)]
    lookup={(r['method'],r['mode'],r['prefix'],r['inserted'][0] if r['mode']=='single' else '',r['query_id']):r for r in outputs}
    def collect(method,mode,deploy,prefix):
        result=[]
        for p in packages[:prefix]:
            edit=p['candidate_id'];native=p['training']['native']
            get=lambda row:lookup[method,mode,1 if mode=='single' else prefix,edit if mode=='single' else '',query_id(row)]
            nr=get(native);nc=score(native,nr[deploy])
            for row in [native]+p['evaluation']:
                r=get(row);out=r[deploy]
                on=False if deploy=='Base' else True if deploy=='FORCED_OWN' else r['route']['activated'] if deploy=='R0' else r['route']['nearest_distance']<=r['route']['radius']*.7696741135364367
                result.append(dict(edit_id=edit,role=row['role'],query_id=query_id(row),image=row['image_sha256'],base_correct=score(row,r['Base']),
                    post_correct=score(row,out),native_correct=nc,on=on,agreement=' '.join(out['raw_answer'].lower().split())==' '.join(r['Base']['raw_answer'].lower().split())))
        return result
    panels=[];expanded=[]
    for method in ('Base',)+METHODS:
        for mode in ('single','sequential'):
            for prefix in ([len(packages)] if mode=='single' else prefixes(len(packages))):
                for deploy in (['Base'] if method=='Base' else ['R0','RC']+(['FORCED_OWN'] if mode=='single' and method!='BalancEdit' else [])):
                    rows=collect('C_NO_H' if method=='Base' else method,mode,deploy,prefix)
                    panels.append(dict(method=method,mode=mode,prefix=prefix,deployment=deploy,metrics=aggregate(rows)))
                    expanded+= [dict(r,method=method,mode=mode,prefix=prefix,deployment=deploy) for r in rows]
    pair_values={}
    for method in METHODS:
        rows=collect(method,'sequential','R0',len(packages));pair_values[method]={}
        for p in packages:
            rs=[r for r in rows if r['edit_id']==p['candidate_id'] and r['role']=='H_eval']
            pair_values[method][p['candidate_id']]=mean(int(r['native_correct'] and r['post_correct']) for r in rs)
    groups={p['candidate_id']:p['training']['native']['source_group'] for p in packages};contrasts=[]
    for other in ('C_NO_H','C_EXTRA','BalancEdit'):
        delta={e:v-pair_values[other][e] for e,v in pair_values['C_FACT'].items()}
        contrasts.append(dict(contrast='C_FACT minus '+other,metric='Final R0 PairCorrect',delta=mean(delta.values()),
            improved=sum(v>0 for v in delta.values()),worsened=sum(v<0 for v in delta.values()),tied=sum(v==0 for v in delta.values()),**interval(delta,groups)))
    # Training diagnostics remain separate from the unseen-source evaluation panels.
    fit=[]
    for method in METHODS[:3]:
        for role in ('native','H_fit','G_fit','U_fit'):
            for deploy in ('Base','R0','RC','FORCED_OWN'):
                rs=[r for r in diagnostics if r['branch']==method and r['role']==role]
                values=[score(r['source'],r[deploy]) for r in rs]
                fit.append(dict(method=method,role=role,deployment=deploy,correct=sum(values),N=len(values),accuracy=mean(values)))
    trajectory=[]
    for method in METHODS:
        for index,p in enumerate(packages,1):
            row=p['training']['native'];q=query_id(row)
            insertion=lookup[method,'sequential',index,'',q];final=lookup[method,'sequential',len(packages),'',q]
            trajectory.append(dict(method=method,ordinal=index,inserted_correct=score(row,insertion['R0']),final_correct=score(row,final['R0']),
                selected_expert_changed=insertion['route']['nearest_logical_edit_id']!=final['route']['nearest_logical_edit_id']))
    anchors=[]
    anchor_ids={p['candidate_id'] for p in packages[:1]}
    for method in METHODS:
        for prefix in prefixes(len(packages)):
            rs=[r for r in collect(method,'sequential','R0',prefix) if r['edit_id'] in anchor_ids]
            anchors.append(dict(method=method,prefix=prefix,anchor_edits=1,metrics=aggregate(rs)))
    receipts=[read(run/'private/edits'/f"e{p['training']['order']:03d}"/'COMPLETE.json') for p in packages]
    result=dict(status='DEV16_SCORED_EXPLORATORY',N=16,formal_eligible=False,methods=list(METHODS),panels=panels,
        primary_contrasts=contrasts,training_diagnostics=fit,native_insert_to_final=trajectory,early_anchors=anchors,
        judge=dict(model='gpt-6-astra',reasoning_effort='high',snapshot=None,protocol=PROTOCOL,new_texts=len(decisions),
            reused_exact_Base_texts=len({score_key(r['source'],r[m]) for r in outputs+diagnostics for m in MODES if r.get(m) is not None}&set(cache['cached'])),
            semantic_retries=0,execution_seconds=None,usage=evidence.get('usage')),
        statistics=dict(bootstrap=10000,seed=20260912,unit='edit',native_source_groups=len(set(groups.values())),
            intervals='Descriptive edit and native-image cluster resampling; shared support remains dependent; not patient-independent confidence',
            all_support_connection='All 16 share the same U_fit source: one connected training-support component; component/patient CI unavailable'),
        limits=['Patient/study independence unknown; AI image review is not physician review','Prior training exposure allowed only for versioned DEV',
            'H_eval has seven unique QA on four images reused in eighteen slots; U_eval one QA/image',
            'No independent T1G/T2G/T1L/T2L probes added; these metrics remain unavailable',
            'Semantic native-target copying was not separately scored under the source-agreement protocol; no substring proxy used',
            'No extra insertion orders or independent training seeds'],
        execution=dict(code=cfg['code_commit'],training_freeze=cfg['freeze_id'],exit_code=0,
            wall_seconds=read(run/'public/EXIT.json')['finished']-read(run/'private/WORKER.json')['started'],
            C_initialization_continuation_diagnostics_seconds=sum(r['seconds'] for r in receipts),
            C_continuation_seconds={m:sum(r['session_seconds'] for r in curves if r['branch']==m) for m in METHODS[:3]},
            C_peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in receipts)/1024**3,
            BE_edits=len(be),BE_steps=sum(r['training']['steps'] for r in be),BE_finite=True,
            timing_limits='BE training, insertion, route and I/O not independently timed; recorded generation seconds overlap reused outputs; no invented decomposition'),
        completed_output_records=len(outputs),failures=0)
    from datetime import datetime
    result['judge']['execution_seconds']=(datetime.fromisoformat(evidence['completed_at_utc'])-datetime.fromisoformat(evidence['started_at_utc'])).total_seconds()
    write_new(run/'private/EXPANDED_SCORED_ROWS.json',expanded)
    write_new(run/'public/RESULTS.json',result)
    return result
