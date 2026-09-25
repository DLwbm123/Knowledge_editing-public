"""Dependency-ordered background queue with hard wall/GPU limits and score gates."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(ROOT))
from metrics import job_report

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def epoch(s):return datetime.fromisoformat(s).timestamp()
M=read(ROOT/'RUN_MANIFEST.json');END=epoch(M['deadline_at']);TRAIN_END=epoch(M['no_new_training_after'])

def budget():
    if time.time()>=END or (ROOT/'STOP').exists():raise TimeoutError('Hard wall or explicit STOP')
    if (ROOT/'JUDGE_FAILURE.json').exists():raise RuntimeError('Judge transport failed; no automatic semantic retry')
    d=read(ROOT/'RESOURCE_LEDGER.json');used=d['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if s.get('ended_epoch') is None)
    if used>=57600:raise TimeoutError('16 GPU resident hour limit')
    pending={p.stem for p in (ROOT/'private/judge/pending').glob('*.json')}
    scored={p.stem for p in (ROOT/'private/judge/scores').glob('*.json')}
    if pending-scored:
        latest=max((p.stat().st_mtime for p in (ROOT/'private/judge/evidence').glob('*.json')),default=time.time())
        if time.time()-latest>1800:raise RuntimeError('No Judge progress for 30 minutes; save state and release GPU')
    return used

def wait_scores():
    began=time.time()
    while True:
        budget()
        pending={p.stem for p in (ROOT/'private/judge/pending').glob('*.json')}
        scores={p.stem for p in (ROOT/'private/judge/scores').glob('*.json')}
        missing=len(pending-scores)
        write(ROOT/'SCORE_COVERAGE.json',dict(expected=len(pending),scored=len(scores&pending),missing=missing,epoch=time.time()))
        if not missing:return
        if time.time()-began>1800:raise TimeoutError('Scoring did not close within 30 minute reserved window')
        time.sleep(10)

def wait_existing():
    receipt=read(ROOT/'BASE_PID.json');pid=receipt['pid'];status=ROOT/'jobs'/receipt.get('job_id','base48')/'STATUS.json'
    while True:
        budget();d=read(status) if status.exists() else {}
        if d.get('status')=='GPU_COMPLETE':break
        if d.get('status')=='FAILED':raise RuntimeError('Base job failed: '+d.get('error',''))
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Base job exited without completion receipt')
        time.sleep(10)
    wait_scores()
    # Only same-round accepted scores create the Base masks, before screen candidates.
    scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')}
    tasks=read(ROOT/'private/TASKS_MATRIX.json')['tasks'];masks=[]
    from worker_v3 import request,input_id
    for t in tasks:
        for row in t['evaluation']:
            out=read(ROOT/'private/base'/f'{input_id(row)}.json');key=request(row,out)
            if key not in scores:raise ValueError('Base mask missing')
            masks.append(dict(edit=t['canonical_edit_id'],task=row['task'],query_id=row['query_id'],base_correct=scores[key],judge_key=key))
    write(ROOT/'private/BASE_MASKS.json',dict(status='FROZEN_BEFORE_SCREEN',rows=masks))

def reconcile_owned_child(pid):
    import fcntl
    with (ROOT/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);d=read(ROOT/'RESOURCE_LEDGER.json')
        for session in d['gpu_sessions']:
            if session['pid']==pid and session.get('ended_epoch') is None:
                elapsed=time.time()-session['started_epoch']
                session.update(ended_epoch=time.time(),resident_seconds=elapsed,closure='owned child exit confirmed by controller')
                d['gpu_seconds_used']+=elapsed
        if all(s.get('ended_epoch') is not None for s in d['gpu_sessions']):d['status']='GPU_SESSION_CLOSED'
        write(ROOT/'RESOURCE_LEDGER.json',d)


def launch(job):
    existing=ROOT/'jobs'/job['id']/'STATUS.json'
    if existing.exists():
        state=read(existing)
        if state['job']!=job:raise ValueError('Existing job binding differs')
        if state['status']=='RUNNING':
            active=read(ROOT/'ACTIVE_JOB.json')
            if active['job']!=job:raise ValueError('Cannot adopt a different active job')
            pid=active['pid']
            try:
                while state['status']=='RUNNING':
                    budget();os.kill(pid,0);time.sleep(5);state=read(existing)
            except BaseException:
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
                # Reconcile after termination only; normal receipts close their own session.
                raise
        if state['status']!='GPU_COMPLETE':raise RuntimeError('Existing job has no successful completion; no training retry')
        wait_scores();reports=job_report(ROOT);write(ROOT/'private/JOB_RESULTS.json',reports)
        return reports[job['id']]
    budget();write(ROOT/'JOB.json',job);write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=job['id'],job=job,epoch=time.time()))
    log=ROOT/'logs'/f"{job['id']}.log"
    with log.open('x') as stream:
        process=subprocess.Popen([str(ROOT/'env/bin/python'),str(ROOT/'matrix_entry.py')],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,cwd=ROOT)
    write(ROOT/'ACTIVE_JOB.json',dict(job=job,pid=process.pid,started_epoch=time.time(),log=str(log)))
    try:
        while process.poll() is None:
            budget();time.sleep(5)
    except BaseException:
        # Signal only this controller's own process group; preserve written checkpoints.
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        reconcile_owned_child(process.pid)
        raise
    reconcile_owned_child(process.pid)
    status=read(ROOT/'jobs'/job['id']/'STATUS.json')
    if process.returncode or status['status']!='GPU_COMPLETE':raise RuntimeError('Job failed: '+job['id']+' '+status.get('error',''))
    wait_scores();reports=job_report(ROOT);write(ROOT/'private/JOB_RESULTS.json',reports)
    return reports[job['id']]

def make_job(name,mode,n,arms,seed=20260924,**kw):return dict(id=name,mode=mode,N=n,arms=arms,seed=seed,**kw)

def choose(report):
    choices=[]
    for comp in report['comparisons']:
        if comp['status']!='COMPLETE':continue
        metrics=comp['metrics'];g=metrics.get('T2G',{}).get('delta_edit_macro');l=metrics.get('T2L',{}).get('delta_edit_macro')
        if g is None or l is None:continue
        summary={s['task']:s for s in report['summary'] if s['arm']==comp['arm'] and s['mode']=='single'}
        native=summary.get('T0',{}).get('accuracy')
        t1=metrics.get('T1G',{}).get('delta_edit_macro')
        locality=metrics.get('T1L',{}).get('right_to_wrong')
        safe=native is not None and native>=.99 and t1 is not None and t1>=-.01 and locality==0
        choices.append(dict(arm=comp['arm'],joint_score=min(g/.02,l/.08),T2G_delta=g,T2L_delta=l,
            engineering_safe=safe,goal_reached=safe and g>=.02 and l>=.08))
    # Fixed simplicity tie order; an exploratory reference never becomes a validated winner.
    preference=['P','E','U','EU','PEU']
    choices.sort(key=lambda c:(not c['engineering_safe'],-c['joint_score'],preference.index(c['arm'])))
    write(ROOT/'CANDIDATE_SELECTION.json',dict(candidates=choices,rule='complete paired edit-macro joint score, engineering safety first, fixed simplicity tie break',
        status='EXPLORATORY_ONLY',independent_confirmation=False))
    if not choices:raise RuntimeError('No complete joint T2G/T2L denominator: cannot rank candidates')
    return choices[0]['arm']

def forecast(arm,n):
    # Actual per-edit training/initialization measurements, with a 30% safety margin.
    measured={}
    for a in ['B0',arm]:
        times=[read(p)['seconds'] for p in (ROOT/'runs/s20260924'/a).glob('e*/TRAINING_COMPLETE.json')]
        if not times:return float('inf')
        measured[a]=sum(times)/len(times)
    return max(0,n-12)*sum(measured.values())*1.3

def effective_seconds():
    return sum(read(p).get('effective_work_seconds',0) for p in (ROOT/'jobs').glob('*/STATUS.json'))


def finalize(status,error=None,best=None,n=12):
    reports=job_report(ROOT);write(ROOT/'private/JOB_RESULTS.json',reports)
    public=ROOT/'public';public.mkdir(exist_ok=True)
    sanitized={k:dict(status=v['status']['status'],job=v['status']['job'],summary=v['summary'],comparisons=[{x:y for x,y in c.items() if x!='changes'} for c in v['comparisons']],consumers=v['consumers']) for k,v in reports.items()}
    write(public/'RESULTS.json',sanitized)
    from metrics import summarize
    scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')}
    diagnostic=[]
    for p in (ROOT/'jobs/diagnostic12').glob('e*.json'):
        diagnostic += [dict(x,arm=x['stage'],prefix=1) for x in read(p)]
    write(public/'STAGE_DIAGNOSTICS.json',dict(summary=summarize(diagnostic,scores),forced_on_is_diagnostic_only=True,
        generated_length_by_stage={stage:dict(count=len(v),mean_tokens=sum(len(x['output']['raw_token_ids']) for x in v)/len(v))
          for stage in sorted({x['stage'] for x in diagnostic}) if (v:=[x for x in diagnostic if x['stage']==stage])}))
    ledger=read(ROOT/'RESOURCE_LEDGER.json');active=[s for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None]
    receipt=dict(status=status,error=error,best_exploratory_arm=best,N=n,independent_confirmation=False,
        elapsed_seconds=time.time()-epoch(M['first_started_at']),gpu_resident_seconds=ledger['gpu_seconds_used'],open_gpu_sessions=len(active),
        judge_items_charged=ledger['judge_submission_attempt_items'],judge_limit=6000,
        effective_queue_work_seconds=effective_seconds(),desired_12h_effective_queue_met=effective_seconds()>=43200,
        effective_time_caveat='Measured worker time after model load includes real training, generation and necessary IO; scoring waits and model loading excluded; not GPU-kernel busy time',
        claim='EXPLORATORY_NO_INDEPENDENT_CONFIRMATION',optional_controls='LoRA-Perf and supervision-matched +Aug are deferred unless the main paired queue leaves sufficient budget')
    if best:
        point=ROOT/'jobs'/f'paired-sequential{n}'/best/'ACTIVE_BANK.pt'
        if not point.exists():point=ROOT/'runs/s20260924'/best/'e001/FINAL.pt'
        if point.exists():receipt['representative_checkpoint']=dict(path=str(point),sha256=hashlib.sha256(point.read_bytes()).hexdigest(),scope='complete deployment bank' if point.name=='ACTIVE_BANK.pt' else 'first-edit checkpoint only')
    write(ROOT/'FINAL.json',receipt)
    safe={k:v for k,v in receipt.items() if k not in ['representative_checkpoint','error']};safe['error_type']=status if error else None
    write(public/'RESOURCE_SUMMARY.json',dict(gpu_resident_seconds=ledger['gpu_seconds_used'],sessions=[{k:v for k,v in x.items() if k in ['started_epoch','ended_epoch','resident_seconds']} for x in ledger['gpu_sessions']],Judge_attempts=[{k:v for k,v in x.items() if k in ['items','status']} for x in ledger['judge_attempts']],physical_requests=ledger['physical_requests']))
    write(public/'FINAL.json',safe);write(public/'CONFIG_LOCK.json',read(ROOT/'CONFIG_LOCK.json'))
    tasks=read(ROOT/'private/TASKS_MATRIX.json')['tasks']
    write(public/'DATA_ROLES.json',dict(native_available=len(tasks),screen=12,expanded_actual=n,CONFIRM=None,exposed_DEV=True,
        support_QA=29,support_source_groups=3,U_counts=[dict(edit_order=t['order'],**t['U_counts']) for t in tasks],
        fit_P='four reviewed native-only semantic rewrites; no evaluation text collisions',evaluation='held out from training, previously exposed DEV; image+text modality retained'))
    # Curves are private numeric artifacts; no medical question/answer fields enter the public table.
    curves=[]
    for p in (ROOT/'runs').glob('s*/*/e*/TRAINING_CURVES.json'):
        data=read(p)
        for stage,values in data.items():
            for v in values:curves.append(dict(seed=p.parts[-4],arm=p.parts[-3],edit_order=p.parts[-2],stage=stage,**{k:v.get(k) for k in ['step','native_ce','fit_ce','U_kl','grad_norm','weighted_native_gradient_norm','weighted_fit_gradient_norm','weighted_U_gradient_norm']}))
    if curves:
        with (public/'TRAINING_CURVES.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    lines=['# 本轮联合优化实验报告','',f'状态：{status}。本轮固定 L30/rank4，Stage17 L21 仅作为历史参考。',
        '旧146面板是已暴露DEV；没有独立CONFIRM，不构成独立验证或临床结论。',
        f'最佳开发参考臂：{best or "未锁定"}；实际配对规模：{n}。',
        f'GPU驻留 {ledger["gpu_seconds_used"]/3600:.3f} 小时；Judge保守计费项 {ledger["judge_submission_attempt_items"]}/6000。',
        'GPU驻留包含模型加载，不等于全部有效训练；没有将CPU等待计入有效实验。',
        '六臂 U 扩展记录实际数量和困难样本缺额；固定隔离辅助池仅3个图像来源，不能声称达到8个独立来源。',
        '结果主项是冻结同轮Base掩码后的edit-macro；micro、分子分母、missing、来源组和配对CI见RESULTS.json。输出token一致性单独列出。',
        '保留全部已完成正负结果。FORCED_ON只用于诊断。目标阈值不等于预期或已实现提升。','']
    if error:lines+=['中止原因见私有 FINAL.json；不自动重跑失败任务。','']
    for name,v in sanitized.items():
        lines += [f'## {name}',f'运行状态：{v["status"]}；生成消费者：{v["consumers"]}。','',
            '|臂|模式|前缀|任务|分子/分母|edit-macro|micro|missing|','|---|---|---:|---|---|---:|---:|---:|']
        for x in v['summary']:
            if x['mode']=='sequential' and x['prefix']!=v['job']['N']:continue
            pct=lambda z:'NA' if z is None else f'{z*100:.2f}%'
            lines.append(f"|{x['arm']}|{x['mode']}|{x['prefix']}|{x['task']}|{x['numerator']}/{x['denominator']}|{pct(x['edit_macro'])}|{pct(x['micro'])}|{x['missing']}|")
        lines+=['']
    (public/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines))
    write(ROOT/'RUN_STATUS.json',dict(status=status,phase='FINAL_REPORT_WRITTEN',best=best,N=n,epoch=time.time()))


def main():
    best=None;n=12
    try:
        wait_existing()
        canary=read(ROOT/'CANARY.json')
        estimate=12*(canary['initial_seconds']+canary['continuation_seconds'])*sum([1,1,1.35,1.7,2.5,2.5])*1.3
        write(ROOT/'BUDGET_FORECAST.json',dict(canary_seconds=canary['seconds'],six_arm_conservative_seconds=estimate,
            available_training_seconds=TRAIN_END-time.time(),GPU_limit_seconds=57600,score_report_reserve_fraction=.25))
        launch(make_job('b0-12','single',12,['B0']))
        launch(make_job('diagnostic12','diagnostic',12,[]))
        screen=dict(comparisons=[],summary=[]);matrix=[]
        baseline_times=[read(p)['seconds'] for p in (ROOT/'runs/s20260924/B0').glob('e*/TRAINING_COMPLETE.json')]
        per_edit=sum(baseline_times)/len(baseline_times)
        for arm,multiplier in [('P',1),('E',1.35),('U',1.7),('EU',2.5),('PEU',2.5)]:
            predicted=12*per_edit*multiplier*1.3
            if time.time()+predicted+600>=TRAIN_END:
                matrix.append(dict(arm=arm,N=12,status='DEFERRED_TIME_BUDGET',predicted_seconds=predicted))
                write(ROOT/'SCREEN_MATRIX_STATUS.json',matrix);continue
            report=launch(make_job(f'screen12-{arm.lower()}','single',12,[arm],baseline_job='b0-12'))
            screen['comparisons']+=report['comparisons'];screen['summary']+=report['summary']
            matrix.append(dict(arm=arm,N=12,status='COMPLETE',predicted_seconds=predicted))
            write(ROOT/'SCREEN_MATRIX_STATUS.json',matrix);write(ROOT/'private/SCREEN_RESULTS.json',screen)
        best=choose(screen)
        proposals=[(48,forecast(best,48)),(36,forecast(best,36))]
        tasks=read(ROOT/'private/TASKS_MATRIX.json')['tasks']
        ledger=read(ROOT/'RESOURCE_LEDGER.json')
        remaining_scores=ledger['judge_submission_attempt_items_limit']-ledger['judge_submission_attempt_items']
        score_bounds={target:2*sum((target-i)*len(t['evaluation']) for i,t in enumerate(tasks[:target]))+2*sum(len(t['evaluation']) for t in tasks[12:target]) for target,_ in proposals}
        write(ROOT/'EXPANSION_FORECAST.json',dict(proposals=proposals,available=TRAIN_END-time.time(),best=best,conservative_Judge_upper_bounds=score_bounds,remaining_Judge_items=remaining_scores))
        for target,cost in proposals:
            if time.time()+cost+600<TRAIN_END and score_bounds[target]<=remaining_scores:
                launch(make_job(f'expand{target}','single',target,['B0',best]));n=target;break
        write(ROOT/'METHOD_LOCK.json',dict(best=best,N=n,layer=30,rank=4,locked_epoch=time.time(),status='EXPLORATORY_REFERENCE_NOT_VALIDATED'))
        launch(make_job(f'paired-sequential{n}','sequential',n,['B0',best]))
        if time.time()+1800<TRAIN_END:
            launch(make_job(f'baseline-single{n}','single',n,['BalancEdit'],baseline_job=f'expand{n}' if n>12 else 'b0-12'))
            launch(make_job(f'baseline-sequential{n}','sequential',n,['BalancEdit'],baseline_job=f'paired-sequential{n}'))
        # New seed if it fits, otherwise order diagnostics reuse frozen experts without training replay.
        unit=forecast(best,36)/24*12/1.3
        for index,seed in enumerate([20260925,20260926,20260927],2):
            if time.time()+unit*1.5+1200>=TRAIN_END or effective_seconds()>=43200:break
            launch(make_job(f'seed{index}-single12','single',12,['B0',best],seed=seed))
            launch(make_job(f'seed{index}-sequential12','sequential',12,['B0',best],seed=seed))
        seq_seconds=read(ROOT/'jobs'/f'paired-sequential{n}'/'STATUS.json')['seconds']
        for label,options in [('reverse',dict(reverse=True)),('order2',dict(order_seed=20260925)),('order3',dict(order_seed=20260926)),('order4',dict(order_seed=20260927))]:
            if time.time()+seq_seconds*1.3+900>=epoch(M['target_at']):break
            if label!='reverse' and effective_seconds()>=43200:break
            launch(make_job(f'{label}-sequential{n}','sequential',n,['B0',best],**options))
        wait_scores();finalize('COMPLETE',best=best,n=n)
    except Exception as e:
        write(ROOT/'CONTROLLER_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),epoch=time.time()))
        finalize('BUDGET_STOP' if isinstance(e,TimeoutError) else 'BLOCKED',str(e),best,n)

if __name__=='__main__':main()
