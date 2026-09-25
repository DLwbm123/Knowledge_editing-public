"""Persisted, budget-bound dependency queue. No scientific retries or clock resets."""
import json,os,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(ROOT))
from report_r2 import read,write,report,choose
MANIFEST=read(ROOT/'RUN_MANIFEST.json')
DEADLINE=datetime.fromisoformat(MANIFEST['deadline_at']).timestamp()
TRAIN_STOP=datetime.fromisoformat(MANIFEST['no_new_training_after']).timestamp()
SEED=20260925

def resources():
    d=read(ROOT/'RESOURCE_LEDGER.json')
    return d,d['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if s.get('ended_epoch') is None)

def guard(training=False,reserve=False):
    if time.time()>= (TRAIN_STOP if training else DEADLINE-600):raise TimeoutError('Persisted wall-clock stop')
    d,used=resources()
    if used>= (43200 if reserve else 57000):raise TimeoutError('GPU reserve/hard stop')
    if d['judge_submission_attempt_items']>=6000:raise TimeoutError('Judge attempts exhausted')
    if (ROOT/'STOP').exists():raise RuntimeError('Explicit STOP')
    if any((ROOT/n).exists() for n in ['JUDGE_FAILURE.json','JUDGE_FAILURE_RECOVERY.json']):raise RuntimeError('Judge technical failure; evidence retained')

def launch(job,gpu):
    guard(training=job['mode'] in ['single','baseline'],reserve=job.get('reserve',False))
    path=ROOT/'jobs'/job['id'];path.mkdir(parents=True,exist_ok=True)
    write(path/'JOB.json',job)
    receipt=path/'STATUS.json'
    if receipt.exists():
        if read(receipt)['status']=='GPU_COMPLETE':return job['id']
        raise RuntimeError('Existing unfinished receipt requires explicit recovery')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    parts=next(x.split(',') for x in lines if int(x.split(',')[0])==gpu)
    if gpu not in [2,3] or int(parts[2])<20000:raise RuntimeError('Authorized GPU unavailable')
    env=dict(os.environ,RUN_ROOT=str(ROOT),RUN_JOB_JSON=str(path/'JOB.json'),CUDA_VISIBLE_DEVICES=parts[1].strip(),
             PINNED_GPU_UUID=parts[1].strip(),TMPDIR=str(ROOT/'tmp'),HF_HOME=str(ROOT/'cache'),OMP_NUM_THREADS='4',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with (ROOT/'logs'/f'{job["id"]}.log').open('a') as log:
        p=subprocess.Popen(['/root/anaconda3/bin/python','/tmp/r2x.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/f'ACTIVE_GPU{gpu}.json',dict(pid=p.pid,job=job['id'],gpu_index=gpu,started_epoch=time.time()))
    return job['id']

def wait_jobs(ids):
    while True:
        guard()
        states=[read(ROOT/'jobs'/i/'STATUS.json') if (ROOT/'jobs'/i/'STATUS.json').exists() else {} for i in ids]
        if any(s.get('status')=='FAILED' for s in states):raise RuntimeError('Worker failure: '+str([(i,s.get('error')) for i,s in zip(ids,states) if s.get('status')=='FAILED']))
        if all(s.get('status')=='GPU_COMPLETE' for s in states):return
        time.sleep(15)

def score_wait(ids):
    from report_r2 import collect
    keys={r[k] for r in collect(ids) for k in ['judge_key','base_judge_key']}
    blocked=set(read(ROOT/'private/judge/JUDGE_MISSING_LOCK.json')['keys'])
    while True:
        guard()
        missing={k for k in keys if not (ROOT/'private/judge/scores'/f'{k}.json').exists()}
        if not missing-blocked:return
        time.sleep(15)

def job(name,orders,arms,mode='single',**kw):
    return dict(id=name,orders=list(orders),arms=arms,mode=mode,seed=SEED,**kw)

def cross_gate():
    import torch
    rows=[]
    for order in [1,2]:
        a=torch.load(ROOT/'jobs/cross-g2'/f'e{order:03d}.pt',weights_only=True)
        b=torch.load(ROOT/'jobs/cross-g3'/f'e{order:03d}.pt',weights_only=True)
        delta=max(float((x-y).abs().max()) for x,y in zip(a['grads'],b['grads']))
        relative=(sum(float((x-y).double().square().sum()) for x,y in zip(a['grads'],b['grads']))/max(1e-30,sum(float(x.double().square().sum()) for x in a['grads'])))**.5
        ok=abs(a['loss']-b['loss'])<=.01 and relative<=.05 and a['base_frozen'] and b['base_frozen'] and a['eos_in_edit_target'] and b['eos_in_edit_target']
        rows.append(dict(order=order,CE_abs_delta=abs(a['loss']-b['loss']),grad_max_abs_delta=delta,grad_relative_l2=relative,
                         same_generation=a['tokens']==b['tokens'],status='PASS' if ok else 'FAIL'))
    write(ROOT/'public/CROSS_GPU_FLOOR.json',rows)
    if any(r['status']!='PASS' for r in rows):raise RuntimeError('Cross-GPU numerical floor gate failed')
    for order in [1,2]:
        base=ROOT/'runs'/f's{SEED}'/'P+S'/f'e{order:03d}'
        if read(base/'OLD_CALLBACK_EQUIVALENCE.json')['status']!='PASS' or read(base/'SHARED_P_W0.json')['status']!='PASS':raise RuntimeError('P+S equivalence gate failed')

def phase(name):write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=name,epoch=time.time()))

def main():
    dev=['r2-canary-bp'];verify=[];seq=[];baseline=[]
    try:
        phase('CANARY_DEPENDENCIES');wait_jobs(dev)
        import hashlib
        stage=ROOT/'source/scripts/medtrace/stage15.py'
        write(ROOT/'private/CANARY_SOURCE_VERSION.json',dict(stage15_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),note='Initial BP canary before additional diagnostics-only logging and configurable VERIFY stopping'))
        stage.write_bytes((ROOT/'stage15_next.py').read_bytes())
        ps=launch(job('r2-canary-ps',[1,2],['P+S']),3);wait_jobs([ps]);dev.append(ps)
        cross=[launch(job('cross-g2',[1,2],[],mode='crosscheck'),2),launch(job('cross-g3',[1,2],[],mode='crosscheck'),3)]
        wait_jobs(cross);cross_gate();score_wait(dev)
        canary=report(dev,'CANARY')
        if canary['missing_judge_keys'] or canary['missing_base_keys']:raise RuntimeError('Canary scoring closure incomplete')
        bp_seconds=read(ROOT/'jobs/r2-canary-bp/STATUS.json')['seconds']
        ps_seconds=read(ROOT/'jobs/r2-canary-ps/STATUS.json')['seconds']
        predicted_dev_remaining=11*(bp_seconds+ps_seconds)
        d,used=resources()
        write(ROOT/'public/BUDGET_FORECAST.json',dict(measured_canary_BP_two_edits_seconds=bp_seconds,measured_PS_two_edits_seconds=ps_seconds,
            DEV_remaining_GPU_seconds=predicted_dev_remaining,DEV_gpu_reserve_cap=43200,global_GPU_cap=57600,
            planned_DEV_consumers=1876,VERIFY_consumers_max=855,sequential_consumers_max=855,baseline_consumers=285,
            counting='Consumers are upper bounds; execution-input and identical Judge payload dedup counted separately',scoring_report_reserve_fraction=.25))
        if used+predicted_dev_remaining>43200:raise TimeoutError('Measured DEV throughput would violate validation reserve')
        phase('DEV24_PAIRED_SCREEN')
        bp=[launch(job('dev-bp-a',range(3,13),['B0','P'],reserve=True),2),launch(job('dev-bp-b',range(13,25),['B0','P'],reserve=True),3)]
        wait_jobs(bp);dev+=bp
        ps=[launch(job('dev-ps-a',range(3,13),['P+S'],reserve=True),3),launch(job('dev-ps-b',range(13,25),['P+S'],reserve=True),2)]
        wait_jobs(ps);dev+=ps;score_wait(dev)
        result=report(dev,'DEV24');selection=choose(result);write(ROOT/'SELECTION_LOCK.json',selection);write(ROOT/'public/SELECTION_LOCK.json',selection)
        phase('VERIFY24_LOCKED_RECIPE')
        candidate=selection['arm'];step=selection['step']
        verify_arms=['B0','P']+(['P+S'] if candidate=='P+S' else [])
        verify=[launch(job('verify-a',range(25,37),verify_arms,steps=[step],train_steps=step),2),launch(job('verify-b',range(37,49),verify_arms,steps=[step],train_steps=step),3)]
        wait_jobs(verify)
        phase('VERIFY24_SPARSE_SEQUENTIAL')
        seq=[launch(job('seq-b0',range(25,49),['B0'],mode='sequential',prefixes=[12,24],selected_step=320),2),
             launch(job('seq-candidate',range(25,49),[candidate],mode='sequential',prefixes=[12,24],selected_step=step),3)]
        wait_jobs(seq);score_wait(verify+seq)
        report(verify,'VERIFY24_SINGLE');report(seq,'VERIFY24_SEQUENTIAL')
        d,used=resources()
        # Standard baseline has a fixed 285-consumer ceiling; no score-dependent expansion.
        if used<43200 and time.time()<TRAIN_STOP-3600 and d['judge_submission_attempt_items']+285<=6000:
            phase('STANDARD_BALANCEDIT')
            baseline=[launch(job('balancedit',range(25,49),['BalancEdit'],mode='baseline'),2)]
            wait_jobs(baseline);score_wait(baseline);report(verify+baseline,'VERIFY24_BASELINE')
        write(ROOT/'RUN_STATUS.json',dict(status='COMPLETE',phase='REPORT_READY',epoch=time.time(),DEV_jobs=dev,VERIFY_jobs=verify,sequential_jobs=seq,baseline_jobs=baseline,
            note='Exploratory. No independent CONFIRM. Optional extra seeds omitted unless budget-qualified separate protocol. Publication pending local review.'))
    except Exception as e:
        (ROOT/'STOP').write_text('Controller stopped: preserve latest checkpoints and original budgets\n')
        write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',phase='PRESERVED_FOR_REVIEW',error=str(e),traceback=traceback.format_exc(),epoch=time.time(),DEV_jobs=dev,VERIFY_jobs=verify,sequential_jobs=seq,baseline_jobs=baseline))
        raise
    finally:
        from report_r2 import final_report
        final_report()

if __name__=='__main__':main()
