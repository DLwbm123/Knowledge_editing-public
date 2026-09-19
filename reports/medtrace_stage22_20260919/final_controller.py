"""Finite Stage23 handoff, one budget owner, insertion-first isolated scoring."""
import os,sys,json,time,shlex,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs,expected
from scripts.medtrace.stage18_score import score_key
from scoring import judge
from final_report import report


def run(cfg):
    d=Path(cfg['report_directory']);root=d/'private/run';p=root/'private';pub=root/'public'
    handle=(p/'final_controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);began=time.time()
    while read(pub/'C_CONTROLLER_STATUS.json')['phase']!='22C_COMPLETE_SCORED_STAGE23_LOCKED':
        if read(pub/'C_CONTROLLER_STATUS.json')['phase']=='NEEDS_ENGINEERING_REVIEW' or time.time()-began>14400:raise RuntimeError('Coverage stage incomplete; audit before final launch')
        write(pub/'FINAL_CONTROLLER_STATUS.json',dict(phase='WAIT_COVERAGE_AND_FROZEN_CANDIDATE',GPU_started=False));time.sleep(45)
    lock=read(d/'private/FINAL_STAGE23_LOCK.json');panel=read(d/'private/CONFIRM_QUALIFIED_FREEZE.json');ledger=read(pub/'BUDGET_LEDGER.json')
    assert ledger['23_items_dispatched']+lock['conservative_new_judgments']<=1200 and ledger['new_judgment_items_dispatched']+lock['conservative_new_judgments']<=2000
    # All historical Base labels are already complete, so no unreserved Base judging is required.
    scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];base={r['query_id']:r for r in read(p/'HISTORICAL_BASE.json')['records']};stream=read(p/'STREAM.json')
    assert all(type(scores.get(score_key(row,base[q]['output']))) is bool for q,row in expected(stream,45,final=True).items())
    supervisor=read(p/'SUPERVISOR_23_TEMPLATE.json');supervisor['final_lock_binding']=lock['binding'];write(p/'SUPERVISOR_23.json',supervisor)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p','30177','root@hb01-ssh.gpuhome.cc'];transport=shlex.join(ssh[:-1]);remote=cfg['remote_run']
    def copy(local,dest):subprocess.run(['rsync','-a','-e',transport,str(local),ssh[-1]+':'+dest],check=True,timeout=60)
    for local in [d/'private/FINAL_STAGE23_LOCK.json',d/'private/CONFIRM_QUALIFIED_FREEZE.json',p/'SUPERVISOR_23.json']:copy(local,remote+'/private/')
    # Only the finite, already authorized qualified images are copied; no new dataset access.
    original=read(d/'private/CONFIRM_CANDIDATE_FREEZE.json');groups=sorted({r['source_group'] for r in original['rows']})
    destinations={r['source_group']:r['image_path'] for r in panel['rows']}
    script='from pathlib import Path\nfor name in '+repr(list(destinations.values()))+':\n Path(name).parent.mkdir(parents=True,exist_ok=True)\n'
    subprocess.run(ssh+['python -'],input=script,text=True,check=True,timeout=30)
    # Use neutral staging aliases in rsync argv while preserving the frozen runtime image bindings.
    for group,dest in destinations.items():
        image=d/'private/confirm_reference_images'/f'image{groups.index(group):02d}.jpg'
        copy(image,remote+'/private/'+image.name)
    mapping={remote+'/private/'+f'image{groups.index(g):02d}.jpg':dest for g,dest in destinations.items()}
    script="""import os,json,subprocess,shutil
from pathlib import Path
root=Path(%r);code=Path(%r);p=root/'private';pidfile=p/'pid_23'
for source,target in %r.items():
 target=Path(target)
 if target.exists():assert target.stat().st_size==Path(source).stat().st_size
 else:shutil.copyfile(source,target)
assert json.loads((root/'public/EXIT_22C.json').read_text())['exit_code']==0
if not (root/'public/EXIT_23.json').exists():
 if pidfile.exists():
  try:os.kill(int(pidfile.read_text()),0)
  except ProcessLookupError:raise RuntimeError('Prior final worker lacks receipt; no blind resume')
 else:
  config=json.loads((p/'SUPERVISOR_23.json').read_text())
  gpu=subprocess.check_output(['nvidia-smi','--id='+str(config['gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert gpu[0].strip()==config['gpu_uuid'] and int(gpu[1].strip())>=20000
  env=dict(os.environ,JOB_CONFIG=str(p/'SUPERVISOR_23.json'),JOB_ENTRYPOINT=str(code/'entry.py'),JOB_ARGV=json.dumps([str(code/'code/reports/medtrace_stage22_20260919/budget.py')]))
  with (p/'worker_23.log').open('ab',buffering=0) as log:
   proc=subprocess.Popen([str(code/'env/bin/python'),str(code/'entry.py'),'main'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  pidfile.write_text(str(proc.pid)+'\\n')
"""%(remote,cfg['remote_code_root'],mapping)
    subprocess.run(ssh+['python -'],input=script,text=True,check=True,timeout=30)
    patterns=['GPU_BUDGET_LEDGER.json','PROGRESS_23.json','EXIT_23.json','GENERATED_23.json','ENVIRONMENT_23.json','PREFIX_23_*.json','REGRESSION_23_*.json','CONFIRM_23_*.json','CONFIRM_BASE_23_READY.json']
    scored={a:max([int(f.stem.split('_')[-3]) for f in p.glob(f'23_{a}_insert_*_SCORE_RECEIPT.json')],default=0) for a in lock['banks']}
    while True:
        subprocess.run(['rsync','-a',*[f'--include={x}' for x in patterns],'--exclude=*','-e',transport,ssh[-1]+':'+remote+'/public/',str(pub)+'/'],check=True,timeout=60)
        subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/OUTPUTS.jsonl',str(p/'OUTPUTS.jsonl')],check=True,timeout=60)
        rows=outputs(root)
        for a in lock['banks']:
            inserted=[r for r in rows if r['arm']=='R_'+a and r['mode']=='insertion']
            if len(inserted)>scored[a]:judge(root,inserted[scored[a]:],f'23_{a}_insert_{len(inserted):03d}',category='23');scored[a]=len(inserted)
            if len(inserted)==45:judge(root,inserted,f'23_{a}_insertions_COMPLETE',category='23')
            for n in reversed(lock['prefixes']):
                if (pub/f'PREFIX_23_{a}_{n:03d}.json').exists():
                    rr=[r for r in rows if r['arm']=='R_'+a and r['mode']=='endpoint' and r['prefix']==n]
                    assert len(rr)==len(expected(stream,n,final=n==45))
                    judge(root,rr,f'23_{a}_prefix_{n:03d}',category='23')
        if (pub/'CONFIRM_BASE_23_READY.json').exists():
            subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/CONFIRM_BASE_23.json',str(p/'CONFIRM_BASE_23.json')],check=True,timeout=60)
            judge(root,read(p/'CONFIRM_BASE_23.json')['records'],'23_CONFIRM_BASE',category='23')
            for a in lock['banks']:
                if (pub/f'CONFIRM_23_{a}.json').exists():
                    rr=[r for r in rows if r['arm']=='R_'+a and r['mode']=='CONFIRM'];assert len(rr)==len(panel['rows'])
                    judge(root,rr,'23_CONFIRM_'+a,category='23')
        result=report(d);write(pub/'FINAL_CONTROLLER_STATUS.json',dict(phase='23_RUNNING_OR_SCORING',insertions_scored=scored,highest_common_scored_N=result['highest_common_scored_N']))
        if (pub/'EXIT_23.json').exists():
            assert read(pub/'EXIT_23.json')['exit_code']==0,'Resource/failure stop: retain completed outputs and charge entire attempt'
            assert result['complete']
            write(pub/'FINAL_CONTROLLER_STATUS.json',dict(phase='23_COMPLETE_SCORED',result=result.get('engineering_gate'),publication='PENDING_ANALYSIS_AND_VERIFIED_PUBLIC_DELIVERY'));break
        time.sleep(45)


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['report_directory'])/'private/run/public/FINAL_CONTROLLER_STATUS.json',dict(phase='NEEDS_ENGINEERING_REVIEW',error=str(e),automatic_semantic_retry=False));raise
