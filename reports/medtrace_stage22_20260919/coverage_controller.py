"""Finite handoff: complete six-arm owner -> reference freeze -> H coverage -> Stage23 lock."""
import os,sys,json,time,shlex,subprocess,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from coverage_report import lock_weights,report,freeze_stage23
from qualify_confirm import execute as qualify
from scoring import judge


def run(cfg):
    directory=Path(cfg['report_directory']);root=directory/'private/run';p=root/'private';pub=root/'public'
    handle=(p/'coverage_controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    while read(pub/'CONTROLLER_STATUS.json')['phase']!='22B_COMPLETE_SCORED':
        status=read(pub/'CONTROLLER_STATUS.json')['phase']
        if status=='NEEDS_ENGINEERING_REVIEW' or time.time()-started>14400:raise RuntimeError('Prior six-arm stage incomplete; no automatic new stage')
        write(pub/'C_CONTROLLER_STATUS.json',dict(phase='WAIT_ALL_SIX_SCORED',GPU_started=False));time.sleep(45)
    assert read(pub/'EXIT_22B.json')['exit_code']==0 and read(pub/'REPORT_STATUS.json')['all_six_scored']
    budget=read(pub/'BUDGET_LEDGER.json')
    assert budget['22C_items_dispatched']+132<=180 and budget['22A_items_dispatched']+7<=100 and budget['new_judgment_items_dispatched']+139+1200<=2000
    lock=lock_weights(directory)
    if not (directory/'private/CONFIRM_QUALIFIED_FREEZE.json').exists():
        write(pub/'C_CONTROLLER_STATUS.json',dict(phase='QUALIFY_SEALED_REFERENCE_ONLY',GPU_started=False));qualify(directory)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p','30177','root@hb01-ssh.gpuhome.cc'];transport=shlex.join(ssh[:-1]);remote=cfg['remote_run']
    def copy(local,dest):subprocess.run(['rsync','-a','-e',transport,str(local),ssh[-1]+':'+dest],check=True,timeout=60)
    copy(pub/'COVERAGE_WEIGHT_LOCK.json',remote+'/public/')
    copy(directory/'private/COVERAGE_POOL_FREEZE.json',remote+'/private/')
    copy(p/'SUPERVISOR_22C.json',remote+'/private/')
    script="""import os,json,subprocess
from pathlib import Path
root=Path(%r);code=Path(%r);p=root/'private';pidfile=p/'pid_22C'
assert json.loads((root/'public/EXIT_22B.json').read_text())['exit_code']==0
if not (root/'public/EXIT_22C.json').exists():
 if pidfile.exists():
  pid=int(pidfile.read_text())
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Prior coverage worker lacks exit receipt; audit before recovery')
 else:
  config=json.loads((p/'SUPERVISOR_22C.json').read_text())
  gpu=subprocess.check_output(['nvidia-smi','--id='+str(config['gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert gpu[0].strip()==config['gpu_uuid'] and int(gpu[1].strip())>=20000,'Unexpected GPU or insufficient free VRAM; leave other processes untouched'
  env=dict(os.environ,JOB_CONFIG=str(p/'SUPERVISOR_22C.json'),JOB_ENTRYPOINT=str(code/'entry.py'),JOB_ARGV=json.dumps([str(code/'code/reports/medtrace_stage22_20260919/budget.py')]))
  with (p/'worker_22C.log').open('ab',buffering=0) as log:
   proc=subprocess.Popen([str(code/'env/bin/python'),str(code/'entry.py'),'main'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  pidfile.write_text(str(proc.pid)+'\\n')
"""%(remote,cfg['remote_code_root'])
    subprocess.run(ssh+['python -'],input=script,text=True,check=True,timeout=30)
    include=['GPU_BUDGET_LEDGER.json','PROGRESS.json','EXIT_22C.json','GENERATED_22C.json','H_COVERAGE_AUDIT.json','ARM_S*.json']
    while True:
        subprocess.run(['rsync','-a',*[f'--include={x}' for x in include],'--exclude=*','-e',transport,ssh[-1]+':'+remote+'/public/',str(pub)+'/'],check=True,timeout=60)
        subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/OUTPUTS.jsonl',str(p/'OUTPUTS.jsonl')],check=True,timeout=60)
        rows=outputs(root)
        for arm in ('S1','S2'):
            if (pub/f'ARM_{arm}.json').exists() and not (p/f'22C_{arm}_SCORE_RECEIPT.json').exists():
                rr=[r for r in rows if r['arm']==arm and r['mode']=='DEV'];assert len(rr)==66;judge(root,rr,'22C_'+arm,category='22C')
        value=report(directory);write(pub/'C_CONTROLLER_STATUS.json',dict(phase='22C_RUNNING_OR_SCORING',weights=lock['weights'],complete_arms=list(value['arms'])))
        if (pub/'EXIT_22C.json').exists():
            assert read(pub/'EXIT_22C.json')['exit_code']==0,'Coverage partial: retain all completed arms, no blind retry'
            assert value['complete']
            subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/H_SELECTION_FREEZE.json',str(p/'H_SELECTION_FREEZE.json')],check=True,timeout=60)
            final=freeze_stage23(directory)
            write(pub/'C_CONTROLLER_STATUS.json',dict(phase='22C_COMPLETE_SCORED_STAGE23_LOCKED',promotion=value['promotion'],stage23_banks=final['banks'],next='Implement/deploy exact45 regression and once-only qualified probe evaluation; no extra budget'))
            break
        time.sleep(45)


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['report_directory'])/'private/run/public/C_CONTROLLER_STATUS.json',dict(phase='NEEDS_ENGINEERING_REVIEW',error=str(e),automatic_semantic_retry=False));raise
