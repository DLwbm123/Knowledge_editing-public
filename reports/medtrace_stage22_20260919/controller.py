"""Finite Stage22A Base -> Stage22B six-arm pipeline, using one cumulative ledger."""
import os,sys,json,time,shlex,fcntl,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring import judge
from report import report
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from scripts.medtrace.stage18_score import score_key,query_id


def run(cfg):
    root=Path(cfg['local_run']);p=root/'private';pub=root/'public';remote=cfg['remote_run'];lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=Path(cfg['Stage21B_publication_receipt']);wait_started=time.time()
    while not receipt.exists():
        write(pub/'CONTROLLER_STATUS.json',dict(phase='WAIT_STAGE21B_PUBLIC_DELIVERY',GPU_started=False))
        if time.time()-wait_started>14400:raise TimeoutError('B public closeout not available within four-hour handoff window')
        time.sleep(45)
    assert read(receipt)['anonymous_access_verified'] is True,'Close B and verify public delivery before next GPU stage'
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p','30177','root@hb01-ssh.gpuhome.cc'];transport=shlex.join(ssh[:-1])
    def pull():
        includes=('GPU_BUDGET_LEDGER.json','PROGRESS.json','GENERATED_*.json','EXIT_*.json','ARM_*.json')
        subprocess.run(['rsync','-a',*[f'--include={x}' for x in includes],'--exclude=*','-e',transport,ssh[-1]+':'+remote+'/public/',str(pub)+'/'],check=True)
        for name in ('DEV_BASE.json','OUTPUTS.jsonl'):
            exists=subprocess.run(ssh+['test -f '+shlex.quote(remote+'/private/'+name)],stdout=subprocess.DEVNULL).returncode==0
            if exists:subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/'+name,str(p/name)],check=True)
    def launch(phase):
        # Neutral visible argv, with a saved PID/exit check; never relaunch a crashed phase silently.
        script="""import os,json,subprocess
from pathlib import Path
root=Path(%r);phase=%r;run=root/'run';pidfile=run/'private'/('pid_'+phase);receipt=run/'public'/('EXIT_'+phase+'.json')
if not receipt.exists():
 if pidfile.exists():
  pid=int(pidfile.read_text())
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Prior phase lost its exit receipt; bounded audit required')
 else:
  c=run/'private'/('SUPERVISOR_'+phase+'.json');env=dict(os.environ,JOB_CONFIG=str(c),JOB_ENTRYPOINT=str(root/'entry.py'),JOB_ARGV=json.dumps([str(root/'code/reports/medtrace_stage22_20260919/budget.py')]))
  with (run/'private'/('worker_'+phase+'.log')).open('ab',buffering=0) as log:
   proc=subprocess.Popen([str(root/'env/bin/python'),str(root/'entry.py'),'main'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  pidfile.write_text(str(proc.pid)+'\\n')
"""%(cfg['remote_root'],phase)
        subprocess.run(ssh+['python -'],input=script,text=True,check=True)
    dev=read(p/'DEV_PANEL.json');launch('22A')
    while not (pub/'EXIT_22A.json').exists():pull();time.sleep(45)
    pull();assert read(pub/'EXIT_22A.json')['exit_code']==0
    base=read(p/'DEV_BASE.json')['records'];assert len(base)==66 and {r['query_id'] for r in base}=={query_id(r) for r in dev['natives']+dev['rows']}
    scores=judge(root,base,'22A_DEV_BASE',category='22A')
    assert all(score_key(r['source'],r['output']) in scores for r in base)
    acceptance=dict(DEV_binding=dev['binding'],Base_queries=66,source_agreement_protocol='Frozen',all_scored=True,selection_method_independent=True)
    write(pub/'BASE_SCORE_ACCEPTANCE.json',acceptance)
    subprocess.run(['rsync','-a','-e',transport,str(pub/'BASE_SCORE_ACCEPTANCE.json'),ssh[-1]+':'+remote+'/public/'],check=True)
    launch('22B')
    while True:
        pull();rows=outputs(root)
        for path in sorted(pub.glob('ARM_*.json')):
            arm=read(path)['arm'];name='22B_'+arm
            if not (p/(name+'_SCORE_RECEIPT.json')).exists():
                rr=[r for r in rows if r['arm']==arm];assert len(rr)==66;judge(root,rr,name,category='22B');report(root)
        write(pub/'CONTROLLER_STATUS.json',dict(phase='22B_RUNNING_OR_SCORING',generated_arms=[read(x)['arm'] for x in sorted(pub.glob('ARM_*.json'))]))
        if (pub/'EXIT_22B.json').exists():
            report(root)
            if read(pub/'EXIT_22B.json')['exit_code']!=0:raise RuntimeError('Preserve partial banks and charged budget; no blind restart')
            assert read(pub/'REPORT_STATUS.json')['all_six_scored'];write(pub/'CONTROLLER_STATUS.json',dict(phase='22B_COMPLETE_SCORED',next='Coverage qualification and single DEV-selected Stage23; follow-on ledger continues'));break
        time.sleep(45)

if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='NEEDS_ENGINEERING_REVIEW',error=str(e),automatic_semantic_retry=False));raise
