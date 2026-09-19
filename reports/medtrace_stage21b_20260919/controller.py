"""Finite B-generation consumer; recurring follow-up is owned by app heartbeat."""
import os,sys,json,time,shlex,fcntl,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring import judge
from report import report
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs,expected


def run(cfg):
    root=Path(cfg['local_run']);p=root/'private';pub=root/'public';remote=cfg['remote_run']
    lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',cfg['ssh_socket'],'-p','30177','root@hb01-ssh.gpuhome.cc'];transport=shlex.join(ssh[:-1]);stream=read(p/'STREAM.json')
    def pull(has_outputs=True):
        subprocess.run(['rsync','-a',*[f'--include={pattern}' for pattern in ('GPU_BUDGET_LEDGER.json','PROGRESS.json','ENVIRONMENT_ACCEPTANCE.json','PREFIX_*.json','INTEGRATION_*.json','GENERATED.json','EXIT_*.json')],'--exclude=*','-e',transport,ssh[-1]+':'+remote+'/public/',str(pub)+'/'],check=True)
        for name in (('OUTPUTS.jsonl',) if has_outputs else ()):
            subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/'+name,str(p/name)],check=True)
    completed=[int(x.stem.split('_')[2]) for x in p.glob('insertions_*_SCORE_RECEIPT.json')]
    scored=max(completed,default=0)
    while True:
        # One low-cost status request per pass; live worker has exclusive ledger ownership.
        state=subprocess.check_output(ssh+['test ! -f '+remote+'/private/OUTPUTS.jsonl || echo outputs; test ! -f '+remote+'/public/EXIT_FACT_FIXED_L31_DOWN.json || echo exit'],text=True).split()
        if 'outputs' in state:
            pull();rows=outputs(root);inserted=[r for r in rows if r['mode']=='insertion']
            if len(inserted)>scored:
                judge(root,inserted[scored:],f'insertions_{scored+1:03d}_{len(inserted):03d}');scored=len(inserted)
            for prefix in sorted(pub.glob('PREFIX_*.json')):
                n=read(prefix)['N']
                label=f'prefix_{n:03d}'
                if (pub/f'PREFIX_{n:03d}.json').exists() and not (p/(label+'_SCORE_RECEIPT.json')).exists():
                    rr=[r for r in rows if r['mode']=='endpoint' and r['prefix']==n]
                    assert len(rr)==len(expected(stream,n,final=n==45 or read(prefix).get("resource_limited_final",False))) and {r['query_id'] for r in rr}==set(expected(stream,n,final=n==45 or read(prefix).get("resource_limited_final",False)))
                    judge(root,rr,label);report(root)
            write(pub/'CONTROLLER_STATUS.json',dict(phase='B_RUNNING_OR_SCORING',insertions_scored=scored))
        if 'exit' in state:
            pull(has_outputs='outputs' in state);receipt=read(pub/'EXIT_FACT_FIXED_L31_DOWN.json');report(root)
            if receipt['exit_code']!=0:raise RuntimeError('GPU worker exited nonzero; preserve ledger and resume state; no blind relaunch')
            assert read(pub/'REPORT_STATUS.json')['B_highest_scored_N']==read(pub/'GENERATED.json')['N'] and read(pub/'REPORT_STATUS.json')['B_all_generated_outputs_scored']
            write(pub/'CONTROLLER_STATUS.json',dict(phase='B_COMPLETE_SCORED' if read(pub/'GENERATED.json')['N']==45 else 'B_RESOURCE_LIMITED_SCORED',GPU_relaunch=False,publication='PENDING_HEARTBEAT_CLOSEOUT',next='Stage22A CPU audit and pre-registered H improvement'));break
        time.sleep(45)


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='NEEDS_ENGINEERING_REVIEW',error=str(e),no_automatic_semantic_retry=True));raise
