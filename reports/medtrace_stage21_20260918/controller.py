"""Finite A-generation consumer; recurring follow-up is owned by app heartbeat."""
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
    def pull():
        subprocess.run(['rsync','-a','--exclude=BUDGET_LEDGER.json','--exclude=CONTROLLER_STATUS.json','-e',transport,ssh[-1]+':'+remote+'/public/',str(pub)+'/'],check=True)
        for name in ('OUTPUTS.jsonl',):
            subprocess.run(['rsync','-a','-e',transport,ssh[-1]+':'+remote+'/private/'+name,str(p/name)],check=True)
    scored=0
    while True:
        # One low-cost status request per pass; live worker has exclusive ledger ownership.
        state=subprocess.check_output(ssh+['test ! -f '+remote+'/private/OUTPUTS.jsonl || echo outputs; test ! -f '+remote+'/public/EXIT_NO_H_HSIC.json || echo exit'],text=True).split()
        if 'outputs' in state:
            pull();rows=outputs(root);inserted=[r for r in rows if r['mode']=='insertion']
            if len(inserted)>scored:
                judge(root,inserted[scored:],f'insertions_{scored+1:03d}_{len(inserted):03d}');scored=len(inserted)
            for n in (11,19,32,45):
                label=f'prefix_{n:03d}'
                if (pub/f'PREFIX_{n:03d}.json').exists() and not (p/(label+'_SCORE_RECEIPT.json')).exists():
                    rr=[r for r in rows if r['mode']=='endpoint' and r['prefix']==n]
                    assert len(rr)==len(expected(stream,n,final=n==45)) and {r['query_id'] for r in rr}==set(expected(stream,n,final=n==45))
                    judge(root,rr,label);report(root)
            write(pub/'CONTROLLER_STATUS.json',dict(phase='A_RUNNING_OR_SCORING',insertions_scored=scored,B='PENDING_LAYER_CLARIFICATION'))
        if 'exit' in state:
            pull();receipt=read(pub/'EXIT_NO_H_HSIC.json');report(root)
            if receipt['exit_code']!=0:raise RuntimeError('GPU worker exited nonzero; preserve ledger and resume state; no blind relaunch')
            assert read(pub/'REPORT_STATUS.json')['A_highest_scored_N']==45 and read(pub/'REPORT_STATUS.json')['A_all_generated_outputs_scored']
            write(pub/'CONTROLLER_STATUS.json',dict(phase='A_COMPLETE_SCORED',B='PENDING_LAYER_CLARIFICATION',GPU_relaunch=False));break
        time.sleep(45)


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    try:run(cfg)
    except Exception as e:
        write(Path(cfg['local_run'])/'public/CONTROLLER_STATUS.json',dict(phase='NEEDS_ENGINEERING_REVIEW',error=str(e),no_automatic_semantic_retry=True));raise
