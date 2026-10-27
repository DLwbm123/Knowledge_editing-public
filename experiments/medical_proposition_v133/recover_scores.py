"""One explicitly authorized additional attempt for the original failed payloads."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from med133_score_contract import PROTOCOL, PROMPT, digest, schema, validate
from score import read, write


def main():
    cfg=read(os.environ['SCORING_RECOVERY_CONFIG'])
    root,original=Path(cfg['local']),Path(cfg['original'])
    root.mkdir(exist_ok=True)
    lock=(root/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'START.json').exists(), 'Recovery already attempted; inspect its receipts'
    contract=read(original/'SCORING_CONTRACT.json')
    assert contract['protocol']==PROTOCOL and contract['prompt']==PROMPT
    assert contract['model']=='gpt-6-astra' and contract['reasoning_effort']=='medium'
    missing=set(read(original/'MISSING.json')['keys'])
    accepted=read(original/'DECISIONS.json')['decisions']
    batches=[b for b in read(original/'BATCH_MANIFEST_A.json')['batches']
             if any(r['opaque_query_id'] in missing for r in b['records'])]
    assert len(batches)==2 and sum(len(b['records']) for b in batches)==80
    assert {r['opaque_query_id'] for b in batches for r in b['records']}==missing
    assert not missing & accepted.keys()
    for b in batches:
        evidence=read(original/'batches'/b['batch_id']/'operator/execution_evidence'/f"{b['batch_id']}.json")
        assert evidence['status']=='FAILED_NO_RETRY' and not evidence['tool_event_types']
    clock=read(original/'CLOCK_LOCAL.json')
    deadline=min(time.time()+600,clock['hard_end']-30)
    assert deadline-time.time()>60 and shutil.disk_usage(root).free>=8*1024**3
    sys.path.insert(0,cfg['source'])
    from scripts.medtrace import stage17_judge as judge
    judge.PROMPT,judge.PROTOCOL,judge.schema,judge.validate=PROMPT,PROTOCOL,schema,validate
    write(root/'START.json',dict(time=time.time(),pid=os.getpid(),argv=sys.argv,
        clock=clock,deadline=deadline,authorized_additional_items=80,original_attempt_items=4614))
    write(root/'SCORING_CONTRACT.json',contract)
    write(root/'BATCH_MANIFEST.json',dict(batches=batches))
    stop=threading.Event()

    def terminate_children():
        rows=subprocess.run(['ps','-Ao','pid=,args='],capture_output=True,text=True).stdout.splitlines()
        for row in rows:
            if str(root/'work') in row and ' exec ' in row:
                os.kill(int(row.strip().split(None,1)[0]),15)

    def guard():
        while not stop.wait(2):
            size=0
            for p in root.rglob('*'):
                try:
                    if p.is_file():size+=p.stat().st_size
                except FileNotFoundError:pass
            if time.time()>=deadline or size>=200_000_000 or shutil.disk_usage(root).free<8*1024**3:
                write(root/'STOP.json',dict(time=time.time(),bytes=size,deadline=deadline))
                terminate_children();return
    thread=threading.Thread(target=guard,daemon=True);thread.start()

    def execute(batch):
        bid=batch['batch_id'];bundle=root/'batches'/bid;work=root/'work'/bid;sibling=root/'sibling'
        for p in [bundle/'operator/execution_evidence',bundle/'operator/responses',bundle/'judge_only',work,sibling]:
            p.mkdir(parents=True,exist_ok=True)
        write(bundle/'operator/MANIFEST.json',{})
        (bundle/'judge_only'/f'{bid}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
        write(bundle/'judge_only'/f'{bid}.schema.json',schema(batch))
        assert (bundle/'judge_only'/f'{bid}.prompt.md').read_text()==(original/'batches'/bid/'judge_only'/f'{bid}.prompt.md').read_text()
        sandbox=work/'preflight.sb';sandbox.write_text(judge.profile(work,[work,sibling],bundle,Path(cfg['source'])))
        checks=judge.isolation_check(work,sibling,bundle,Path(cfg['source']),sandbox)
        preflight=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sandbox),'/usr/bin/curl',
            '--proxy','http://127.0.0.1:7897','--connect-timeout','8','--max-time','15',
            '-sS','-o','/dev/null','-w','%{http_code}','https://chatgpt.com/backend-api/codex/models'],capture_output=True,text=True)
        write(bundle/'operator/PREFLIGHT.json',dict(checks=checks,exit_code=preflight.returncode,HTTP=preflight.stdout))
        assert all(checks.values()) and preflight.returncode==0 and preflight.stdout in ('200','401','403')
        assert not (root/'STOP.json').exists() and time.time()<deadline
        write(root/'attempts'/f'{bid}.json',dict(time=time.time(),items=len(batch['records']),original_batch=bid))
        try:
            judge.run_batch(bundle,batch,work,[work,sibling,original],Path(cfg['source']),Path(cfg['cli']),
                            explicit_proxy=True,reasoning_effort='medium')
            value=read(bundle/'operator/responses'/f'{bid}.json');validate(batch,value)
            return value['decisions']
        except Exception:
            write(bundle/'operator/FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
            return []

    code=1
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(execute,batches))
        recovered={}
        for rows in values:
            for value in rows:
                key=value['opaque_query_id'];assert key in missing and key not in recovered
                recovered[key]=value
        merged=dict(accepted);merged.update(recovered)
        assert all(merged[k]==v for k,v in accepted.items())
        charged=sum(read(p)['items'] for p in (root/'attempts').glob('*.json'))
        assert charged<=80
        write(root/'RECOVERED_DECISIONS.json',dict(decisions=recovered))
        write(root/'MERGED_DECISIONS.json',dict(decisions=merged))
        write(root/'MISSING.json',dict(keys=sorted(missing-recovered.keys())))
        write(root/'COMPLETE.json',dict(time=time.time(),status='COMPLETE' if len(recovered)==80 else 'INCOMPLETE',
            recovered=len(recovered),remaining=len(missing-recovered.keys()),additional_attempt_items=charged,
            cumulative_attempt_items=4614+charged,new_training=0,new_generation=0,original_accepted_unchanged=True))
        code=0 if len(recovered)==80 else 1
    finally:
        stop.set();thread.join()
        write(root/'EXIT.json',dict(time=time.time(),code=code))


if __name__=='__main__':main()
