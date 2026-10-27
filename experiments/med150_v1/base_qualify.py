"""One isolated, bounded qualification of four frozen Base answers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def size(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total


def guard(cfg):
    local=Path(cfg['local']);clock=read(cfg['clock'])
    assert time.time()<clock['hard_end']-620
    with (local/'GUARD_ONCE').open('x'):pass
    with (local/'worker.log').open('a') as log:
        child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_ENTRY']],env=dict(os.environ,ENTRY_MODE='judge'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=local)
    write(local/'WORKER_IDENTITY.json',dict(pid=child.pid,time=time.time(),argv=[sys.executable,os.environ['NEUTRAL_ENTRY']]))
    start=time.time()
    try:
        while child.poll() is None:
            used=size(local)+size(Path(cfg['private_root']))+1024**3
            free=os.statvfs(local).f_bavail*os.statvfs(local).f_frsize
            write(local/'GUARD_HEALTH.json',dict(time=time.time(),new_bytes_upper_bound=used,free_bytes=free,hard_end=clock['hard_end']))
            assert used<47_000_000_000 and free>8*1024**3 and time.time()<clock['hard_end']-30
            assert time.time()-start<600, 'Judge hard timeout; no automatic retry'
            time.sleep(5)
    except BaseException:
        write(local/'STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait();write(local/'GUARD_EXIT.json',dict(time=time.time(),code=code,original_clock=clock))
    raise SystemExit(code)


def judge_four(cfg):
    local=Path(cfg['local']);clock=read(cfg['clock']);assert time.time()<clock['hard_end']-620
    sys.path[:0]=[cfg['common_source'],cfg['source']]
    from med133_score_contract import PROTOCOL,PROMPT,digest,schema,validate
    from scripts.medtrace import stage17_judge as judge
    judge.PROTOCOL,judge.PROMPT,judge.schema,judge.validate=PROTOCOL,PROMPT,schema,validate
    old_contract=read(cfg['accepted_contract'])
    assert all(old_contract[k]==v for k,v in dict(protocol=PROTOCOL,prompt=PROMPT,model='gpt-6-astra',reasoning_effort='medium').items())
    old=read(cfg['accepted_decisions'])['decisions'];candidates=read(cfg['candidates'])['items'];assert len(candidates)==4
    records=[];keys=[];known={}
    for v in candidates:
        assert all(v['checks'].values())
        native=v['task']['native'];bind=v['historical_binding'];answer=v['old_base_output']
        assert bind['prompt_ids']==answer['prompt_token_ids'] and answer['error'] is None and answer['raw_generated_token_ids']
        output=dict(raw_answer=answer['model_answer_raw'],raw_token_ids=answer['raw_generated_token_ids'])
        key=digest(dict(query_id=native['query_id'],question=native['question'],reference=native['reference'],
                        image_sha256=native['image_sha256'],prompt_ids=bind['prompt_ids'],answer=output))
        keys.append(key)
        if key in old:known[key]=old[key]
        else:records.append(dict(opaque_query_id=key,question=native['question'],gold_answer=native['reference'],raw_base_answer=output['raw_answer']))
    assert len(set(keys))==4
    result=dict(known)
    if records:
        batch=dict(batch_id=digest([PROTOCOL,'MED150_BASE_QUALIFICATION',records]),records=records)
        bid=batch['batch_id'];bundle=local/'bundle';work=local/'work';sibling=local/'sibling';source=Path(cfg['source'])
        for path in (bundle/'operator/execution_evidence',bundle/'operator/responses',bundle/'judge_only',work,sibling):path.mkdir(parents=True,exist_ok=True)
        (bundle/'operator/MANIFEST.json').write_text('{}')
        (bundle/'judge_only'/f'{bid}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
        (bundle/'judge_only'/f'{bid}.schema.json').write_text(json.dumps(schema(batch)))
        sandbox=work/'preflight.sb';sandbox.write_text(judge.profile(work,[work,sibling],bundle,source))
        checks=judge.isolation_check(work,sibling,bundle,source,sandbox)
        proxy=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sandbox),'/usr/bin/curl','--proxy','http://127.0.0.1:7897',
            '--connect-timeout','8','--max-time','15','-sS','-o','/dev/null','-w','%{http_code}',
            'https://chatgpt.com/backend-api/codex/models'],capture_output=True,text=True)
        write(local/'PREFLIGHT.json',dict(checks=checks,exit_code=proxy.returncode,HTTP=proxy.stdout))
        assert all(checks.values()) and proxy.returncode==0 and proxy.stdout in ('200','401','403')
        write(local/'ATTEMPT.json',dict(time=time.time(),items=len(records),batch_id=bid,automatic_retry=False))
        judge.run_batch(bundle,batch,work,[work,sibling],source,Path(cfg['cli']),explicit_proxy=True,reasoning_effort='medium')
        response=read(bundle/'operator/responses'/f'{bid}.json');validate(batch,response)
        for row in response['decisions']:result[row['opaque_query_id']]=row
    assert set(result)==set(keys)
    values=[result[k] for k in keys]
    status='PASS_ALL_FOUR_BASE_CORE_INCORRECT' if all(not v['core_correct'] for v in values) else 'FAIL_BASE_ALREADY_CORRECT'
    write(Path(cfg['private_root'])/'BASE_QUALIFICATION.json',dict(time=time.time(),status=status,decisions=values,
        new_attempt_items=len(records),old_valid_reused=len(known),prompt=PROTOCOL,clock=clock))
    write(local/'COMPLETE.json',dict(time=time.time(),status=status,valid=len(values),new_attempt_items=len(records),unknown=0))


if __name__=='__main__':
    cfg=read(os.environ['QUALIFICATION_CONFIG']);local=Path(cfg['local'])
    if os.environ.get('ENTRY_MODE','guard')=='guard':guard(cfg)
    else:
        try:judge_four(cfg)
        except BaseException:
            write(local/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),no_retry=True));raise
