"""One restartable common CORE/FULL queue; accepted/failed attempts never repeat."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from med133_score_contract import PROTOCOL,PROMPT,digest,schema,validate


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

def remote(code):
    result=subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,text=True,capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-3000:])
    return json.loads(result.stdout)


def main():
    cfg=read(os.environ['SCORING_CONFIG']);local=Path(cfg['local']);local.mkdir(parents=True,exist_ok=True)
    lock=(local/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running=subprocess.run(['ps','-Ao','pid=,args='],capture_output=True,text=True).stdout
    assert not any(str(local/'work') in line and ' exec ' in line for line in running.splitlines()), 'An isolated batch is still running; do not duplicate it'
    write(local/'IDENTITY.json',dict(pid=os.getpid(),time=time.time(),argv=sys.argv))
    source=Path(cfg['source']);sys.path.insert(0,str(source))
    from scripts.medtrace import stage17_judge as judge
    import med133_report as report
    judge.PROMPT,judge.PROTOCOL,judge.schema,judge.validate=PROMPT,PROTOCOL,schema,validate
    contract=dict(protocol=PROTOCOL,prompt=PROMPT,model='gpt-6-astra',reasoning_effort='medium',
                  semantic_retries=0,transport_retries=0,max_attempt_items=22482,workers=4)
    if (local/'SCORING_CONTRACT.json').exists():assert read(local/'SCORING_CONTRACT.json')==contract
    else:write(local/'SCORING_CONTRACT.json',contract)
    old=Path(cfg['accepted_small_epoch']);oldcontract=read(old/'SCORING_CONTRACT.json')
    assert all(oldcontract[k]==contract[k] for k in ('protocol','prompt','model','reasoning_effort'))
    decisions=read(old/'DECISIONS.json')['decisions'].copy()
    if (local/'DECISIONS.json').exists():
        previous=read(local/'DECISIONS.json')['decisions']
        for key,value in previous.items():
            assert key not in decisions or decisions[key]==value;decisions[key]=value
    missing=set(read(local/'MISSING.json')['keys']) if (local/'MISSING.json').exists() else set()
    envfile=cfg['envfile'];root=cfg['remote']
    prefix='import os,json,pathlib,sys\nos.environ.update(json.loads(pathlib.Path('+repr(envfile)+').read_text()))\nr=pathlib.Path('+repr(root)+')\n'
    charged=sum(v['items'] for v in (read(p) for p in (local/'attempts').glob('*.json')))
    charge_lock=threading.Lock()
    for stage in ('A','B'):
        if (local/f'STAGE_{stage}_REPORT_COMPLETE.json').exists():continue
        while True:
            state=remote(prefix+"print(json.dumps(dict(clock=json.loads((r/'CLOCK.json').read_text()),complete=(r/'STAGE_"+stage+"_GENERATION_COMPLETE.json').exists(),stop=(r/'STOP.json').exists(),failure=(r/'FAILURE.json').exists())))")
            if state['complete']:break
            if state['stop'] or state['failure']:raise RuntimeError('Generator stopped; preserve partial scoring')
            assert time.time()<state['clock']['hard_end']-30
            time.sleep(30)
        if stage=='A' and not (local/'CANONICAL_BACKUP_RECEIPT.json').exists():
            backup=remote(prefix+"import base64\np=r/'retained/MED146.pt'\nprint(json.dumps(dict(bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode(),manifest=json.loads((r/'EXPERT_LIBRARY_FROZEN.json').read_text()))))")
            payload=base64.b64decode(backup['data']);assert len(payload)==backup['bytes'] and backup['manifest']['save_load_exact']
            target=Path(cfg['backup']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.stat().st_size==len(payload)
            else:target.write_bytes(payload)
            write(local/'CANONICAL_BACKUP_RECEIPT.json',dict(time=time.time(),path=str(target),bytes=len(payload),experts=146,source_manifest=backup['manifest'],copy_completed=True))
        packetfile=local/f'PACKET_{stage}.json'
        if packetfile.exists():packet=read(packetfile)
        else:
            packet=remote(prefix+"sys.path.insert(0,str(r/'tools'))\nimport med133_score_packet as score_packet\nprint(json.dumps(score_packet.gather("+repr(stage)+")))")
            write(packetfile,packet)
        consumers=packet.pop('consumers');payloads={}
        for v in consumers:
            key=v['payload_key'];record=dict(opaque_query_id=key,question=v['question'],gold_answer=v['reference'],raw_base_answer=v['answer']['raw_answer'])
            assert key not in payloads or payloads[key]==record;payloads[key]=record
        frozen=[{k:x for k,x in v.items() if k not in ('answer',)} for v in consumers]
        write(local/f'CONSUMERS_{stage}_FROZEN.json',dict(consumers=frozen,unique=len(payloads),protocol=PROTOCOL,prompt_binding=digest(PROMPT)))
        manifest=local/f'BATCH_MANIFEST_{stage}.json'
        if manifest.exists():
            saved=read(manifest);assert saved['input_binding']==packet['input_binding'];batches=saved['batches']
            for batch in batches:
                assert all(payloads[r['opaque_query_id']]==r for r in batch['records'])
        else:
            rows=[payloads[k] for k in sorted(payloads) if k not in decisions and k not in missing]
            batches=[dict(batch_id=digest([PROTOCOL,stage,rows[offset:offset+40]]),records=rows[offset:offset+40]) for offset in range(0,len(rows),40)]
            write(manifest,dict(batches=batches,input_binding=packet['input_binding']))
        def execute(batch):
            nonlocal charged
            bid=batch['batch_id'];bundle=local/'batches'/bid;work=local/'work'/bid;sibling=local/'sibling'
            for folder in ('operator/execution_evidence','operator/responses','judge_only'):(bundle/folder).mkdir(parents=True,exist_ok=True)
            (bundle/'operator/MANIFEST.json').write_text('{}')
            response=bundle/'operator/responses'/f'{bid}.json';attempt=local/'attempts'/f'{bid}.json'
            if attempt.exists():
                if response.exists():value=read(response);validate(batch,value);return value['decisions'],[]
                return [],[v['opaque_query_id'] for v in batch['records']]
            assert time.time()<packet['clock']['hard_end']-620
            work.mkdir(parents=True,exist_ok=True);sibling.mkdir(parents=True,exist_ok=True)
            (bundle/'judge_only'/f'{bid}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
            (bundle/'judge_only'/f'{bid}.schema.json').write_text(json.dumps(schema(batch)))
            sandbox=work/'preflight.sb';sandbox.write_text(judge.profile(work,[work,sibling],bundle,source))
            checks=judge.isolation_check(work,sibling,bundle,source,sandbox);assert all(checks.values())
            preflight=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sandbox),'/usr/bin/curl','--proxy','http://127.0.0.1:7897',
                '--connect-timeout','8','--max-time','15','-sS','-o','/dev/null','-w','%{http_code}','https://chatgpt.com/backend-api/codex/models'],capture_output=True,text=True)
            write(bundle/'operator/PREFLIGHT.json',dict(checks=checks,exit_code=preflight.returncode,HTTP=preflight.stdout))
            assert preflight.returncode==0 and preflight.stdout in ('200','401','403')
            with charge_lock:
                charged+=len(batch['records']);assert charged<=contract['max_attempt_items']
                write(attempt,dict(items=len(batch['records']),time=time.time(),batch_id=bid))
                write(local/'COST.json',dict(attempt_items=charged,time=time.time()))
            def terminate():
                for pid in subprocess.run(['pgrep','-P',str(os.getpid())],capture_output=True,text=True).stdout.split():
                    cmd=subprocess.run(['ps','-p',pid,'-o','args='],capture_output=True,text=True).stdout
                    if str(work/'final.json') in cmd:os.kill(int(pid),15)
            timer=threading.Timer(600,terminate);timer.daemon=True;timer.start()
            try:
                judge.run_batch(bundle,batch,work,[work,sibling],source,Path(cfg['cli']),explicit_proxy=True,reasoning_effort='medium')
                value=read(response);validate(batch,value);return value['decisions'],[]
            except Exception as error:
                write(bundle/'operator/FAILURE_NO_RETRY.json',dict(time=time.time(),error=repr(error)))
                return [],[v['opaque_query_id'] for v in batch['records']]
            finally:timer.cancel()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(execute,batch) for batch in batches]
            for future in as_completed(futures):
                accepted,failed=future.result()
                for d in accepted:
                    key=d['opaque_query_id'];assert key not in decisions or decisions[key]==d;decisions[key]=d;missing.discard(key)
                missing.update(failed)
                write(local/'DECISIONS.json',dict(decisions=decisions))
                write(local/'MISSING.json',dict(keys=sorted(missing)))
        cost=dict(Judge_new_attempt_items=charged,accepted_small_epoch_available=len(read(old/'DECISIONS.json')['decisions']),
                  generated_consumers=2956 if stage=='A' else 7494)
        meta=remote(prefix+'''
import statistics,time
load=lambda p:json.loads(p.read_text())
a=load(r/'CURVATURE_ADMISSION.json');anchors=[v for v in a['anchors'] if v['status']=='MAPPED']
training=[load(p) for p in (r/'training').glob('MED_*.json')];assert len(training)==146
curves=[v for t in training for v in t['curve']];calibration=[load(p) for p in (r/'calibration').glob('*.json')]
g=load(r/'STAGE_STAGE_GENERATION_COMPLETE.json');clock=load(r/'CLOCK.json');worker=load(r/'WORKER_IDENTITY.json');health=load(r/'GUARD_HEALTH.json')
print(json.dumps(dict(counts=g['counts'],experiment_elapsed_seconds=g['time']-clock['original_start'],
 current_GPU_worker_residency_seconds=g['time']-worker['start'],cuda_max_memory_allocated=g['cuda_max_memory_allocated'],
 owned_remote_new_bytes_last_sample=health['new_bytes'],
 curvature=dict(mapped=a['mapped'],common_fallback=a['common_fallback'],full_rank_min=min(v['full_rank'] for v in anchors),full_rank_max=max(v['full_rank'] for v in anchors),max_relative_MED_trace_error=max(abs(v['MED_trace']/v['full_trace']-1) for v in anchors)),
 mechanism=dict(scientific_updates=len(curves),Fisher_cap_active_fraction=statistics.mean(v['balance']['active'] for v in curves),
 anti_parallel_removed_groups=sum(v['anti_parallel']['removed_groups'] for v in curves),
 initial_native_NLL_mean=statistics.mean(t['curve'][0]['source_NLL'][0] for t in training),terminal_native_NLL_mean=statistics.mean(t['curve'][-1]['source_NLL'][0] for t in training),
 lambda_min=min(v['coefficient'] for v in calibration),lambda_max=max(v['coefficient'] for v in calibration),
 interpretation='Descriptive mechanism audit only; no146 GEN/FULL_CTX comparison'))))
'''.replace('STAGE_STAGE','STAGE_'+stage))
        cost.update(meta)
        result,report_text,owner_panels=report.summarize(packet,frozen,decisions,missing,stage,cost)
        write(local/f'RESULTS_{stage}.json',result);write(local/f'PRIVATE_OWNER_AUDIT_{stage}.json',owner_panels)
        (local/f'REPORT_{stage}_ZH.md').write_text(report_text)
        receipt=dict(time=time.time(),engineering_complete=True,scoring_terminal=True,status=result['status'],
            single_consumers=2956,unknown_consumers=result['grading']['unknown_consumers'],scientific_gates_do_not_control_stage_B=True)
        reportname='STAGE_A_REPORT_COMPLETE.json' if stage=='A' else 'FINAL_REPORT_COMPLETE.json'
        remote(prefix+'value=json.loads('+repr(json.dumps(receipt))+')\n(r/'+repr(reportname)+').write_text(json.dumps(value,indent=2))\nprint(json.dumps(value))')
        write(local/f'STAGE_{stage}_REPORT_COMPLETE.json',receipt)
        # Only aggregate results enter the public workspace; raw answers remain private.
        destination=Path(cfg['workspace'])/'reports/med146_single_sequential_v133_20261027';destination.mkdir(exist_ok=True)
        write(destination/f'RESULTS_{stage}.json',result);(destination/f'REPORT_{stage}_ZH.md').write_text(report_text)
    write(local/'COMPLETE.json',dict(time=time.time(),attempt_items=charged,failed_payloads=len(missing)))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;cfg=read(os.environ['SCORING_CONFIG']);write(Path(cfg['local'])/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
    finally:
        cfg=read(os.environ['SCORING_CONFIG']);write(Path(cfg['local'])/'EXIT.json',dict(time=time.time(),code=code))
