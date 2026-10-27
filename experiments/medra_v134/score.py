"""Isolated four-arm grading and deterministic screening; no scheduled monitor."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import base64
import csv
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

from med133_score_contract import PROTOCOL,PROMPT,digest,schema,validate


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def remote(code):
    p=subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


def publish(cfg,paths):
    workspace=Path(cfg['workspace'])
    env=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',
             ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    def git(*args):return subprocess.check_output(['git','-C',str(workspace),*args],env=env,text=True).strip()
    try:
        assert git('branch','--show-current')==cfg['branch']
        assert not git('diff','--cached','--name-only'), 'Unrelated staged work; preserve'
        for p in paths:
            s=p.read_text()
            assert not any(v in s for v in ('/Users/','/data/bmw/','/remote-home/','core9q-','raw_token_ids','image_sha256','PRIVATE KEY'))
        relative=[str(p.relative_to(workspace)) for p in paths]
        git('add','--',*relative)
        if git('diff','--cached','--name-only'):git('commit','-m','Report finite MED-RA screening and authorized evaluation')
        sha=git('rev-parse','HEAD')
        git('-c','http.proxy=http://127.0.0.1:7897','push','origin','HEAD:'+cfg['branch'])
        report_path=next(p for p in reversed(relative) if p.endswith('.md'))
        url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/'+report_path
        p=subprocess.run(['/usr/bin/curl','--proxy','http://127.0.0.1:7897','--fail','--silent','--show-error',
                          '--connect-timeout','10','--max-time','30',url],capture_output=True,text=True,env=env,check=True)
        assert p.stdout
        write(Path(cfg['local'])/'PUBLICATION_RECEIPT.json',dict(status='PASS',commit=sha,anonymous_read=True,url=url))
    except Exception:
        write(Path(cfg['local'])/'PUBLICATION_PENDING.json',dict(time=time.time(),error=traceback.format_exc()))


def main():
    cfg=read(os.environ['SCORING_CONFIG']);local=Path(cfg['local']);root=cfg['remote'];local.mkdir(exist_ok=True)
    lock=(local/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(local/'IDENTITY.json',dict(pid=os.getpid(),time=time.time(),argv=sys.argv))
    sys.path.insert(0,cfg['source']);sys.path.insert(0,cfg['common_source']);sys.path.insert(0,str(Path(cfg['score_source']).parent))
    from scripts.medtrace import stage17_judge as judge
    import ra_report as report
    judge.PROMPT,judge.PROTOCOL,judge.schema,judge.validate=PROMPT,PROTOCOL,schema,validate
    old_contract=read(cfg['accepted_contract'])
    assert all(old_contract[k]==v for k,v in dict(protocol=PROTOCOL,prompt=PROMPT,model='gpt-6-astra',reasoning_effort='medium').items())
    contract=dict(protocol=PROTOCOL,prompt=PROMPT,model='gpt-6-astra',reasoning_effort='medium',
                  semantic_retries=0,transport_retries=0,max_attempt_items=7494,workers=4)
    if (local/'SCORING_CONTRACT.json').exists():assert read(local/'SCORING_CONTRACT.json')==contract
    else:write(local/'SCORING_CONTRACT.json',contract)
    original=read(cfg['accepted_decisions'])['decisions'];decisions=dict(original)
    if (local/'DECISIONS.json').exists():
        for k,v in read(local/'DECISIONS.json')['decisions'].items():
            assert k not in decisions or decisions[k]==v;decisions[k]=v
    missing=set(read(local/'MISSING.json')['keys']) if (local/'MISSING.json').exists() else set()
    charged=sum(read(p)['items'] for p in (local/'attempts').glob('*.json'));charge_lock=threading.Lock()
    prefix='import os,json,sys\nfrom pathlib import Path\nos.environ.update(json.loads(Path('+repr(cfg['envfile'])+').read_text()))\nr=Path('+repr(root)+')\n'
    public=Path(cfg['workspace'])/'reports/medra_v134_20261027';published=[]
    final_status=None
    for stage in ('A','B'):
        if (local/f'STAGE_{stage}_REPORT_COMPLETE.json').exists():
            old=read(local/f'STAGE_{stage}_REPORT_COMPLETE.json')
            final_status=old['status']
            if stage=='A' and not old['sequential_gate_pass']:final_status=old['status'];break
            continue
        while True:
            state=remote(prefix+"print(json.dumps(dict(clock=json.loads((r/'CLOCK.json').read_text()),ready=(r/'STAGE_"+stage+"_GENERATION_COMPLETE.json').exists(),stop=(r/'STOP.json').exists(),failure=(r/'FAILURE.json').exists())))")
            if state['ready']:break
            assert not state['stop'] and not state['failure'] and time.time()<state['clock']['hard_end']-30
            time.sleep(30)
        if stage=='A' and not (local/'CANONICAL_BACKUP_RECEIPT.json').exists():
            value=remote(prefix+"import base64\np=r/'retained/MED_RA146.pt'\nprint(json.dumps(dict(data=base64.b64encode(p.read_bytes()).decode(),bytes=p.stat().st_size)))")
            data=base64.b64decode(value['data']);assert len(data)==value['bytes']
            backup=Path(cfg['backup']);backup.parent.mkdir(parents=True,exist_ok=True)
            assert not backup.exists();backup.write_bytes(data)
            import torch
            package=torch.load(backup,map_location='cpu',weights_only=True)
            assert len(package['states'])==len(package['R0_entries'])==146
            write(local/'CANONICAL_BACKUP_RECEIPT.json',dict(time=time.time(),bytes=len(data),experts=146,CPU_load_PASS=True))
        packetfile=local/f'PACKET_{stage}.json'
        if packetfile.exists():packet=read(packetfile)
        else:
            packet=remote(prefix+"sys.path.insert(0,str(r/'tools'))\nimport ra_score_packet as score_packet\nprint(json.dumps(score_packet.gather("+repr(stage)+")))")
            write(packetfile,packet)
        consumers=packet.pop('consumers');payloads={}
        for v in consumers:
            if v['arm']!='RA':assert v['payload_key'] in original, 'Frozen comparator has no accepted same-contract score'
            row=dict(opaque_query_id=v['payload_key'],question=v['question'],gold_answer=v['reference'],raw_base_answer=v['answer']['raw_answer'])
            assert v['payload_key'] not in payloads or payloads[v['payload_key']]==row;payloads[v['payload_key']]=row
        frozen=[{k:x for k,x in v.items() if k!='answer'} for v in consumers]
        write(local/f'CONSUMERS_{stage}_FROZEN.json',dict(consumers=frozen,input_binding=packet['input_binding']))
        manifest=local/f'BATCH_MANIFEST_{stage}.json'
        if manifest.exists():
            batches=read(manifest)['batches']
            assert all(payloads[v['opaque_query_id']]==v for b in batches for v in b['records'])
        else:
            pending=[payloads[k] for k in sorted(payloads) if k not in decisions and k not in missing]
            batches=[dict(batch_id=digest([PROTOCOL,'RA134',stage,pending[i:i+40]]),records=pending[i:i+40]) for i in range(0,len(pending),40)]
            write(manifest,dict(batches=batches,input_binding=packet['input_binding']))
        def execute(batch):
            nonlocal charged
            bid=batch['batch_id'];bundle=local/'batches'/bid;work=local/'work'/bid;sibling=local/'sibling'
            for p in (bundle/'operator/execution_evidence',bundle/'operator/responses',bundle/'judge_only',work,sibling):p.mkdir(parents=True,exist_ok=True)
            response=bundle/'operator/responses'/f'{bid}.json';attempt=local/'attempts'/f'{bid}.json'
            if attempt.exists():
                if response.exists():value=read(response);validate(batch,value);return value['decisions'],[]
                return [],[v['opaque_query_id'] for v in batch['records']]
            assert time.time()<packet['clock']['hard_end']-620
            write(bundle/'operator/MANIFEST.json',{})
            (bundle/'judge_only'/f'{bid}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
            write(bundle/'judge_only'/f'{bid}.schema.json',schema(batch))
            sandbox=work/'preflight.sb';sandbox.write_text(judge.profile(work,[work,sibling],bundle,Path(cfg['source'])))
            checks=judge.isolation_check(work,sibling,bundle,Path(cfg['source']),sandbox)
            preflight=subprocess.run(['/usr/bin/sandbox-exec','-f',str(sandbox),'/usr/bin/curl','--proxy','http://127.0.0.1:7897',
                '--connect-timeout','8','--max-time','15','-sS','-o','/dev/null','-w','%{http_code}','https://chatgpt.com/backend-api/codex/models'],capture_output=True,text=True)
            write(bundle/'operator/PREFLIGHT.json',dict(checks=checks,exit_code=preflight.returncode,HTTP=preflight.stdout))
            assert all(checks.values()) and preflight.returncode==0 and preflight.stdout in ('200','401','403')
            with charge_lock:
                charged+=len(batch['records']);assert charged<=7494
                write(attempt,dict(time=time.time(),items=len(batch['records'])))
                write(local/'COST.json',dict(new_attempt_items=charged,historical_MED_attempt_items=4694))
            def terminate():
                for row in subprocess.run(['ps','-Ao','pid=,args='],capture_output=True,text=True).stdout.splitlines():
                    if str(work/'final.json') in row and ' exec ' in row:os.kill(int(row.strip().split(None,1)[0]),15)
            timer=threading.Timer(600,terminate);timer.daemon=True;timer.start()
            try:
                judge.run_batch(bundle,batch,work,[work,sibling,Path(cfg['accepted_decisions']).parent],Path(cfg['source']),Path(cfg['cli']),
                                explicit_proxy=True,reasoning_effort='medium')
                value=read(response);validate(batch,value);return value['decisions'],[]
            except Exception:
                write(bundle/'operator/FAILURE_NO_RETRY.json',dict(time=time.time(),error=traceback.format_exc()))
                return [],[v['opaque_query_id'] for v in batch['records']]
            finally:timer.cancel()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(execute,b) for b in batches]):
                valid,failed=future.result()
                for v in valid:
                    k=v['opaque_query_id'];assert k not in decisions or decisions[k]==v;decisions[k]=v
                missing.update(failed)
                write(local/'DECISIONS.json',dict(decisions=decisions));write(local/'MISSING.json',dict(keys=sorted(missing)))
        metadata=remote(prefix+"g=json.loads((r/'STAGE_"+stage+"_GENERATION_COMPLETE.json').read_text())\nsel=[json.loads(p.read_text()) for p in (r/'selection').glob('*.json')]\nprint(json.dumps(dict(counts=g['counts'],cuda_max_memory_allocated=g['cuda_max_memory_allocated'],selection=[{k:v[k] for k in ['owner','alpha','q','q0','trace','raw_lambda','raw_operator_norm','final_operator_norm','raw_energy','final_energy','final_risk','source_floor_pass','risk_pass','violation_fallback']} for v in sel],original_clock=json.loads((r/'CLOCK.json').read_text()),curvature_parity=json.loads((r/'CURVATURE_REBUILD_PARITY.json').read_text()))))")
        cost=dict(new_Judge_attempt_items=charged,historical_MED_Judge_attempt_items=4694,inherited_raw_training_updates=23360,
            new_gradient_training_updates=0,inherited_payloads_used=len(set(payloads)&original.keys()),**metadata)
        result,text,owner=report.summarize(packet,frozen,decisions,missing,stage,cost)
        write(local/f'RESULTS_{stage}.json',result);write(local/f'OWNER_AUDIT_{stage}.json',owner)
        write(local/f'ITEM_AUDIT_{stage}.json',report.item_audit(packet,consumers,decisions,stage))
        report.write_owner_tables(public,packet,owner,stage)
        write(public/f'RESULTS_{stage}.json',result);(public/f'REPORT_{stage}_ZH.md').write_text(text)
        published.extend([public/f'RESULTS_{stage}.json',public/f'REPORT_{stage}_ZH.md'])
        gate=dict(time=time.time(),engineering_complete=True,single_consumers=2956,scoring_complete=result['grading']['unknown_consumers']==0,
            sequential_gate_pass=result['sequential_gate_pass'],gate=result['sequential_gate'],status=result['status'])
        remote(prefix+'v=json.loads('+repr(json.dumps(gate))+')\n(r/'+repr('STAGE_A_REPORT_COMPLETE.json' if stage=='A' else 'FINAL_REPORT_COMPLETE.json')+').write_text(json.dumps(v,indent=2))\nprint(json.dumps(v))')
        write(local/f'STAGE_{stage}_REPORT_COMPLETE.json',gate)
        final_status=result['status']
        if stage=='A' and not result['sequential_gate_pass']:
            remote(prefix+'v=json.loads('+repr(json.dumps(gate))+')\n(r/\'FINAL_REPORT_COMPLETE.json\').write_text(json.dumps(v,indent=2))\nprint(json.dumps(v))');break
    write(local/'COMPLETE.json',dict(time=time.time(),status=final_status,new_attempt_items=charged,missing_payloads=len(missing)))
    # Publication is a finite completion step, not a recurring monitor.
    for _ in range(12):
        terminal=remote(prefix+"print(json.dumps(dict(worker=(r/'WORKER_EXIT.json').exists(),guard=(r/'GUARD_EXIT.json').exists())))")
        if terminal['worker'] and terminal['guard']:break
        time.sleep(5)
    published=[public/name for s in ('A','B') for name in (f'RESULTS_{s}.json',f'REPORT_{s}_ZH.md',f'PER_EDIT_{s}.csv',f'PER_SOURCE_{s}.csv') if (public/name).exists()]
    publish(cfg,published)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;cfg=read(os.environ['SCORING_CONFIG']);write(Path(cfg['local'])/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        try:remote("import json\nfrom pathlib import Path\np=Path("+repr(cfg['remote'])+")\nv={'error':'Local scoring stopped; preserve all completed work'}\n(p/'SCORE_FAILURE.json').write_text(json.dumps(v))\nprint('{}')")
        except Exception:pass
        raise
    finally:
        cfg=read(os.environ['SCORING_CONFIG']);write(Path(cfg['local'])/'EXIT.json',dict(time=time.time(),code=code))
