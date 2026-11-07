"""Dependency-bound post-generation scoring, durable receipts and public closeout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def main(cfg):
    original=Path(cfg['root']);private=Path(cfg['private_root']);clock=read(cfg['clock']);w=Path(cfg['worktree'])
    with (original/'FINISH_ONCE').open('x'):pass
    remote=cfg['remote_paths']['root']
    def ssh(code,timeout=60):
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',cfg['ssh_host'],cfg['remote_python'],'-'],
            input=code,text=True,capture_output=True,timeout=timeout)
        assert result.returncode==0,result.stderr[-2000:]
        return json.loads(result.stdout)
    probe='from pathlib import Path\nimport json\nr=Path('+repr(remote)+')\nprint(json.dumps({n:json.loads((r/n).read_text()) if (r/n).exists() else None for n in ["GUARD_EXIT.json","PIPELINE_COMPLETE.json","PIPELINE_FAILURE.json","EVAL_GENERATION_COMPLETE.json"]}))'
    while True:
        assert time.time()<clock['hard_end']-660
        status=ssh(probe)
        assert not status['PIPELINE_FAILURE.json'],status['PIPELINE_FAILURE.json']
        if status['PIPELINE_COMPLETE.json'] and status['GUARD_EXIT.json']:
            assert status['GUARD_EXIT.json']['code']==0
            break
        write(original/'FINISH_DEPENDENCY_WAIT.json',dict(time=time.time(),waiting_for='GUARDED_T4_TRAINING_AND_BANK'))
        time.sleep(30)
    values=ssh('from pathlib import Path\nimport json\nr=Path('+repr(remote)+')\nprint(json.dumps([json.loads(p.read_text()) for p in sorted((r/"outputs").glob("*.json"))]))',300)
    admission=read(Path(cfg['report'])/'TRAINING_ADMISSION.json')
    assert len(values)==admission['total_consumers'] and len({v['consumer_id'] for v in values})==len(values)
    method_root=original/'eval';method_root.mkdir()
    sys.path.insert(0,cfg['common_source'])
    from med133_score_contract import PROTOCOL,digest
    contract=read(cfg['accepted_contract']);fields=('core_correct','full_supported','unsupported_expansion','extra_contradiction')
    def semantic(row):
        return digest(dict(protocol=PROTOCOL,model='gpt-6-astra',reasoning='medium',
            question=row['question'],gold_answer=row['gold_answer'],raw_base_answer=row['raw_base_answer']))
    payloads,consumers={},[]
    for value in values:
        b=value['judge_input'];a=value['answer']
        row=dict(question=b['question'],gold_answer=b['reference'],raw_base_answer=a['raw_answer'])
        key=semantic(row);payloads[key]=dict(row,opaque_query_id=key)
        consumers.append({**{k:value[k] for k in ('mode','owner','role','query_id','consumer_id')},'key':key})
    votes={};failures=set()
    for prior in [str(original)]+cfg['prior_score_roots']:
        p=Path(prior)
        if not (p/'PACKET_PRIVATE.json').exists() or not (p/'DECISIONS.json').exists():continue
        packet,decisions=read(p/'PACKET_PRIVATE.json'),read(p/'DECISIONS.json')
        for oldkey,raw in packet['payloads'].items():
            key=semantic(raw.get('record',raw))
            if key not in payloads:continue
            if oldkey in decisions['decisions']:votes.setdefault(key,set()).add(tuple(decisions['decisions'][oldkey][f] for f in fields))
            elif oldkey in decisions.get('unknown',[]):failures.add(key)
    known,pending,unknown={},[],[]
    for key in sorted(payloads):
        values_=votes.get(key,set())
        if len(values_)>1 or not values_ and key in failures:unknown.append(key)
        elif values_:known[key]=dict(opaque_query_id=key,**dict(zip(fields,next(iter(values_)))))
        else:pending.append(key)
    write(method_root/'PACKET_PRIVATE.json',dict(payloads=payloads,consumers=consumers,known_valid=known,
        pending=pending,inherited_unknown=unknown,contract=contract))
    write(private/'GENERATION_OUTPUTS.private.json',values)
    method_cfg=dict(cfg,root=str(method_root),stage='method')
    write(method_root/'config.json',method_cfg)
    env=dict(os.environ,T4_CONFIG=str(method_root/'config.json'),T4_MODE='guard')
    result=subprocess.run([sys.executable,cfg['neutral_entry']],env=env,cwd=method_root,
        stdout=(method_root/'guard.log').open('x'),stderr=subprocess.STDOUT,
        timeout=max(1,int(clock['hard_end']-time.time()-30)),start_new_session=True)
    assert result.returncode==0 and read(method_root/'GUARD_EXIT.json')['code']==0
    for source,destination in [(original,private/'qualification'),(method_root,private/'evaluation')]:
        destination.mkdir(exist_ok=True)
        for name in ['clock.json','config.json','EVENTS_FROZEN.private.json','BASE_BINDINGS.private.json',
            'PACKET_PRIVATE.json','DECISIONS.json','BASE_MASK_FROZEN.json','COMPLETE.json','GUARD_EXIT.json','COST.json','BATCH_MANIFEST.json']:
            if (source/name).is_file():shutil.copy2(source/name,destination/name)
    write(Path(cfg['report'])/'EXECUTION_AUDIT.json',dict(status='COMPLETE',remote_guard_exit=0,
        method_score_guard_exit=0,qualification_new_attempt_items=read(original/'COST.json')['attempt_items'],
        method_new_attempt_items=read(method_root/'COST.json')['attempt_items'] if (method_root/'COST.json').exists() else 0,
        generations=status['EVAL_GENERATION_COMPLETE.json'],clock=clock,main150_modified=False,
        source_only_publication=True,raw_material_private=True))
    # Publish only this run's aggregate report. Refuse to absorb unrelated staged work.
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=w,text=True).splitlines()
    assert not staged,'Unrelated staged work prevents automatic publication'
    relative=str(Path(cfg['report']).relative_to(w))
    subprocess.run(['git','add',relative],cwd=w,check=True)
    subprocess.run(['git','commit','-m','Report fixed T4 MDU single and actual final-bank CORE results'],cwd=w,check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=w,text=True).strip()
    network_env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',
        ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','HEAD:refs/heads/'+branch],cwd=w,env=network_env,check=True)
    remote_head=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+branch],cwd=w,env=network_env,text=True).split()[0]
    assert remote_head==revision
    url=cfg['public_raw_base']+'/'+revision+'/'+relative+'/RESULTS.json'
    probe=subprocess.run(['curl','--proxy','http://127.0.0.1:7897','--connect-timeout','8','--max-time','30',
        '-sS','-o','/dev/null','-w','%{http_code}',url],text=True,capture_output=True,env=network_env)
    assert probe.returncode==0 and probe.stdout=='200'
    write(private/'PUBLICATION_RECEIPT.json',dict(revision=revision,branch=branch,remote_verified=True,anonymous_HTTP=200))
    write(original/'FINISH_COMPLETE.json',dict(time=time.time(),revision=revision))


if __name__=='__main__':
    config=read(os.environ['T4_CONFIG'])
    try:main(config)
    except BaseException:
        write(Path(config['root'])/'FINISH_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
