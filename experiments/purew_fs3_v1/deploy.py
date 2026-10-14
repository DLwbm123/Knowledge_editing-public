"""Build from audited executors; one install after finite CPU validation and publication."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
W=Path(os.environ.get('WORKSPACE',Path(__file__).resolve().parents[2]))
HERE=W/'experiments/purew_fs3_v1';OUT=W/'reports/purew_fs3_v1_20261014'
STATE=W/'.local_run/fs3_v1_20261014/execution';BRANCH='experiment/purew-fs3-v1-20261014'


def replace_once(source,old,new):
    assert source.count(old)==1,(old,source.count(old));return source.replace(old,new)


def package():
    medical=W/'experiments/purew_medical_evidence100_v1'
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','finish.py','consumers.py')}
    for name in ('runtime.py','access.py','curvature.py'):
        if name not in files:files[name]=(medical/name).read_text()
    files['base_controller.py']=(medical/'controller.py').read_text()
    files['base_evaluator.py']=(medical/'evaluator.py').read_text()
    s=(medical/'learner.py').read_text();s=replace_once(s,'range(start_owner,101)','range(start_owner,4)')
    s=replace_once(s,"dict(status='COMPLETE',attempt_slots=16000)","dict(status='COMPLETE',attempt_slots=480)")
    s=replace_once(s,'                    admission(runtime,weight,full,removed,shape,frozen)',"                    assert c.read(R/'public'/('REAL_VLM_ADMISSION_'+ARM+'.json'))['status']=='PASS'")
    files['base_learner.py']=s
    s=(medical/'checkpoint.py').read_text();s=replace_once(s,'import gzip','import gzip\nimport io\nimport shutil')
    s=replace_once(s,"with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=True)",
        "buffer=io.BytesIO()\n        with gzip.open(p,'rb') as f:shutil.copyfileobj(f,buffer,1024*1024)\n        buffer.seek(0)\n        try:return torch.load(buffer,map_location='cpu',weights_only=True)\n        finally:buffer.close()")
    files['checkpoint.py']=s;files['base_queue.py']=(medical/'queue_medical.py').read_text()
    files['base_run.py']=(W/'experiments/purew_original146_v1/run146.py').read_text()
    files['logic.py']=(W/'experiments/purew_protect100_v1/logic.py').read_text()
    return dict(files=files,protocol=json.loads((OUT/'PROTOCOL.json').read_text()),authorization=json.loads((OUT/'EXECUTION_AUTHORIZATION.json').read_text()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),branch=BRANCH)


def remote(code,extra=None,timeout=55):
    data=json.dumps(code)+'\n'+(json.dumps(extra)+'\n' if extra is not None else '')
    STATE.mkdir(parents=True,exist_ok=True)
    try:
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',
            "/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=data,text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as e:
        (STATE/('REMOTE_TIMEOUT_'+str(int(time.time()))+'.json')).write_text(json.dumps(dict(timeout=timeout,stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout,stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr)))
        raise
    STATE.mkdir(parents=True,exist_ok=True)
    if p.returncode:
        (STATE/('REMOTE_FAILURE_'+str(int(time.time()))+'.json')).write_text(json.dumps(dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)))
        raise RuntimeError(p.stderr)
    return p.stdout


def validate():
    result=remote((HERE/'integration_check.py').read_text(),package())
    (STATE/'CPU_OUTPUT.txt').write_text(result);data=json.loads(result.strip().splitlines()[-1]);assert data['status']=='PASS'
    (OUT/'CPU_VALIDATION.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))


def launch():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'Publish frozen code first'
    data=package();data['CPU_validation']=json.loads((OUT/'CPU_VALIDATION.json').read_text());assert data['CPU_validation']['status']=='PASS'
    private=W/'.local_run/medical_evidence100_v1_20261014'
    src=json.loads((private/'SOURCE_PACKET_PRIVATE.json').read_text());data['source']=dict(src,rows=src['rows'][:3])
    with (STATE/'INSTALL_ATTEMPT.json').open('x') as f:json.dump(dict(commit=data['commit'],epoch=time.time()),f)
    result=remote((HERE/'bootstrap.py').read_text(),data)
    (STATE/'INSTALL_RESPONSE.txt').write_text(result);startup=json.loads(result.strip().splitlines()[-1])
    os.environ['WORKSPACE']=str(W);sys.path.insert(0,str(HERE));import consumers
    consumers.start_consumers(startup)


if __name__=='__main__':{'validate':validate,'launch':launch}[sys.argv[1]]()
