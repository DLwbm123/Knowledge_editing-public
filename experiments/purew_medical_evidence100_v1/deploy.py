"""Validated published package, one install attempt, and finite score/delivery consumers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
W=Path(os.environ.get('WORKSPACE',Path(__file__).resolve().parents[2]))
HERE=W/'experiments/purew_medical_evidence100_v1';OUT=W/'reports/purew_medical_evidence100_v1_20261014'
STATE=W/'.local_run/medical_evidence100_v1_20261014/execution';BRANCH='experiment/purew-medical-evidence100-v1-20261014'


def package():
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','finish.py','consumers.py')}
    files['base_run.py']=(W/'experiments/purew_original146_v1/run146.py').read_text()
    files['logic.py']=(W/'experiments/purew_protect100_v1/logic.py').read_text()
    return dict(files=files,protocol=json.loads((OUT/'PROTOCOL.json').read_text()),
                authorization=json.loads((OUT/'EXECUTION_AUTHORIZATION.json').read_text()),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),branch=BRANCH)


def remote(code, extra=None):
    data=json.dumps(code)+'\n'+(json.dumps(extra)+'\n' if extra is not None else '')
    try:
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',
               "/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
               input=data,text=True,capture_output=True,timeout=300)
    except subprocess.TimeoutExpired as error:
        STATE.mkdir(parents=True,exist_ok=True)
        (STATE/('CPU_TIMEOUT_'+str(int(time.time()))+'.json')).write_text(json.dumps(dict(error=str(error),
            stdout=str(error.stdout),stderr=str(error.stderr),epoch=time.time())))
        raise
    if p.returncode:
        STATE.mkdir(parents=True,exist_ok=True)
        (STATE/('REMOTE_FAILURE_'+str(int(time.time()))+'.json')).write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
        raise RuntimeError(p.stderr)
    return p.stdout


def validate():
    STATE.mkdir(parents=True,exist_ok=True)
    result=remote((HERE/'integration_check.py').read_text(),package())
    (STATE/'CPU_OUTPUT.txt').write_text(result)
    data=json.loads(result.strip().splitlines()[-1]);assert data['status']=='PASS'
    (OUT/'CPU_VALIDATION.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))


def launch():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(), 'Publish first'
    payload=package();payload['CPU_validation']=json.loads((OUT/'CPU_VALIDATION.json').read_text())
    private=W/'.local_run/medical_evidence100_v1_20261014'
    payload['source']=json.loads((private/'SOURCE_PACKET_PRIVATE.json').read_text())
    payload['review']=json.loads((private/'SOURCE_REVIEW_PRIVATE.json').read_text())
    assert payload['CPU_validation']['status']=='PASS'
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'INSTALL_ATTEMPT.json').open('x') as f:json.dump(dict(commit=payload['commit'],epoch=time.time()),f)
    result=remote((HERE/'bootstrap.py').read_text(),payload)
    (STATE/'INSTALL_RESPONSE.txt').write_text(result)
    startup=json.loads(result.strip().splitlines()[-1])
    os.environ['WORKSPACE']=str(W)
    sys.path.insert(0,str(HERE));import consumers
    consumers.start_consumers(startup)


if __name__=='__main__':{'validate':validate,'launch':launch}[sys.argv[1]]()
