"""Published-source, once-only quick deployment and finite result consumer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
W=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
STATE=W/'.local_run/ecw_v2_20261014';OUT=W/'reports/purew_ecw_v1_20261014/revision2'


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'Publish source first'
    protocol=json.loads((OUT/'QUICK_PROTOCOL.json').read_text());cpu=json.loads((OUT/'CPU_VALIDATION.json').read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    payload=dict(protocol=protocol,CPU=cpu,commit=sha,files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ['bootstrap.py','deploy.py','finish.py','cpu_check.py','consumers.py']})
    with (STATE/'INSTALL_ATTEMPT.json').open('x') as f:json.dump(dict(commit=sha,epoch=time.time(),run_root=protocol['run_root']),f)
    code=(HERE/'bootstrap.py').read_text()
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
            input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True,timeout=55)
    except subprocess.TimeoutExpired as exc:
        (STATE/'INSTALL_UNCERTAIN.json').write_text(json.dumps(dict(epoch=time.time(),reason='SSH timeout; inspect existing remote run; never reinstall',stdout=str(exc.stdout),stderr=str(exc.stderr))))
        raise
    (STATE/'INSTALL_RESPONSE.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
    assert result.returncode==0,result.stderr
    startup=json.loads(result.stdout);(STATE/'STARTUP.json').write_text(json.dumps(startup,indent=2))
    os.environ['WORKSPACE']=str(W)
    from consumers import start_consumers
    start_consumers(startup)



if __name__=='__main__':main()
