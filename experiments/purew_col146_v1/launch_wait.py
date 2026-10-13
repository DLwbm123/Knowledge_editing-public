"""Freeze the published package locally and start one neutral finite dependency wait."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

W=Path(os.environ['WORKSPACE']);H=W/'experiments/purew_col146_v1';S=W/'.local_run/col146'
assert not (S/'WAIT_START.json').exists() and not (S/'INSTALL_ATTEMPT.json').exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
publication=json.loads((S/'SOURCE_PUBLICATION.json').read_text())
assert publication['commit']==commit and publication['remote_match'] and publication['anonymous']==200
files={p.name:p.read_text() for p in H.glob('*.py') if p.name not in ('bootstrap.py','deferred.py','finish.py','consumers.py','integration_check.py','launch_wait.py')}
files['base_run.py']=(W/'experiments/purew_original146_v1/run146.py').read_text()
package=dict(protocol=json.loads((W/'reports/purew_col146_v1_20261013/PROTOCOL.json').read_text()),
    files=files,norm_schedule=json.loads((W/'.local_run/C_NORM_SCHEDULE.json').read_text()),commit=commit,branch='experiment/purew-cpol146-v1-20261013')
(S/'FROZEN_PACKAGE.json').write_text(json.dumps(package,ensure_ascii=False))
folder=Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'));entry=folder/'e.py'
entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
env=dict(os.environ,WORKSPACE=str(W),TASK_FILE=str(H/'deferred.py'),
    HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',
    https_proxy='http://127.0.0.1:7897',http_proxy='http://127.0.0.1:7897',all_proxy='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(folder/'wait.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
identity=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],text=True).strip()
assert not any(s in identity for s in ('wangbomin','Knowledge_editing','purew'))
receipt=dict(pid=child.pid,entry=str(entry),identity=identity,log=str(folder/'wait.log'),commit=commit,epoch=time.time(),GPU_updates=0)
(S/'WAIT_START.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
