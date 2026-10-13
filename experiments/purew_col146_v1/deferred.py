"""Finite read-only dependency wait, then one installation and score/delivery consumers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

W=Path(os.environ['WORKSPACE']);S=W/'.local_run/col146';H=W/'experiments/purew_col146_v1'
sys.path.insert(0,str(H))


def remote(code):
    for attempt in range(3):
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','python3 -'],input=code,text=True,capture_output=True,timeout=60)
        if p.returncode==255 and attempt<2:
            time.sleep(2**(attempt+1));continue
        p.check_returncode();return json.loads(p.stdout)


def write(name,data):
    p=S/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def main():
    lock=(S/'WAIT.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (S/'INSTALL_ATTEMPT.json').exists(), 'Uncertain install requires manual receipt reconciliation'
    package=json.loads((S/'FROZEN_PACKAGE.json').read_text());p=package['protocol'];old=p['reference_run']
    probe=package['files']['dependency.py']+"\nimport time,subprocess\nr=Path("+repr(old)+")\nm=json.loads((r/'RUN_MANIFEST.json').read_text())\nprint(json.dumps(dict(ready=ready(r),old_deadline=m['deadline_epoch'],observed=time.time())))\n"
    while True:
        result=remote(probe);write('WAIT_STATUS.json',dict(status='WAITING_ORIGINAL_GPU_COMPLETION',**result))
        if result['ready']:break
        if result['observed']>result['old_deadline']+1800:raise TimeoutError('Original GPU did not retire within original deadline; review only, no new run')
        time.sleep(300)
    # Refuse a changed or unpublished executable rather than silently following HEAD.
    assert subprocess.check_output(['git','status','--porcelain','--','experiments/purew_col146_v1'],cwd=W,text=True).strip()==''
    for name,source in package['files'].items():
        local=H/name
        if local.exists():assert local.read_text()==source
    branch=package['branch'];commit=package['commit']
    actual=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+branch],cwd=W,text=True).split()[0]
    assert actual==commit, 'Published branch changed; inspect and rebind executable package explicitly'
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+commit+'/reports/purew_col146_v1_20261013/PROTOCOL.json'
    assert json.load(urllib.request.urlopen(url,timeout=45))==p
    write('INSTALL_ATTEMPT.json',dict(epoch=time.time(),commit=commit,attempt=1))
    stdin=json.dumps((H/'bootstrap.py').read_text())+'\n'+json.dumps(package)+'\n'
    installed=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=stdin,text=True,capture_output=True,timeout=300)
    write('INSTALL_RESPONSE.json',dict(returncode=installed.returncode,stdout=installed.stdout,stderr=installed.stderr))
    installed.check_returncode();startup=json.loads(installed.stdout)
    from consumers import start_consumers
    start_consumers(startup)
    write('WAIT_COMPLETE.json',dict(status='INSTALLED_AND_CONSUMERS_STARTED',epoch=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        write('WAIT_FAILURE.json',dict(epoch=time.time(),traceback=traceback.format_exc(),no_implicit_retry=True));raise
