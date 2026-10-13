"""Finite dependency tail: all four workers, report, then source-only public delivery."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request

W=Path(os.environ['WORKSPACE']);R=os.environ['RUN_ROOT']


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','python3 -'],input=code,text=True,capture_output=True,timeout=300,check=True)
    return p.stdout


def main():
    state=json.loads((W/'.local_run/col146/SCORERS.json').read_text())
    while True:
        live=[]
        for row in state['workers']:
            current=subprocess.run(['ps','-ww','-p',str(row['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
            live.append(current.split()[:5]==row['identity'].split()[:5] and current.split()[-2:]==['-u',row['entry']])
        if not any(live):break
        time.sleep(30)
    while True:
        status=json.loads(remote("import json,pathlib,time\nr=pathlib.Path("+repr(R)+")\nd=json.loads((r/'RESOURCE_LEDGER.json').read_text())\nm=json.loads((r/'RUN_MANIFEST.json').read_text())\nprint(json.dumps(dict(active=any(not x.get('ended_epoch') for x in d['gpu_sessions']),deadline=m['deadline_epoch'],now=time.time(),ended=(r/'private/EXIT_controller.json').exists() or (r/'private/failures/controller.json').exists())))\n"))
        if status['ended'] and not status['active']:break
        assert status['now']<status['deadline']+120, 'GPU stages require exit/lease review before reporting'
        time.sleep(30)
    code="import os,runpy\nos.environ['RUN_ROOT']="+repr(R)+"\nrunpy.run_path("+repr(R+'/private/tools/report_col.py')+",run_name='__main__')\n"
    output=remote(code);(W/'.local_run/col146/REPORT_OUTPUT.txt').write_text(output)
    report=W/'reports/purew_col146_v1_20261013'
    data=json.loads(remote("import json,pathlib\nr=pathlib.Path("+repr(R)+")\nprint(json.dumps({p.name:p.read_text() for p in (r/'public').iterdir() if p.name in ['RESULTS.json','REPORT_ZH.md','REAL_VLM_ADMISSION.json','BASELINE_REUSE_ADMISSION.json','CPU_VALIDATION.json']}))\n"))
    for name,text in data.items():
        if name.endswith('.json'):
            obj=json.loads(text)
            assert not any(x in text for x in ('raw_answer','raw_token_ids','opaque_Base_id','image_path','query_id','wangbomin@'))
        (report/name).write_text(text)
    paths=[str((report/name).relative_to(W)) for name in data]
    subprocess.run(['git','add','--',*paths],cwd=W,check=True)
    subprocess.run(['git','commit','-m','Report frozen original146 C O L controls with retained missing bounds'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','-c','push.default=current','push','origin'],cwd=W,check=True,timeout=180)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    actual=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+os.environ['BRANCH']],cwd=W,text=True).split()[0]
    assert actual==sha
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/reports/purew_col146_v1_20261013/RESULTS.json'
    assert json.load(urllib.request.urlopen(url,timeout=45))['status'] in ('COMPLETE_WITH_MISSING','PARTIAL_NOT_ALL_CONSUMERS_COMPLETE')
    receipt=dict(status='DELIVERED',commit=sha,remote_match=True,anonymous_status=200,epoch=time.time())
    (W/'.local_run/col146/PUBLICATION.json').write_text(json.dumps(receipt,indent=2))
    remote('import pathlib,json\np=pathlib.Path('+repr(R+'/private/PUBLICATION.json')+');p.write_text('+repr(json.dumps(receipt))+')\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (W/'.local_run/col146/FINITE_TAIL_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        raise
