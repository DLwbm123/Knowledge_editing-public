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
    state=json.loads((W/'.local_run/SCORERS.json').read_text())
    while True:
        live=[]
        for row in state['workers']:
            current=subprocess.run(['ps','-ww','-p',str(row['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
            live.append(current.split()[:5]==row['identity'].split()[:5] and current.split()[-2:]==['-u',row['entry']])
        if not any(live):break
        time.sleep(30)
    code="import os,runpy\nos.environ['RUN_ROOT']="+repr(R)+"\nrunpy.run_path("+repr(R+'/private/tools/report146.py')+",run_name='__main__')\n"
    output=remote(code);(W/'.local_run/REPORT_OUTPUT.txt').write_text(output)
    report=W/'reports/purew_original146_v1_20261012'
    data=json.loads(remote("import json,pathlib\nr=pathlib.Path("+repr(R)+")\nprint(json.dumps({p.name:p.read_text() for p in (r/'public').iterdir() if p.name in ['RESULTS.json','REPORT_ZH.md','REAL_VLM_ADMISSION.json','BASELINE_REUSE_ADMISSION.json','NATIVE_RELOAD_PERFORMANCE.json','CPU_VALIDATION.json']}))\n"))
    for name,text in data.items():
        if name.endswith('.json'):
            obj=json.loads(text)
            assert not any(x in text for x in ('raw_answer','raw_token_ids','opaque_Base_id','image_path','query_id','wangbomin@'))
        (report/name).write_text(text)
    paths=[str(p.relative_to(W)) for p in report.iterdir() if p.is_file()]
    subprocess.run(['git','add','--',*paths],cwd=W,check=True)
    subprocess.run(['git','commit','-m','Report frozen original146 pure W comparison with retained TT and BalancEdit'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','-c','push.default=current','push','origin'],cwd=W,check=True,timeout=180)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    actual=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+os.environ['BRANCH']],cwd=W,text=True).split()[0]
    assert actual==sha
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/reports/purew_original146_v1_20261012/RESULTS.json'
    assert json.load(urllib.request.urlopen(url,timeout=45))['status'].startswith('COMPLETE')
    receipt=dict(status='DELIVERED',commit=sha,remote_match=True,anonymous_status=200,epoch=time.time())
    (W/'.local_run/PUBLICATION.json').write_text(json.dumps(receipt,indent=2))
    remote('import pathlib,json\np=pathlib.Path('+repr(R+'/private/PUBLICATION.json')+');p.write_text('+repr(json.dumps(receipt))+')\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (W/'.local_run/FINITE_TAIL_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        raise
