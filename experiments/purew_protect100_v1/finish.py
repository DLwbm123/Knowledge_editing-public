"""Finite score/report/publication consumer; never marks an unfinished run successful."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request

W=Path(os.environ['WORKSPACE']);R=os.environ['RUN_ROOT'];STATE=W/'.local_run/protect100'


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',
          "/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
          input=json.dumps(code)+'\n',text=True,capture_output=True,timeout=300,check=True)
    return p.stdout


def alive(row):
    p=subprocess.run(['ps','-ww','-p',str(row['pid']),'-o','lstart=,args='],capture_output=True,text=True)
    text=p.stdout.strip()
    return p.returncode==0 and text.split()[:5]==row['identity'].split()[:5] and text.split()[-2:]==['-u',row['entry']]


def main():
    if (STATE/'PUBLICATION_FINAL.json').exists():return
    workers=json.loads((STATE/'SCORERS.json').read_text())['workers']
    while any(alive(row) for row in workers):time.sleep(30)
    status=json.loads(remote("from pathlib import Path\nimport json\nr=Path("+repr(R)+")\nd=json.loads((r/'RESOURCE_LEDGER.json').read_text())\nprint(json.dumps(dict(active=any(not s.get('ended_epoch') for s in d['gpu_sessions']),done=(r/'private/EXIT_controller.json').exists(),failed=(r/'private/failures/controller.json').exists(),scientific_stop=(r/'public/ADMISSION_REJECTED.json').exists(),reviewed=(r/'private/STOP_REVIEWED.json').exists())))\n"))
    if status['active'] or not (status['done'] or status['scientific_stop'] or status['reviewed']):
        (STATE/'TAIL_REVIEW_REQUIRED.json').write_text(json.dumps(dict(status=status,epoch=time.time())))
        return
    output=remote("import os,runpy\nos.environ['RUN_ROOT']="+repr(R)+"\nrunpy.run_path("+repr(R+'/private/tools/protect_report.py')+",run_name='__main__')\n")
    (STATE/'REPORT_OUTPUT.txt').write_text(output)
    names=['RESULTS.json','REPORT_ZH.md','REAL_VLM_ADMISSION.json','ADMISSION_REJECTED.json','NATIVE_RELOAD.json','CPU_VALIDATION.json']
    data=json.loads(remote("from pathlib import Path\nimport json\np=Path("+repr(R+'/public')+")\nprint(json.dumps({n:(p/n).read_text() for n in "+repr(names)+" if (p/n).exists()}))\n"))
    report=W/'reports/purew_protect100_v1_20261013'
    for name,text in data.items():
        assert not any(x in text for x in ('raw_answer','raw_token_ids','opaque_Base_id','image_path','query_id','wangbomin@'))
        (report/name).write_text(text)
    paths=[str((report/name).relative_to(W)) for name in data]
    env=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    subprocess.run(['git','add','--',*paths],cwd=W,check=True)
    subprocess.run(['git','commit','-m','Deliver continuous pure-W KL protection results and retained costs'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin',os.environ['BRANCH']],cwd=W,env=env,check=True,timeout=180)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    actual=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+os.environ['BRANCH']],cwd=W,env=env,text=True).split()[0]
    assert actual==sha
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':env['HTTP_PROXY'],'https':env['HTTPS_PROXY']}))
    with opener.open('https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/reports/purew_protect100_v1_20261013/RESULTS.json',timeout=45) as response:
        assert response.status==200 and json.load(response)['status'] in ('COMPLETE_WITH_MISSING','PARTIAL_NOT_ALL_CONSUMERS_COMPLETE')
    receipt=dict(status='DELIVERED',commit=sha,remote_match=True,anonymous_status=200,epoch=time.time())
    (STATE/'PUBLICATION_FINAL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    remote("from pathlib import Path\np=Path("+repr(R+'/private/PUBLICATION.json')+")\np.write_text("+repr(json.dumps(receipt))+"+'\\n')\n")


if __name__=='__main__':
    try:main()
    except BaseException:
        (STATE/'FINITE_TAIL_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        raise
