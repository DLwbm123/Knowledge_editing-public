"""Finite quick-result publisher, not a recurring monitor."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import traceback
W=Path(os.environ['WORKSPACE']);STATE=W/'.local_run/ecw_v1_20261014';R=os.environ['RUN_ROOT']


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n',text=True,capture_output=True,timeout=55,check=True)
    return json.loads(p.stdout)


def main():
    if (STATE/'PUBLICATION_FINAL.json').exists():return
    while True:
        d=remote("from pathlib import Path\nimport json\nr=Path("+repr(R)+");l=json.loads((r/'RESOURCE_LEDGER.json').read_text());print(json.dumps(dict(active=any(not x.get('ended_epoch') for x in l['gpu_sessions']),done=(r/'private/EXIT_controller.json').exists(),failed=bool(list((r/'private/failures').glob('*.json'))),report=(r/'public/QUICK_RESULTS.json').exists())))\n")
        if not d['active'] and d['report'] and (d['done'] or d['failed']):break
        if not d['active'] and (d['done'] or d['failed']) and not d['report']:
            (STATE/'FINISH_REVIEW_REQUIRED.json').write_text(json.dumps(dict(epoch=time.time(),state=d,reason='Stage exited without report; preserve evidence and inspect')))
            return
        time.sleep(20)
    names=['QUICK_RESULTS.json','QUICK_REPORT_ZH.md','BASE_IDENTITY.json','REAL_VLM_ADMISSION.json','NATIVE_RELOAD.json','MEDICAL_MECHANICAL.json']
    data=remote("from pathlib import Path\nimport json\nr=Path("+repr(R+'/public')+");print(json.dumps({n:(r/n).read_text() for n in "+repr(names)+" if (r/n).exists()}))\n")
    folder=W/'reports/purew_ecw_v1_20261014';paths=[]
    for n,text in data.items():
        assert not any(s in text for s in ['raw_answer','raw_token_ids','query_id','image_path','opaque_Base_id','wangbomin@'])
        p=folder/n;p.write_text(text);paths.append(str(p.relative_to(W)))
    env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    subprocess.run(['git','add','--',*paths],cwd=W,check=True);subprocess.run(['git','commit','-m','Deliver bounded ECW GPU4 implementation validation and retained costs'],cwd=W,check=True)
    subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin',os.environ['BRANCH']],cwd=W,env=env,check=True,timeout=120)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    actual=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:7897','ls-remote','origin','refs/heads/'+os.environ['BRANCH']],cwd=W,env=env,text=True,timeout=45).split()[0];assert actual==sha
    os.environ.update({k:env[k] for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','no_proxy']})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':env['HTTP_PROXY'],'https':env['HTTPS_PROXY']}))
    with opener.open('https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+sha+'/reports/purew_ecw_v1_20261014/QUICK_RESULTS.json',timeout=45) as response:assert response.status==200
    (STATE/'PUBLICATION_FINAL.json').write_text(json.dumps(dict(commit=sha,remote_match=True,anonymous_status=200,epoch=time.time())))


if __name__=='__main__':
    try:main()
    except BaseException:(STATE/'FINISH_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())));raise
