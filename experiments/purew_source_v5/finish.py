"""One finite dependent chain: qualification, admission, training, score, delivery."""
import os,json,time,traceback,subprocess,sys,urllib.request,html,fcntl
from pathlib import Path
W=Path(os.environ['WORKSPACE']);OUT=W/'reports/purew_source_v5_20261011'
sys.path.insert(0,str(W/'experiments/purew_source_v5'));import deploy
STATE=W/'.local_run/source_v5_finish.json'

def write(status,**extra):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,epoch=time.time(),**extra),indent=2));p.replace(STATE)

def wait_for(name):
    while True:
        s=deploy.snapshot()
        if s[name]:return s
        assert not s['failures'],'Owned stage failed; no implicit retry'
        assert s['now']<s['deadline'],'Frozen deadline reached; never reset'
        time.sleep(20)

def publication():
    r=deploy.remote('import pathlib,json\np=pathlib.Path('+repr(deploy.R+'/public/RESULTS.json')+')\nprint(p.read_text())\n')
    def privacy(value):
        if isinstance(value,dict):
            assert not set(value)&{'question','reference','image_path','query_id','source_qid','raw_answer','raw_token_ids','paths'}
            for x in value.values():privacy(x)
        elif isinstance(value,list):
            for x in value:privacy(x)
    privacy(r);(OUT/'RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    parts=['<h1>新源题 U/H-source 有限验证</h1><p>零新增推理参数。不同于历史源题＋GFIT合同；T2G-style同图问法分为结构改写与原命题完整保留的输入重述，不声称完整M3Bench复现或临床独立验证。</p>','<pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre>']
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>U/H-source</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.7 system-ui}pre{white-space:pre-wrap}</style>'+''.join(parts)+'</html>')
    st=dict(status='EXPERIMENT_TERMINAL_PUBLICATION_PENDING',scientific_status=r['status'],scoring_terminal=True,publication_complete=False,formal_updates=0 if r['status']=='STOP_DATA_QUALIFICATION_NOT_MET' else 7680)
    (OUT/'EXECUTION_STATUS.json').write_text(json.dumps(st,indent=2)+'\n')
    admission=deploy.remote('import pathlib,json\np=pathlib.Path('+repr(deploy.R+'/public/DATA_ADMISSION.json')+')\nprint(p.read_text() if p.exists() else "{}")\n')
    if admission:(OUT/'DATA_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()=='experiment/purew-source-v5-20261011'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=W).returncode==0
    names=['RESULTS.json','REPORT.html','EXECUTION_STATUS.json']+(['DATA_ADMISSION.json'] if admission else [])
    subprocess.run(['git','add','--pathspec-from-file=-'],input=''.join('reports/purew_source_v5_20261011/'+x+'\n' for x in names),cwd=W,text=True,check=True)
    msg=W/'.local_run/message.txt';msg.write_text('Publish anonymous source-history experiment results and full denominators\n')
    subprocess.run(['git','commit','-F',str(msg)],cwd=W,check=True)
    proxy='http://127.0.0.1:7897';env=dict(os.environ,HTTP_PROXY=proxy,HTTPS_PROXY=proxy,ALL_PROXY=proxy,NO_PROXY='',no_proxy='')
    subprocess.run(['git','-c','http.proxy='+proxy,'push','origin','HEAD'],cwd=W,env=env,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    ref=subprocess.check_output(['git','-c','http.proxy='+proxy,'ls-remote','origin','refs/heads/experiment/purew-source-v5-20261011'],cwd=W,env=env,text=True);assert ref.split()[0]==head
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
    with op.open('https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+head+'/reports/purew_source_v5_20261011/RESULTS.json',timeout=60) as response:assert response.status==200 and json.load(response)['status']==r['status']
    receipt=dict(status='PUBLICATION_VERIFIED',commit=head,anonymous_access=True,proxy=proxy,finite_delivery_complete=True)
    (OUT/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');st.update(status='FINITE_DELIVERY_COMPLETE',publication_complete=True);(OUT/'EXECUTION_STATUS.json').write_text(json.dumps(st,indent=2)+'\n')
    subprocess.run(['git','add','--pathspec-from-file=-'],input='reports/purew_source_v5_20261011/PUBLICATION_RECEIPT.json\nreports/purew_source_v5_20261011/EXECUTION_STATUS.json\n',cwd=W,text=True,check=True)
    msg.write_text('Record verified finite public delivery\n');subprocess.run(['git','commit','-F',str(msg)],cwd=W,check=True);subprocess.run(['git','-c','http.proxy='+proxy,'push','origin','HEAD'],cwd=W,env=env,check=True)
    deploy.remote('import pathlib,json\np=pathlib.Path('+repr(deploy.R+'/private/PUBLICATION_RECEIPT.json')+')\np.write_text('+repr(json.dumps(receipt))+')\nprint(json.dumps({"saved":True}))\n')
    write('FINITE_DELIVERY_COMPLETE',publication=receipt)

def main():
    with (W/'.local_run/source_v5_finish.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists() and json.loads(STATE.read_text())['status']=='FINITE_DELIVERY_COMPLETE':return
        write('WAITING_FOR_SOURCE_BASE_QUALIFICATION');wait_for('qual_generated');deploy.score('qualification')
        write('SOURCE_BASE_SCORING');wait_for('qual_scored');admission=deploy.admit_and_launch()
        if admission['status']=='STOP_DATA_QUALIFICATION_NOT_MET':publication();return
        write('ADMITTED_GPU_CHAIN_RUNNING',admission=admission);wait_for('generated');deploy.score('formal')
        write('FORMAL_SCORING');wait_for('scored')
        deploy.remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(deploy.R)+',QUEUE_REQUEST=\'{"action":"report"}\')\nrunpy.run_path('+repr(deploy.R+'/private/tools/score.py')+',run_name="__main__")\n')
        publication()

if __name__=='__main__':
    try:main()
    except BaseException as e:write('STOPPED_WITH_RETAINED_EVIDENCE',error=repr(e),traceback=traceback.format_exc());raise
