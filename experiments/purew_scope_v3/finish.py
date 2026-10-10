"""One-shot dependent scoring/report/publication tail, not a recurring monitor."""
import json,os,runpy,subprocess,sys,time,traceback,urllib.request
from pathlib import Path

WORK=Path(os.environ['WORKSPACE']);OUT=WORK/'reports/purew_scope_v3_20261010';RUN='/data/bmw/Knowledge_editing/outputs/purew-scope-v3-20261010/run'
sys.path.insert(0,str(WORK/'experiments/purew_scope_v3'));import deploy
STATE=WORK/'.local_run/scope_v3_finish.json'

def write(status,**extra):
    value=dict(status=status,epoch=time.time(),**extra);temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(STATE)

def snapshot():
    return deploy.remote('import json,pathlib,time\np=pathlib.Path('+repr(RUN)+')\nm=json.loads((p/"RUN_MANIFEST.json").read_text())\nprint(json.dumps(dict(deadline=m["deadline_epoch"],generated=(p/"private/GENERATION_COMPLETE.json").is_file(),reported=(p/"private/REPORT_COMPLETE.json").is_file(),scored=(p/"private/judge_scope_v3_astra_medium/ALL_WORKERS_COMPLETE.json").is_file(),failures=[f.name for f in (p/"private/failures").glob("*.json")],now=time.time())))\n')

def publication():
    result=deploy.remote('import json,pathlib\np=pathlib.Path('+repr(RUN)+')\nprint((p/"public/RESULTS.json").read_text())\n')
    def privacy(x):
        if isinstance(x,dict):
            assert not set(x)&{'question','reference','R0','image_path','query_id','opaque_query_id','record','paths'}
            for y in x.values():privacy(y)
        elif isinstance(x,list):
            for y in x:privacy(y)
    privacy(result);(OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    os.environ['WORKSPACE']=str(WORK);runpy.run_path(str(WORK/'experiments/purew_scope_v3/render_report.py'),run_name='__main__')
    audit=dict(status='COMPLETED_WITH_MISSING' if result['missing_consumers'] else 'COMPLETE',structural_checks_passed=result['structural_checks_passed'],scoring_status=result['status'],resource=result['resource'],payload_status=result['payload_status'],old_failures_not_retried=True,results_not_clinical_confirmation=True)
    (OUT/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    st=json.loads((OUT/'EXECUTION_STATUS.json').read_text());st.update(status='EXPERIMENT_TERMINAL_PUBLICATION_PENDING',training_complete=True,generation_complete=True,scoring_terminal=True,scoring_complete=result['status']=='COMPLETE',DEV_success_by_arm=result['DEV_success_by_arm'],missing_consumers=result['missing_consumers']);(OUT/'EXECUTION_STATUS.json').write_text(json.dumps(st,indent=2)+'\n')
    branch='experiment/purew-scope-v3-20261010'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=WORK,text=True).strip()==branch,'Publication checkout changed; retain result and request review'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=WORK).returncode==0,'Other staged work; do not commit it'
    names=['RESULTS.json','REPORT.html','COMPLETION_AUDIT.json','EXECUTION_STATUS.json']
    subprocess.run(['git','add','--pathspec-from-file=-'],input=''.join('reports/purew_scope_v3_20261010/'+n+'\n' for n in names),cwd=WORK,text=True,check=True)
    subprocess.run(['git','commit','-m','Publish anonymous experiment results and completion audit'],cwd=WORK,check=True)
    proxy='http://127.0.0.1:7897';env=dict(os.environ,HTTPS_PROXY=proxy,HTTP_PROXY=proxy,ALL_PROXY=proxy,NO_PROXY='',no_proxy='')
    subprocess.run(['git','-c','http.proxy='+proxy,'push','origin','HEAD'],cwd=WORK,env=env,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip();refs=subprocess.check_output(['git','-c','http.proxy='+proxy,'ls-remote','origin'],cwd=WORK,text=True,env=env)
    assert head+'\trefs/heads/'+branch in refs
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}))
    url='https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/'+head+'/reports/purew_scope_v3_20261010/RESULTS.json'
    with op.open(url,timeout=60) as resp:assert resp.status==200 and json.load(resp)['payload_status']==result['payload_status']
    receipt=dict(status='PUBLICATION_VERIFIED',commit=head,anonymous_access=True,proxy=proxy,finite_delivery_complete=True)
    (OUT/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');st.update(status='FINITE_DELIVERY_COMPLETE',publication_complete=True);(OUT/'EXECUTION_STATUS.json').write_text(json.dumps(st,indent=2)+'\n')
    subprocess.run(['git','add','--pathspec-from-file=-'],input='reports/purew_scope_v3_20261010/PUBLICATION_RECEIPT.json\nreports/purew_scope_v3_20261010/EXECUTION_STATUS.json\n',cwd=WORK,text=True,check=True)
    subprocess.run(['git','commit','-m','Record verified public delivery'],cwd=WORK,check=True);subprocess.run(['git','-c','http.proxy='+proxy,'push','origin','HEAD'],cwd=WORK,env=env,check=True)
    deploy.remote('import json,pathlib\np=pathlib.Path('+repr(RUN+'/private/PUBLICATION_RECEIPT.json')+')\np.write_text('+repr(json.dumps(receipt))+')\nprint(json.dumps({"saved":True}))\n')
    write('FINITE_DELIVERY_COMPLETE',publication=receipt)


def main():
    import fcntl
    with (WORK/'.local_run/scope_v3_finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists() and json.loads(STATE.read_text())['status']=='FINITE_DELIVERY_COMPLETE':return
        write('WAITING_FOR_GPU_CONSUMERS')
        while True:
            x=snapshot()
            if x['generated']:break
            assert not x['failures'],'GPU engineering failure; preserve state and await diagnosis'
            assert x['now']<x['deadline'],'Frozen deadline reached; never extend'
            time.sleep(30)
        deploy.score();write('SCORING_DEPENDENT_STAGE')
        while True:
            x=snapshot()
            if x['scored']:break
            assert x['now']<x['deadline'];time.sleep(30)
        # Wait only for the four owned workers to finish archiving before reporting.
        workers=json.loads((WORK/'.local_run/scope_v3_scorer.json').read_text())['workers']
        for worker in workers:
            while True:
                identity=subprocess.run(['ps','-ww','-p',str(worker['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
                if not deploy.scorer_alive(worker,identity):break
                assert time.time()<x['deadline'];time.sleep(2)
        deploy.remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(RUN)+',QUEUE_REQUEST=\'{"action":"report"}\')\nrunpy.run_path('+repr(RUN+'/private/tools/score.py')+',run_name="__main__")\n')
        publication()

if __name__=='__main__':
    try:main()
    except BaseException as error:write('STOPPED_WITH_RETAINED_EVIDENCE',error=repr(error),traceback=traceback.format_exc());raise
