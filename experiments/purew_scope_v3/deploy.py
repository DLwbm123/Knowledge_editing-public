"""Local entry: deploy the frozen GPU chain, or start the existing isolated scorer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

WORK=Path(os.environ['WORKSPACE']).resolve()
RUN='/data/bmw/Knowledge_editing/outputs/purew-scope-v3-20261010/run'
HERE=WORK/'experiments/purew_scope_v3'


def remote(code):
    return json.loads(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000','python3 -'],
        input=code,capture_output=True,text=True,check=True,timeout=60).stdout)


def scorer_alive(worker, identity):
    fields=identity.split()
    # macOS Python execs its framework binary; birth time and neutral entry remain fixed.
    return len(fields)==8 and fields[:5]==worker['process_identity'].split()[:5] and fields[-2:]==['-u',worker['entry']]


def gpu():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--','experiments/purew_scope_v3','reports/purew_scope_v3_20261010'],cwd=WORK,text=True).strip(),'Publish frozen files before launch'
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','check.py','finish.py')}
    files['native_eval.py']=(WORK/'experiments/purew_scope_v2/run.py').read_text()
    files['propagation_math.py']=(WORK/'experiments/purew_scope_v2/propagation_math.py').read_text()
    files['journal.py']=(WORK/'experiments/purew_scope_v2/journal.py').read_text()
    payload=dict(files=files,protocol=json.loads((WORK/'reports/purew_scope_v3_20261010/PROTOCOL.json').read_text()),commit=commit)
    code=json.dumps((HERE/'bootstrap.py').read_text())+'\n'+json.dumps(payload)+'\n'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000',
        "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=code,capture_output=True,text=True,timeout=300)
    if p.returncode:raise RuntimeError(p.stderr[-3000:])
    print(p.stdout)


def score():
    result=remote('import json,pathlib\np=pathlib.Path('+repr(RUN)+')\nassert (p/"private/GENERATION_COMPLETE.json").is_file()\nprint(json.dumps({"generated":True,"reported":(p/"private/REPORT_COMPLETE.json").is_file()}))\n')
    if result['reported']:print(json.dumps(dict(status='REPORT_ALREADY_COMPLETE')));return
    state=WORK/'.local_run/scope_v3_scorer.json'
    remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(RUN)+',QUEUE_REQUEST=\'{"action":"ingest"}\')\nrunpy.run_path('+repr(RUN+'/private/tools/score.py')+',run_name="__main__")\n')
    prior=json.loads(state.read_text()) if state.exists() else {}
    root=Path(prior['root']) if prior else Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'))
    workers=[]
    for worker in range(4):
        old=next((w for w in prior.get('workers',[]) if w['worker']==worker),None)
        if old:
            found=subprocess.run(['ps','-ww','-p',str(old['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
            if scorer_alive(old,found):
                workers.append(old);continue
        folder=root/str(worker);folder.mkdir(mode=0o700,exist_ok=True)
        if old:
            (folder/('RESTART_'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(old,reason='Recorded worker ended; recover retained accepted responses, never resubmit reserved Judge inputs')))
        env=dict(os.environ,RUN_ROOT=RUN,SCORER_STATE=str(folder),JUDGE_SHARED_ROOT=str(root),
            SCORER_SOURCE=str(WORK/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),
            SCORER_SCRIPT=str(WORK/'experiments/medtrace_full_method_comparison_v2_20261003/scorer.py'),
            JUDGE_QUEUE_FILE='score.py',JUDGE_RESULT_FOLDER='judge_scope_v3_astra_medium/worker'+str(worker),
            JUDGE_EPOCH='PUREW_SCOPE_V3_20261010',JUDGE_EFFORT='medium',JUDGE_WORKER=str(worker))
        assert Path(env['SCORER_SOURCE']).is_dir()
        entry=root/(str(worker)+'.py');entry.write_text("import os,runpy\nrunpy.run_path(os.environ['SCORER_SCRIPT'],run_name='__main__')\n")
        child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(folder/'log.txt').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        identity=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],text=True).strip()
        workers.append(dict(worker=worker,pid=child.pid,process_identity=identity,started=time.time(),entry=str(entry),log=str(folder/'log.txt')))
        value=dict(root=str(root),workers=workers,epoch=time.time())
        state.parent.mkdir(parents=True,exist_ok=True)
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(state)
    value=dict(root=str(root),workers=workers,epoch=time.time())
    temp=state.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(state)
    remote('import json,pathlib\np=pathlib.Path('+repr(RUN+'/private/SCORER_ACTIVE_START.json')+')\np.write_text('+repr(json.dumps(value))+')\nprint(json.dumps({"saved":True}))\n')
    time.sleep(2)
    for w in workers:
        command=subprocess.check_output(['ps','-ww','-p',str(w['pid']),'-o','args='],text=True).strip()
        assert command and not any(x in command for x in ('wangbomin','Knowledge_editing','scope'))
    print(json.dumps(dict(status='SCORERS_STARTED',state=value)))


if __name__=='__main__':
    {'gpu':gpu,'score':score}[os.environ.get('DEPLOY_STAGE','gpu')]()
