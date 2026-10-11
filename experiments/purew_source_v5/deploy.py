"""Published source-only finite chain; neutral entry points and isolated scorers."""
import os,json,subprocess,sys,tempfile,time
from pathlib import Path
W=Path(os.environ['WORKSPACE']);HERE=W/'experiments/purew_source_v5'
R='/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run'

def remote(code):return json.loads(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=300).stdout)

def gpu():
    assert not subprocess.check_output(['git','status','--porcelain','--','experiments/purew_source_v5','reports/purew_source_v5_20261011'],cwd=W,text=True).strip()
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','finish.py')}
    files['journal.py']=(W/'experiments/purew_scope_v2/journal.py').read_text();files['shared_math.py']=(W/'experiments/purew_scope_v4/shared_math.py').read_text()
    payload=dict(files=files,protocol=json.loads((W/'reports/purew_source_v5_20261011/PROTOCOL.json').read_text()),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip())
    code=json.dumps((HERE/'bootstrap.py').read_text())+'\n'+json.dumps(payload)+'\n'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=code,text=True,capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    return json.loads(p.stdout)

def snapshot():
    return remote('import pathlib,json,time\np=pathlib.Path('+repr(R)+')\nprint(json.dumps(dict(deadline=json.loads((p/"RUN_MANIFEST.json").read_text())["deadline_epoch"],now=time.time(),qual_generated=(p/"private/QUAL_GENERATION_COMPLETE.json").is_file(),qual_scored=(p/"private/judge_source_v5_qualification/ALL_WORKERS_COMPLETE.json").is_file(),admitted=(p/"private/DATA_ADMISSION.json").is_file(),generated=(p/"private/GENERATION_COMPLETE.json").is_file(),scored=(p/"private/judge_source_v5_formal/ALL_WORKERS_COMPLETE.json").is_file(),failures=[x.name for x in (p/"private/failures").glob("*.json")],data_stop=(p/"private/DATA_GATE_STOP.json").is_file())))\n')

def score(phase):
    assert phase in ('qualification','formal');root='judge_source_v5_'+phase
    x=remote('import pathlib,json\np=pathlib.Path('+repr(R+'/private/'+root)+')\nprint(json.dumps(dict(done=(p/"ALL_WORKERS_COMPLETE.json").is_file())))\n')
    if x['done']:return dict(status='SCORING_ALREADY_TERMINAL')
    script='qual_queue.py' if phase=='qualification' else 'score.py'
    remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(R)+',QUEUE_REQUEST=\'{"action":"ingest"}\')\nrunpy.run_path('+repr(R+'/private/tools/'+script)+',run_name="__main__")\n')
    state=W/'.local_run'/('source_v5_scorer_'+phase+'.json');prior=json.loads(state.read_text()) if state.exists() else {}
    folder=Path(prior['root']) if prior else Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'))
    workers=[]
    for i in range(4):
        old=next((x for x in prior.get('workers',[]) if x['worker']==i),None)
        if old:
            identity=subprocess.run(['ps','-ww','-p',str(old['pid']),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
            if identity.split()[:5]==old['process_identity'].split()[:5] and identity.split()[-2:]==['-u',old['entry']]:workers.append(old);continue
            ended=remote('import pathlib,json\np=pathlib.Path('+repr(R+'/private/'+root+'/worker'+str(i)+'/SCORER_DONE.json')+')\nprint(json.dumps(json.loads(p.read_text()) if p.exists() else {}))\n')
            if ended:
                assert ended['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING' and ended['state']['worker']==i;workers.append(old);continue
            raise RuntimeError('Owned scorer ended without completion; inspect retained evidence, never blindly restart')
        leaf=folder/str(i);leaf.mkdir(mode=0o700,exist_ok=True);entry=folder/(str(i)+'.py')
        entry.write_text("import os,runpy\nrunpy.run_path(os.environ['SCORER_SCRIPT'],run_name='__main__')\n")
        env=dict(os.environ,RUN_ROOT=R,SCORER_STATE=str(leaf),JUDGE_SHARED_ROOT=str(folder),SCORER_SOURCE=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),SCORER_SCRIPT=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/scorer.py'),JUDGE_QUEUE_FILE=script,JUDGE_RESULT_FOLDER=root+'/worker'+str(i),JUDGE_EPOCH='PUREW_SOURCE_V5_20261011_'+phase.upper(),JUDGE_EFFORT='medium',JUDGE_WORKER=str(i))
        child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(leaf/'log.txt').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        identity=subprocess.run(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
        assert identity and not any(x in identity for x in ('wangbomin','Knowledge_editing','scope'))
        workers.append(dict(worker=i,pid=child.pid,process_identity=identity,entry=str(entry),log=str(leaf/'log.txt')))
        state.write_text(json.dumps(dict(root=str(folder),workers=workers),indent=2))
    return dict(status='SCORERS_STARTED',phase=phase,workers=workers)

def admit_and_launch():
    result=remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(R)+',REFERENCE_PARENT="/data/bmw/Knowledge_editing/outputs/purew-scope-v4-20261011/run")\nrunpy.run_path('+repr(R+'/private/tools/admit.py')+',run_name="__main__")\n')
    if result['status']!='PASS':return result
    return remote('import json,os,pathlib,subprocess,tempfile,time\np=pathlib.Path('+repr(R)+')\nimport sys;sys.path.insert(0,str(p/"private/tools"));import journal\nif (p/"private/CONTROLLER_START.json").exists() and journal.alive(json.loads((p/"private/CONTROLLER_START.json").read_text())):\n print(json.dumps(dict(status="ALREADY_RUNNING")))\nelse:\n env=dict(os.environ,**json.loads((p/"private/LAUNCH_ENV.json").read_text()),ACTION="source_controller",TASK_FILE=str(p/"private/tools/run.py"))\n fd,entry=tempfile.mkstemp(prefix="e.",suffix=".py",dir=env["TMPDIR"]);os.close(fd)\n pathlib.Path(entry).write_text("import os,sys,runpy\\nsys.path.insert(0,os.environ[\'RUN_ROOT\']+\'/private/tools\')\\nrunpy.run_path(os.environ[\'TASK_FILE\'],run_name=\'__main__\')\\n")\n child=subprocess.Popen([env["TRAIN_PYTHON"],"-u",entry],env=env,stdout=(p/"logs/controller.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True)\n start=dict(pid=child.pid,start_ticks=pathlib.Path(f"/proc/{child.pid}/stat").read_text().rsplit(")",1)[1].split()[19],entry=entry,epoch=time.time())\n journal.write(p/"private/CONTROLLER_START.json",start);time.sleep(2);assert journal.alive(start)\n print(json.dumps(dict(status="FORMAL_CONTROLLER_STARTED",start=start)))\n')
