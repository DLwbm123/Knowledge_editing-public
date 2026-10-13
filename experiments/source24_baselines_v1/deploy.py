"""Published source-only finite chain; neutral entry points and isolated scorers."""
import os,json,subprocess,sys,tempfile,time
from pathlib import Path
W=Path(os.environ['WORKSPACE']);HERE=W/'experiments/source24_baselines_v1'
R='/data/bmw/Knowledge_editing/outputs/source24-baselines-v1-20261011/run'

def remote(code):return json.loads(subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=300).stdout)

def gpu():
    assert not subprocess.check_output(['git','status','--porcelain','--','experiments/source24_baselines_v1','reports/source24_baselines_v1_20261011'],cwd=W,text=True).strip()
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','finish.py')}
    files['tt_cores.py']=(W/'experiments/medtrace_tt_only_budget_h_v1_20261005/structures.py').read_text()
    payload=dict(files=files,protocol=json.loads((W/'reports/source24_baselines_v1_20261011/PROTOCOL.json').read_text()),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip())
    code=json.dumps((HERE/'bootstrap.py').read_text())+'\n'+json.dumps(payload)+'\n'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=code,text=True,capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError(p.stderr[-5000:])
    return json.loads(p.stdout)

def snapshot():
    return remote('import pathlib,json,time\np=pathlib.Path('+repr(R)+')\nparent=pathlib.Path("/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run")\nprint(json.dumps(dict(deadline=json.loads((p/"RUN_MANIFEST.json").read_text())["deadline_epoch"],now=time.time(),generated=(p/"private/GENERATION_COMPLETE.json").is_file(),scored=(p/"private/judge_source24_baselines/ALL_WORKERS_COMPLETE.json").is_file(),parent_scored=(parent/"private/judge_source_v5_formal/ALL_WORKERS_COMPLETE.json").is_file(),failures=[x.name for x in (p/"private/failures").glob("*.json")])))\n')

def score(phase):
    assert phase in ('qualification','formal');root='judge_source24_baselines'
    x=remote('import pathlib,json\np=pathlib.Path('+repr(R+'/private/'+root)+')\nprint(json.dumps(dict(done=(p/"ALL_WORKERS_COMPLETE.json").is_file())))\n')
    if x['done']:return dict(status='SCORING_ALREADY_TERMINAL')
    script='bench_queue.py'
    remote('import os,runpy\nos.environ.update(RUN_ROOT='+repr(R)+',QUEUE_REQUEST=\'{"action":"ingest"}\')\nrunpy.run_path('+repr(R+'/private/tools/'+script)+',run_name="__main__")\n')
    state=W/'.local_run'/('source24_baseline_scorer_'+phase+'.json');prior=json.loads(state.read_text()) if state.exists() else {}
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
        env=dict(os.environ,RUN_ROOT=R,SCORER_STATE=str(leaf),JUDGE_SHARED_ROOT=str(folder),SCORER_SOURCE=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),SCORER_SCRIPT=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/scorer.py'),JUDGE_QUEUE_FILE=script,JUDGE_RESULT_FOLDER=root+'/worker'+str(i),JUDGE_EPOCH='SOURCE24_BASELINES_V1_20261011',JUDGE_EFFORT='medium',JUDGE_WORKER=str(i))
        child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(leaf/'log.txt').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        identity=subprocess.run(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],capture_output=True,text=True).stdout.strip()
        assert identity and not any(x in identity for x in ('wangbomin','Knowledge_editing','scope'))
        workers.append(dict(worker=i,pid=child.pid,process_identity=identity,entry=str(entry),log=str(leaf/'log.txt')))
        state.write_text(json.dumps(dict(root=str(folder),workers=workers),indent=2))
    return dict(status='SCORERS_STARTED',phase=phase,workers=workers)

