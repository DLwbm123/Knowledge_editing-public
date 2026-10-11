"""First installation freezes deadline; CPU dependency owns only new assets."""
import os,sys,json,pathlib,shutil,time,tempfile,subprocess
payload=json.loads(sys.stdin.readline())
P=pathlib.Path('/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run')
R=pathlib.Path('/data/bmw/Knowledge_editing/outputs/source24-baselines-v1-20261011/run')
assert json.loads((P/'private/DATA_ADMISSION.json').read_text())['status']=='PASS'
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(P)],text=True))['filesystems'][0]
assert mount['target']!='/' and shutil.disk_usage(P).free>=payload['protocol']['minimum_free_bytes']
R.mkdir(parents=True,exist_ok=True)
for folder in ('private/tools','logs','public'):(R/folder).mkdir(parents=True,exist_ok=True)
for path in (P/'private/tools').glob('*.py'):
    dest=R/'private/tools'/path.name
    if not dest.exists():shutil.copy2(path,dest)
files=dict(payload['files']);files['source_run.py']=(P/'private/tools/run.py').read_text()
for name,source in files.items():
    path=R/'private/tools'/name
    if path.exists():assert path.read_text()==source
    else:path.write_text(source)
sys.path.insert(0,str(R/'private/tools'));import journal as j
with j.exclusive(R/'private/locks/install.lock'):
    for name in ('source','official_llava','cpu_gate','derived_inputs'):
        dest=R/'private'/name
        if not dest.exists():dest.symlink_to((P/'private'/name).resolve(),target_is_directory=True)
        assert dest.resolve()==(P/'private'/name).resolve()
    for name in ('SOURCE_COMMIT.json','EVAL_BINDINGS.json','ROWS.json','DATA_ADMISSION.json'):
        if not (R/'private'/name).exists():shutil.copy2(P/'private'/name,R/'private'/name)
    env=dict(os.environ,**json.loads((P/'private/LAUNCH_ENV.json').read_text()));env.update(RUN_ROOT=str(R),REFERENCE_PARENT=str(P),PYTHONPATH=str(R/'private/tools'))
    if not (R/'RUN_MANIFEST.json').exists():
        now=time.time();manifest=json.loads((P/'RUN_MANIFEST.json').read_text())
        manifest.update(started_epoch=now,starting_epoch=now,deadline_epoch=now+86400,Judge_limit=4500,GPU_seconds_limit=10**18,GPU_time_limit_enabled=False,owned_weight_limit_bytes=payload['protocol']['owned_weight_limit_bytes'],min_free_bytes=payload['protocol']['minimum_free_bytes'],stage='source24_baselines_v1',hourly_monitor_authorized=False)
        j.write(R/'RUN_MANIFEST.json',manifest);j.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=10**18))
        ledger=json.loads((P/'RESOURCE_LEDGER.json').read_text());j.write(R/'private/INHERITED_COST_SNAPSHOT.json',ledger)
        ledger.update(Judge_attempts=0,gpu_seconds_used=0,gpu_sessions=[],weights_observed_peak_bytes=0)
        for key in list(ledger):
            if isinstance(ledger[key],list):ledger[key]=[]
        j.write(R/'RESOURCE_LEDGER.json',ledger)
        config=json.loads((P/'PLAN_CONFIG.json').read_text());config['hardware']['allowed_GPUs']=[5];config['hardware']['UUIDs']={'5':payload['protocol']['GPU_UUID']};j.write(R/'PLAN_CONFIG.json',config)
        protocol=payload['protocol'];assert json.loads((P/'private/GPU_SOURCE_VERSION.json').read_text())['commit']==protocol['inherited_GPU_install_commit']
        j.write(R/'private/PARENT_EXECUTION_VERSIONS.json',dict(installation=json.loads((P/'private/GPU_SOURCE_VERSION.json').read_text()),repairs={p.name:json.loads(p.read_text()) for p in (P/'private').glob('ENGINEERING*.json')}))
        j.write(R/'private/FROZEN_PROTOCOL.json',protocol)
        sys.path.insert(0,str(R/'private/source'));from scripts.medtrace.stage17_prepare import digest
        parent_lock=digest(json.loads((P/'private/LOCK.json').read_text()));assert parent_lock==protocol['source_lock']
        j.write(R/'private/PARENT_BINDING.json',dict(scientific_lock=parent_lock,rows_binding=digest(json.loads((P/'private/ROWS.json').read_text()))))
        j.write(R/'private/LOCK.json',dict(protocol=protocol,parent_binding=json.loads((R/'private/PARENT_BINDING.json').read_text())))
        j.write(R/'public/PROTOCOL.json',protocol);j.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],source='published finite native baseline bank comparison'))
        for path in (P/'private/outputs/BASE').glob('*/*.json'):
            dest=R/'private/outputs/BASE'/path.relative_to(P/'private/outputs/BASE');d=json.loads(path.read_text());assert d['lock']==parent_lock
            d.update(lock=digest(json.loads((R/'private/LOCK.json').read_text())),reused_generation='PARENT_BASE');j.write(dest,d)
        assert len(list((R/'private/outputs/BASE').glob('*/*.json')))==261
        j.write(R/'private/LAUNCH_ENV.json',env)
    else:assert json.loads((R/'private/FROZEN_PROTOCOL.json').read_text())==payload['protocol']
    tests=json.loads(subprocess.check_output([env['TRAIN_PYTHON'],'-c','import bench_check,json;print(json.dumps(bench_check.run()))'],env=env,text=True));assert tests['status']=='PASS';j.write(R/'public/CPU_VALIDATION.json',tests)
    start=R/'private/DEPENDENCY_START.json'
    if start.exists():assert j.alive(json.loads(start.read_text())),'Inspect ended dependency; no implicit restart'
    else:
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=env['TMPDIR']);os.close(fd)
        pathlib.Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env.update(TASK_FILE=str(R/'private/tools/dependency.py'),ACTION='baseline_dependency')
        child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/dependency.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=child.pid,start_ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,epoch=time.time());j.write(start,receipt);time.sleep(2);assert j.alive(receipt)
    command=subprocess.check_output(['ps','-ww','-p',str(json.loads(start.read_text())['pid']),'-o','args='],text=True).strip();assert not any(x in command for x in ('wangbomin','Knowledge_editing','scope','baseline'))
    print(json.dumps(dict(status='DEPENDENCY_QUEUED',start=json.loads(start.read_text()),argv=command,CPU=tests,deadline=json.loads((R/'RUN_MANIFEST.json').read_text())['deadline_epoch'],GPU_training_started=False)))
