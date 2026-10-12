"""Install only owned assets and start the source-qualification GPU stage."""
import os,json,sys,pathlib,shutil,subprocess,tempfile,time
payload=json.loads(sys.stdin.readline())
P=pathlib.Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v4-20261011/run')
R=pathlib.Path('/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run')
assert json.loads((P/'private/PUBLICATION_RECEIPT.json').read_text())['finite_delivery_complete']
assert all(x.get('ended_epoch') for x in json.loads((P/'RESOURCE_LEDGER.json').read_text())['gpu_sessions'])
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(P)],text=True))['filesystems'][0]
assert mount['target']!='/' and shutil.disk_usage(P).free>=payload['protocol']['minimum_free_bytes']
R.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(dir=R) as f:
    f.write(b'probe');f.flush();os.fsync(f.fileno());f.seek(0);assert f.read()==b'probe'
for d in ('private/tools','logs','public'):(R/d).mkdir(parents=True,exist_ok=True)
for p in (P/'private/tools').glob('*.py'):
    if p.name in payload['files']:continue
    dest=R/'private/tools'/p.name
    if not dest.exists():shutil.copy2(p,dest)
for name,code in payload['files'].items():
    path=R/'private/tools'/name
    if path.exists():assert path.read_text()==code
    else:path.write_text(code)
sys.path.insert(0,str(R/'private/tools'))
import journal as j,cohort
with j.exclusive(R/'private/locks/install.lock'):
    for name in ('source','official_llava','cpu_gate','derived_inputs'):
        path=R/'private'/name
        if not path.exists():path.symlink_to((P/'private'/name).resolve(),target_is_directory=True)
        assert path.is_symlink() and path.resolve()==(P/'private'/name).resolve()
    for name in ('SOURCE_COMMIT.json','EVAL_BINDINGS.json'):
        if not (R/'private'/name).exists():shutil.copy2(P/'private'/name,R/'private'/name)
    sys.path.insert(0,str(R/'private/source'))
    old=json.loads((P/'RESOURCE_LEDGER.json').read_text())
    if not (R/'RUN_MANIFEST.json').exists():
        now=time.time();manifest=json.loads((P/'RUN_MANIFEST.json').read_text())
        manifest.update(starting_epoch=now,started_epoch=now,deadline_epoch=now+86400,physical_GPUs=[5],Judge_limit=old['Judge_attempts']+3900,GPU_seconds_limit=10**18,GPU_time_limit_enabled=False,stage='purew_source_v5',hourly_monitor_authorized=False)
        j.write(R/'RUN_MANIFEST.json',manifest);j.write(R/'RESOURCE_LEDGER.json',old);j.write(R/'private/INHERITED_COST.json',old)
        j.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=10**18))
        config=json.loads((P/'PLAN_CONFIG.json').read_text());config['hardware']['allowed_GPUs']=[5];config['hardware']['UUIDs']={'5':payload['protocol']['GPU_UUID']};j.write(R/'PLAN_CONFIG.json',config)
    protocol_path=R/'private/FROZEN_PROTOCOL.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==payload['protocol']
    else:j.write(protocol_path,payload['protocol'])
    pool=json.loads(pathlib.Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v4-audit-20261011/run/private/AUDIT_PACKET.json').read_text())['role_eligible_candidates']
    packet=cohort.construct(pool)
    path=R/'private/COHORT_CANDIDATES.json'
    if path.exists():assert json.loads(path.read_text())==packet
    else:j.write(path,packet)
    from scripts.medtrace.stage17_prepare import digest
    initial=dict(payload['protocol'],candidate_packet_binding=digest(packet))
    if not (R/'private/LOCK.json').exists():j.write(R/'private/LOCK.json',initial)
    j.write(R/'public/PROTOCOL.json',payload['protocol'])
    j.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],epoch=time.time(),source='published finite new-source U/H comparison'))
    env=dict(os.environ,**json.loads((P/'private/LAUNCH_ENV.json').read_text()))
    env.update(RUN_ROOT=str(R),REFERENCE_PARENT=str(P),PUREW_PARENT='/data/bmw/Knowledge_editing/outputs/purew-scope-v1-20261009/run',PYTHONPATH=str(R/'private/tools'))
    j.write(R/'private/LAUNCH_ENV.json',env)
    assert pathlib.Path(env['TRAIN_PYTHON']).is_file()
    tests=json.loads(subprocess.check_output([env['TRAIN_PYTHON'],'-c','import check,json;print(json.dumps(check.run()))'],env=env,text=True))
    j.write(R/'public/CPU_VALIDATION.json',tests);assert tests['status']=='PASS'
    if (R/'private/QUAL_GENERATION_COMPLETE.json').exists():result=dict(status='QUAL_GENERATION_ALREADY_COMPLETE')
    elif (R/'private/START_CHAIN_source_qualify_0.json').exists() and j.alive(json.loads((R/'private/START_CHAIN_source_qualify_0.json').read_text())):result=dict(status='QUALIFICATION_ALREADY_RUNNING')
    else:
        row=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert row[0]==payload['protocol']['GPU_UUID'] and int(row[1])>=60000
        env.update(ACTION='source_qualify',PARTITION='0',TASK_FILE=str(R/'private/tools/run.py'))
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=env['TMPDIR']);os.close(fd)
        pathlib.Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        log=R/'logs/qualification.log';child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=log.open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=child.pid,start_ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,log=str(log),epoch=time.time())
        j.write(R/'private/START_CHAIN_source_qualify_0.json',receipt);time.sleep(2);assert j.alive(receipt)
        command=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip();assert not any(x in command for x in ('wangbomin','Knowledge_editing','scope'))
        result=dict(status='SOURCE_BASE_QUALIFICATION_STARTED',start=receipt,argv=command,free_MiB=int(row[1]),mount=mount)
    print(json.dumps(result))
