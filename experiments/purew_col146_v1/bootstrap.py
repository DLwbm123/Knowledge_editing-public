"""One installation after the predecessor barrier, with no historical mutation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

payload=json.loads(sys.stdin.readline())
protocol=payload['protocol'];R=Path(protocol['new_run_root']);P=Path(protocol['reference_run'])


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    ns={};exec(payload['files']['dependency.py'],ns);assert ns['ready'](P)
    assert not R.exists(), 'Never reinstall or reset activation clock'
    assert protocol['arm_order']==['O','L'] and protocol['qualification']['independent_of_P']
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(P)],text=True))['filesystems'][0]
    assert mount['target']=='/data' and shutil.disk_usage(P).free>=10*1024**3
    R.mkdir(parents=True)
    for name in ('private/tools','private/failures','logs','public'):(R/name).mkdir(parents=True)
    probe=R/'private/.probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    for path in (P/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
    for name,source in payload['files'].items():(R/'private/tools'/name).write_text(source)
    for name in ('source','official_llava','cpu_gate','derived_inputs','legacy_stage17'):
        (R/'private'/name).symlink_to((P/'private'/name).resolve(),target_is_directory=True)
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','METHOD_LOCK.json','EVAL_BINDINGS.json','SOURCE_COMMIT.json'):
        shutil.copy2(P/'private'/name,R/'private'/name)
    write(R/'private/FROZEN_PROTOCOL.json',protocol);write(R/'public/PROTOCOL.json',protocol)
    write(R/'private/C_NORM_SCHEDULE.json',payload['norm_schedule'])
    now=time.time();b=protocol['budgets']
    write(R/'RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+b['wall_hours']*3600,
        GPU_seconds_limit=b['GPU_process_hours']*3600,Judge_limit=b['maximum_new_Judge'],
        min_free_bytes=b['minimum_free_bytes'],owned_weight_limit_bytes=b['owned_weight_limit_bytes']))
    write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=b['GPU_process_hours']*3600))
    write(R/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
    write(R/'PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[5],UUIDs={'5':protocol['GPU_UUID']})))
    write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],published_branch=payload['branch']))
    tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp')
    env=dict(os.environ,**json.loads((P/'private/LAUNCH_ENV.json').read_text()))
    env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
        M3BENCH_LLAVA_SOURCE=str(R/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(R/'private/official_llava'),ACTION='controller',TASK_FILE=str(R/'private/tools/controller.py'))
    write(R/'private/LAUNCH_ENV.json',env);sys.path.insert(0,str(R/'private/tools'));os.environ.update(env)
    import common as c
    from validate import selfcheck
    write(R/'public/CPU_VALIDATION.json',selfcheck())
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    digest=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest(payload['norm_schedule'])==protocol['norm_budget']['binding']
    assert digest([dict(order=t['order'],question=t['native']['question'],answer=t['native']['reference']) for t in tasks])==protocol['qualification']['source_binding']
    assert digest([{k:t[k] for k in ('order','seed','native','fit_questions')} for t in tasks])=='4c143c67f04709f7087636d5f3e8e87c0008d159055c2eaf4cdf5980fa0db583'
    assert c.read(P/'public/BASELINE_REUSE_ADMISSION.json')['status']=='PASS'
    assert c.read(P/'public/REAL_VLM_ADMISSION.json')['status']=='PASS'
    assert c.read(P/'private/single_R0_COMPLETE.json')['committed']==23360
    for i,x in enumerate(payload['norm_schedule'],1):
        assert x['order']==i and len(x['proposal_norm'])==len(x['actual_native_norm'])==160
        assert all(0<v<=.050001 for v in x['proposal_norm'])
    write(R/'private/BASELINE_REUSE_ADMISSION.json',dict(status='PASS',C_single_updates=23360,source_and_schedule_bound=True,original_qualification_reused=True,P_not_required=True))
    shutil.copy2(R/'private/BASELINE_REUSE_ADMISSION.json',R/'public/BASELINE_REUSE_ADMISSION.json')
    for arm in ('O','L'):
        A=R/'private/arms'/arm
        for name in ('private','public'):(A/name).mkdir(parents=True)
        for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','FROZEN_PROTOCOL.json','GPU_SOURCE_VERSION.json'):
            (A/'private'/name).symlink_to(R/'private'/name)
        (A/'PLAN_CONFIG.json').symlink_to(R/'PLAN_CONFIG.json')
    # Initialize and validate all inherited C consumers before any GPU activation.
    import queue_col
    q=queue_col.q;q.ROOT.mkdir(parents=True,exist_ok=True)
    q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=protocol['epoch'],model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
    db=q.connect();queue_col.ingest(db);db.close()
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd)
    Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    start=dict(pid=child.pid,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().split()[21],entry=entry,epoch=time.time())
    write(R/'private/START_CONTROLLER.json',start)
    argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
    assert not any(x in argv for x in ('wangbomin','Knowledge_editing','purew'))
    print(json.dumps(dict(status='INSTALLED',start=start,argv=argv,activation_epoch=now,run_root=str(R))))


try:main()
except Exception:
    if R.exists():write(R/'private/failures/install.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
    raise
