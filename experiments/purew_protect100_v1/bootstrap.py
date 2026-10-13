"""Install once under a separate budget; parent assets and paused checkpoints stay read-only."""
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
P=Path(payload['protocol']['reference_run'])
R=Path('/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    assert not R.exists(), 'Installation is one attempt; inspect unknown outcomes instead of reinstalling'
    assert payload['authorization']['status']=='USER_AUTHORIZED_IMPLEMENT_ADMIT_AND_LAUNCH'
    pause=json.loads((P/'public/PAUSE_PREFIX100_RECEIPT.json').read_text())
    assert pause['status']=='PAUSED_USER_REQUEST' and (P/'private/active/bank_R0.pt').exists()
    assert not any(not s.get('ended_epoch') for s in json.loads((P/'RESOURCE_LEDGER.json').read_text())['gpu_sessions'])
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(P)],text=True))['filesystems'][0]
    assert mount['target']=='/data' and shutil.disk_usage(P).free>=10*1024**3
    prep=R.parent/'preparation'
    data=json.loads((prep/'QUALIFICATION_PRIVATE.json').read_text())
    q=json.loads((prep/'QUALIFICATION.json').read_text())
    ns={};exec(payload['files']['qualify.py'],ns)
    protocol=payload['protocol']
    assert q['status']=='PASS' and ns['binding'](data['basis'])==protocol['qualification']['basis_binding']
    assert ns['binding'](data['targets'])==protocol['norm']['binding']
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
    write(R/'public/EXECUTION_AUTHORIZATION.json',payload['authorization'])
    write(R/'private/BASIS_BUFFER.json',data['basis']);write(R/'private/C_REFERENCE.json',data['targets'])
    write(R/'public/DATA_QUALIFICATION.json',q);write(R/'public/CPU_VALIDATION.json',payload['CPU_validation'])
    assert payload['CPU_validation']['status']=='PASS'
    now=time.time();budget=protocol['budgets']
    write(R/'RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+budget['wall_hours']*3600,
        GPU_seconds_limit=budget['GPU_process_hours']*3600,Judge_limit=budget['maximum_new_Judge'],
        min_free_bytes=budget['minimum_free_bytes'],owned_weight_limit_bytes=budget['owned_weight_limit_bytes']))
    write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=budget['GPU_process_hours']*3600))
    write(R/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
    write(R/'PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[5],UUIDs={'5':protocol['GPU_UUID']})))
    write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],published_branch=payload['branch']))
    tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp')
    env=dict(os.environ,**json.loads((P/'private/LAUNCH_ENV.json').read_text()))
    env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
        M3BENCH_LLAVA_SOURCE=str(R/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(R/'private/official_llava'),
        ACTION='controller',TASK_FILE=str(R/'private/tools/controller.py'))
    write(R/'private/LAUNCH_ENV.json',env);os.environ.update(env);sys.path.insert(0,str(R/'private/tools'))
    import queue_protect as queue
    queue.q.ROOT.mkdir(parents=True,exist_ok=True)
    queue.q.write(queue.q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=protocol['epoch'],model='gpt-6-astra',reasoning_effort='medium',
        protocol=queue.q.PROTOCOL,prompt=queue.q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
    db=queue.q.connect();queue.ingest(db);db.close()
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd)
    Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
    start=dict(pid=child.pid,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,epoch=time.time(),argv=argv)
    write(R/'private/START_CONTROLLER.json',start)
    assert not any(x in argv for x in ('wangbomin','Knowledge_editing','purew'))
    print(json.dumps(dict(status='INSTALLED_ADMISSION_RUNNING',start=start,activation_epoch=now,run_root=str(R))))


try:main()
except BaseException:
    if R.exists():write(R/'private/failures/install.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
    raise
