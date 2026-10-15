"""One new finite run; all old state read-only and no training-stage replay."""
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
payload=json.loads(sys.stdin.readline());protocol=payload['protocol']
R=Path(protocol['run_root']);P=Path(protocol['reference_run']);M=Path(protocol['M_reference_run'])


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def main():
    assert not R.exists(),'One installation; inspect unknown outcome, never repeat'
    assert payload['CPU_validation']['status']=='PASS' and payload['authorization']['status']=='USER_AUTHORIZED_IMPLEMENT_ADMIT_AND_LAUNCH'
    pause=json.loads((M/'public/PAUSE_FS_RECEIPT.json').read_text());assert pause['first3_reference_W_exact'] and pause['GPU_leases_closed']
    assert json.loads((P/'public/PAUSE_PREFIX100_RECEIPT.json').read_text())['status']=='PAUSED_USER_REQUEST'
    assert all(x.get('ended_epoch') for old in (P,M,Path(protocol['K_reference_run'])) for x in json.loads((old/'RESOURCE_LEDGER.json').read_text())['gpu_sessions'])
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(M)],text=True))['filesystems'][0];assert mount['target']=='/data'
    assert shutil.disk_usage(M).free>=protocol['budgets']['minimum_free_bytes']+protocol['budgets']['owned_weight_limit_bytes']
    u,free=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert u==protocol['GPU_UUID'] and int(free)>=45*1024
    R.mkdir(parents=True)
    for name in ('private/tools','private/failures','private/operator','logs','public'):(R/name).mkdir(parents=True)
    probe=R/'private/.probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    for path in (M/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
    for name,source in payload['files'].items():(R/'private/tools'/name).write_text(source)
    for name in ('source','official_llava','cpu_gate','derived_inputs','legacy_stage17'):(R/'private'/name).symlink_to((P/'private'/name).resolve(),target_is_directory=True)
    for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','METHOD_LOCK.json','EVAL_BINDINGS.json','SOURCE_COMMIT.json'):shutil.copy2(P/'private'/name,R/'private'/name)
    shutil.copytree(M/'private/learner_gate',R/'private/learner_gate')
    write(R/'private/operator/source.json',payload['source'])
    write(R/'private/FROZEN_PROTOCOL.json',protocol);write(R/'public/PROTOCOL.json',protocol)
    write(R/'public/EXECUTION_AUTHORIZATION.json',payload['authorization']);write(R/'public/CPU_VALIDATION.json',payload['CPU_validation'])
    write(R/'private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],published_branch=payload['branch']))
    write(R/'private/M_REFERENCE_PROOF.json',pause)
    # Old output bindings stay unchanged; no source/generation/score replay.
    for marker in (M/'private/outputs/M').glob('*/*/*/COMPLETE.json'):
        d=json.loads(marker.read_text());phase=d['phase']
        if phase['mode']=='bank_R0' and phase['prefix']!=1:continue
        if phase['mode']=='insertion_R0' and phase['owner']>3:continue
        shutil.copytree(marker.parent,R/'private/outputs/M'/marker.parent.relative_to(M/'private/outputs/M'))
    old_protocol=json.loads((M/'private/FROZEN_PROTOCOL.json').read_text())
    now=time.time();budget=protocol['budgets']
    write(R/'RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+budget['wall_hours']*3600,GPU_seconds_limit=budget['GPU_process_hours']*3600,
        Judge_limit=budget['maximum_new_Judge'],min_free_bytes=budget['minimum_free_bytes'],owned_weight_limit_bytes=budget['owned_weight_limit_bytes']))
    write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=budget['GPU_process_hours']*3600))
    write(R/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
    write(R/'PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[5],UUIDs={'5':u})))
    tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp');env=dict(os.environ,**json.loads((P/'private/LAUNCH_ENV.json').read_text()))
    env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
        M3BENCH_LLAVA_SOURCE=str(R/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(R/'private/official_llava'),ACTION='controller',TASK_FILE=str(R/'private/tools/controller.py'))
    write(R/'private/LAUNCH_ENV.json',env);os.environ.update(env);sys.path.insert(0,str(R/'private/tools'))
    import torch
    torch.set_num_threads(1)
    buffer=io.BytesIO()
    with gzip.open(M/'private/active/M/state.pt','rb') as f:shutil.copyfileobj(f,buffer,1024*1024)
    buffer.seek(0);state=torch.load(buffer,map_location='cpu',weights_only=True);buffer.close()
    assert state['owner']==4 and state['slot']==51 and torch.equal(state['native'].float(),state['edit_initial'])
    torch.save(dict(native=state['native']),R/'private/M_reference_native.pt');del state
    import common as c
    write(R/'private/M_PARENT_PROTOCOL_BINDING.json',dict(protocol=c.digest(old_protocol)))
    import queue_fs as queue
    queue.q.ROOT.mkdir(parents=True,exist_ok=True)
    queue.q.write(queue.q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=protocol['epoch'],model='gpt-6-astra',reasoning_effort='medium',protocol=queue.q.PROTOCOL,prompt=queue.q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
    db=queue.q.connect();queue.ingest(db);db.close()
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd)
    Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip();assert not any(x in argv for x in ('wangbomin','Knowledge_editing','purew'))
    start=dict(pid=child.pid,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,epoch=time.time(),argv=argv)
    write(R/'private/START_CONTROLLER.json',start)
    print(json.dumps(dict(status='INSTALLED_ADMISSION_RUNNING',start=start,activation_epoch=now,run_root=str(R))))


try:main()
except BaseException:
    if R.exists():write(R/'private/failures/install.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
    raise
