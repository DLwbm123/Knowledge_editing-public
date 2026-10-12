"""Install owned metadata/helpers; historical source and outputs remain read-only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

payload=json.loads(sys.stdin.readline())
R=Path('/data/bmw/Knowledge_editing/outputs/purew-original146-v1-20261012/run')
P=Path(payload['protocol']['common_parent'])
TT=Path(json.loads((P/'PLAN_CONFIG.json').read_text())['TT_parent'])
assert not R.exists(), 'Existing run requires explicit receipt/state review; no blind reinstall'
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(P)],text=True))['filesystems'][0]
assert mount['target']=='/data' and shutil.disk_usage(P).free>=10*1024**3
R.mkdir(parents=True)
for name in ('private/tools','private/tmp','logs','public'):(R/name).mkdir(parents=True)
probe=R/'private/.write_probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
for path in (TT/'private/tools').glob('*.py'):shutil.copy2(path,R/'private/tools'/path.name)
for name,source in payload['files'].items():(R/'private/tools'/name).write_text(source)
for name in ('source','official_llava','cpu_gate','derived_inputs'):
    (R/'private'/name).symlink_to((TT/'private'/name).resolve(),target_is_directory=True)
(R/'private/legacy_stage17').symlink_to((P/'private/legacy_stage17').resolve(),target_is_directory=True)
for name in ('BENCHMARK146_QUEUE.json','EVAL_LEDGER.json','METHOD_LOCK.json'):
    shutil.copy2(P/'private'/name,R/'private'/name)
for name in ('EVAL_BINDINGS.json','SOURCE_COMMIT.json'):
    shutil.copy2(TT/'private'/name,R/'private'/name)
def write(name,d):
    p=R/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
write('private/FROZEN_PROTOCOL.json',payload['protocol']);write('public/PROTOCOL.json',payload['protocol'])
now=time.time();budgets=payload['protocol']['budgets']
write('RUN_MANIFEST.json',dict(starting_epoch=now,deadline_epoch=now+budgets['wall_hours']*3600,
    GPU_seconds_limit=budgets['GPU_process_hours']*3600,Judge_limit=budgets['maximum_new_Judge'],
    min_free_bytes=budgets['minimum_free_bytes'],owned_weight_limit_bytes=budgets['owned_weight_limit_bytes']))
write('STAGE_CAP.json',dict(GPU_seconds_limit=budgets['GPU_process_hours']*3600))
write('RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_sessions=[],Judge_attempts=0,Astra_purew146_batches=[],weights_observed_peak_bytes=0))
write('private/INHERITED_COSTS.json',dict(common=json.loads((P/'RESOURCE_LEDGER.json').read_text()),TT=json.loads((TT/'RESOURCE_LEDGER.json').read_text()),baseline_parent=str(json.loads((P/'PLAN_CONFIG.json').read_text())['baseline_parent']),rule='Separate historical ledgers; no double-counting aggregate histories'))
write('PLAN_CONFIG.json',dict(hardware=dict(allowed_GPUs=[5],UUIDs={'5':payload['protocol']['GPU_UUID']})))
write('private/GPU_SOURCE_VERSION.json',dict(commit=payload['commit'],published_branch=payload['branch']))
tmp=tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp')
env=dict(os.environ,**json.loads((TT/'private/LAUNCH_ENV.json').read_text()))
env.update(RUN_ROOT=str(R),TMPDIR=tmp,PYTHONPATH=str(R/'private/official_llava')+':'+str(R/'private/source')+':'+str(R/'private/tools'),
    M3BENCH_LLAVA_SOURCE=str(R/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(R/'private/official_llava'),ACTION='controller',TASK_FILE=str(R/'private/tools/run146.py'))
write('private/LAUNCH_ENV.json',env)
sys.path.insert(0,str(R/'private/tools'));os.environ.update(env)
from contract import selfcheck
write('public/CPU_VALIDATION.json',selfcheck())
fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=tmp);os.close(fd)
Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs/controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
start=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],entry=entry,epoch=time.time())
write('private/START_CONTROLLER.json',start)
time.sleep(2)
cmd=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
assert child.poll() is None and not any(x in cmd for x in ('wangbomin','Knowledge_editing','purew'))
print(json.dumps(dict(status='BASELINE_REUSE_AUDIT_RUNNING',start=start,argv=cmd,run_root=str(R),deadline_epoch=now+budgets['wall_hours']*3600)))
