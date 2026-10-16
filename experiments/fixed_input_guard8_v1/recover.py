"""One authorized recovery from the pre-generation path error; never reset clocks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def install(payload):
    root = Path('/data/bmw/Knowledge_editing/outputs/fixed-input-guard8-v1-20261016/run')
    parent = Path('/data/bmw/Knowledge_editing/outputs/purew-odds-hinge-first1-v1-20261016/run')
    read = lambda p: json.loads(Path(p).read_text())
    assert not (root/'private/RECOVERY1.json').exists(), 'Inspect existing recovery; never launch twice'
    assert read(root/'public/PROGRESS.json')['status'] == 'STOPPED_FAILURE_NO_RETRY'
    assert payload['cpu']['status'] == 'PASS'
    ledger = read(root/'RESOURCE_LEDGER.json')
    manifest = read(root/'RUN_MANIFEST.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    assert time.time() < manifest['deadline_epoch'] and ledger['gpu_seconds_used'] < manifest['GPU_seconds_limit']
    assert not any('/data/bmw/tmp/e.d0nrqj6u/e.py' in s for s in subprocess.check_output(['ps','-ww','-eo','args'],text=True).splitlines())
    uuid, free = subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert uuid == 'GPU-578b486c-6eae-13b0-23be-353ed09220ec' and int(free) >= 60000
    mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(root)],text=True))['filesystems'][0]
    assert mount['target'] == '/data' and shutil.disk_usage(root).free > 24*1024**3
    probe=root/'private/.recovery_probe';probe.write_bytes(b'recovery');assert probe.read_bytes()==b'recovery';probe.unlink()
    target = (parent/'private/derived_inputs').resolve()
    assert target.is_dir() and str(target).startswith('/data/bmw/')
    def mapped(p):
        if '/derived_inputs/' in p:return target/p.split('/derived_inputs/',1)[1]
        if '/DataP/' in p:return Path('/data/bmw/DataP')/p.split('/DataP/',1)[1]
        return Path(p)
    tasks=read(root/'private/BENCHMARK146_QUEUE.json')['tasks'];queries=read(root/'private/EVAL_LEDGER.json')['queries']
    ids=set(read(root/'private/protection_ids.json'))
    for t in tasks[:8]:
        ids.add(t['edit_id'])
        for e in t['events']:
            if e['task'] in ('T1G','T2G'):ids.update(e['all_probe_query_ids'])
    paths={mapped(queries[q]['image_path']) for q in ids}
    paths.update(mapped(tasks[i-1]['native']['image_path']) for i in (1,50,100))
    for p in paths:
        with p.open('rb') as f:assert f.read(1)
    old_config=read(root/'private/config.json');previous=old_config['implementation_commit']
    assert previous == 'e42ad24f94e345918ce9aebcfd2a44bf116bd856'
    bindings=read(root/'private/EVAL_BINDINGS.json')
    outputs=[p for folder in ('outputs','evaluation') for p in (root/'private'/folder).rglob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    markers=[p for folder in ('outputs','evaluation') for p in (root/'private'/folder).rglob('*.STARTED.json')]
    assert len(outputs)==len(markers)==58
    for marker in markers:assert marker.with_name(marker.name.replace('.STARTED.json','.json')).exists()
    for p in outputs:
        d=read(p);b=d['binding'];phase=b['phase'];row=b['input']
        assert b['judge_input']==bindings[row['opaque_Base_id']]
        assert phase['arm']=='U' and (phase.get('stage')=='BASE_IDENTITY' or (phase.get('mode'),phase.get('prefix'),phase.get('code'))==('base',0,previous))
        assert read(p.with_suffix('.STARTED.json'))['phase']==phase
    budgets={str(p.relative_to(root)):read(p) for p in (root/'private/fixed8').glob('*/*budget.json')}
    assert sum(b['counts']['generation_attempts'] for b in budgets.values())==58
    assert all(b['counts']['attempt_slots']==b['counts']['admission_slots']==0 for b in budgets.values())
    assert read(root/'public/CALIBRATION_COMPLETE.json')['status']=='PASS'
    assert (root/'private/calibration_inputs.pt').is_file()
    record=dict(authorized=True,previous_commit=previous,repair_commit=payload['commit'],epoch=time.time(),
                resume_from='base_U_0',reused_generations=58,readable_panel_images=len(paths),
                original_manifest=manifest,original_resources=ledger,original_budgets=budgets)
    with (root/'private/RECOVERY1.json').open('x') as f:json.dump(record,f,indent=2)
    link=root/'private/derived_inputs'
    assert not link.exists() and not link.is_symlink()
    link.symlink_to(target,target_is_directory=True)
    for name,content in payload['files'].items():
        assert name in ('launch.py','bootstrap.py')
        path=root/'private/implementation/experiments/fixed_input_guard8_v1'/name
        archive=root/'private/recovery1_source_before'/name;archive.parent.mkdir(exist_ok=True)
        shutil.copy2(path,archive)
        compile(content,name,'exec');path.write_text(content)
    old_config['implementation_commit']=payload['commit']
    (root/'private/config.json').write_text(json.dumps(old_config,indent=2)+'\n')
    assert read(root/'RUN_MANIFEST.json')==manifest and read(root/'RESOURCE_LEDGER.json')==ledger
    assert all(read(root/p)==b for p,b in budgets.items())
    env=read(root/'private/LAUNCH_ENV.json')
    entry=Path(env['TMPDIR'])/'r.py'
    with entry.open('x') as f:f.write("import runpy\nrunpy.run_module('experiments.fixed_input_guard8_v1.launch',run_name='__main__')\n")
    env.update(NEUTRAL_ENTRY=str(entry),ENTRY_MODE='controller',RESUME_FROM='base_U_0',RECOVERY_ID='1',ACTION='controller_recovery1')
    (root/'private/RECOVERY1_ENV.json').write_text(json.dumps(env,indent=2))
    with (root/'logs/controller.recovery1.log').open('ab') as log:
        process=subprocess.Popen([env['TRAIN_PYTHON'],'-u',str(entry)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    argv=subprocess.check_output(['ps','-ww','-p',str(process.pid),'-o','args='],text=True).strip()
    assert not any(s in argv for s in ('wangbomin','Knowledge_editing','guard','purew'))
    receipt=dict(pid=process.pid,argv=argv,epoch=time.time(),repair_commit=payload['commit'],resume_from='base_U_0',
                 reused_generations=58,prior_forwards=2345,prior_gpu_seconds=ledger['gpu_seconds_used'],
                 unchanged_deadline=manifest['deadline_epoch'],readable_panel_images=len(paths),status='RECOVERY_STARTED')
    (root/'private/START_RECOVERY1.json').write_text(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':print(json.dumps(install(json.loads(sys.stdin.readline()))))
