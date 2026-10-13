"""Prepare a new isolated run from the terminal parent's read-only bindings."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parent=Path(os.environ['GUARD_PARENT']).resolve()
    run=Path(os.environ['NEW_RUN_ROOT'])
    source=Path(os.environ['NEW_TOOLS'])
    assert parent.is_dir() and not run.exists()
    def read(path):return json.loads(path.read_text())
    def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    assert read(parent/'private/PHASE_A_REPORT_COMPLETE.json')['decision']=='STOP_MECHANISM_GATE_NOT_MET'
    assert read(parent/'private/SCHEDULE_EXIT1.json')['exit_code']==0
    ledger=read(parent/'RESOURCE_LEDGER.json')
    assert all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    # Reject a live process still owning the parent run; no old training is adopted.
    for path in Path('/proc').glob('[0-9]*/environ'):
        if path.parent.name==str(os.getpid()):continue
        try:entries=path.read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        assert ('RUN_ROOT='+str(parent)).encode() not in entries, 'Parent still active'
    run.mkdir(parents=True)
    for name in ['logs','public','private/tools','private/work','private/active','private/probes',
                 'private/results_joint','private/results/C','private/candidates/C','private/reconstruction','private/C_FISHER_PARITY']:
        (run/name).mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(run).free>=8*1024**3
    with tempfile.NamedTemporaryFile(dir=run) as probe:
        probe.write(b'fixed-write-probe');probe.flush()
        assert Path(probe.name).read_bytes()==b'fixed-write-probe'
    write(run/'private/STORAGE_ADMISSION.json',dict(mount=subprocess.check_output(['findmnt','-T',str(run),'-J'],text=True),
                                                  free_bytes=shutil.disk_usage(run).free,write_read=True))
    for path in (parent/'private/tools').glob('*.py'):shutil.copy2(path,run/'private/tools'/path.name)
    for name in ['joint_math.py','joint_geometry.py','joint_run.py']:shutil.copy2(source/name,run/'private/tools'/name)
    for name in ['SOURCE_COMMIT.json','EVAL_BINDINGS.json','SCOPE_LOCK.json','OPTIMIZER160_LOCK.json',
                 'HELD_BASE.json','BENCHMARK146_QUEUE.json']:
        shutil.copy2(parent/'private'/name,run/'private'/name)
    for name in ['derived_inputs','cpu_gate','official_llava','source']:
        (run/'private'/name).symlink_to((parent/'private'/name).resolve(),target_is_directory=True)
    for owner in range(1,9):
        for directory,suffix in [('candidates/C','.pt'),('results/C','.json')]:
            (run/'private'/directory/f'{owner}{suffix}').symlink_to((parent/'private'/directory/f'{owner}{suffix}').resolve())
    targets=[]
    for owner in range(1,9):
        curve=read(parent/'private/prefix_curve'/f'{owner}.json')
        selected=next(x for x in curve if x['decision']['status']=='SKIPPED_INFEASIBLE' and x['decision']['pure_source_gain']>0)
        targets.append(dict(owner=owner,step=selected['step'],source_curve=str(parent/'private/prefix_curve'/f'{owner}.json')))
    assert sum(x['step'] for x in targets)==341
    write(run/'private/TARGETS.json',dict(selection='first_positive_pure_gain_rejection',rows=targets,parent=str(parent)))
    manifest=read(parent/'RUN_MANIFEST.json')
    manifest.update(stage='root_joint_fixed_states',starting_epoch=time.time(),hourly_monitor_authorized=False,
                    physical_GPUs=[0,1,2,3,4,7],max_training_GPUs=3,max_training_workers=3)
    write(run/'RUN_MANIFEST.json',manifest)
    plan=read(parent/'PLAN_CONFIG.json')
    plan.update(stage='root_joint_fixed_states',steps=341,arms=['PURE','ORIGINAL_GUARD','JOINT_ROOT'],
                full_trajectory=False,local_problems=32,maximum_non_generation_forwards=12000,maximum_NN_backwards=3071)
    write(run/'PLAN_CONFIG.json',plan)
    inherited=dict(gpu_seconds_used=ledger['gpu_seconds_used'],Judge_attempts=ledger['Judge_attempts'])
    write(run/'private/INHERITED_COST.json',inherited)
    write(run/'RESOURCE_LEDGER.json',dict(inherited,gpu_sessions=[],weights_observed_peak_bytes=0))
    write(run/'STAGE_CAP.json',dict(stage='FIXED_ROOT_DIAGNOSTICS',GPU_seconds_limit=10**18,GPU_time_limit_enabled=False))
    env=read(parent/'private/LAUNCH_ENV.json')
    env={k:str(v).replace(str(parent),str(run)) for k,v in env.items()}
    env.update(RUN_ROOT=str(run),GUARD_PARENT=str(parent),PYTHONPATH=str(run/'private/tools'),
               M3BENCH_LLAVA_SOURCE=str(run/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(run/'private/official_llava'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(run/'private/LAUNCH_ENV.json',env)
    write(run/'private/GPU_SOURCE_VERSION.json',dict(commit=os.environ['NEW_COMMIT'],parent='0d7da3b',stage='root_joint'))
    write(run/'public/PROGRESS.json',dict(status='PREPARED_CPU_ADMISSION_PENDING',complete=False))
    print(json.dumps(dict(prepared=True,target_count=8,reconstruction_steps=341,source_commit=os.environ['NEW_COMMIT'])))


if __name__=='__main__':main()
