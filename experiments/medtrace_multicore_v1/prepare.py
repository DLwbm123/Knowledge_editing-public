"""Independent run; copy code and bindings, reuse historical C read-only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parent=Path(os.environ['REFINE_PARENT']).resolve();run=Path(os.environ['NEW_RUN_ROOT'])
    tools=Path(os.environ['NEW_TOOLS']);read=lambda p:json.loads(p.read_text())
    assert read(parent/'private/FULL_COMPLETE.json')['status']=='COMPLETE'
    assert read(parent/'private/FULL_HANDOFF_EXIT1.json')['exit_code']==0
    assert read(parent/'private/FULL_DELIVERY.json')['remote_SHA_match']
    ledger=read(parent/'RESOURCE_LEDGER.json');assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    assert not run.exists();run.mkdir(parents=True)
    folders=['logs','public','private/tools','private/work','private/probes','private/active','private/paired',
        'private/diagnostic_states','private/reconstruction','private/C_FISHER_PARITY','private/RESUME_REAL',
        'private/full_resume','private/candidates/C','private/results/C','private/candidates/ROOT_FIX',
        'private/candidates/MULTI','private/results/ROOT_FIX','private/results/MULTI','private/outputs']
    for folder in folders:(run/folder).mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(run).free>=8*1024**3
    with tempfile.NamedTemporaryFile(dir=run) as f:
        f.write(b'paired-storage-probe');f.flush();assert Path(f.name).read_bytes()==b'paired-storage-probe'
    write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    write(run/'private/STORAGE_ADMISSION.json',dict(free_bytes=shutil.disk_usage(run).free,
        mount=subprocess.check_output(['findmnt','-T',str(run),'-J'],text=True),write_read=True))
    for path in (parent/'private/tools').glob('*.py'):shutil.copy2(path,run/'private/tools'/path.name)
    for path in tools.glob('*.py'):
        if path.name!='prepare.py':shutil.copy2(path,run/'private/tools'/path.name)
    for name in ['SOURCE_COMMIT.json','EVAL_BINDINGS.json','SCOPE_LOCK.json','OPTIMIZER160_LOCK.json',
                 'HELD_BASE.json','TARGETS.json','REFERENCE_EXACT.json']:
        shutil.copy2(parent/'private'/name,run/'private'/name)
    for name in ['source','official_llava','derived_inputs','cpu_gate']:
        (run/'private'/name).symlink_to((parent/'private'/name).resolve(),target_is_directory=True)
    for owner in range(1,9):
        for folder,suffix in [('candidates/C','.pt'),('results/C','.json')]:
            origin=(parent/'private'/folder/f'{owner}{suffix}').resolve();assert origin.is_file()
            (run/'private'/folder/f'{owner}{suffix}').symlink_to(origin)
    inherited={k:ledger[k] for k in ['gpu_seconds_used','Judge_attempts']}
    write(run/'private/INHERITED_COST.json',inherited)
    write(run/'RESOURCE_LEDGER.json',dict(inherited,gpu_sessions=[],weights_observed_peak_bytes=0))
    manifest=read(parent/'RUN_MANIFEST.json')
    manifest.update(stage='multicore_paired',starting_epoch=time.time(),deadline_epoch=time.time()+48*3600,
        physical_GPUs=[0,1,2,3,4,7],max_training_workers=3,Judge_limit=inherited['Judge_attempts']+2316,
        GPU_time_limit_enabled=False,owned_weight_limit_bytes=2*1024**3,min_free_bytes=8*1024**3)
    write(run/'RUN_MANIFEST.json',manifest)
    plan=read(parent/'PLAN_CONFIG.json')
    plan.pop('full_consumer',None);plan.pop('eligible_candidates',None)
    plan.update(stage='multicore_paired',arms=['ROOT_FIX','MULTI'],parameters=7168,
        fixed_states=8,paired_fraction_cases=32,steps_rebuilt=341,
        mechanism_gate='ADDITIONAL_TRUE_PAIRED_CASE_AND_IMPROVED_FIRST_SELECTION',
        total_NN_backward_limit=144593,operator_backward_limit=15152,
        maximum_NN_backwards=144593,full_scientific_steps=2560,
        local_solve_limit=45328,
        maximum_non_generation_forwards=600000,maximum_generations=2316,maximum_new_Judge=2316)
    write(run/'PLAN_CONFIG.json',plan)
    write(run/'STAGE_CAP.json',dict(GPU_seconds_limit=10**18,GPU_time_limit_enabled=False))
    env=read(parent/'private/LAUNCH_ENV.json')
    env={k:str(v).replace(str(parent),str(run)) for k,v in env.items()}
    env.update(RUN_ROOT=str(run),REFINE_PARENT=str(parent),PYTHONPATH=str(run/'private/tools'),
        M3BENCH_LLAVA_SOURCE=str(run/'private/official_llava'),M3BENCH_EXPECTED_LLAVA_SOURCE=str(run/'private/official_llava'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(run/'private/LAUNCH_ENV.json',env)
    write(run/'private/METHOD_LOCK.json',dict(arms=['ROOT_FIX','MULTI'],inner_margin=1e-5,
        hard_Fisher_fraction=.1,real_source_gain_fraction=.9,step_upper=1.002,
        fractions=[1.,.5,.25,.125],maximum_repairs=3,trust_fractions=[.25,.125,.0625],
        root_dimensions=512,tail_directions='NORMALIZED_SOURCE_AND_FISHER_PER_G2_G3_G4',
        rank_and_parameters_unchanged=True,source_clock_frozen_in_diagnostic=True,source_only_RMS=True,
        full_first_selection='MULTI_FIRST_ROOT_FALLBACK_WITHOUT_JUDGE_SELECTION',source_commit=os.environ['NEW_COMMIT']))
    write(run/'private/GPU_SOURCE_VERSION.json',dict(commit=os.environ['NEW_COMMIT'],upstream_execution='be310287487a72576157bbb84b491434d9a3ab42'))
    write(run/'public/PROGRESS.json',dict(status='PREPARED_CPU_ADMISSION_PENDING',training_started=False))
    print(json.dumps(dict(prepared=True,fixed_states=8,reconstruction_proposals=341,inherited=inherited)))


if __name__=='__main__':main()
