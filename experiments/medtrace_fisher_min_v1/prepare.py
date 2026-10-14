"""Reuse preparation bindings; inherit the latest completed chain's real costs."""
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parent=Path(os.environ['MULTICORE_PARENT'])
    read=lambda p:json.loads(p.read_text())
    assert read(parent/'private/COMPLETE.json')['status']=='COMPLETE'
    assert read(parent/'private/EXIT1.json')['exit_code']==0
    assert read(parent/'private/DELIVERY1.json')['remote_SHA_match']
    assert read(parent/'public/PAIRED_FULL_RESULTS.json')['complete_no_missing']
    ledger=read(parent/'RESOURCE_LEDGER.json');assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    oldenv=read(parent/'private/LAUNCH_ENV.json')
    os.environ['REFINE_PARENT']=oldenv['REFINE_PARENT']
    spec=importlib.util.spec_from_file_location('paired_prepare',Path(os.environ['NEW_TOOLS'])/'paired_prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
    run=Path(os.environ['NEW_RUN_ROOT'])
    write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    inherited={k:ledger[k] for k in ('gpu_seconds_used','Judge_attempts')}
    write(run/'private/INHERITED_COST.json',inherited)
    write(run/'RESOURCE_LEDGER.json',dict(inherited,gpu_sessions=[],weights_observed_peak_bytes=0))
    manifest=read(run/'RUN_MANIFEST.json')
    manifest.update(stage='fisher_min_mechanism',Judge_limit=inherited['Judge_attempts'],starting_epoch=time.time(),deadline_epoch=time.time()+48*3600)
    write(run/'RUN_MANIFEST.json',manifest)
    plan=read(run/'PLAN_CONFIG.json')
    plan.update(stage='fisher_min_mechanism',arms=['MULTI','FISHER_MIN'],
        mechanism_gate='REAL_LOWER_FISHER_SAME_FRACTION_AND_FIRST_FRACTION_WITH_MULTIPARITY',
        total_NN_backward_limit=3743,maximum_NN_backwards=3743,operator_backward_limit=256,
        local_solve_limit=256,maximum_non_generation_forwards=60128,maximum_generations=0,maximum_new_Judge=0,
        full_consumer_implemented=False)
    write(run/'PLAN_CONFIG.json',plan)
    env=read(run/'private/LAUNCH_ENV.json');env.update(MULTICORE_PARENT=str(parent))
    write(run/'private/LAUNCH_ENV.json',env)
    write(run/'private/METHOD_LOCK.json',dict(source_commit=os.environ['NEW_COMMIT'],
        objective='.5*normalized_Fisher+.5e-6*normalized_distance_to_fraction_proposal',
        hard_Fisher_fraction=.1,real_source_gain_fraction=.9,step_upper=1.002,inner_margin=1e-5,
        rank_parameters=7168,root_dimensions=512,tail_directions=6,fractions=[1.,.5,.25,.125],
        trust_fractions=[.25,.125,.0625],source_only_RMS=True,heldout_accessed=False,
        meaningful_Fisher_ratio_decrease=1e-4,full_started=False))
    write(run/'public/PROGRESS.json',dict(status='PREPARED_CPU_ADMISSION_PENDING',full_training_started=False))


if __name__=='__main__':main()
