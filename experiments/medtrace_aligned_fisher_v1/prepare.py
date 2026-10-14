"""Prepare one finite direction test, inheriting the fully delivered parent."""
import importlib.util
import json
import os
from pathlib import Path


def main():
    parent=Path(os.environ['MULTICORE_PARENT'])
    spec=importlib.util.spec_from_file_location('fisher_prepare',Path(os.environ['NEW_TOOLS'])/'fisher_prepare.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.main()
    run=Path(os.environ['NEW_RUN_ROOT']);read=lambda p:json.loads(p.read_text())
    write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    env=read(run/'private/LAUNCH_ENV.json')
    env.update(BASE_PARENT=str(parent),MULTICORE_PARENT=read(parent/'private/LAUNCH_ENV.json')['MULTICORE_PARENT'])
    write(run/'private/LAUNCH_ENV.json',env)
    manifest=read(run/'RUN_MANIFEST.json');manifest['stage']='aligned_fisher_mechanism';write(run/'RUN_MANIFEST.json',manifest)
    plan=read(run/'PLAN_CONFIG.json');plan.update(stage='aligned_fisher_mechanism',arms=['FISHER_MIN','ALIGNED_MIN'],
        alignment_floor=.9,mechanism_gate='REAL_RESTORED_ALIGNMENT_PAIR_AND_FIRST_FRACTION',full_consumer_implemented=False)
    write(run/'PLAN_CONFIG.json',plan)
    method=read(run/'private/METHOD_LOCK.json');method.update(alignment_floor=.9,alignment_inner_margin=1e-5,
        alignment_external_tolerance=1e-7,extra_rule='operator_dot(candidate-before,pure-before)>=.9*R^2',
        original_external_rules_unchanged=True)
    write(run/'private/METHOD_LOCK.json',method)


if __name__=='__main__':main()
