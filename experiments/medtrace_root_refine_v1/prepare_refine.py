"""Prepare one new bounded correction run; terminal parent files stay read-only."""
import json
import os
from pathlib import Path
import shutil
import time
import prepare
from prior_pairs import read_parent,paired


def main():
    parent=Path(os.environ['JOINT_PARENT']).resolve()
    read=lambda p:json.loads(p.read_text())
    def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    assert read(parent/'EXIT.json')['exit_code']==0
    assert read(parent/'public/PROGRESS.json')['publication_complete']
    assert read(parent/'public/RESULTS.json')['decision']=='NO_VERIFIED_SEARCH_MISS_WITNESS'
    ledger=read(parent/'RESOURCE_LEDGER.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    for path in Path('/proc').glob('[0-9]*/environ'):
        if path.parent.name==str(os.getpid()):continue
        try:entries=path.read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        assert ('RUN_ROOT='+str(parent)).encode() not in entries,'Parent still active'
    rows=read_parent(parent)
    selection=[dict(owner=row['owner'],step=row['step'],fractions=[t['fraction'] for t in row['fractions']
               if t['solver']['status']=='LOCAL_KKT_PASS']) for row in rows]
    assert sum(len(x['fractions']) for x in selection)==6
    assert sum(bool(x['fractions']) for x in selection)==5
    assert sum(x['step'] for x in selection)==341
    prepare.main()
    run=Path(os.environ['NEW_RUN_ROOT']);source=Path(os.environ['NEW_TOOLS'])
    for name in ('refine_math.py','refine_run.py','prior_pairs.py'):
        shutil.copy2(source/name,run/'private/tools'/name)
    for name in ('refinement_states','PARENT_PARITY'):(run/'private'/name).mkdir()
    write(run/'private/PARENT_SELECTION.json',dict(rows=selection,parent=str(parent),
           parent_commit=read(parent/'private/TERMINAL_DELIVERY.json')['commit'],
           selection='All six prior LOCAL_KKT_PASS candidates, including failed quarter fraction',
           maximum_repairs=3,trust_radii_over_full_step=[.25,.125,.0625]))
    write(run/'public/PAIRED_PRIOR_RESULTS.json',paired(rows))
    env=read(run/'private/LAUNCH_ENV.json');env['JOINT_PARENT']=str(parent)
    write(run/'private/LAUNCH_ENV.json',env)
    inherited=dict(gpu_seconds_used=ledger['gpu_seconds_used'],Judge_attempts=ledger['Judge_attempts'])
    write(run/'private/INHERITED_COST.json',inherited)
    write(run/'RESOURCE_LEDGER.json',dict(inherited,gpu_sessions=[],weights_observed_peak_bytes=0))
    plan=read(run/'PLAN_CONFIG.json')
    plan.update(stage='root_refine_fixed_states',arms=['PURE','ORIGINAL_GUARD','JOINT_ROOT','BOUNDED_REFINEMENT'],
                eligible_candidates=6,maximum_repairs_per_candidate=3,maximum_NN_backwards=3125,
                maximum_non_generation_forwards=12000,maximum_local_solves=50)
    write(run/'PLAN_CONFIG.json',plan)
    manifest=read(run/'RUN_MANIFEST.json');manifest.update(stage='root_refine_fixed_states',starting_epoch=time.time())
    assert manifest['deadline_epoch']>time.time()+3600
    write(run/'RUN_MANIFEST.json',manifest)
    write(run/'private/GPU_SOURCE_VERSION.json',dict(commit=os.environ['NEW_COMMIT'],parent='5614048',stage='root_refine'))
    print(json.dumps(dict(prepared=True,fixed_states=8,eligible_candidates=6,inherited_GPU_hours=inherited['gpu_seconds_used']/3600)))


if __name__=='__main__':main()
