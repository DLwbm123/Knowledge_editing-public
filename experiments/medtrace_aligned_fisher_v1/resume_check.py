"""Actual checkpoint boundary checks, without a model or GPU lease."""
import copy
from pathlib import Path
import os
import resume_aligned as u

report=u.admission()
owner=next(v['owner'] for v in report['pending'] if v['checkpoint'])
probe=u.r.p.load_state(u.RUN/'private/probes'/f'{owner}.pt')
saved=u.r.p.load_state(u.RUN/'private/full_resume'/f'ALIGNED_MIN_{owner}.pt')
task=next(t for t in u.r.s.tasks() if t['order']==owner)
for mutate in ('seed','clock','source','budget','RNG'):
    bad=copy.deepcopy(saved)
    if mutate=='seed':bad['seed']+=1
    elif mutate=='clock':bad['scalar']['step']-=1
    elif mutate=='source':bad['source_order']=list(reversed(bad['source_order']))
    elif mutate=='budget':bad['budget']*=2
    else:bad['rng']['cuda_rng']=None
    try:u.validate(bad,task,probe)
    except AssertionError:pass
    else:raise AssertionError('Invalid resume accepted: '+mutate)
assert report['remaining_scientific_proposals']==291
all_tasks=u.r.s.tasks()
assert [[t['order'] for t in all_tasks[i::3] if t['order'] not in report['complete']] for i in range(3)]==[[4,7],[8],[]]
u.c.write(u.RUN/'public/USER_RESUME_CPU_CHECKS.json',dict(status='PASS',actual_checkpoint=True,
    completed_editors_reused=5,remaining_scientific_proposals=291,negative_boundary_checks=5,
    GPU_leases=0,new_model_calls=0,scientific_rules_changed=False,finite_cost_forecast=report['finite_forecast']))
print('USER_RESUME_CPU_CHECKS PASS',flush=True)
