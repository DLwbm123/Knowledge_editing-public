"""CPU checks for persistent loading and cumulative per-trajectory counters."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import runtime146 as worker
import sys
sys.path.insert(0,str(Path(__file__).parent))
import native_fmin
import selected_training146 as selected


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    a,b=selected.assignments(0),selected.assignments(1)
    assert len(a)==len(b)==72 and set(a).isdisjoint(b) and sorted(a+b)==list(range(3,147))
    runtime,bindings=object(),object()
    with patch.object(worker.r.p,'lease',side_effect=AssertionError('Borrow must not acquire a lease')), \
         patch.object(worker.r,'load',side_effect=AssertionError('Borrow must not load a model')):
        with worker.network(0,(runtime,bindings)) as loaded:assert loaded==(runtime,bindings)
    import selected_queue146 as queue
    assert queue.base.ADMISSION_FILE=='SELECTED_RUNTIME_ADMISSION1.json'
    assert queue.base.SCOPE_FILE=='SELECTED_SCOPE1.json'
    assert queue.base.GENERATION_FILE=='SELECTED_GENERATION_COMPLETE1.json'
    try:worker.stage_admission('FISHER_MIN',True)
    except FileNotFoundError:pass
    else:raise AssertionError('Unprovisioned full lane admitted')
    initial=dict(worker.r.COUNTS);initial['local_solves']=91
    worker.r.COUNTS.update(initial)
    worker.configure();assert worker.r.COUNTS['local_solves']==91
    initial=dict(worker.r.COUNTS)
    with patch.object(worker.c,'budget'):
        native_fmin.configure()
        start=dict(worker.r.COUNTS)
        worker.r.count();native_fmin.p.solve_count()
        assert worker.r.COUNTS['non_generation_forwards']==start['non_generation_forwards']+1
        assert worker.r.COUNTS['local_solves']==start['local_solves']+1
        worker.r.COUNTS['non_generation_forwards']=start['non_generation_forwards']+100000
        try:worker.r.count()
        except AssertionError:pass
        else:raise AssertionError('Per-trajectory forward cap lost in persistent worker')
    packet=worker.packet();rows=__import__('trajectory').source_rows(packet,1)
    scope=worker.c.read(worker.RUN/'private/NATIVE_FMIN_SCOPE1.json')
    calibration=worker.c.read(worker.RUN/'private/calibration146/1.json')
    binding=dict(arm='FISHER_MIN',seed=packet['original_tasks'][0]['seed'],source_recipe=__import__('contract').SOURCE_RECIPE,
        source_order=[v['query_id'] for v in rows],source_binding=calibration['source_binding'],
        scope_binding=worker.c.digest(scope),source_version=worker.c.read(worker.RUN/'private/NATIVE_FMIN_SOURCE1.json')['commit'],
        budget=.1*calibration['C_operator_penalty'],steps=160,parameters=7168)
    path=worker.RUN/'private/resume146/FISHER_MIN_1.pt'
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(worker.RUN.resolve())
    old=torch.load(path,map_location='cpu',weights_only=False)
    native_fmin.validate(old,binding)
    saved=copy.deepcopy(old);offset={k:1000 for k in old['counts']}
    saved['counts_start']=offset;saved['counts']={k:v+offset[k] for k,v in old['counts'].items()}
    native_fmin.validate(saved,binding)
    saved['scalar']['step']+=1
    try:native_fmin.validate(saved,binding)
    except AssertionError:pass
    else:raise AssertionError('Wrong RMS clock accepted')
    worker.r.COUNTS.clear();worker.r.COUNTS.update(initial)
    result=dict(status='PASS',two_disjoint_72_owner_partitions=True,pilot_owners_excluded=True,
        borrowed_network_no_extra_lease_or_load=True,cumulative_counts_not_reset=True,
        finite_per_trajectory_forward_cap=True,own_snapshot_and_cumulative_counts_binding=True,
        altered_RMS_clock_rejected=True,unprovisioned_full_lane_refused=True,selected_queue_does_not_use_legacy_admission=True,new_GPU=0,new_VLM=0,new_Judge=0,
        execution_admission_written=False,full146_launched=False)
    Path(os.environ['SELECTED_TRAINING_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
