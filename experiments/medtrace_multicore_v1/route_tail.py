"""Audited zero-output route repair; never re-enter training or forced evaluation."""
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
import paired_run as p

c, r, RUN = p.c, p.r, p.RUN


def admission():
    assert c.read(RUN/'private/EXIT1.json')['exit_code']==1
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['candidates']==16
    ledger=c.read(RUN/'RESOURCE_LEDGER.json')
    assert len(ledger['gpu_sessions'])==11 and all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    failed=c.read(RUN/'private/FAILURE_paired_route_0.json')
    assert 'BENCHMARK146_QUEUE.json' in failed['traceback']
    assert all(v==0 for v in failed['counts'].values())
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==1878
    assert not list((RUN/'private/outputs').glob('NATURAL_*/*/*.json'))
    assert not (RUN/'private/ROUTES.json').exists()
    assert ledger['Judge_attempts']==c.read(RUN/'private/INHERITED_COST.json')['Judge_attempts']
    parent=Path(os.environ['REFINE_PARENT'])
    queue=c.read(parent/'private/BENCHMARK146_QUEUE.json')
    target=RUN/'private/BENCHMARK146_QUEUE.json'
    if target.exists(): assert c.read(target)==queue
    else: shutil.copy2(parent/'private/BENCHMARK146_QUEUE.json',target)
    routers=[]
    original=c.read(r.s.PARENT/'private/BENCHMARK146_QUEUE.json')['tasks'][:8]
    for before, task in zip(original,r.s.tasks()):
        path=r.p.initial(before).parent.parent/'ROUTER.pt'
        router=r.p.load_state(path)
        assert len(router['entries'])==1 and router['entries'][0]['logical_edit_id']==before['edit_id']
        routers.append(str(path))
        for arm in p.ARMS:
            result=c.read(RUN/'private/results'/arm/f"{task['order']}.json")
            assert len(result['curve'])==160 and result['scalar_optimizer_state']['step']==160 and result['save_load_exact']
            final=r.p.load_state(r.s.point(task,arm))
            resume=r.p.load_state(RUN/'private/full_resume'/f"{arm}_{task['order']}.pt")
            assert resume['step']==160 and resume['scalar']==final['scalar_optimizer_state']
            assert all(torch.equal(resume['state'][k],final['states']['FINAL'][k]) for k in r.mm.KEYS)
    assert all(c.read(RUN/'private/RESUME_REAL'/f'{arm}_{i}.json')['status']=='PASS'
               for arm in p.ARMS for i in range(1,9))
    out=dict(status='PASS',frozen_candidates=16,terminal_step=160,save_load_and_resume_exact=True,
             historical_queue_binding=True,original_router_entries=8,forced_outputs_retained=1878,
             natural_outputs_before_repair=0,new_Judge=0,extra_failed_GPU_leases=1,
             training_restarted=False,scientific_rules_changed=False,epoch=time.time())
    c.write(RUN/'private/ROUTE_REPAIR_ADMISSION1.json',dict(out,router_paths=routers))
    return out


def tail():
    with (RUN/'private/ROUTE_TAIL_ONCE1').open('x'): pass
    out=admission()
    archive=RUN/'private/route_zero_output_recovery1';archive.mkdir()
    for name in ('FAILURE_paired_route_0.json','FAILURE_paired_controller_0.json','EXIT1.json'):
        (RUN/'private'/name).rename(archive/name)
    c.write(RUN/'private/import_recovery1/ADMISSION.json',out)
    c.write(RUN/'public/ROUTE_REPAIR_ADMISSION.json',out)
    c.write(RUN/'public/PROGRESS.json',dict(status='NATURAL_ROUTE_REPAIR_RUNNING',training_complete=True))
    p.j.stage('paired_route_recovery1',[0],file='route_tail.py')
    assert len(list((RUN/'private/outputs').glob('NATURAL_*/*/*.json')))==438
    p.scoring_tail()
    p.final_report()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='route_repair_cpu': print(admission())
        elif action=='route_tail_recovery1': tail()
        elif action=='paired_route_recovery1':
            import module_route
            module_route.main(expected_candidates=16)
        else: raise ValueError(action)
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION','0')}.json",
                dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
