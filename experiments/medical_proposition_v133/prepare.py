"""Freeze MED-only original146 and its original single/insertion/prefix consumers."""
import collections
import json
import os
from pathlib import Path
import sys
import time


def main():
    root, parent, small = map(lambda k:Path(os.environ[k]), ('EXPERIMENT_ROOT','PARENT_RUN','SMALL_RUN'))
    assert not root.exists(), 'Never overwrite an admitted experiment'
    sys.path[:0]=[str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import module_run as r
    import trajectory
    import contract
    import torch
    from m3bench_repro.editors.routing import MemoryRouter
    packet=r.c.read(parent/'private/M13_SOURCE_PACKET.json')
    assert contract.source_admission(packet)['status']=='PASS'
    tasks=packet['original_tasks']; assert [t['order'] for t in tasks]==list(range(1,147))
    old=r.c.read(small/'INPUTS_FROZEN.json'); oldlock=r.c.read(small/'PROTOCOL_LOCK.json')
    basis,_=r.q.split(); assert len(basis)==61
    assert r.c.digest(old['mapping'])==oldlock['mapping_binding']
    for v in old['mapping']:assert v['row']==basis[v['index']]
    source_rows={str(t['order']):trajectory.source_rows(packet,t['order']) for t in tasks}
    # Shared images are not automatically conflicts; the frozen source admission
    # remains authoritative. Surface any exact question/reference contradiction.
    conflicts=[]
    for t in tasks:
        for row in source_rows[str(t['order'])]:
            for b in basis:
                if (row['image_sha256'],row['question'])==(b['image_sha256'],b['question']) and row['reference']!=b['reference']:
                    conflicts.append(dict(owner=t['order'],basis=b['query_id']))
    assert not conflicts, ('Unresolved source-fact conflicts',conflicts)
    reference=Path(os.environ['REFERENCE_RUN'])
    plan=[dict(v,method='MED') for v in r.c.read(reference/'EVAL_PLAN.json') if v['method']=='M13P_160']
    original_single=[dict(v,method='MED') for v in r.c.read(parent/'private/EVAL_PLAN146.json') if v['method']=='M13P']
    assert [v for v in plan if v['mode']=='single']==original_single
    assert len(plan)==7494 and len({contract.consumer_id(v) for v in plan})==7494
    assert collections.Counter(v['mode'] for v in plan)==dict(single=2956,bank=4392,insertion=146)
    entries=[]; bindings=r.c.read(parent/'private/ROUTER_BINDINGS146.json')
    for t in tasks:
        c=r.p.load_state(parent/'private/candidates/C'/f"{t['order']}.pt")['states']
        init=r.p.tr.TT4(t['seed'],8,8).state_dict()
        assert all(torch.equal(init[k],c['BASE'][k].cpu()) for k in init)
        reference=r.c.read(parent/'private/results/C'/f"{t['order']}.json")
        assert len(reference['curve'])==160 and len(reference['gain'])==1
        saved=r.p.load_state(Path(bindings[str(t['order'])]))
        assert len(saved['entries'])==1 and saved['entries'][0]['logical_edit_id']==t['edit_id']
        entries.append(saved['entries'][0])
    for n in (1,50,100,146):
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries[:n]),device='cpu')
        for key in (entries[0]['key'],entries[n-1]['key']):
            d=router.route(torch.as_tensor(key))
            assert not d.activated or d.logical_edit_id in {t['edit_id'] for t in tasks[:n]}
    # Fallbacks and aliases have exactly the already admitted V132 identities.
    frozen=dict(tasks=tasks,mapping=old['mapping'],unmapped=old['unmapped'],source_rows=source_rows,
                plan=plan,base_mask=packet['base_mask'],router_bindings=bindings)
    root.mkdir(parents=True)
    assert os.stat(root).st_dev==os.stat(parent).st_dev
    probe=root/'probe';probe.write_text('133');assert probe.read_text()=='133';probe.unlink()
    assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>10*1024**3
    def write(name,value):r.c.write(root/name,value)
    write('INPUTS_FROZEN.json',frozen)
    budgets=dict(max_elapsed_seconds=86400,new_bytes_limit=45_000_000_000,minimum_free_bytes=8*1024**3,
                 forwards=9_000_000,backwards=53000,updates=24000,operator_backwards=24000,
                 generations=7600,vision_forwards=40000)
    write('PROTOCOL_LOCK.json',dict(experiment='MED146_SINGLE_SEQUENTIAL_V133',seed=oldlock['seed'],GEN_seed=oldlock['GEN_seed'],
        arms=['MED'],owners=list(range(1,147)),steps=160,training_updates=23360,
        input_binding=r.c.digest(frozen),mapping_binding=oldlock['mapping_binding'],numeric=oldlock['numeric'],
        budgets=budgets,allowed_physical_GPUs=[0],minimum_free_MiB=45056,
        source_reference='a5eb69254bb0086d59fad44b09537dc83aba8203',
        stages=['TRAIN_ALL_146','SINGLE2956','SINGLE_SCORE_REPORT','INSERT146_AND_BANK4392','FINAL_SCORE_REPORT'],
        stage_B_condition='StageA engineering/scoring integrity only; scientific negative proceeds',
        retention='Keep one canonical all146 MED FINAL library and independent small backup; never retire automatically',
        cache='Real Base R0 per consumer; selected state/input/prefill/decoder/generation binding equality required',
        score_protocol='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1',
        estimates=dict(training_updates=23360,curvature_backwards=1116,training_backwards=46720,
                       unique_generation_upper=7494,actual_generation_attempt_cap=7600,judge_consumer_upper=22482,
                       recovery_updates_reservation=640,curvature_rebuild_reservation=1,
                       wall_hours=24,peak_owned_GB=4,source_only=True),
        sealed_V111_opened=False,independent_confirmation=False,prepared_time=time.time()))
    write('DATA_ADMISSION.json',dict(status='PASS',owners=146,mapped=31,fallback=30,conflicts=conflicts,
        cold_initializers_exact=146,original_R0_entries=146,original_consumers=7494,CPU_prefix_routes_checked=True,
        original_mapping_exact=True,no_GPU_calls=True,no_Judge=True))
    print(json.dumps(dict(status='PREPARED',consumers=7494,updates=23360,root=str(root))))

if __name__=='__main__':main()
