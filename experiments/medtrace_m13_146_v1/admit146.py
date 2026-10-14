"""Operational CPU admission for an explicitly provisioned new RUN; never launches."""
import os
from pathlib import Path
import shutil
import tempfile
import torch
import contract
import trajectory
from runtime146 import c,r,RUN


def check_router(entry,original):
    assert entry['logical_edit_id']==original['logical_edit_id'] and entry['radius']==original['radius']
    assert tuple(entry.get('label',()))==tuple(original.get('label',()))
    assert torch.equal(torch.as_tensor(entry['key'],dtype=torch.float32),torch.as_tensor(original['key'],dtype=torch.float32))


def admit():
    assert RUN.resolve().is_relative_to(Path('/data/bmw/Knowledge_editing/outputs'))
    packet=c.read(RUN/'private/M13_SOURCE_PACKET.json')
    assert contract.source_admission(packet)['status']=='PASS','All original146 source evidence is mandatory'
    authorization=c.read(RUN/'private/USER_LAUNCH_AUTHORIZATION.json')
    assert authorization['launch_authorized'] is True and authorization['original_questions_unchanged'] is True
    env=c.read(RUN/'private/LAUNCH_ENV.json');upstream=Path(env['ALIGNED_CONTROL_PARENT'])
    assert c.read(upstream/'private/COMPLETE.json')['status']=='COMPLETE'
    assert c.read(upstream/'private/FULL_EXIT1.json')['exit_code']==0
    delivery=Path(env['ALIGNED_DELIVERY_RECEIPT']);assert delivery.is_relative_to(upstream/'private')
    assert c.read(delivery)['remote_SHA_match']
    for file in ('RUNTIME_CPU_CHECKS.json','REPORT_CPU_CHECKS.json','TRAJECTORY_CPU_CHECKS.json','QUEUE_CPU_CHECKS.json'):
        assert c.read(RUN/'private/admission_checks'/file)['status']=='PASS'
    for task in packet['original_tasks']:
        initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
        assert sum(v.numel() for v in initial.values())==7168
        for row in trajectory.source_rows(packet,task['order']):assert Path(row['image_path']).is_file()
    for rows in packet['reviews']['owners'].values():
        for row in rows:
            if row.get('admitted'):assert all(Path(v['path']).is_file() for v in row['source_evidence'])
    plan=contract.evaluation_plan(packet,c.read(RUN/'private/EVAL_QUERIES146.json'))
    router=c.read(RUN/'private/ROUTER_BINDINGS146.json')
    original_router=c.read(RUN/'private/ORIGINAL_ROUTER_MANIFEST146.json')
    assert set(router)=={str(i) for i in range(1,147)}
    for task in packet['original_tasks']:
        saved=r.p.load_state(Path(router[str(task['order'])]))
        assert len(saved['entries'])==1 and saved['entries'][0]['logical_edit_id']==task['edit_id']
        assert saved['native_binding']==contract.original.task_binding(task)['native']
        assert original_router[str(task['order'])]['native_binding']==saved['native_binding']
        check_router(saved['entries'][0],original_router[str(task['order'])]['entry'])
    assert shutil.disk_usage(RUN).free>=8*1024**3
    with tempfile.NamedTemporaryFile(dir=RUN) as file:
        file.write(b'm13 storage probe');file.flush();assert Path(file.name).read_bytes()==b'm13 storage probe'
    manifest=c.read(RUN/'RUN_MANIFEST.json')
    assert set(manifest['physical_GPUs'])<=set((0,1,2,3,4,7))
    assert manifest['owned_weight_limit_bytes']==2*1024**3 and manifest['min_free_bytes']==8*1024**3
    assert c.read(RUN/'PLAN_CONFIG.json')['maximum_counts']==contract.CAPS
    scope=dict(method_flags={k:list(v) for k,v in contract.FLAGS.items()},methods=contract.METHODS,
        steps=160,parameters=7168,source_weights=[.25,.25,.5],prefixes=contract.PREFIXES,
        soft_Fisher_coefficient_fraction=.1,hard_Fisher_budget=False,source_rejection=False,
        task_binding=[contract.original.task_binding(t) for t in packet['original_tasks']])
    with (RUN/'private/M13_146_ADMISSION_ONCE').open('x'):pass
    c.write(RUN/'private/SCOPE_LOCK.json',scope);c.write(RUN/'private/EVAL_PLAN146.json',plan)
    c.write(RUN/'private/ORIGINAL146_TASKS.json',packet['original_tasks'])
    c.write(RUN/'private/ORIGINAL_BASE_MASK146.json',packet['base_mask'])
    c.write(RUN/'private/SOURCE_ADMISSION146.json',dict(status='PASS',packet_binding=c.digest(packet)))
    c.write(RUN/'private/M13_146_RUNTIME_ADMISSION.json',dict(status='PASS',launch_authorized=True,
        source_commit=c.read(RUN/'private/GPU_SOURCE_VERSION.json')['commit'],packet_binding=c.digest(packet),
        scope_binding=c.digest(scope),plan_binding=c.digest(plan),physical_source_paths_checked=True,
        cold_initialization_exact=True,original_R0_bindings_checked=True,storage_write_read=True,
        live_network_verified=False,live_network_gate='First source logits must verify actual zero-function Base before updates'))


if __name__=='__main__':admit()
