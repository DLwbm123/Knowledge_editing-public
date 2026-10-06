"""Create one independent run from a frozen PR27 run; no GPU or old-run writes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def prepare(plan,source_dir,report_dir,commit,data_summary):
    root=Path(plan['run_root']);old=Path(plan['parent_run'])
    assert not root.exists(),'Never reset an existing run or its clock'
    assert read(old/'private/CONTROLLER_COMPLETE.json')['status']=='GENERATION_COMPLETE'
    assert read(old/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    assert shutil.disk_usage(root.parent.parent).free>10*1024**3
    root.mkdir(parents=True)
    probe=root/'.write_probe';probe.write_text('bounded-write');assert probe.read_text()=='bounded-write';probe.unlink()
    for name in ('private/tools','private/teacher','private/tmp','private/edits','private/base','private/mechanical','private/judge_common','logs','public'):(root/name).mkdir(parents=True,exist_ok=True)
    for name in ('source','official_llava','cpu_gate','legacy_stage17','derived_inputs'):(root/'private'/name).symlink_to(old/'private'/name,target_is_directory=True)
    for name in ('H_AVAILABLE.json','SOURCE_COMMIT.json'):shutil.copy2(old/'private'/name,root/'private'/name)
    for p in (old/'private/base').glob('*.json'):shutil.copy2(p,root/'private/base'/p.name)
    dependencies=['legacy_worker.py','audit.py','admit.py','updates.py','structures.py','protocol.py','legacy_queue.py','judge_queue.py','legacy_metrics.py']
    for name in dependencies:shutil.copy2(old/'private/tools'/name,root/'private/tools'/name)
    for src,dest in [('worker.py','baseline_worker.py'),('controller.py','controller_parent.py'),('reporting.py','reporting_parent.py'),('qwen_queue.py','qwen_parent.py'),('qwen_scorer.py','qwen_scorer.py')]:
        shutil.copy2(old/'private/tools'/src,root/'private/tools'/dest)
    qp=root/'private/tools/qwen_parent.py';qp.write_text(qp.read_text().replace("epoch='MEDTRACE_TT_SVD_H8_QWEN_C32_20261005_V1'","epoch='MEDTRACE_TT_DIRECTION_H_20261006_V1'"))
    # Restrict Judge to idle GPUs too; this changes admission, not scoring identity.
    scorer=root/'private/tools/qwen_scorer.py';text=scorer.read_text()
    text=text.replace("if int(free)>=60000:gpu=g;break","if int(free)>=60000 and judge_idle(g,u):gpu=g;break")
    text=text.replace('def caps():',"def judge_idle(gpu,uuid):\n    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()\n    usage=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')\n    return uuid not in [r.strip() for r in rows] and int(usage[0])==0 and int(usage[1])<100\n\ndef caps():")
    scorer.write_text(text)
    for p in Path(source_dir).glob('*.py'):shutil.copy2(p,root/'private/tools'/p.name)
    for p in Path(report_dir).glob('*'):
        if p.suffix in ('.json','.md'):shutil.copy2(p,root/p.name)
    for name in ('QWEN_JUDGE_AMENDMENT.json','RUNTIME_REPRODUCIBILITY_AMENDMENT.json'):shutil.copy2(old/name,root/name)
    shutil.copy2(old/'private/judge_common/RUNTIME.private.json',root/'private/judge_common/RUNTIME.private.json')
    epoch=time.time()
    write(root/'RUN_MANIFEST.json',dict(stage=plan['stage'],starting_epoch=epoch,deadline_epoch=epoch+7*24*3600,GPU_seconds_limit=24*3600,Judge_limit=6000,owned_weight_limit_bytes=2*1024**3,min_free_bytes=8*1024**3,physical_GPUs=[3,4,5,6],reset_allowed=False,plan_commit=commit,wall_limit='7-day capacity-wait safety bound; scientific compute cap24GPUh'))
    write(root/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0.,gpu_sessions=[],Judge_attempts=0,Judge_batches=[]))
    versions={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'private/tools').glob('*.py')}
    write(root/'private/GPU_SOURCE_VERSION.json',dict(commit=commit,tool_sha256=versions,reason='Explicit protocol-required locked source binding'))
    l=read(root/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');H=read(root/'private/H_AVAILABLE.json');by={t['edit_id']:t for t in l['tasks']}
    assert sorted(by[e]['order'] for e in H)==plan['H8_orders'] and sum(map(len,H.values()))==12
    inherited=read(old.parent.parent/'medtrace-h8-causal-v1-20261004/run/private/H_SEMANTIC_BINDING_AUDIT.json')
    assert not inherited['blocked'];evidence={r['review_id']:r for r in inherited['rows']};rows=[]
    for eid,hs in H.items():
        for h in hs:
            ev=h['evidence'];prior=evidence[ev['review_id']];source=ev['source'];original=read(source['source_file'])
            record=next(r for r in original if str(r['qid'])==str(source['qid']))
            assert source['original_source_row']==record and h['reference']==record['answer'] and h['question']==by[eid]['native']['question']
            assert all(prior['checks'].values()) and ev['admitted_H_fit'] and ev['source_binding']=='PASS'
            rows.append(dict(order=by[eid]['order'],edit_id=eid,review_id=ev['review_id'],source_group=h['source_group'],original_QA=source,scope_evidence=ev['evidence'],H_scope='SOURCE_BOUND_NOT_CLINICAL_SIGNOFF',label_source=prior,patient='UNKNOWN',Base_is_not_scope_qualification=True))
    write(root/'private/SCOPE_AUDIT.json',dict(rows=rows,summary=dict(relations=12,edits=8,original_QA=7,source_components=5,patient='UNKNOWN',scope_status='PRIOR_SOURCE_BOUND_EVIDENCE_REVALIDATED',clinical_signoff=False,clear_input_or_role_errors=0)))
    inventory=[]
    for order in plan['H8_orders']:
        folder=old/'private/edits'/f'e{order:03d}'
        paths=[folder/'ROUTER.pt']+[folder/f'{s}_s{k}_WARMUP/W0.pt' for k in range(3) for s in ('TT44','TT84','TT48','TT88')]+list(folder.glob('*/final.pt'))
        for p in paths:assert p.is_file();inventory.append(dict(path=str(p),bytes=p.stat().st_size,readonly=True))
    assert len(inventory)==96+240+8
    write(root/'private/PARENT_REUSE_INVENTORY.json',dict(items=inventory,checks='existence+known counts; TT state/ancestry checked on use; not a bytewise copy audit'))
    data=dict(status='BLOCKED_DATA',source_stage='h_source_grounded_v1_20261006',source_commit='e844d1c',candidate_verified_eval_relations=data_summary['eval_verified']['relations'],candidate_verified_eval_edits=data_summary['eval_verified']['edits'],qualified_same_modality_same_body_hard_H=data_summary['near_boundary_confirmed'],patient_independence='UNKNOWN',T1L='NO_QUALIFIED_EXTERNAL_DENOMINATOR',T2L='NO_QUALIFIED_EXTERNAL_DENOMINATOR',external_independent_audit=data_summary['API']['status'],sealed_eval_read=False,reason='Current source audit has zero confirmed same-modality/same-body hard H; clinical referent/modality uncertainty and qualified locality denominator unresolved. Existing isolated candidates are not a complete independently qualified paired panel.',next_collection_target=dict(edits=24,source_groups=12),no_old_H_substitution=True)
    assert data['qualified_same_modality_same_body_hard_H']==0
    write(root/'private/P3_DATA_GATE.json',data)
    env=read(old/'private/LAUNCH_ENV.json');env['RUN_ROOT']=str(root);env['VLLM_RPC_BASE_PATH']=str(Path('/data/bmw/tmp')/('ipc.'+str(os.getpid())))
    Path(env['VLLM_RPC_BASE_PATH']).mkdir(exist_ok=True)
    write(root/'private/LAUNCH_ENV.json',env)
    write(root/'private/CPU_ADMISSION.json',dict(status='PASS',parent_commit=plan['parent_commit'],source_binding=True,scope_rows=12,retained_parent_weights=len(inventory),GPU_started=False))
    return root


if __name__=='__main__':
    import sys
    payload=json.load(sys.stdin);root=prepare(**payload);print(root)
