"""CPU-only aggregate of all fixed non-teacher mechanism cells."""
import json
from pathlib import Path
import statistics


def analyze(root):
    root=Path(root)
    read=lambda p:json.loads(p.read_text())
    rows=[read(p) for p in sorted((root/'cells').glob('*.json'))]
    assert {(d['owner'],d['anchor']) for d in rows}=={(i,a) for i in range(1,9) for a in ('M1','M13')}
    assert all(d['status']=='PASS' for d in rows)
    completion=read(root/'MAIN_COMPLETE.json')
    assert completion['counts']['backwards']==80 and completion['counts']['updates']==32
    assert all(read(root/f'EXIT_geometry_{p}.json')['exit_code']==0 for p in range(2))
    mechanism=read(root/'MECHANICAL.json');assert mechanism['status']=='PASS'
    guard=read(root/'RESOURCE_GUARD_EXIT.json');assert guard['status']=='ALL_REGISTERED_PROCESSES_EXITED' and guard['trigger'] is None
    limits=read(root/'RESOURCE_LIMITS.json')
    targets=read(root/'RESOURCE_TARGETS.json')['targets']
    gpu=sum(t['ended_epoch']-t['started_epoch'] for t in targets if t.get('gpu') is not None)
    assert all(completion['counts'][k]<=v for k,v in limits['counts'].items())
    result=dict(status='COMPLETED_LOCAL_PROTECTION_TRADEOFF',cells=16,groups={},per_edit=[],
        counts=completion['counts'],gpu_process_seconds=gpu,all_owned_processes_exited=True,
        same_prefix_logits_max_abs=mechanism['logits_max_abs'],new_model_checkpoints=0,
        teacher_calls=0,heldout_calls=0,
        limitation='Local source-only finite steps at historical saturated endpoints; not T2G attribution, preservation success, or paper-method reproduction.',
        engineering=dict(zero_model_bootstrap_failure_preserved=True,
            guard_self_target_registration_fixed_after_model_completion=True,
            no_model_replay=True,original_clock_and_counts_preserved=True))
    for anchor in ('M1','M13'):
        cells=[d for d in rows if d['anchor']==anchor]
        diffs=[d['updates']['P']['weighted_source_gain']-d['updates']['U']['weighted_source_gain'] for d in cells]
        u=statistics.mean(d['updates']['U']['weighted_source_gain'] for d in cells)
        p=statistics.mean(d['updates']['P']['weighted_source_gain'] for d in cells)
        result['groups'][anchor]=dict(source_gain_U_mean=u,source_gain_P_mean=p,
            P_less_gain_than_U=sum(x<0 for x in diffs),P_more_gain_than_U=sum(x>0 for x in diffs),
            raw_equal=sum(x==0 for x in diffs),mean_P_minus_U_gain=statistics.mean(diffs),
            aggregate_gain_ratio_P_U=p/u if u else None,
            operator_cosine_mean=statistics.mean(d['operator_cosine_P_U'] for d in cells),
            P_lower_Fisher_than_U=sum(d['updates']['P']['Fisher_penalty']<d['updates']['U']['Fisher_penalty'] for d in cells),
            P_positive_source_gain=sum(d['updates']['P']['weighted_source_gain']>0 for d in cells),
            U_positive_source_gain=sum(d['updates']['U']['weighted_source_gain']>0 for d in cells))
    for d in rows:
        result['per_edit'].append(dict(owner=d['owner'],anchor=d['anchor'],
            NLL_before=d['weighted_source_NLL_before'],gain_U=d['updates']['U']['weighted_source_gain'],
            gain_P=d['updates']['P']['weighted_source_gain'],operator_cosine=d['operator_cosine_P_U'],
            Fisher_before=d['Fisher_penalty_before'],Fisher_U=d['updates']['U']['Fisher_penalty'],Fisher_P=d['updates']['P']['Fisher_penalty']))
    (root/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (root/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status='PASS',cells=16,no_selection=True)))
    return result
