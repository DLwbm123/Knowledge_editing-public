"""Separate same-precision editing effects from total changes to the deployment Base."""
import os
import json
import time
from pathlib import Path
import common as c
import probe_report as stats
import bounded_report as gate

RUN=Path(os.environ['RUN_ROOT']);ARMS=('RAW','BOUNDED','MATCHED_RAW')


def summarize(rows,field='candidates'):
    assert rows
    return {a:dict(KL=stats.mean([r[field][a]['KL'] for r in rows]),
        lost_correct_tokens=sum(r[field][a]['lost_correct_tokens'] for r in rows)) for a in ARMS}


def main():
    gate.selfcheck()
    source=[c.read(f) for f in sorted((RUN/'private/source_results').glob('*.json'))]
    held=[c.read(f) for f in sorted((RUN/'private/results').glob('*.json'))]
    freeze=c.read(RUN/'private/CANDIDATES_FROZEN.json');lock=c.read(RUN/'private/PROBE_LOCK.json')
    assert len(source)==8 and len({s['expert_order'] for s in source})==8
    assert all(s['lock']==c.digest(lock) for s in source+held)
    assert all(s['state_restored_exact'] and s['baseline_repeat_exact'] and not s['Base_gradient'] for s in source+held)
    assert not list((RUN/'private').glob('FAILURE*'))
    assert len(held)==(8 if freeze['source_gate_passed'] else 0)
    source_count=sum(c.read(f)['forwards'] for f in (RUN/'private').glob('SOURCE_COUNTS_*.json'))
    held_counts=[c.read(f) for f in (RUN/'private').glob('HELD_COUNTS_*.json')]
    assert source_count==sum(s['forward_calls'] for s in source)
    assert 2784<=source_count<=3424 and sum(s['backward_calls'] for s in source)==15672
    assert sum(s['forwards'] for s in held_counts)==(4424 if held else 0)
    assert sum(h['forward_calls'] for h in held)+sum(x['shared_FP16_Base_forwards'] for x in held_counts)==sum(x['forwards'] for x in held_counts)
    per=[]
    for s in source:
        edit=[r for r in s['diagnostics'] if r['role']=='EDIT'];assert len(edit)==5
        gain={a:-sum((.5 if r['index']==0 else .125)*r['candidates'][a]['NLL_change'] for r in edit) for a in ARMS}
        assert gain['RAW']>1e-8 and gain['BOUNDED']>=.9*gain['RAW']
        assert s['map_norms']['BOUNDED']<=s['map_norms']['RAW']*(1+1e-6)+1e-10
        assert abs(s['function_matching']['actual']-s['function_matching']['target'])<=1e-10+1e-3*s['function_matching']['target']
        per.append(dict(expert_order=s['expert_order'],edit_progress=gain,
            actual_edit_retention=gain['BOUNDED']/gain['RAW'],source_gate_passed=s['source_gate_passed'],
            edit_lost_tokens={a:sum(r['candidates'][a]['lost_correct_tokens'] for r in edit) for a in ARMS},
            geometry=s['geometry'],matching=s['function_matching'],basis_audit=s['basis_audit']))
    edit={a:stats.mean([r['edit_progress'][a] for r in per]) for a in ARMS}
    edit_lost=sum(r['edit_lost_tokens']['BOUNDED'] for r in per)
    protection=None;lost=None;endtoend=None;migration=None;secondary=None
    decision='SOURCE_GATE_NOT_MET'
    if held:
        assert all(len(h['diagnostics'])==96 for h in held)
        by_order={h['expert_order']:h for h in held};primary=[]
        for s in per:
            rs=[r for r in by_order[s['expert_order']]['diagnostics'] if r['semantic_correct']]
            assert len(rs)==63;s['heldout_KL']={a:stats.mean([r['candidates'][a]['KL'] for r in rs]) for a in ARMS};primary.extend(rs)
        groups=sorted({r['group'] for r in primary});assert len(groups)==29 and len(primary)==504
        protection={a:stats.mean([r['heldout_KL'][a] for r in per]) for a in ARMS}
        protection.update(expert_CI=stats.interval([r['heldout_KL']['BOUNDED']-r['heldout_KL']['MATCHED_RAW'] for r in per]),
            source_CI=stats.interval([stats.mean([r['candidates']['BOUNDED']['KL']-r['candidates']['MATCHED_RAW']['KL'] for r in primary if r['group']==g]) for g in groups]))
        lost={a:sum(r['candidates'][a]['lost_correct_tokens'] for r in primary) for a in ARMS}
        decision=gate.decide(protection,edit,[r['actual_edit_retention'] for r in per],edit_lost,lost)
        endtoend=summarize(primary,'end_to_end')
        migration=dict(KL=stats.mean([r['baseline_migration']['KL'] for r in primary]),
            lost_correct_tokens=sum(r['baseline_migration']['lost_correct_tokens'] for r in primary))
        secondary=summarize([r for h in held for r in h['diagnostics']])
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==(12 if held else 6) and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    assert ledger['Judge_attempts']==before['Judge_attempts']==12412
    result=dict(status='COMPLETE',decision=decision,source_gate_passed=freeze['source_gate_passed'],
        primary_panel='ORIGINAL_DEPLOYMENT_BASE_CORRECT_63',primary_queries=63,primary_sources=29,
        baseline_semantic_migration=dict(original_correct=63,FP32_correct=62,denominator=63,new_scoring=False),
        protection=protection,edit_progress=edit,response_lost_tokens=lost,edit_lost_tokens=edit_lost,
        source_fit_diagnostic=summarize([r for s in source for r in s['diagnostics'] if r['role']=='BASIS_FIT']),
        end_to_end_from_original_Base=endtoend,baseline_numerical_migration=migration,all96_secondary=secondary,
        per_expert=per,matching_is_map_norm_not_edit_gain=True,free_generation_evaluated=False,
        independent_confirmation=False,clinical_protection=False,full_training_success=False,
        resource=dict(forward_calls=source_count+sum(x['forwards'] for x in held_counts),backward_calls=15672,
            new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,
            cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,cumulative_Judge=ledger['Judge_attempts'],
            temporary_updates=8,new_generations=0,new_Judge=0,temporary_candidate_packages=8))
    c.write(RUN/'public/RESULTS.json',result)
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status='PASS',experts=8,source_freeze_before_heldout=True,
        Base_checks=sum(s['zero_expert_Base_checks'] for s in source+held),source_forward_calls=source_count,
        held_forward_calls=sum(x['forwards'] for x in held_counts),backward_calls=15672,sessions_ended=len(ledger['gpu_sessions'])))
    (RUN/'public/REPORT_ZH.md').write_text('# 统一FP32真实编辑约束机制实验\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n原FP32资格失败保留；原63主面板不改为62。KL使用原固定回答前缀，端到端KL单独实测而非相减；无自由生成保护或临床确认结论。\n')
    paths=[RUN/'private/candidates'/f"{s['expert_order']}.pt" for s in source]
    assert all(x.is_file() and not x.is_symlink() and x.resolve().parent==(RUN/'private/candidates').resolve() for x in paths)
    total=sum(x.stat().st_size for x in paths)
    if decision=='LOCAL_MECHANISM_SIGNAL':
        lifecycle=dict(status='RETAINED_FOR_CONDITIONAL_CONSUMER',packages=8,bytes=total,
            consumer='Fixed 8-expert free-generation comparison after separate preregistration, at most 2464 outputs',historical_artifacts_deleted=0)
    else:
        c.write(RUN/'private/CANDIDATE_DELETION_PLAN.json',dict(paths=[str(x) for x in paths],bytes=total,reason='All registered local consumers complete; conditional generation gate did not pass'))
        for x in paths:x.unlink()
        lifecycle=dict(status='COMPLETE',packages_deleted=8,bytes_deleted=total,historical_artifacts_deleted=0,
            reconstruction_requires_new_authorization=True)
    c.write(RUN/'public/LIFECYCLE.json',lifecycle)
    p=dict(status='COMPLETE',decision=decision,epoch=time.time());c.write(RUN/'private/REPORT_COMPLETE.json',p)


if __name__=='__main__':main()
