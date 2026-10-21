"""Post-run structural audit and paired summaries; no model calls or metric tuning."""
import csv
import json
import math
from pathlib import Path
from .diagnose import FIRST_D


def summarize(root,old_path):
    root=Path(root);new=json.loads((root/'RESULT.json').read_text())
    old=json.loads(Path(old_path).read_text());attempts=[];accepted=[];pairs=[];checks=0
    assert new['status']=='COMPLETE' and old['passed']==8
    assert len(new['samples'])==len(old['samples'])==8 and new['all_base_restored']
    for sample,previous in zip(new['samples'],old['samples']):
        alias=sample['sample'];assert alias==previous['sample']
        assert sample['status']=='EVALUATED' and sample['base_restored']
        for field in ('native','fp32'):
            assert sample['baseline'][field]==previous['baseline'][field];checks+=1
        assert sample['baseline']['anchor']==previous['baseline']['anchor'];checks+=1
        assert sample['reference']==previous['reference'];checks+=1
        method=sample['method'];baseline=sample['baseline']['native'];current=baseline
        assert method['baseline']==baseline
        nfull=nhalf=ncorr=0;primary_count=half_count=primary_corr=half_corr=0
        for row in method['steps']:
            assert row['start_native_nll']==current
            primary=row['primary'];half=row.get('half');step=row['step']
            assert (half is not None)==(not primary['final_verdict']['accepted'])
            chosen='primary' if half is None else 'half'
            assert row['selected_attempt']==chosen
            if half is not None:assert primary['rollback_pass']
            for kind,trial in [('primary',primary)]+([('half',half)] if half is not None else []):
                proposal=trial['proposal'];verdict=trial['final_verdict'];corr=trial.get('correction')
                assert proposal['native_before']==current
                requested=FIRST_D/(16 if kind=='primary' else 32)
                assert proposal['requested_reduction']==requested
                assert 0<proposal['effective_reduction']<=requested
                assert proposal['clipped']==(proposal['effective_reduction']<requested)
                assert (corr is not None)==(not trial['proposal_verdict']['gates']['null'])
                if corr is not None:
                    ncorr+=1
                    assert corr['native_before']==proposal['native_after']
                null=verdict['native_nll'][1]-baseline[1]
                assert abs(null-verdict['cumulative_null_change'])<1e-12
                cost=verdict['combined_native_fisher_cost']
                expected=dict(null=abs(null)<=.01,visual=verdict['native_nll'][0]<current[0],
                    trust=math.isfinite(cost) and cost<=.001)
                if corr is not None and corr['status']!='APPLIED':expected['valid_correction']=False
                assert verdict['gates']==expected and verdict['accepted']==all(expected.values())
                assert (trial['status']=='ACCEPTED')==verdict['accepted']
                checks+=8
                primary_count+=kind=='primary';half_count+=kind=='half'
                primary_corr+=kind=='primary' and corr is not None
                half_corr+=kind=='half' and corr is not None
                attempts.append(dict(sample=alias,step=step,attempt=kind,status=trial['status'],
                    requested=requested,effective=proposal['effective_reduction'],clipped=proposal['clipped'],
                    correction=corr is not None,correction_status=corr['status'] if corr else 'NA',
                    start_visual=current[0],start_null=current[1],main_visual=proposal['native_after'][0],
                    main_null=proposal['native_after'][1],final_visual=verdict['native_nll'][0],
                    final_null=verdict['native_nll'][1],null_drift=null,null_margin=.01-abs(null),
                    old_fisher_cost=cost,visual_pass=expected['visual'],null_pass=expected['null'],
                    trust_pass=expected['trust'],accepted=verdict['accepted']))
            selected=row[chosen]
            if row['final_verdict']['accepted']:
                nfull+=chosen=='primary';nhalf+=chosen=='half'
                current=row['final_verdict']['native_nll']
                accepted.append(dict(sample=alias,step=step,selected=chosen,visual_nll=current[0],
                    null_nll=current[1],null_drift=current[1]-baseline[1],null_margin=.01-abs(current[1]-baseline[1]),
                    requested=selected['proposal']['requested_reduction'],effective=selected['proposal']['effective_reduction'],
                    old_fisher_cost=row['final_verdict']['combined_native_fisher_cost']))
            else:
                assert chosen=='half' and half['rollback_pass'] and row['rollback_pass']
                assert row is method['steps'][-1]
        assert method['accepted_steps']==nfull+nhalf
        assert method['accepted_primary_steps']==nfull and method['accepted_half_steps']==nhalf
        assert method['primary_proposals']==primary_count and method['half_proposals']==half_count
        assert method['primary_corrections']==primary_corr and method['half_corrections']==half_corr
        assert method['correction_attempts']==ncorr and method['final_accepted_native_nll']==current
        own=[r for r in accepted if r['sample']==alias]
        assert len(method['accepted_trajectory'])==len(own)
        for actual,expected in zip(method['accepted_trajectory'],own):
            assert actual['native_nll']==[expected['visual_nll'],expected['null_nll']]
            assert actual['requested_reduction']==expected['requested'] and actual['effective_reduction']==expected['effective']
            checks+=3
        equivalent=nfull+nhalf/2
        assert method['equivalent_d16_steps']==equivalent
        assert abs(method['accepted_effective_reduction']-sum(r['effective'] for r in own))<1e-12
        assert abs(method['accepted_requested_reduction']-sum(r['requested'] for r in own))<1e-12
        joint=len(own)==16 and current[0]<=sample['reference']['native_after'][0] and current[0]<baseline[0] and all(abs(r['null_drift'])<=.01 for r in own)
        assert method['joint_gate']==sample['joint_pass']==joint
        assert method['solver_baseline']['status']=='PASS'
        margin=min((r['null_margin'] for r in own),default=None)
        om=previous['method'];old_margin=previous['metrics']['boundary_margin']
        pairs.append(dict(sample=alias,old_pass=previous['joint_pass'],new_pass=joint,
            old_steps=om['accepted_steps'],new_steps=len(own),old_full=om['accepted_primary_steps'],new_full=nfull,
            old_half=om['accepted_half_steps'],new_half=nhalf,old_equivalent=om['equivalent_d16_steps'],new_equivalent=equivalent,
            old_final_visual=om['final_accepted_native_nll'][0],new_final_visual=current[0],
            visual_new_minus_old=current[0]-om['final_accepted_native_nll'][0],
            old_margin=old_margin,new_margin=margin,old_corrections=om['correction_attempts'],new_corrections=ncorr,
            failed_gates=','.join(sample['metrics']['failures']) or 'NA'))
    total=dict(status='COMPLETE',denominator=8,passed=sum(p['new_pass'] for p in pairs),
        primary_proposals=sum(a['attempt']=='primary' for a in attempts),half_proposals=sum(a['attempt']=='half' for a in attempts),
        accepted_full=sum(a['selected']=='primary' for a in accepted),accepted_half=sum(a['selected']=='half' for a in accepted),
        accepted_steps=len(accepted),corrections=sum(a['correction'] for a in attempts),
        primary_corrections=sum(a['correction'] and a['attempt']=='primary' for a in attempts),
        half_corrections=sum(a['correction'] and a['attempt']=='half' for a in attempts),
        rejected_primary=sum(a['attempt']=='primary' and not a['accepted'] for a in attempts),
        rejected_half=sum(a['attempt']=='half' and not a['accepted'] for a in attempts),
        rejected_primary_visual=sum(a['attempt']=='primary' and not a['visual_pass'] for a in attempts),
        rejected_primary_null=sum(a['attempt']=='primary' and not a['null_pass'] for a in attempts),
        rejected_primary_both=sum(a['attempt']=='primary' and not a['visual_pass'] and not a['null_pass'] for a in attempts),
        equivalent_d16=sum(p['new_equivalent'] for p in pairs),old_equivalent_d16=sum(p['old_equivalent'] for p in pairs),
        accepted_requested=sum(a['requested'] for a in accepted),accepted_effective=sum(a['effective'] for a in accepted),
        all_candidates_unclipped=not any(a['clipped'] for a in attempts),
        minimum_null_margin=min(a['null_margin'] for a in accepted),
        near_boundary_samples=sum(p['new_margin']<.001 for p in pairs),
        max_old_fisher_cost=max(a['old_fisher_cost'] for a in attempts),
        final_visual_improved_samples=sum(p['visual_new_minus_old']<0 for p in pairs),
        final_visual_mean_new_minus_old=sum(p['visual_new_minus_old'] for p in pairs)/8,
        model_forward_calls=new['model_forward_calls'],elapsed_seconds=new['elapsed_seconds'],
        peak_allocated_bytes=new['peak_allocated_bytes'],all_base_restored=new['all_base_restored'],
        generation_calls=new['generation_calls'],judge_calls=new['judge_calls'],configurations=1)
    assert total['passed']==new['passed']
    (root/'SUMMARY.json').write_text(json.dumps(total,indent=2)+'\n')
    audit=dict(status='PASS',posthoc=True,structural_scalar_checks=checks,
        historical_runtime_checks=sum(len(s['historical_checks']) for s in new['samples']),
        old_baselines_and_references_exact=True,actual_native_gates_recomputed=True,
        old_trust_scalar_gate_recomputed=True,weight_quadratic_not_independently_recomputed_without_private_weights=True,
        accepted_state_continuity=True,same_start_fallback=True,two_scales_only=True,
        original_fisher_for_clipping_and_gates='verified in source and CPU tests',no_extra_model_calls=True)
    (root/'PROTOCOL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    for name,rows in [('PAIRED_COMPARISON.tsv',pairs),('FULL_ATTEMPTS.tsv',attempts),('ACCEPTED_TRAJECTORY.tsv',accepted)]:
        with (root/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    return total,pairs,audit
