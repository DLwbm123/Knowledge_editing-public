"""Preregistered phase-A gate; training BASIS diagnostics are not heldout success."""
import json
import os
import time
import guard_probe as run
import probe_report as stats
c,RUN=run.c,run.RUN


def eligible(gates):
    return all(gates.values())


def main():
    assert not (RUN/'private/PHASE_A_REPORT_COMPLETE.json').exists()
    assert len(list((RUN/'private/PREFIX_COMPLETE').glob('*.json')))==8
    diagnostics=[c.read(p) for p in sorted((RUN/'private/diagnostics').glob('*.json'))]
    assert len(diagnostics)==16 and {(x['owner'],x['node']) for x in diagnostics}=={(i,n) for i in range(1,9) for n in run.NODES}
    basis,_=run.r.q.split();groups=sorted({v['source_group'] for v in basis})
    cells=[];owner_diff=[];source_differences={group:[] for group in groups}
    for owner in range(1,9):
        differences=[]
        for node in run.NODES:
            m=c.read(RUN/'private/nodes'/f'M1REF_{owner}_{node}.json')['metrics']
            g=c.read(RUN/'private/nodes'/f'GUARD_{owner}_{node}.json')['metrics']
            delta=g['basis_mean_KL']-m['basis_mean_KL'];differences.append(delta)
            for row,a,b in zip(basis,m['basis_values'],g['basis_values']):source_differences[row['source_group']].append(b['KL']-a['KL'])
            cells.append(dict(owner=owner,node=node,M1_source_gain=m['source_gain'],GUARD_source_gain=g['source_gain'],
                              M1_BASIS_KL=m['basis_mean_KL'],GUARD_BASIS_KL=g['basis_mean_KL'],
                              M1_lost_correct_tokens=m['basis_lost_correct_tokens'],GUARD_lost_correct_tokens=g['basis_lost_correct_tokens']))
        owner_diff.append(stats.mean(differences))
    curves={i:c.read(RUN/'private/prefix_curve'/f'{i}.json') for i in range(1,9)}
    ratios=[row['decision']['norm_ratio'] for curve in curves.values() for row in curve]
    ci=dict(expert=stats.interval(owner_diff),source=stats.interval([stats.mean(v) for v in source_differences.values()]))
    gates=dict(positive_M1_source_gain=all(x['M1_source_gain']>0 for x in cells),
               source_progress_90percent=all(x['GUARD_source_gain']>=.9*x['M1_source_gain']-1e-6 for x in cells),
               no_cell_token_loss_worse=all(x['GUARD_lost_correct_tokens']<=x['M1_lost_correct_tokens'] for x in cells),
               lower_BASIS_KL_both_intervals=ci['expert'][1]<0 and ci['source'][1]<0,
               no_cell_KL_worse=all(x['GUARD_BASIS_KL']<=x['M1_BASIS_KL']+1e-10 for x in cells),
               mean_step_norm_90percent=stats.mean(ratios)>=.9,
               each_owner_step_norm_90percent=all(stats.mean([row['decision']['norm_ratio'] for row in curve])>=.9 for curve in curves.values()))
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    totals={key:sum(x[key] for x in counts) for key in run.r.COUNTS}
    assert totals['updates']==1280 and totals['backwards']==5840 and totals['operator_backwards']==16
    assert totals['non_generation_forwards']<=24000 and totals['generation_attempts']==0
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==4 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    assert ledger['Judge_attempts']==inherited['Judge_attempts']
    # A passing diagnostic admits implementation/strict continuation, not a new method success claim.
    passed=eligible(gates)
    result=dict(status='PHASE_A_COMPLETE',decision='ADMITTED_CONDITIONAL_FULL_VALIDATION' if passed else 'STOP_MECHANISM_GATE_NOT_MET',
                complete=True,full_validation_complete=False,independent_confirmation=False,gates=gates,cells=cells,
                BASIS_KL_difference_intervals=ci,mean_operator_step_ratio=stats.mean(ratios),
                skipped_proposals=sum(row['decision']['status']=='SKIPPED_INFEASIBLE' for curve in curves.values() for row in curve),
                diagnostics=[{k:v for k,v in row.items() if k!='metrics'} for row in diagnostics],counts=totals,
                resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
                              cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=0,cumulative_Judge=ledger['Judge_attempts']),
                next_consumer='REGISTERED_FULL_81_160_IMPLEMENTATION_PENDING' if passed else 'NONE',
                boundary='Source/BASIS teacher forcing only; no generation, heldout measurement or Judge')
    c.write(RUN/'public/PHASE_A_RESULTS.json',result)
    if not passed:
        paths=[RUN/'private/OPERATOR_FISHER.pt']+list((RUN/'private/prefix/GUARD').glob('*.pt'))
        assert len(paths)==9 and all(p.is_file() and not p.is_symlink() for p in paths)
        c.write(RUN/'private/PHASE_A_CONSUMED.json',dict(files=9,bytes=sum(p.stat().st_size for p in paths),reason='Complete failed mechanism gate; no downstream consumers admitted',copies=False,reconstruction_requires_retraining=True))
        for p in paths:p.unlink()
    c.write(RUN/'private/PHASE_A_REPORT_COMPLETE.json',dict(decision=result['decision'],epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status=result['decision'],phase_A_complete=True,
                                         full_validation_complete=False,publication_complete=False))
    print(json.dumps(dict(decision=result['decision'],gates=gates,resource=result['resource'])))


if __name__=='__main__':main()
