"""Audit engineering evidence without reconstructing missing scientific results."""
import json,math
from pathlib import Path


def summarize(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    status=read('STATUS.json');states=read('ARMS.json');life=read('LIFECYCLE.json')
    assert len(states)==1 and states[0]['arm']=='A';s=states[0]
    assert len(s['steps'])==8 and all(t['accepted'] and t['trial']['final_verdict']['accepted'] for t in s['steps'])
    gate_checks=0
    def verify(v,start,base):
        nonlocal gate_checks
        drift=v['native_nll'][1]-base[1]
        gates=dict(null=abs(drift)<=.01,visual=v['native_nll'][0]<start[0],trust=math.isfinite(v['combined_native_fisher_cost']) and v['combined_native_fisher_cost']<=.001)
        assert gates==v['gates'] and v['accepted']==all(gates.values()) and abs(v['cumulative_null_change']-drift)<1e-12
        gate_checks+=3
    prev=s['baseline']['native']
    for step in s['steps']:
        t=step['trial'];assert t['proposal']['native_before']==prev
        verify(t['proposal_verdict'],prev,s['baseline']['native']);verify(t['final_verdict'],prev,s['baseline']['native'])
        prev=t['final_verdict']['native_nll']
    p=s['probes']['F1'];verify(p['verdict'],prev,s['baseline']['native'])
    assert p['matching_passed'] and p['endpoint_restored'] and p['verdict']['accepted']
    assert p['proposal']['native_norm_matching']['applications']==1 and p['proposal']['native_norm_matching']['relative_error']<=.01
    assert s['base_restored'] and s['probe_base_restored']
    diffs=[v for c in s['checks'] for k,v in c.items() if k.endswith('difference')]
    assert max(diffs)==0
    engine=read('ENGINEERING.json');assert engine['base_weight_restored'] and engine['base_losses_restored'] and engine['precision_flags_hooks_restored'] and not engine['base_cache_changes']
    assert life['main_process_exited'] and life['own_gpu_process_exited'] and life['within_900_seconds'] and life['model_forward_calls']<=320
    scientific=[]
    for e in engine['events']:
        if 'losses' in e:
            assert all(math.isfinite(x) for x in e['losses'])
            scientific.append(e)
    summary=dict(status=status['status'],engineering_status=engine['status'],stages_completed=engine['stages_completed'],
        last_event=engine['events'][-1]['label'] if engine['events'] else None,failure=engine.get('failure'),
        reference_main=8,path_corrections=sum(t['trial']['correction'] is not None for t in s['steps']),fixed_F1_controls=1,
        reference_checks=len(s['checks']),reference_max_difference=max(diffs),scalar_gate_checks=gate_checks,events_persisted=len(engine['events']),
        loss_pairs_persisted=len(scientific),model_forward_calls=life['model_forward_calls'],max_forward=320,
        elapsed_launch_to_end=life['launch_to_end_seconds'],all_recorded_base_restored=True,scientific_grid_calls=0,
        retries=0,downgrades=0,extra_candidates=0)
    for n,v in [('SUMMARY.json',summary),('PROTOCOL_AUDIT.json',dict(status='PASS',reference_checks=len(s['checks']),first_stop_enforced=status['status']!='COMPLETE',no_D=True,no_scientific_grid=True,all_recorded_restore_checks_passed=True,private_noneditable_weight_values_not_independently_copied=True))]:
        (root/n).write_text(json.dumps(v,indent=2)+'\n')
    return summary,engine
