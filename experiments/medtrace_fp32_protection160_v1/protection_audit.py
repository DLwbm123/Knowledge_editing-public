"""Audit completed frozen consumers and receipts, without model calls or rescoring."""
import os
import json
import sqlite3
from pathlib import Path
from collections import Counter
import common as c

RUN = Path(os.environ['RUN_ROOT'])


def main():
    result = c.read(RUN/'public/RESULTS.json')
    lock = c.read(RUN/'private/PROTECTION_LOCK.json')
    assert c.read(RUN/'private/PROTECTION_REPORT_COMPLETE.json')['status'] == 'COMPLETE'
    raw = c.read(RUN/'private/RAW_REFERENCE_READY.json')
    frozen = c.read(RUN/'private/CANDIDATES_FROZEN.json')
    assert raw['source_exact'] == 40 and frozen['candidates'] == 8 and frozen['source_numerical_gate']
    starts = [c.read(p) for p in (RUN/'private').glob('START_CHAIN_*.json')]
    for s in starts:
        stat = Path('/proc')/str(s['pid'])/'stat'
        assert not stat.exists() or stat.read_text().split()[21] != s['start_ticks'] or stat.read_text().split()[2] == 'Z'
    assert not list((RUN/'private').glob('FAILURE*'))
    source = [c.read(RUN/'private/source_results'/f'{i}.json') for i in range(1, 9)]
    for s in source:
        assert s['updates'] == 160 and s['backwards'] == 8928 and len(s['curve']) == 160
        assert s['all_four_cores_updated'] and s['mechanical']['status'] == s['terminal_prefix_check']['status'] == 'PASS'
        assert s['endpoint']['gain_ratio'] >= .9
        assert [v['step'] for v in s['refreshes']] == [1, 41, 81, 121]
        assert all(v['rows'] == 1952 and v['questions'] == 61 and v['tokens'] == 1514 for v in s['refreshes'])
        for step in s['curve']:
            assert step['coordinate_step_norm'] <= step['RAW_coordinate_step_norm'] + step['roundoff_bound'] + 1e-10
            if step['mode'] == 'PROTECTED_OR_RESTORED':
                assert step['accepted_actual_step_gain'] >= .9 * step['RAW_actual_step_gain']
                assert step['solver']['KKT_residual'] <= 1e-7
    folder = RUN/'private/judge_protection_astra_medium'
    db = sqlite3.connect('file:'+str(folder/'queue.sqlite')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    consumers = list(db.execute('SELECT c.*,p.status,p.correct,p.binding FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(consumers) == 923 and all(r['status'] == 'FORMAT_VALID' for r in consumers)
    hb = c.read(RUN/'private/HELD_BASE.json')
    assert len(hb) == 96 and sum(v['correct'] for v in hb.values()) == 62
    base_scores = {v['path']: v['correct'] for v in hb.values()}
    parent = Path(c.read(RUN/'private/LAUNCH_ENV.json')['DAMAGE_PARENT'])
    old_scores = {v['path']: v['correct'] for v in c.read(parent/'private/DAMAGE_SEMANTIC.json')}
    panels = {i: Counter() for i in range(1, 9)}
    paired = Counter()
    unique_damage = set()
    outputs = []
    role_scores = Counter()
    t2 = Counter()
    primary = 0
    for r in consumers:
        d = c.read(r['path']); b = json.loads(r['binding']); i = d['binding']['input']
        assert d['lock'] == c.digest(lock)
        assert all(i[k] == b[k] for k in ('query_id', 'image_sha256', 'question', 'reference')) and d['R0'] == b['output']
        assert all(d['binding']['judge_input'][k] == b[k] for k in ('prompt_ids', 'attention_mask', 'runtime', 'generation'))
        assert b['judge']['model'] == 'gpt-6-astra' and b['judge']['reasoning_effort'] == 'medium'
        assert d['EOS'] and not d['at_cap'] and any(v['active_residual_norm'] > 0 for v in d['generation_trace'])
        outputs.append(d); role_scores[d['role']] += r['correct']
        owner = d['expert_order']
        if d['role'] == 'T2G': t2[owner] += r['correct']
        if d['role'] != 'HELDOUT': continue
        before, after = base_scores[d['baseline_path']], r['correct']
        n = panels[owner]; n['queries'] += 1; n['Base_correct'] += before; n['candidate_correct'] += after
        n[('retained' if after else 'new_damage') if before else ('repaired' if after else 'still_wrong')] += 1
        if before and not after: unique_damage.add(i['query_id'])
        if d['original_primary']: primary += after
        oldpath = parent/'private/audit_outputs/RAW'/str(owner)/(i['query_id']+'.json')
        old = old_scores[str(oldpath)]
        paired['both_correct' if old == after == 1 else 'both_wrong' if old == after == 0 else 'RAW_only_correct' if old else 'PROTECTED_only_correct'] += 1
    for owner, panel in panels.items():
        assert panel['queries'] == 96 and panel['Base_correct'] == 62
        reported = result['per_owner'][owner-1]['panels']['HELDOUT']
        assert all(panel[k] == reported[k] for k in panel)
    assert primary == result['panels']['ORIGINAL_PRIMARY']['candidate_correct']
    assert all(role_scores[k] == v['candidate_correct'] for k, v in result['panels'].items() if k in role_scores)
    assert result['source_gate'] == (role_scores['NATIVE'] == 8 and role_scores['FIT'] == 32)
    assert result['generalization_gate'] == (role_scores['T1G'] == 32 and all(t2[i] >= lock['baseline_T2G_by_owner'][i-1] for i in range(1, 9)))
    ledger = c.read(RUN/'RESOURCE_LEDGER.json'); before = c.read(RUN/'private/INHERITED_COST.json')
    batches = ledger['Astra_protection_batches']; keys = [k for b in batches for k in b['keys']]
    assert len(keys) == len(set(keys)) == ledger['Judge_attempts']-before['Judge_attempts'] == result['resource']['new_Judge']
    assert all(b['status'] == 'FORMAT_VALID' and b['model'] == 'gpt-6-astra' and b['reasoning_effort'] == 'medium' and not b['transport_failure'] for b in batches)
    evidence = list((folder/'evidence').glob('*.json')); assert len(evidence) == len(batches)
    for path in evidence:
        e = c.read(path)['evidence']
        assert e['status'] == 'FORMAT_VALID' and e['actual_model'] == 'gpt-6-astra' and e['reasoning_effort'] == 'medium'
        assert not e['tool_event_types'] and not e.get('errors') and all(e['isolation_checks'].values())
    assert len(ledger['gpu_sessions']) == 18 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    assert min(s['started_epoch'] for s in ledger['gpu_sessions'] if s['action'] == 'protection_eval') > frozen['epoch']
    counts = [c.read(RUN/'private'/f'PROTECTION_EVAL_COUNTS_{i}.json') for i in range(6)]
    src = raw['counts']+frozen['counts']
    assert sum(v['updates'] for v in src) == 2560 and sum(v['backwards'] for v in src) == 73984
    assert sum(v['generations'] for v in src+counts) == 963 and sum(v['prefix_forwards'] for v in counts) == 1344
    raw_outputs = [c.read(p) for p in (RUN/'private/outputs/RAW').glob('*/*.json')]
    assert len(raw_outputs) == 40 and all(d['EOS'] and not d['at_cap'] for d in raw_outputs)
    forwards = sum(v['forwards'] for v in src+counts)
    non_generation = 3046+sum(v['non_generation_forwards'] for v in source)+1344
    assert forwards == non_generation+sum(v['forwards'] for v in outputs+raw_outputs) == result['generation']['forwards']
    deleted = c.read(RUN/'private/DELETION_PLAN.json')
    active = [c.read(p) for p in (RUN/'private/retired_active').glob('*.json')]
    assert len(deleted['paths']) == 16 and len(active) == 8
    retired = deleted['paths']+[p for r in active for p in r['paths']]
    assert len(retired) == 32 and all(not Path(p).exists() for p in retired)
    assert not list((RUN/'private/active').glob('*.pt'))
    receipt = dict(status='PASS',source_reproduction_outputs=40,all_candidates_frozen_before_eval=True,
        updates=2560,backwards=73984,forwards=forwards,non_generation_forwards=non_generation,new_generations=963,
        new_generation_EOS=963,at_cap=0,scored_consumers=923,payload_status=result['payload_status'],new_Judge=len(keys),
        inherited_payloads=sum(result['payload_status'].values())-len(keys),Judge_batches=len(batches),all_actual_Astra_medium_isolated=True,
        no_payload_retry=True,missing=0,GPU_sessions_ended=18,recorded_remote_processes_ended=len(starts),
        candidate_packages_deleted=16,active_files_deleted=16,historical_deletions=0,per_owner_held=[dict(owner=i,**n) for i,n in panels.items()],
        distinct_held_queries_damaged=len(unique_damage),paired_held=dict(paired),automatic_next_experiment=False,
        scientific_decision=result['decision'],source_numerical_constraints_pass=True)
    c.write(RUN/'public/REVIEW_AUDIT.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
