"""CPU-only initial gate; no training, judging, or historical mutation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT / path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()

def identity(row):
    # Prompt/context versions are explicit; answers and display IDs are excluded.
    return digest([row['image_sha256'], row['question'],
                   'unchanged-frozen-llava-question-template', row.get('context', '')])

def old_id(row):
    return digest([row['image_sha256'], row['question']])

def mask_u(prefix, h_arrival, u_arrival, semantic_conflict):
    return prefix >= max(h_arrival, u_arrival) and semantic_conflict

def self_check():
    for h, u in [(8, 40), (6, 42), (12, 45)]:
        assert not mask_u(u-1, h, u, True)
        assert mask_u(u, h, u, True)
        assert not mask_u(u, h, u, False)
    assert not mask_u(10, 12, 5, True)  # No future reference masking.
    assert mask_u(12, 12, 5, True)
    row = dict(image_sha256='image', question='q')
    assert identity(row) == identity(dict(row, reference='changed', source_group='other'))
    assert identity(row) != identity(dict(row, context='different'))

def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def main():
    self_check()
    stream = read('reports/medtrace_stage22_20260919/private/run/private/STREAM.json')
    panel = read('reports/medtrace_stage24c_20260919/private/run/private/DEV_UNION.json')['rows']
    memberships = read('reports/medtrace_stage24c_20260919/private/run/private/PANEL_MEMBERSHIPS.json')
    arrivals = {}
    lineage = set()
    for task in stream['tasks'][:19]:
        t = task['order']
        rows = [('native', task['native'])]
        rows += [('fit', dict(task['native'], question=q)) for q in task['fit_questions']]
        for role in ['H_fit', 'U_fit']:
            rows += [(role, r) for r in task[role]]
        for role, row in rows:
            key = identity(row)
            item = arrivals.setdefault(key, dict(first_prefix=t, events=[], source=row['source_group']))
            item['events'].append(dict(prefix=t, role=role))
        # Known native paraphrase lineage, not a claim to detect unknown semantic duplicates.
        lineage.add((task['native']['image_sha256'], task['native']['question']))
        lineage.update((task['native']['image_sha256'], q) for q in task['fit_questions'])
    details = []
    for row in panel:
        key = identity(row)
        details.append(dict(query_id=old_id(row), input_fingerprint=key,
                            exact_training_overlap=key in arrivals,
                            known_fit_lineage_overlap=(row['image_sha256'], row['question']) in lineage,
                            source=row['source_group']))
    by_id = {r['query_id']: r for r in details}
    counts = {}
    for name, ids in memberships.items():
        unique = set(ids)
        retained = [by_id[q] for q in unique if not by_id[q]['exact_training_overlap']
                    and not by_id[q]['known_fit_lineage_overlap']]
        counts[name] = dict(original_QA=len(unique), no_exact_or_known_fit_overlap_QA=len(retained),
                            sources=len({r['source'] for r in retained}))
    write('private/INPUT_IDENTITY_AUDIT.json', dict(inputs=arrivals, evaluation=details))
    write('IDENTITY_AUDIT_SUMMARY.json', dict(panel_QA=len(panel), training_inputs=len(arrivals),
          panels=counts, status='PRELIMINARY_EXACT_AND_EXPLICIT_FIT_LINEAGE_ONLY',
          limitation='Not final truly-untrained freeze: broader recorded paraphrase provenance audit pending budget gate. No output-based filtering.'))
    nonnative = set().union(*(set(v) for k, v in memberships.items() if k != 'native')) - set(memberships['native'])
    native_checks = 3 * sum(range(1, 20))
    endpoint_checks = 3 * 2 * len(nonnative)
    ledger = read('reports/medtrace_stage22_20260919/private/run/public/BUDGET_LEDGER.json')
    reservation = native_checks + endpoint_checks
    write('BUDGET_PREFLIGHT.json', dict(status='BLOCKED_CONSERVATIVE_JUDGE_RESERVATION',
          stage_Judge_cap=800, shared_Judge_cap=ledger['limit'], shared_Judge_spent=ledger['new_judgment_items_dispatched'],
          shared_Judge_remaining=ledger['limit']-ledger['new_judgment_items_dispatched'],
          arms=3, N=19, native_after_insert_consumers=native_checks,
          nonnative_unique_QA=len(nonnative), fixed_prefixes=[11,19], endpoint_nonnative_consumers=endpoint_checks,
          subtotal_consumers=reservation, no_cache_reservation_shortfall=reservation-800,
          maintenance_native_checks='Additional; not yet scheduled/countable',
          support_scoring_and_acceptance='Additional; excluded from this subtotal',
          caveat='930 is an output-consumer/no-cache reservation subtotal, NOT a lower bound on actual new semantic judgments. Exact reuse can reduce cost only when established; no new-output hit rate is guaranteed.',
          GPU_new_seconds=0, Judge_new_items=0, formal_training_started=False,
          next_gate='A defensible bound on exact reuse under immutable expert/router bindings, or explicit phase allocation revision within the existing shared total; do not silently relax checks or increase caps.'))
    write('TARGET_POLICY.json', dict(name='VERIFIED_REFERENCE_OVER_BASE', status='POLICY_ONLY_NOT_TRAINER_ACCEPTANCE',
          rules=['Only already-arrived verified references take priority.',
                 'Preserve immutable Base teachers; mask conflicting U in every arm only after reference availability.',
                 'Semantic agreement preserves U despite textual difference.',
                 'No verified reference: preserve Base.',
                 'Conflicting legitimate references without version rule stop affected candidate.'],
          CPU_temporal_tests='PASS', cases=[dict(H_first=h,U_first=u,masked_at=u) for h,u in [(8,40),(6,42),(12,45)]],
          limitation='Historical temporal cases only; no N45 training, new semantic judgments, or implicit validation of U-only reference provenance.'))
    print(json.dumps(dict(status='BUDGET_GATE_BLOCKED', subtotal=reservation, panel_counts=counts), ensure_ascii=False))

if __name__ == '__main__':
    main()
