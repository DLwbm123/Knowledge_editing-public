"""Summarize frozen, fully reviewed Base answers; never select replacements."""
import collections
import json
from pathlib import Path
import re
import sys


def main(private, destination):
    private, destination = Path(private), Path(destination)
    outputs = [json.loads(p.read_text()) for p in sorted((private/'qualification/outputs').glob('*.json'))]
    assert len(outputs) == 80
    rows = json.loads((private/'SOURCE_ROWS_ADMITTED.json').read_text())
    expected = {r['qid']: r for r in rows}
    assert {x['query']['qid'] for x in outputs} == set(expected)
    assert all(x['EOS'] and not x['at_cap'] for x in outputs)
    review, groups = [], collections.defaultdict(dict)
    image_bindings = collections.defaultdict(set)
    for x in outputs:
        row, answer = x['query'], x['raw_answer']
        original = expected[row['qid']]
        assert all(row[k] == original[k] for k in ['question', 'reference', 'role', 'source_group'])
        assert x['binding']['question'] == row['question']
        assert x['binding']['image_path'].endswith('/'+row['source_group']+'.png')
        # Reuse the runtime's recorded binding; do not recalculate file hashes.
        image_bindings[row['source_group']].add(x['binding']['image_sha256'])
        if row['role'] in ['target', 'brief_control']:
            lower = answer.lower()
            value = 'bilateral' if 'bilateral' in lower or 'both sides' in lower else 'left' if re.search(r'\bleft\b', lower) else 'right' if re.search(r'\bright\b', lower) else 'UNKNOWN'
        else:
            value = 'Yes' if answer.startswith('Yes,') else 'No' if answer.startswith('No,') else 'UNKNOWN'
        assert value != 'UNKNOWN', 'Full answer needs additional adjudication'
        core = value == row['reference']
        # These two qualifiers were reviewed against complete admitted reports.
        extra = row['source_group'] == 'p01' and row['role'] == 'protect_fit'
        extra |= row['source_group'] == 'p06' and row['role'] == 'target'
        entry = dict(qid=row['qid'], group=row['source_group'], role=row['role'],
            reference=row['reference'], answer=answer, core_fact_correct=core,
            unsupported_addition=extra,
            addition_note='Wrong right-sided qualifier; source specifies left' if row['source_group']=='p01' and extra else 'Lower lobe is more specific than source basal; not independently established' if extra else None,
            source_supported_qualifier=False,
            style_only_difference=core and not extra,
            full_source_supported=core and not extra)
        groups[row['source_group']][row['role']] = entry
        review.append(entry)
    assert len(image_bindings) == 20 and all(len(v)==1 for v in image_bindings.values())
    assert len({next(iter(v)) for v in image_bindings.values()}) == 20
    qualified = [g for g,r in groups.items() if not r['target']['core_fact_correct']
                 and not r['brief_control']['core_fact_correct']
                 and r['protect_fit']['core_fact_correct'] and r['protect_holdout']['core_fact_correct']]
    exit_receipt = json.loads((private/'qualification/EXIT_worker.json').read_text())
    assert exit_receipt['exit_code'] == 0
    assert exit_receipt['counts']['updates'] == exit_receipt['counts']['backwards'] == 0
    start = json.loads((private/'qualification/START_worker.json').read_text())
    result = dict(status='CLOSED_NO_QUALIFIED_CORRECTION_PROTECTION_PAIRS', original_patients=24,
        source_excluded=4, source_admitted=20, source_admitted_laterality={'left':7,'right':5,'bilateral':8},
        questions=80, all_EOS=True, all_bindings_match=True,
        core_correct={role:sum(r['role']==role and r['core_fact_correct'] for r in review) for role in ['target','brief_control','protect_fit','protect_holdout']},
        full_source_supported={role:sum(r['role']==role and r['full_source_supported'] for r in review) for role in ['target','brief_control','protect_fit','protect_holdout']},
        target_errors_fixed_by_brief=sum(not r['target']['core_fact_correct'] and r['brief_control']['core_fact_correct'] for r in groups.values()),
        target_correct_broken_by_brief=sum(r['target']['core_fact_correct'] and not r['brief_control']['core_fact_correct'] for r in groups.values()),
        persistent_target_errors=sum(not r['target']['core_fact_correct'] and not r['brief_control']['core_fact_correct'] for r in groups.values()),
        both_protections_correct=sum(r['protect_fit']['core_fact_correct'] and r['protect_holdout']['core_fact_correct'] for r in groups.values()),
        qualified_pairs=len(qualified), counts=exit_receipt['counts'],
        gpu_process_seconds=exit_receipt['time']-start['started_epoch'],
        scoring='Assistant full-text review of all 80 outputs; no independent clinical adjudication; deterministic summary parser validated against those readings',
        no_method_training=True, no_replacement=True, no_test_answer_use=True)
    (private/'FULL_TEXT_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
