"""Read existing V62 outputs and reviewed annotations; never run a model or rescore."""
import argparse
import collections
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def analyze(root, review, original, out):
    tasks = read(root / 'DATA_ADMISSION.json')['tasks']
    labels = read(root / 'BLIND_VERDICTS.json')
    mapping = {(x['arm'], x['owner'], x['query']): x['blind_id']
               for x in read(root / 'BLIND_MAPPING.json')}
    details, anonymous = [], []
    descriptions = {
        (3, 'ABSTAIN'): 'Both decline image identification; neither gives the target finding.',
        (5, 'UNCERTAIN'): 'TT identifies finding but omits side; LoRA declines both side and finding.',
        (7, 'UNCERTAIN'): 'TT eventually affirms contrast but omits both routes; LoRA supplies both.',
        (7, 'INCORRECT'): 'TT reverses contrast polarity; LoRA has correct polarity and both routes.',
        (8, 'UNCERTAIN', True): 'TT gives full list then malformed numeric continuation reaches cap; LoRA terminates normally.',
        (8, 'UNCERTAIN', False): 'TT gives one of three unranked causes; LoRA gives full list. Singular main-cause wording is ambiguous.',
        (10, 'INCORRECT'): 'TT substitutes a different symptom list; LoRA has empty decoded answer with EOS.',
    }
    for task in tasks:
        owner = task['order']
        for row in task['eval_rows']:
            if 'T2G' not in row['roles']:
                continue
            lab = [labels[mapping[(arm, owner, row['query_id'])]]['label']
                   for arm in ['TT_FIXED160', 'NORMALIZED_LORA_R1_FIXED160']]
            if lab == ['CORRECT', 'CORRECT']:
                continue
            outputs = [read(root / str(i) / task['key'] / 'FIXED160' / 'outputs' /
                            (row['query_id'] + '.json')) for i in range(2)]
            key = (owner, lab[0], outputs[0]['at_cap']) if owner == 8 else (owner, lab[0])
            entry = dict(case=f'D{len(details)+1:02}', owner=owner, TT_label=lab[0],
                         LoRA_label=lab[1], TT_EOS=outputs[0]['EOS'], LoRA_EOS=outputs[1]['EOS'],
                         TT_at_cap=outputs[0]['at_cap'], LoRA_at_cap=outputs[1]['at_cap'],
                         TT_empty=not outputs[0]['raw_answer'].strip(),
                         LoRA_empty=not outputs[1]['raw_answer'].strip(), explanation=descriptions[key])
            anonymous.append(entry)
            details.append(dict(**entry, query=row['query_id'], question=row['question'],
                                reference=row['reference'], answers=[x['raw_answer'] for x in outputs]))
    assert len(details) == 7
    assert sum(x['LoRA_label'] == 'CORRECT' for x in details) == 4
    write(review / 'SEVEN_CASES_PRIVATE.json', details)
    write(out / 'SEVEN_CASES.json', anonymous)

    scope = read(review / 'SCOPE_CLASSIFICATION.json')
    reviewed = read(review / 'LOCALITY_REVIEW_VERDICTS.json')
    expected = {(t['order'], q['query_id']) for t in tasks for q in t['eval_rows']
                if set(q['roles']) & {'T1L', 'T2L'}}
    assert {(r['owner'], r['query']) for r in scope} == expected
    covered = [(x['owner'], q) for x in reviewed for q in x['queries']]
    assert len(covered) == len(set(covered)) == len(expected) == 89
    categories = dict(collections.Counter(x['category'] for x in scope))
    assert categories == {'APPLICABLE': 53, 'UNDETERMINED': 36}
    arms = {}
    for arm in ['BASE', 'TT160', 'LORA160']:
        unsupported = {(x['owner'], q) for x in reviewed if x['arms'][arm]['unsupported_source_assertion'] for q in x['queries']}
        baseline = {(x['owner'], q) for x in reviewed if x['arms']['BASE']['unsupported_source_assertion'] for q in x['queries']}
        arms[arm] = dict(confirmed_mispropagation_eligible=0, confirmed_mispropagation_count=None,
                         confirmed_mispropagation_rate=None, newly_confirmed_vs_Base=None,
                         unsupported_source_assertions=len(unsupported),
                         new_unsupported_vs_Base=len(unsupported-baseline),
                         unsupported_unique_inputs=len({q for _, q in unsupported}),
                         unsupported_edit_units=len({o for o, _ in unsupported}),
                         disjoint_reference_target_substitutions=sum(x['count'] for x in reviewed
                             if x['arms'][arm]['disjoint_target_source_substitution']))
    scope_result = dict(status='NO_EXPLICIT_EXCLUSION_DENOMINATOR', consumers=89,
                        unique_inputs=len({x['query'] for x in scope}), edit_units=4,
                        category_counts=categories, explicitly_inapplicable=0,
                        undetermined_subgroups=dict(collections.Counter(x['subcategory'] for x in scope if x['category']=='UNDETERMINED')),
                        disjoint_target_diagnostic_consumers=2, disjoint_target_unique_inputs=1,
                        arms=arms, note='Unsupported additions are not clinically verified false facts; original C/I/U/A unchanged.')
    write(out / 'SCOPE_RESULTS.json', scope_result)
    original_tasks = read(original / 'ORIGINAL146_TASKS.json')
    plan = read(original / 'EVAL_PLAN146.json')
    assert len(original_tasks) == 146
    assert {x['owner'] for x in plan if x['method']=='C' and x['mode']=='single'} == set(range(1,147))
    first_images = {t['native']['image_sha256'] for t in original_tasks[:10]}
    first_sources = {t['native']['source_group'] for t in original_tasks[:10]}
    roster = [dict(edit=t['order'], source_group=t['native']['source_group'],
                   existing_image_identifier=t['native']['image_sha256'],
                   historical_development_exposure=True, V62_exposed=t['order']<=10,
                   same_image_as_V62=t['native']['image_sha256'] in first_images,
                   same_source_group_as_V62=t['native']['source_group'] in first_sources,
                   eligible_independent_confirmation=False) for t in original_tasks]
    write(review / 'ORIGINAL146_EXPOSURE_PRIVATE.json', roster)
    public_roster = [{k:v for k,v in x.items() if k not in ['source_group','existing_image_identifier']} for x in roster]
    write(out / 'ORIGINAL146_EXPOSURE.json', public_roster)
    exposure = dict(original_edits=146, original_source_groups=len({x['source_group'] for x in roster}),
                    original_images=len({x['existing_image_identifier'] for x in roster}),
                    V62_edits=10, remaining_edits=136,
                    remaining_unique_images=len({x['existing_image_identifier'] for x in roster[10:]}),
                    remaining_same_image_as_V62_edits=sum(x['same_image_as_V62'] for x in roster[10:]),
                    remaining_same_source_as_V62_edits=sum(x['same_source_group_as_V62'] for x in roster[10:]),
                    image_groups_shared_between_partitions=len(first_images & {x['existing_image_identifier'] for x in roster[10:]}),
                    verified_independent_confirmation_edits=0,
                    reason='All146 previously used for development; new recipe does not erase old exposure.',
                    evidence=['medtrace_post32_closeout_v1_20261007/LORA146_REVIEW_ZH.md',
                              'medtrace_curriculum146_v1_20261017/CURRICULUM_REPORT_ZH.md'])
    write(out / 'EXPOSURE_SUMMARY.json', exposure)
    print(json.dumps(dict(scope=scope_result, exposure=exposure), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'review', 'original', 'out']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    analyze(a.root, a.review, a.original, a.out)
