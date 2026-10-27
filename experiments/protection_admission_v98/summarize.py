"""Aggregate complete, manually reviewed paired-polarity outputs."""
import collections
import json
from pathlib import Path
import sys

ROLES = ['target','brief_control','negative_direct','negative_reverse','positive_direct','positive_reverse']


def main(private, destination):
    private, destination = Path(private), Path(destination)
    source = {r['qid']:r for r in json.loads((private/'SOURCE_ROWS.json').read_text())}
    outputs = {x['query']['qid']:x for x in [json.loads(p.read_text()) for p in (private/'qualification/outputs').glob('*.json')]}
    reviewed = json.loads((private/'FULL_TEXT_REVIEW.json').read_text())
    review = {r['qid']:r for r in reviewed}
    assert len(source)==len(outputs)==len(review)==90 and source.keys()==outputs.keys()==review.keys()
    assert all(x['EOS'] and not x['at_cap'] for x in outputs.values())
    groups = collections.defaultdict(dict)
    binding_images = collections.defaultdict(set)
    for qid,row in source.items():
        x, verdict = outputs[qid],review[qid]
        assert all(x['query'][k]==row[k] for k in ['question','reference','role','source_group'])
        assert x['binding']['question']==row['question']
        assert Path(x['binding']['image_path']).name==Path(row['image_path']).name
        assert verdict['answer']==x['raw_answer'] and verdict['role']==row['role']
        assert verdict['core'] in ['CORRECT','WRONG','UNKNOWN']
        image_role = 'positive' if row['role'].startswith('positive') else 'negative'
        binding_images[(row['source_group'],image_role)].add(x['binding']['image_sha256'])
        groups[row['source_group']][row['role']] = dict(row,verdict=verdict)
    assert len(binding_images)==30 and all(len(v)==1 for v in binding_images.values())
    assert len({next(iter(v)) for v in binding_images.values()})==30

    def okay(row):
        return row['verdict']['core']=='CORRECT' and not row['verdict']['unsupported_medical_addition']

    persistent, operational, strong = [], [], []
    negative_stable, positive_stable = [], []
    for group,r in groups.items():
        assert set(r)==set(ROLES)
        target_wrong = all(r[role]['verdict']['core']=='WRONG' for role in ['target','brief_control'])
        negative_ok = all(okay(r[role]) for role in ['negative_direct','negative_reverse'])
        positive_ok = all(okay(r[role]) for role in ['positive_direct','positive_reverse'])
        if target_wrong:persistent.append(group)
        if negative_ok:negative_stable.append(group)
        if positive_ok:positive_stable.append(group)
        if target_wrong and negative_ok:operational.append(group)
        if target_wrong and negative_ok and positive_ok:strong.append(group)
    coverage = dict(target_answers=sorted({groups[g]['target']['reference'] for g in strong}),
        target_findings=sorted({groups[g]['target']['target_label'] for g in strong}),
        protection_findings=sorted({groups[g]['target']['protection_label'] for g in strong}))
    admitted = len(strong)>=8 and all(len(v)>=2 for v in coverage.values())
    exit_receipt = json.loads((private/'qualification/EXIT_worker.json').read_text())
    start = json.loads((private/'qualification/START_worker.json').read_text())
    assert exit_receipt['exit_code']==0 and exit_receipt['counts']['updates']==exit_receipt['counts']['backwards']==0
    result = dict(status='TASK_ADMITTED_FOR_SEPARATE_PROTOCOL' if admitted else 'TASK_NOT_ADMITTED',
        source_candidates=18,source_excluded=3,admitted_source_pairs=15,patients=30,questions=90,
        target_side_counts={'left':6,'right':7,'bilateral':2},
        core_correct={role:sum(r['verdict']['core']=='CORRECT' for g in groups.values() for k,r in g.items() if k==role) for role in ROLES},
        unknown={role:sum(r['verdict']['core']=='UNKNOWN' for g in groups.values() for k,r in g.items() if k==role) for role in ROLES},
        fully_supported={role:sum(okay(r) for g in groups.values() for k,r in g.items() if k==role) for role in ROLES},
        persistent_target_errors=len(persistent),stable_negative_protection=len(negative_stable),
        stable_positive_controls=len(positive_stable),operational_pairs=len(operational),strong_pairs=len(strong),
        exact_answer_equal_across_polarity={form:sum(g['negative_'+form]['verdict']['answer']==g['positive_'+form]['verdict']['answer'] for g in groups.values()) for form in ['direct','reverse']},
        operational_coverage=dict(target_answers=sorted({groups[g]['target']['reference'] for g in operational}),
            target_findings=sorted({groups[g]['target']['target_label'] for g in operational}),
            protection_findings=sorted({groups[g]['target']['protection_label'] for g in operational})),
        strong_coverage=coverage,task_admitted=admitted,counts=exit_receipt['counts'],
        gpu_process_seconds=exit_receipt['time']-start['started_epoch'],
        all_EOS=True,all_bindings_match=True,replacement=False,method_training=False,
        review='All complete answers reviewed by assistant against source reports; no new independent clinical adjudication')
    (private/'QUALIFIED_GROUPS_PRIVATE.json').write_text(json.dumps(dict(persistent=persistent,
        stable_negative=negative_stable,stable_positive=positive_stable,operational=operational,strong=strong),indent=2))
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main(*sys.argv[1:])
