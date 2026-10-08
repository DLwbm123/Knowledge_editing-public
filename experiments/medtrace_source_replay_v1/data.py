"""Source-annotated image facts, split before model outputs by image source."""
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ('Organ', 'Position', 'Modality', 'Plane', 'Quantity')
KINDS = ('yes', 'no', 'open')


def select(rows, excluded):
    groups = defaultdict(lambda: defaultdict(list))
    answers = defaultdict(set)
    for row in rows:
        if row['q_lang'] == 'en':
            answers[row['img_name'], row['question']].add(row['answer'].strip().lower())
    for row in rows:
        group = 'SLAKE:' + row['img_name'].split('/')[0]
        if row['q_lang'] != 'en' or group in excluded or row['content_type'] not in TYPES:
            continue
        if len(answers[row['img_name'], row['question']]) != 1 or not row['answer'].strip():
            continue
        answer = row['answer'].strip().lower()
        groups[group][answer if answer in KINDS[:2] else 'open'].append(row)
    eligible = sorted(g for g, strata in groups.items() if set(strata) == set(KINDS))
    assert len(eligible) >= 96
    random.Random(20261008).shuffle(eligible)
    selected = []
    for role, names in [('FIT', eligible[:64]), ('CHECK', eligible[64:96])]:
        types = Counter()
        for index, group in enumerate(names):
            for kind in KINDS:
                row = min(groups[group][kind], key=lambda x: (types[x['content_type']], x['qid']))
                types[row['content_type']] += 1
                selected.append(dict(source=row, group=group, role=role, kind=kind, group_index=index))
    assert len(selected) == 288 and len({x['group'] for x in selected}) == 96
    assert Counter((x['role'], x['kind']) for x in selected) == {
        ('FIT', k): 64 for k in KINDS} | {('CHECK', k): 32 for k in KINDS}
    return selected, len(eligible)


def main():
    run = Path(os.environ['RUN_ROOT'])
    parent = Path(os.environ['SOURCE_PARENT'])
    base = Path(os.environ['BASE_ROOT'])
    read = lambda p: json.loads(p.read_text())
    excluded = {x['source_group'] for x in read(parent/'private/EVAL_LEDGER.json')['queries'].values()}
    excluded.update(x['source_group'] for rows in read(base/'private/U_ROLES.json').values() for x in rows)
    source = Path(os.environ['DATA_ROOT'])/'knowledge_editing/data/m3bench/SLAKE/train.json'
    selected, eligible = select(read(source), excluded)
    roles = {'FIT': [], 'CHECK': []}
    for item in selected:
        row, role = item['source'], item['role']
        image = source.parent/'imgs'/row['img_name']
        assert image.is_file()
        roles[role].append(dict(query_id='SR_'+role+'_'+str(row['qid']), dataset='SLAKE',
            image_path=str(image), original_image_path=str(image), source_group=item['group'],
            question=row['question'], reference=row['answer'], role='REPLAY_'+role,
            answer_kind=item['kind'], source_qid=row['qid'], source_file=str(source), source_role='train',
            source_annotation={k: row[k] for k in ('content_type', 'location', 'modality', 'answer_type')},
            forced_expert_index=item['group_index'] if role == 'CHECK' else None,
            patient_id='UNKNOWN', annotation_status='DATASET_SOURCE_ANNOTATION_NOT_CLINICIAN_REVIEWED'))
    fit_groups = {x['source_group'] for x in roles['FIT']}
    check_groups = {x['source_group'] for x in roles['CHECK']}
    assert not (fit_groups & check_groups or (fit_groups | check_groups) & excluded)
    for role, rows in roles.items():
        (run/'private'/('REPLAY_'+role+'.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    public = dict(status='PASS_SOURCE_ANNOTATED_DEVELOPMENT', eligible_balanced_groups=eligible,
        fit_groups=64, fit_queries=192, check_groups=32, check_queries=96,
        fit_answers={k: 64 for k in KINDS}, check_answers={k: 32 for k in KINDS},
        source_group_overlap=0, previous_evaluation_and_U_group_overlap=0,
        train_source='SLAKE official train annotations', content_types=list(TYPES),
        annotation_review='SOURCE_ONLY', patient_independence='UNKNOWN',
        independent_confirmation=False, historical_exposure_outside_current_campaign='NOT_EXHAUSTIVELY_AUDITED')
    (run/'public/DATA_ADMISSION.json').write_text(json.dumps(public, indent=2)+'\n')
    print(json.dumps(public))


if __name__ == '__main__':
    main()
