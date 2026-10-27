"""Aggregate explicit full-answer reviews; never infer correctness from loss."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def summarize(root):
    complete = read(root/'MAIN_COMPLETE.json')
    assert read(root/'EXIT_worker.json')['exit_code'] == 0
    assert read(root/'RESOURCE_GUARD_EXIT.json')['trigger'] is None
    assert complete['outputs'] == 372 and complete['retrieval'] == 93
    assert complete['counts']['updates'] == 3840 and complete['counts']['backwards'] == 8200
    tasks = read(root/'TASKS.json')
    ids = {t['group']: i+1 for i,t in enumerate(tasks)}
    base_review = read(root.parent/'BASE_REVIEW_FROZEN.json')['rows']
    visual = {x['qid']: x for x in read(root.parent/'SOURCE_VISUAL_REVIEW_FROZEN.json')['rows']}
    qualified = {x['qid'] for x in base_review if x['strict_pass'] and
                 visual[x['qid']]['status'] == 'REFERENCE_COMPATIBLE_NONEXPERT'}
    assert len(qualified) == 3
    metrics = ('strict_reference_scope', 'requested_fact_match', 'source_supported_qualifier_sensitivity')
    scores = {}
    for group in read(root/'FULL_TEXT_REVIEW.json')['groups']:
        for arm, owner, qid, role in group['members']:
            key = (arm, owner, qid)
            assert key not in scores
            raw = read(root/'outputs'/arm/owner/(qid+'.json'))
            assert (raw['query']['question'], raw['query']['reference'], raw['raw_answer'], raw['EOS'], raw['at_cap']) == (
                group['question'], group['reference'], group['answer'], group['EOS'], group['at_cap'])
            scores[key] = dict(role=role, qualified=raw['query']['qid'] in qualified,
                exact_source_answer=raw['raw_answer'].strip() == tasks[ids[owner]-1]['training'][0]['reference'],
                **{m: bool(group[m] and raw['EOS'] and not raw['at_cap']) for m in metrics})
    assert len(scores) == len(list((root/'outputs').glob('*/*/*.json'))) == 372
    routes = read(root/'ROUTES_FROZEN.json')['routes']
    records = []
    for task in tasks:
        owner = task['group']
        for row in task['evaluation']:
            qid = row['query_id']
            selected = routes[owner][qid]['logical_edit_id']
            assert selected in (owner, None)
            for arm in ('Base', 'TT_CE', 'TT_BAL', 'LoRA_BAL'):
                for mode in ('FORCED', 'NATURAL'):
                    effective = arm if mode == 'FORCED' or selected else 'Base'
                    records.append(dict(owner=ids[owner], arm=arm, mode=mode, **scores[(effective,owner,qid)]))
            retrieved = read(root/'retrieval'/owner/(qid+'.json'))
            assert retrieved['route'] == routes[owner][qid]
            if selected:
                assert retrieved['raw_answer'] == task['training'][0]['reference']
                # All source targets are identical here; OTHER answers request different relations.
                passed = row['evaluation_role'] != 'OTHER'
                result = dict(role=row['evaluation_role'], qualified=row['qid'] in qualified,
                    exact_source_answer=True, **{m: passed for m in metrics})
            else:
                assert retrieved['raw_answer'] == read(root/'outputs'/'Base'/owner/(qid+'.json'))['raw_answer']
                result = scores[('Base',owner,qid)]
            records.append(dict(owner=ids[owner], arm='Retrieval', mode='NATURAL', **result))

    summary = {}
    for mode in ('FORCED', 'NATURAL'):
        summary[mode] = {}
        for arm in ('Base','TT_CE','TT_BAL','LoRA_BAL','Retrieval'):
            rows = [x for x in records if x['mode'] == mode and x['arm'] == arm]
            if not rows:
                continue
            assert len(rows) == 93
            role_results = {}
            for role in ('NATIVE','FIT','T2G','OTHER','QUALIFIED_PROTECTION'):
                selected = [x for x in rows if x['qualified']] if role == 'QUALIFIED_PROTECTION' else [x for x in rows if x['role'] == role]
                role_results[role] = dict(n=len(selected), **{m: sum(x[m] for x in selected) for m in metrics})
            role_results['other_exact_source_answer'] = sum(x['role']=='OTHER' and x['exact_source_answer'] for x in rows)
            summary[mode][arm] = role_results

    paired = []
    for owner in range(1,9):
        row = dict(owner=owner)
        for arm in ('Base','TT_CE','TT_BAL','LoRA_BAL'):
            selected = [x for x in records if x['mode']=='FORCED' and x['owner']==owner and x['arm']==arm]
            row[arm] = {role:sum(x['strict_reference_scope'] for x in selected if x['role']==role) for role in ('NATIVE','FIT','T2G','OTHER')}
        row['BAL_minus_CE_T2G'] = row['TT_BAL']['T2G']-row['TT_CE']['T2G']
        paired.append(row)
    training = list((root/'training').glob('*/*.json'))
    assert len(training) == 24 and len(list((root/'lifecycle').glob('*/*.json'))) == 24
    costs = {arm:sum(read(p)['seconds'] for p in training if p.stem==arm) for arm in ('TT_CE','TT_BAL','LoRA_BAL')}
    base_correct = {key[1:]: value for key,value in scores.items() if key[0]=='Base' and value['role']=='OTHER' and value['strict_reference_scope']}
    losses = {arm:sum(not scores[(arm,*key)]['strict_reference_scope'] for key in base_correct) for arm in ('TT_CE','TT_BAL','LoRA_BAL')}
    return dict(status='COMPLETE_NOT_PROMOTED', full_text_review_consumers=372, retrieval_consumers=93,
        summary=summary, paired_by_image_group=paired,
        promotion=dict(T2G_net_gain=0, required_net_gain=3, passed=False,
            reason='No TT_BAL gain over TT_CE; all eight paired differences zero. No confirmation expansion.'),
        all_other_Base_strict_correct=11, all_other_Base_strict_correct_losses=losses,
        route_activations={role:sum(routes[t['group']][r['query_id']]['logical_edit_id'] is not None
            for t in tasks for r in t['evaluation'] if r['evaluation_role']==role) for role in ('NATIVE','FIT','T2G','OTHER')},
        cost=dict(counts=complete['counts'], train_seconds_by_arm=costs,
            stage_b_process_seconds=read(root/'EXIT_worker.json')['time']-read(root/'START.json')['worker']['started_epoch'],
            phase_a_charged_seconds=130, final_guard_observed_bytes=read(root/'RESOURCE_GUARD_HEALTH.json')['owned_bytes'],
            retired_checkpoint_bytes=sum(read(p)['retired_bytes'] for p in (root/'lifecycle').glob('*/*.json'))),
        caveats=['Single exposed TRAIN seed; eight images are not eight independent patients.',
            'All modality facts were already correct in Base; source-scope compliance is not factual correction.',
            'Reference-only scope penalizes an added known modality qualifier; source-supported sensitivity preserves all three qualified facts.',
            'Eight sequence references are not acquisition-verified; their retention counts are reference comparisons only.',
            'Method identities were visible to the sole assistant reviewer; no independent clinical judgment.',
            'R0 is a single-edit route per image, not an eight-edit bank. No T1G, DEV or CONFIRM evaluation.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
