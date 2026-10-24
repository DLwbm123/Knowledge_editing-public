"""Validate four arms and blind-grade only new prompt outputs; preserve prior frozen labels."""
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from protocol import ARMS, INSTRUCTION, brief_row, verify_original, gate_ready


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def rows(root):
    config = read(root/'DATA_ADMISSION.json')
    assert config['instruction'] == INSTRUCTION
    result = []
    for i, task in enumerate(config['tasks']):
        for q in task['eval_rows']:
            for arm in ARMS:
                path = root/str(i % 2)/task['key']/'outputs'/arm/(q['query_id']+'.json')
                d = read(path)
                expected = brief_row(q) if arm.endswith('_SHORT') else q
                assert all(d['binding'][k] == expected[k] for k in ('question', 'reference', 'image_path'))
                result.append(dict(case=task['key'], arm=arm, q=q, answer=d['raw_answer'],
                    tokens=d['raw_token_ids'], eos=d['EOS'], at_cap=d['at_cap']))
    assert len(result) == 124
    return result


def validate(root):
    assert gate_ready(root)
    for part in ('0', '1'):
        assert read(root/part/'EXIT.json')['code'] == 0
        complete = read(root/part/'MODEL_COMPLETE.json')
        assert complete['status'] == 'PASS'
        assert complete['counts']['updates'] == 320 and complete['counts']['generations'] == (60 if part=='0' else 64)
        assert complete['counts']['candidate_sequences'] == 0
        release = read(root/part/'INTERVENTION_RELEASE.json')
        assert release['status'] == 'PASS' and release['original_outputs_exact'] == 62
        assert release['time'] >= max(read(root/p/'ORIGINAL_PARITY.json')['time'] for p in ('0','1'))
    rr = rows(root)
    for i, task in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
        folder = root/str(i % 2)/task['key']
        trained = read(folder/'JOINT/TRAIN_COMPLETE.json')
        assert trained['steps'] == 160 and trained['parameters'] == 7168 and trained['save_load_exact']
        assert read(folder/'LIFECYCLE.json')['status'] == 'RETIRED'
        assert read(folder/'ORIGINAL_PARITY.json')['status'] == 'PASS'
    return dict(status='PASS', original_outputs_exact=62, brief_outputs=62, total_outputs=len(rr))


def blind(root):
    validate(root)
    rr = rows(root)
    keys = list(dict.fromkeys((r['q']['query_id'], r['answer']) for r in rr if r['arm'].endswith('_SHORT')))
    random.Random(350021).shuffle(keys)
    packet, mapping = [], {}
    for i, key in enumerate(keys):
        code = f'Q{i+1:03d}'
        indices = [n for n, r in enumerate(rr) if r['arm'].endswith('_SHORT') and (r['q']['query_id'], r['answer']) == key]
        r = rr[indices[0]]
        mapping[code] = indices
        packet.append(dict(code=code, question=r['q']['question'], reference=r['q']['reference'], answer=r['answer']))
    assert not (root/'BLIND_PACKET.json').exists()
    write(root/'BLIND_PACKET.json', packet)
    write(root/'BLIND_MAPPING.json', mapping)
    return dict(unique=len(packet), new_output_prior_verdict_reuse=0)


def summarize(root):
    validate(root)
    assert (root/'VERDICTS_FROZEN.json').is_file()
    rr = rows(root)
    mapping, verdicts = read(root/'BLIND_MAPPING.json'), read(root/'BLIND_VERDICTS.json')
    assert set(mapping) == set(verdicts)
    for code, indices in mapping.items():
        v = verdicts[code]
        assert v['label'] in {'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'} and v['reason']
        for i in indices:
            rr[i]['label'] = v['label']
    # Explicit original-output bindings supplied after blind verdict freeze.
    prior = read(root/'PRIOR_LABELS.json')
    for r in rr:
        if not r['arm'].endswith('_SHORT'):
            p = prior[r['arm']][r['q']['query_id']]
            assert p['tokens'] == r['tokens'] and p['answer'] == r['answer']
            r['label'] = p['label']
    arms = {a:{r['q']['query_id']:r for r in rr if r['arm']==a} for a in ARMS}
    qualified = {t['key'] for t in read(root/'DATA_ADMISSION.json')['tasks']
        if arms['BASE'][t['train_rows'][0]['query_id']]['label'] != 'CORRECT'
        and arms['BASE'][t['train_rows'][1]['query_id']]['label'] == 'CORRECT'}
    result = dict(status='COMPLETE', qualified_cases=len(qualified), independent_confirmation=False, clinical_expert_review=False, groups={})
    for group in ('SOURCE','HELDOUT','LOCALITY_CLINICAL','LOCALITY_TECHNICAL'):
        ids = [q for q,r in arms['BASE'].items() if r['q']['eval_group']==group]
        entry = dict(n=len(ids), labels={a:dict(Counter(arms[a][q]['label'] for q in ids)) for a in ARMS}, paired={})
        for original in ('BASE','JOINT'):
            short = original+'_SHORT'
            entry['paired'][original] = dict(
                repairs=sum(arms[original][q]['label']!='CORRECT' and arms[short][q]['label']=='CORRECT' for q in ids),
                losses=sum(arms[original][q]['label']=='CORRECT' and arms[short][q]['label']!='CORRECT' for q in ids),
                prior_correct=sum(arms[original][q]['label']=='CORRECT' for q in ids))
        entry['base_C_retention'] = {a:dict(support=sum(arms['BASE'][q]['label']=='CORRECT' for q in ids),
            lost=sum(arms['BASE'][q]['label']=='CORRECT' and arms[a][q]['label']!='CORRECT' for q in ids)) for a in ('JOINT','JOINT_SHORT')}
        entry['roles'] = {role:{a:dict(n=sum(arms[a][q]['q']['relation_role']==role for q in ids),
            correct=sum(arms[a][q]['q']['relation_role']==role and arms[a][q]['label']=='CORRECT' for q in ids)) for a in ARMS}
            for role in sorted({arms['BASE'][q]['q']['relation_role'] for q in ids})}
        qualified_ids = [q for q in ids if arms['BASE'][q]['case'] in qualified]
        entry['source_qualified'] = {a:dict(n=len(qualified_ids), correct=sum(arms[a][q]['label']=='CORRECT' for q in qualified_ids)) for a in ARMS}
        result['groups'][group] = entry
    result['all_four_heldout_correct'] = {a:sum(all(r['label']=='CORRECT' for r in arms[a].values()
        if r['case']==t['key'] and r['q']['eval_group']=='HELDOUT') for t in read(root/'DATA_ADMISSION.json')['tasks']) for a in ARMS}
    result['stability'] = {a:dict(empty=sum(not r['answer'].strip() for r in arms[a].values()),
        cap=sum(r['at_cap'] for r in arms[a].values()), eos=sum(r['eos'] for r in arms[a].values()),
        mean_tokens=statistics.mean(len(r['tokens']) for r in arms[a].values()),
        median_tokens=statistics.median(len(r['tokens']) for r in arms[a].values())) for a in ARMS}
    diagnostic = read(root/'PRIOR_DIAGNOSTIC_CELLS.json')
    gaps = [d for d in diagnostic if d['arms']['JOINT']['candidate_correct_top'] and d['arms']['JOINT']['free_label']!='CORRECT']
    assert len(gaps)==4 and len({d['case'] for d in gaps})==3
    result['predeclared_four_gap_subset'] = dict(n=4, cases=3, corrected=sum(arms['JOINT_SHORT'][d['query_id']]['label']=='CORRECT' for d in gaps))
    held = result['groups']['HELDOUT']['paired']['JOINT']
    result['limited_development_improvement_gate'] = bool(held['repairs']>held['losses'] and held['losses']==0
        and result['groups']['SOURCE']['paired']['JOINT']['losses']==0
        and result['groups']['LOCALITY_CLINICAL']['paired']['JOINT']['losses']==0)
    write(root/'RESULTS.json', result)
    return result


if __name__ == '__main__':
    import sys
    operation, path = sys.argv[1:]
    print(json.dumps(globals()[operation](Path(path)), indent=2))
