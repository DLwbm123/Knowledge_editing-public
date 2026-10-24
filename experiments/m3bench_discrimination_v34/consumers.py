"""Frozen V34 bindings, blinded free-answer grading, and separate candidate readouts."""
import json
import math
import random
from collections import Counter
from pathlib import Path

LABELS = {'CORRECT', 'INCORRECT', 'UNCERTAIN', 'ABSTAIN'}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2)+'\n')


def rows(root):
    result = []
    for i, task in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
        for q in task['eval_rows']:
            for arm in ('BASE', 'JOINT'):
                d = read(root/str(i % 2)/task['key']/'outputs'/arm/(q['query_id']+'.json'))
                assert all(d['binding'][k] == q[k] for k in ('question', 'reference', 'image_path'))
                result.append(dict(case=task['key'], arm=arm, q=q, answer=d['raw_answer'],
                    tokens=d['raw_token_ids'], eos=d['EOS'], at_cap=d['at_cap']))
    assert len(result) == 62
    return result


def validate(root):
    for part in ('0', '1'):
        assert read(root/part/'EXIT.json')['code'] == 0
        assert read(root/part/'MODEL_COMPLETE.json')['status'] == 'PASS'
    rr = rows(root)
    n = 0
    for i, task in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
        folder = root/str(i % 2)/task['key']
        trained = read(folder/'JOINT/TRAIN_COMPLETE.json')
        assert trained['steps'] == 160 and trained['parameters'] == 7168 and trained['save_load_exact']
        assert read(folder/'LIFECYCLE.json')['status'] == 'RETIRED'
        for arm in ('BASE', 'JOINT'):
            for q in task['candidate_rows']:
                d = read(folder/'scores'/arm/(q['query_id']+'.json'))
                assert all(d['binding'][k] == q[k] for k in ('question', 'reference', 'image_path'))
                assert d['parent_query_id'] == q['parent_query_id'] and d['candidate_type'] == q['candidate_type']
                s = d['scores']
                assert len(s['target_ids']) == len(s['token_logp']) and s['answer_tokens'] > 0
                assert all(math.isfinite(x) and x <= 1e-6 for x in s['token_logp'])
                assert math.isclose(sum(s['token_logp']), s['sum_with_eos'], rel_tol=1e-5, abs_tol=1e-4)
                assert math.isclose(s['mean_no_eos']*s['answer_tokens'], s['sum_no_eos'], rel_tol=1e-5, abs_tol=1e-4)
                assert s['answer_tokens']+len(s['eos_logp']) == len(s['target_ids'])
                assert math.isclose(s['sum_no_eos']+sum(s['eos_logp']), s['sum_with_eos'], rel_tol=1e-5, abs_tol=1e-4)
                assert s['critical_span']['status'] == 'BOUND'
                n += 1
    assert n == 96
    return dict(status='PASS', free_outputs=len(rr), candidate_sequences=n)


def blind(root):
    validate(root)
    rr = rows(root)
    keys = list(dict.fromkeys((r['q']['query_id'], r['answer']) for r in rr))
    random.Random(340021).shuffle(keys)
    packet, mapping = [], {}
    for i, key in enumerate(keys):
        code = f'Q{i+1:03d}'
        indices = [n for n, r in enumerate(rr) if (r['q']['query_id'], r['answer']) == key]
        r = rr[indices[0]]
        mapping[code] = indices
        packet.append(dict(code=code, question=r['q']['question'], reference=r['q']['reference'], answer=r['answer']))
    assert not (root/'BLIND_PACKET.json').exists()
    write(root/'BLIND_PACKET.json', packet)
    write(root/'BLIND_MAPPING.json', mapping)
    return dict(unique=len(packet), previous_verdict_reuse=0)


def summarize(root):
    validate(root)
    assert (root/'VERDICTS_FROZEN.json').is_file()
    rr, verdicts, mapping = rows(root), read(root/'BLIND_VERDICTS.json'), read(root/'BLIND_MAPPING.json')
    assert set(verdicts) == set(mapping)
    for code, indices in mapping.items():
        v = verdicts[code]
        assert v['label'] in LABELS and v['reason']
        for i in indices:
            rr[i]['label'] = v['label']
    base = {r['q']['query_id']: r for r in rr if r['arm'] == 'BASE'}
    joint = {r['q']['query_id']: r for r in rr if r['arm'] == 'JOINT'}
    qualified = set()
    for task in read(root/'DATA_ADMISSION.json')['tasks']:
        u, p = task['train_rows']
        if base[u['query_id']]['label'] != 'CORRECT' and base[p['query_id']]['label'] == 'CORRECT':
            qualified.add(task['key'])
    result = dict(status='COMPLETE_FIXED_REFERENCE_DIAGNOSIS', qualified_cases=len(qualified),
                  independent_confirmation=False, clinical_expert_review=False, groups={}, diagnostic_cells={})
    for group in ('SOURCE', 'HELDOUT', 'LOCALITY_CLINICAL', 'LOCALITY_TECHNICAL'):
        subset = [r for r in joint.values() if r['q']['eval_group'] == group]
        entry = dict(n=len(subset), base_labels=dict(Counter(base[r['q']['query_id']]['label'] for r in subset)),
                     joint_labels=dict(Counter(r['label'] for r in subset)))
        entry['base_C_support'] = sum(base[r['q']['query_id']]['label'] == 'CORRECT' for r in subset)
        entry['base_C_lost'] = sum(base[r['q']['query_id']]['label'] == 'CORRECT' and r['label'] != 'CORRECT' for r in subset)
        entry['repairs'] = sum(base[r['q']['query_id']]['label'] != 'CORRECT' and r['label'] == 'CORRECT' for r in subset)
        entry['roles'] = {}
        for role in sorted({r['q']['relation_role'] for r in subset}):
            selected = [r for r in subset if r['q']['relation_role'] == role]
            entry['roles'][role] = dict(n=len(selected),
                base_correct=sum(base[r['q']['query_id']]['label']=='CORRECT' for r in selected),
                joint_correct=sum(r['label']=='CORRECT' for r in selected),
                base_C_lost=sum(base[r['q']['query_id']]['label']=='CORRECT' and r['label']!='CORRECT' for r in selected))
        selected = [r for r in subset if r['case'] in qualified]
        entry['source_qualified'] = dict(n=len(selected), joint_correct=sum(r['label']=='CORRECT' for r in selected))
        result['groups'][group] = entry
    result['heldout_all_four_correct_cases'] = sum(all(r['label'] == 'CORRECT' for r in joint.values()
        if r['case'] == task['key'] and r['q']['eval_group'] == 'HELDOUT') for task in read(root/'DATA_ADMISSION.json')['tasks'])
    diagnostics = []
    for i, task in enumerate(read(root/'DATA_ADMISSION.json')['tasks']):
        folder = root/str(i % 2)/task['key']/'scores'
        for q in task['eval_rows']:
            if q['eval_group'] != 'HELDOUT':
                continue
            item = dict(case=task['key'], query_id=q['query_id'], arms={})
            for arm in ('BASE', 'JOINT'):
                ss = {k: read(folder/arm/(q['query_id']+'_'+k+'.json'))['scores'] for k in ('CORRECT', 'OTHER_RELATION', 'WRONG_VALUE')}
                margins = {k: ss['CORRECT']['mean_no_eos']-ss[k]['mean_no_eos'] for k in ('OTHER_RELATION', 'WRONG_VALUE')}
                label = (base if arm == 'BASE' else joint)[q['query_id']]['label']
                item['arms'][arm] = dict(margins=margins, free_label=label,
                    candidate_correct_top=all(v > 0 for v in margins.values()),
                    lengths={k: s['answer_tokens'] for k, s in ss.items()})
            diagnostics.append(item)
    for arm in ('BASE', 'JOINT'):
        result['diagnostic_cells'][arm] = dict(Counter(
            ('free_C' if d['arms'][arm]['free_label'] == 'CORRECT' else 'free_nonC')+'_'+
            ('candidate_top' if d['arms'][arm]['candidate_correct_top'] else 'candidate_not_top') for d in diagnostics))
    result['stability'] = {arm: dict(empty=sum(not r['answer'].strip() for r in rr if r['arm']==arm),
        cap=sum(r['at_cap'] for r in rr if r['arm']==arm), eos=sum(r['eos'] for r in rr if r['arm']==arm)) for arm in ('BASE','JOINT')}
    write(root/'PRIVATE_DIAGNOSTIC_CELLS.json', diagnostics)
    write(root/'RESULTS.json', result)
    return result


if __name__ == '__main__':
    import sys
    op, path = sys.argv[1:]
    print(json.dumps(globals()[op](Path(path)), indent=2))
