"""Select isolated, task-matched U candidates using native source metadata only."""
import hashlib
import json
import os
from pathlib import Path
import re
from difflib import SequenceMatcher
import unicodedata

ROOT = Path(os.environ['RUN_ROOT'])
DATA = Path(os.environ['M3BENCH_ROOT'])
# Native-only manual review: similar words did not establish matching task semantics.
ALLOW_HARD = {6: {'2171'}, 19: {'902'}, 23: set()}
REJECT_HARD = {6: {'1028'}, 7: {'1753'}, 17: {'1136'}, 19: {'2082'},
               23: {'2065','2160'}, 43: {'1999'}}
STOP = {'what', 'which', 'where', 'this', 'that', 'the', 'are', 'is', 'in', 'on', 'of', 'a', 'an',
        'how', 'to', 'image', 'picture', 'organ', 'located', 'there', 'present', 'does', 'do', 'be'}


def words(question):
    return set(re.findall(r'[a-z]+', question.casefold())) - STOP


def source_group(dataset, row):
    return ('SLAKE:' + row['img_name'].split('/')[0] if dataset == 'SLAKE'
            else 'VQA-RAD:' + row['image_name'])


def source_path(dataset, row):
    return (DATA / 'SLAKE/imgs' / row['img_name'] if dataset == 'SLAKE'
            else DATA / 'VQA-RAD/images' / row['image_name'])


def equivalent(a, b):
    return ' '.join(str(a).casefold().split()) == ' '.join(str(b).casefold().split())


def canonical_answer(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(value)).casefold() if c.isalnum())


def prepare():
    tasks = json.loads((ROOT / 'private/TASKS_MATRIX.json').read_text())['tasks']
    slake_root = DATA / 'SLAKE'
    slake = {s: json.loads((slake_root / f'{s}.json').read_text()) for s in ('train', 'validation', 'test')}
    vqa = json.loads((DATA / 'VQA-RAD/VQA_RAD Dataset Public.json').read_text())
    used = [r for t in tasks for r in [t['native'], *t['evaluation'], *t['U_fit'], *t['U_expanded']]]
    used += json.loads((ROOT / 'private/AUXILIARY_POOL.json').read_text())
    used_groups = {r['source_group'] for r in used}
    used_text = {' '.join(r['question'].casefold().split()) for r in used}
    # The new pressure/verify pools are validation/test; U may use SLAKE train only.
    heldout_slake = {source_group('SLAKE', r) for s in ('validation', 'test') for r in slake[s]}
    used_cases = {r['image_case_url'] for r in vqa if source_group('VQA-RAD', r) in used_groups}
    pools = {
        'SLAKE': [r for r in slake['train'] if source_group('SLAKE', r) not in used_groups | heldout_slake
                  and ' '.join(r['question'].casefold().split()) not in used_text],
        'VQA-RAD': [r for r in vqa if not r['phrase_type'].startswith('test')
                    and source_group('VQA-RAD', r) not in used_groups and r['image_case_url'] not in used_cases
                    and ' '.join(r['question'].casefold().split()) not in used_text],
    }
    catalog = {'SLAKE': sum(slake.values(), []), 'VQA-RAD': vqa}
    out = []
    counts = []
    for task in tasks:
        native = task['native']
        dataset = native['dataset']
        meta = next((r for r in catalog[dataset] if source_group(dataset, r) == native['source_group']
                     and equivalent(r['question'], native['question'])), None)
        if meta is None:
            raise ValueError(f'Native source metadata missing for edit order {task["order"]}')
        candidates = []
        for row in pools[dataset]:
            if (canonical_answer(row['answer']) == canonical_answer(native['reference'])
                    or native['reference'].casefold() in row['question'].casefold()):
                continue
            if dataset == 'SLAKE':
                if row['q_lang'] != meta['q_lang'] or row['answer_type'] != meta['answer_type']:
                    continue
                same_relation = (row['content_type'] == meta['content_type']
                                 and (meta['triple'][1] == '_' or row['triple'][1] == meta['triple'][1]))
                diverse_label = row['content_type']
            else:
                if row['answer_type'].strip() != meta['answer_type'].strip():
                    continue
                same_relation = row['question_type'] == meta['question_type'] and row['image_organ'] == meta['image_organ']
                diverse_label = row['question_type']
            overlap = len(words(row['question']) & words(native['question']))
            ratio = SequenceMatcher(None, row['question'].casefold(), native['question'].casefold()).ratio()
            semantic = overlap >= 1 or ratio >= .54
            hard = same_relation and semantic
            candidates.append((row, hard, same_relation, overlap, ratio, diverse_label))
        hard = sorted((x for x in candidates if x[1]), key=lambda x: (-x[3], -x[4], str(x[0].get('qid'))))
        selected = []
        groups = {r['source_group'] for r in task['U_fit']}
        texts = {r['question'].casefold() for r in task['U_fit']}
        evidence = []
        def take(item, kind):
            row = item[0]
            group = source_group(dataset, row)
            if group in groups or row['question'].casefold() in texts:
                return False
            path = source_path(dataset, row)
            if not path.is_file():
                raise FileNotFoundError(path)
            selected.append(dict(dataset=dataset, question=row['question'], reference=str(row['answer']),
                                 source_group=group, source_qid=str(row['qid']), image_path=str(path),
                                 image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_role='isolated_auxiliary_train',
                                 U_kind=kind))
            groups.add(group); texts.add(row['question'].casefold())
            evidence.append(dict(kind=kind, same_relation=item[2], shared_content_terms=item[3],
                                 question_similarity=round(item[4], 3), distinct_source=True, distinct_answer=True))
            return True
        for item in hard:
            if sum(x['U_kind'] == 'hard' for x in selected) >= 2:
                break
            if (str(item[0]['qid']) not in REJECT_HARD.get(task['order'],set())
                    and (task['order'] not in ALLOW_HARD or str(item[0]['qid']) in ALLOW_HARD[task['order']])):
                take(item, 'hard')
        diverse = sorted((x for x in candidates if x not in hard),
                         key=lambda x: (x[5] == (meta['content_type'] if dataset == 'SLAKE' else meta['question_type']),
                                        x[5], str(x[0].get('qid'))))
        diverse_types=set()
        for item in diverse:
            if sum(x['U_kind'] == 'diverse' for x in selected) >= 2:
                break
            if item[5] not in diverse_types and take(item, 'diverse'):
                diverse_types.add(item[5])
        counts.append(dict(order=task['order'], old=len(task['U_fit']), hard=sum(x['U_kind'] == 'hard' for x in selected),
                           diverse=sum(x['U_kind'] == 'diverse' for x in selected), evidence=evidence))
        out.append(dict(task, U_new=selected))
    p = ROOT / 'private/TASKS_R2.json'
    p.write_text(json.dumps(dict(tasks=out,source='native source metadata plus isolated train auxiliary pool',
                                 U_old_in_previous_expanded=all(all(any(e['image_sha256'] == o['image_sha256']
                                      and equivalent(e['question'], o['question']) for e in t['U_expanded']) for o in t['U_fit']) for t in tasks)),
                            ensure_ascii=False, indent=2))
    (ROOT / 'U_SUPPORT_AUDIT.json').write_text(json.dumps(dict(counts=counts,hard_2=sum(c['hard']==2 for c in counts),
                                                              diverse_2=sum(c['diverse']==2 for c in counts)), indent=2))
    print(json.dumps({'hard_2':sum(c['hard']==2 for c in counts),'diverse_2':sum(c['diverse']==2 for c in counts),
                      'first24_hard_2':sum(c['hard']==2 for c in counts[:24]),
                      'first24_diverse_2':sum(c['diverse']==2 for c in counts[:24])}))


if __name__ == '__main__':
    prepare()
