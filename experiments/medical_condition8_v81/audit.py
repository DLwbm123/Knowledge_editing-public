"""Prepare blind review packets and freeze reviewed source targets; no model calls."""
import argparse
import json
from pathlib import Path
import random
import re
import time


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    assert not p.exists(), f'Immutable output already exists: {p}'
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def targets_pack(root):
    assert all(read(root/f'COMPLETE_teacher_{p}.json')['status']=='PASS' for p in range(2))
    files = sorted((root/'teacher').glob('*/*/*.json'))
    assert len(files)==120
    random.Random(20261023).shuffle(files)
    public=[]; mapping={}
    for index, p in enumerate(files):
        d=read(p); key=f'T{index:03d}'
        public.append(dict(id=key, question=d['question'], reference=d['reference'],
                           response=d['raw_answer'], EOS=d['EOS'], at_cap=d['at_cap']))
        mapping[key]=str(p.relative_to(root))
    write(root/'BLIND_TARGETS.json',public)
    write(root/'BLIND_TARGET_MAPPING.json',mapping)


def targets_freeze(root):
    packet=read(root/'BLIND_TARGETS.json')
    labels=read(root/'TARGET_REVIEW_LABELS.json')
    assert len(labels)==len({d['id'] for d in labels})==120
    by_id={d['id']:d for d in labels}
    assert set(by_id)=={d['id'] for d in packet}
    for label in labels:
        assert label['label'] in ('C','I','U','A') and label['reason']
        assert isinstance(label['complete'],bool) and isinstance(label['extra_claims'],bool)
    # The immutable blind judgments must precede opening the arm mapping.
    write(root/'TARGET_LABELS_FROZEN.json',labels)
    by_path={}
    mapping=read(root/'BLIND_TARGET_MAPPING.json')
    for d in packet:
        label=by_id[d['id']]
        assert label['label'] in ('C','I','U','A') and label['reason']
        assert isinstance(label['complete'],bool) and isinstance(label['extra_claims'],bool)
        by_path[mapping[d['id']]]=dict(label,
            accepted=label['label']=='C' and label['complete'] and not label['extra_claims']
                     and d['EOS'] and not d['at_cap'])
    targets={};tokens={};rows=read(root/'SOURCE_ROWS.json');decisions=[];changed=0
    for owner in range(1,9):
        key=str(owner)
        targets[key]={a:[] for a in ('S0','S1','S2','S3')}
        tokens[key]={a:[] for a in ('S0','S1','S2','S3')}
        for view,row in enumerate(rows[key]):
            paths={a:f'teacher/{a}/{owner}/{view}.json' for a in ('S1','S2','S3')}
            passed=all(by_path[p]['accepted'] for p in paths.values())
            decisions.append(dict(owner=owner,view=view,common_fallback=not passed,
                                  reviews={a:by_path[p] for a,p in paths.items()}))
            for arm in ('S0','S1','S2','S3'):
                use=passed and arm!='S0'
                candidate=read(root/paths[arm]) if use else None
                targets[key][arm].append(candidate['raw_answer'] if use else row['reference'])
                tokens[key][arm].append(candidate['raw_token_ids'] if use else None)
                changed += int(use and normalize(candidate['raw_answer'])!=normalize(row['reference']))
    write(root/'TARGETS_FROZEN.json',targets)
    write(root/'TARGET_TOKENS_FROZEN.json',tokens)
    write(root/'TARGET_ADMISSION.json',dict(status='PASS' if changed else 'NO_VALID_NONIDENTICAL_TARGETS',
        candidate_count=120,inputs=40,common_fallback_inputs=sum(d['common_fallback'] for d in decisions),
        normalized_nonidentical_targets=changed,decisions=decisions,time=time.time(),
        reviewer='assistant blinded against source reference; not clinical validation'))
    if not changed:
        write(root/'STOP',dict(reason='SCIENTIFIC_TARGET_ADMISSION_FALSE',time=time.time()))
        return
    write(root/'TARGET_REVIEW_COMPLETE.json',dict(status='PASS',time=time.time(),all8=True,
          no_evaluation_material_used=True,normalized_nonidentical_targets=changed))


def evaluation_pack(root):
    assert read(root/'MAIN_COMPLETE.json')['outputs']==620
    files=sorted((root/'outputs').glob('*/*/*.json'))
    assert len(files)==620
    groups={}
    for p in files:
        d=read(p);q=d['query']
        groups.setdefault((q['question'],q['reference'],d['raw_answer']),[]).append(str(p.relative_to(root)))
    items=list(groups.items());random.Random(810023).shuffle(items)
    packet=[];mapping={}
    for i,((question,reference,answer),paths) in enumerate(items):
        key=f'E{i:04d}'
        packet.append(dict(id=key,question=question,reference=reference,response=answer))
        mapping[key]=paths
    write(root/'BLIND_EVALUATION.json',packet)
    write(root/'BLIND_EVALUATION_MAPPING.json',mapping)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['targets-pack','targets-freeze','evaluation-pack'])
    p.add_argument('root',type=Path)
    args=p.parse_args()
    {'targets-pack':targets_pack,'targets-freeze':targets_freeze,
     'evaluation-pack':evaluation_pack}[args.action](args.root)
