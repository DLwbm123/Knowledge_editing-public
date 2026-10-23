"""Full-answer hidden-arm review packet for independent route consumers."""
import json
from pathlib import Path
import random


def evaluation_pack(root,expected):
    root=Path(root);files=sorted((root/'outputs').glob('*/*/*.json'));assert len(files)==expected
    groups={}
    for p in files:
        d=json.loads(p.read_text());q=d['query']
        key=(d['binding']['image_sha256'],q['question'],q['reference'],d['raw_answer'],d['EOS'],d['at_cap'])
        groups.setdefault(key,[]).append(str(p.relative_to(root)))
    rows=list(groups.items());random.Random(870023).shuffle(rows);packet=[];mapping={}
    for i,((image,question,reference,response,eos,cap),paths) in enumerate(rows):
        key=f'R{i:04d}';packet.append(dict(id=key,question=question,reference=reference,response=response,EOS=eos,at_cap=cap));mapping[key]=paths
    assert not (root/'BLIND_EVALUATION.json').exists()
    (root/'BLIND_EVALUATION.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (root/'BLIND_EVALUATION_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (root/'REVIEW_READY.json').write_text(json.dumps(dict(consumers=expected,unique=len(packet),status='FULL_HIDDEN_ARM_REVIEW_PENDING')))
