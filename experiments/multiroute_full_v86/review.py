"""CPU review consumers; never infer correctness from token confidence."""
import json
from pathlib import Path
import random


def evaluation_pack(root):
    root=Path(root);read=lambda p:json.loads(p.read_text())
    assert read(root/'MAIN_COMPLETE.json')['outputs']==775
    files=sorted((root/'outputs').glob('*/*/*.json'));assert len(files)==775
    groups={}
    for p in files:
        d=read(p);q=d['query']
        key=(d['binding']['image_sha256'],q['question'],q['reference'],d['raw_answer'],d['EOS'],d['at_cap'])
        groups.setdefault(key,[]).append(str(p.relative_to(root)))
    rows=list(groups.items());random.Random(860023).shuffle(rows)
    packet=[];mapping={}
    for i,((image,question,reference,response,eos,cap),paths) in enumerate(rows):
        key=f'V{i:04d}';mapping[key]=paths
        packet.append(dict(id=key,question=question,reference=reference,response=response,EOS=eos,at_cap=cap))
    assert not (root/'BLIND_EVALUATION.json').exists()
    (root/'BLIND_EVALUATION.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    (root/'BLIND_EVALUATION_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (root/'REVIEW_READY.json').write_text(json.dumps(dict(status='HIDDEN_ARM_FULL_REVIEW_REQUIRED',unique=len(packet),consumers=775)))
