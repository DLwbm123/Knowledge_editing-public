"""Shared frozen prompt and full-original-output barrier."""
import json
from pathlib import Path

INSTRUCTION = 'Answer the question briefly with only the requested factual finding. Do not add explanations or information not requested.'
ARMS = ('BASE', 'JOINT', 'BASE_SHORT', 'JOINT_SHORT')


def brief_row(row, instruction=INSTRUCTION):
    assert instruction == INSTRUCTION
    return dict(row, question=row['question']+'\n'+instruction)


def verify_original(actual, prior):
    for key in ('question', 'reference', 'image_path'):
        assert actual['binding'][key] == prior['binding'][key], key
    for key in ('raw_token_ids', 'raw_answer', 'EOS', 'at_cap'):
        assert actual[key] == prior[key], key


def gate_ready(root):
    root = Path(root)
    for part, expected in (('0', 30), ('1', 32)):
        path = root/part/'ORIGINAL_PARITY.json'
        if not path.exists():
            return False
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'PASS' and receipt['outputs'] == expected
    return True


def source_rows(task):
    sources = task['train_rows']
    assert len(sources)==2 and len({q['query_id'] for q in sources})==2
    allowed = {q['query_id']:q for q in task['eval_rows'] if q['eval_group']=='SOURCE'}
    assert len(allowed)==2
    for q in sources:
        assert q['query_id'] in allowed
        assert all(q[k]==allowed[q['query_id']][k] for k in ('question','reference','image_path'))
    augmented = [dict(brief_row(q), query_id=q['query_id']+'_CTX') for q in sources]
    assert not {q['query_id'] for q in augmented} & {q['query_id'] for q in task['eval_rows']}
    return [*sources, *augmented]


def projected_gradient(grads, losses):
    import torch
    assert len(grads)==len(losses)==4 and all(g.shape==grads[0].shape for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    values=torch.as_tensor(losses,dtype=torch.float64,device=grads[0].device)
    assert values.shape==(4,) and torch.isfinite(values).all()
    weights=torch.softmax(values,dim=0)
    projected=[]
    for i,g in enumerate(grads):
        h=g.clone()
        for offset in (1,2,3):
            other=grads[(i+offset)%4];norm2=torch.dot(other,other)
            dot=torch.dot(h,other)
            if norm2>0 and dot<0:h=h-dot/norm2*other
        projected.append(h)
    gradient=(torch.stack(projected)*weights[:,None]).sum(0)
    assert torch.isfinite(gradient).all()
    return gradient,weights
