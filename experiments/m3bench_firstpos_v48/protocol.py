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


def adaptive_gradient(grads, losses):
    import torch
    assert len(grads)==len(losses)==4 and all(g.shape==grads[0].shape for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    values=torch.as_tensor(losses,dtype=torch.float64,device=grads[0].device)
    assert values.shape==(4,) and torch.isfinite(values).all()
    weights=torch.softmax(values,dim=0)
    gradient=(torch.stack(grads)*weights[:,None]).sum(0)
    assert torch.isfinite(gradient).all()
    return gradient,weights


def swapped_rows(task,tasks,short=False):
    assert len(tasks)==4 and len({t['key'] for t in tasks})==4
    paths=[t['eval_rows'][0]['image_path'] for t in tasks]
    assert len(set(paths))==4
    for t,path in zip(tasks,paths):assert all(q['image_path']==path for q in t['eval_rows'])
    index=next(i for i,t in enumerate(tasks) if t['key']==task['key'])
    target=paths[(index+1)%4]
    return [dict(brief_row(q) if short else q,image_path=target) for q in task['eval_rows']]


def distribution_shift(original,foreign):
    import torch,math
    assert original.shape==foreign.shape and original.ndim==2 and original.shape[0]>0
    assert torch.isfinite(original).all() and torch.isfinite(foreign).all()
    lp=original.double().log_softmax(-1);lq=foreign.double().log_softmax(-1)
    p,q=lp.exp(),lq.exp();lm=torch.logaddexp(lp,lq)-math.log(2.)
    tv=.5*(p-q).abs().sum(-1)
    js=.5*((p*(lp-lm)).sum(-1)+(q*(lq-lm)).sum(-1))
    assert (tv>=0).all() and (tv<=1+1e-12).all() and (js>=-1e-12).all() and (js<=math.log(2)+1e-12).all()
    changed=(original.argmax(-1)!=foreign.argmax(-1))
    return dict(tokens=len(tv),tv_mean=float(tv.mean()),js_nats_mean=float(js.mean()),argmax_changed=int(changed.sum()),argmax_changed_fraction=float(changed.double().mean()))


def paired_gradient(grads,losses,arm):
    import torch
    assert arm in ('COVER','ADAPT')
    if arm=='COVER':return torch.stack(grads).mean(0),torch.full((4,),.25,dtype=grads[0].dtype)
    return adaptive_gradient(grads,losses)

def position_shift(left,right):
    assert left.shape[0]>1
    return dict(all=distribution_shift(left,right),first=distribution_shift(left[:1],right[:1]),continuation=distribution_shift(left[1:],right[1:]))
