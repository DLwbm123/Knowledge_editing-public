"""Frozen source-span supervision; no evaluation references or medical inference."""
import torch
import torch.nn.functional as F


def token_groups(tokenizer, ids, reference, spans):
    eos=tokenizer.eos_token_id
    assert len(ids)>1 and ids[-1]==eos and ids.count(eos)==1
    content=ids[:-1]
    decode=lambda values:tokenizer.decode(values,skip_special_tokens=True,clean_up_tokenization_spaces=False).lstrip()
    text=decode(content)
    assert text in (reference,reference+' '), 'Source target differs beyond the exact template separator'
    assert spans and all(text[s['start']:s['end']]==s['text'] for s in spans)
    assert all(a['end']<=b['start'] for a,b in zip(spans,spans[1:]))
    boundaries=[0]
    for j in range(1,len(content)+1):
        prefix=decode(content[:j]);common=0
        for a,b in zip(prefix,text):
            if a!=b:break
            common+=1
        assert common>=boundaries[-1], 'Non-monotone tokenizer prefix'
        boundaries.append(common)
    assert boundaries[-1]==len(text)
    groups=[]
    for start,end in zip(boundaries,boundaries[1:]):
        if start>=len(reference):
            assert text[start:end].isspace() or start==end
            groups.append(-1);continue
        overlap=[max(0,min(end,s['end'])-max(start,s['start'])) for s in spans]
        if max(overlap)>0:group=max(range(len(spans)),key=lambda i:overlap[i])
        else:group=max([i for i,s in enumerate(spans) if s['start']<=start] or [0])
        groups.append(group)
    assert set(groups)-{-1}==set(range(len(spans))), 'Source unit has no target tokens'
    return groups


def weights_for(groups):
    n=sum(g>=0 for g in groups);k=max(groups)+1
    assert set(groups)-{-1}==set(range(k))
    if k==1:return torch.ones(len(groups)+1,dtype=torch.float32)
    counts=[groups.count(i) for i in range(k)]
    # Multipliers in the ordinary mean over all T targets, including EOS.
    return torch.tensor([n/(k*counts[g]) if g>=0 else 1. for g in groups]+[1.],dtype=torch.float32)


def balanced_loss(logits,labels,weights):
    target=labels[:,1:].contiguous().view(-1)
    valid=target!=-100
    assert weights.numel()==int(valid.sum()) and torch.isfinite(weights).all() and (weights>0).all()
    z=logits[:,:-1].contiguous().view(-1,logits.shape[-1])
    losses=F.cross_entropy(z,target,ignore_index=-100,reduction='none')[valid]
    return (losses*weights.to(losses.device)).mean()


class Objective:
    def __init__(self,model,weights):
        self.labels=None;self.weights=weights;self.native=None;self.changed=False
        def replace(_module,_args,out):
            if self.labels is None:return out
            self.native=out.loss.detach()
            if torch.equal(self.weights,torch.ones_like(self.weights)):
                self.changed=False;return out
            out.loss=balanced_loss(out.logits,self.labels,self.weights);self.changed=True
            return out
        self.handle=model.register_forward_hook(replace)
    def close(self):
        self.labels=None;self.handle.remove()


def selfcheck():
    torch.manual_seed(26)
    z=torch.randn(1,7,19,requires_grad=True);labels=torch.tensor([[-100,-100,1,2,3,4,5]])
    native=F.cross_entropy(z[:,:-1].contiguous().view(-1,19),labels[:,1:].contiguous().view(-1))
    same=balanced_loss(z,labels,torch.ones(5))
    assert torch.equal(native,same)
    assert torch.equal(torch.autograd.grad(native,z,retain_graph=True)[0],torch.autograd.grad(same,z)[0])
    groups=[0,0,0,1];w=weights_for(groups)
    torch.testing.assert_close(w[:-1].sum(),torch.tensor(4.))
    assert w[-1]==1 and all(abs(float(w[:-1][torch.tensor(groups)==g].sum())-2)<1e-6 for g in (0,1))
    assert torch.equal(weights_for([0,0,0]),torch.ones(4))
    boundary=weights_for([0,0,0,1,-1]);assert boundary[-1]==1 and boundary[-2]==1
    torch.testing.assert_close(boundary.sum(),torch.tensor(6.))
    return dict(status='PASS',all_one_native_loss_gradient_exact=True,equal_group_mass=True,EOS_mass_unchanged=True,single_unit_exact=True)
