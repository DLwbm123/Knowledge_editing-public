"""Source-answer concept spans only; no diagnosis alternatives or evaluation labels."""
import math
import torch
import torch.nn.functional as F

class MedicalAdmissionRejected(RuntimeError):pass

def source_concepts(reference):
    concepts=tuple(part.strip() for part in reference.split(','))
    if len(concepts)!=2 or not all(concepts) or concepts[0].casefold()==concepts[1].casefold():
        raise MedicalAdmissionRejected('Frozen source answer must provide two unambiguous concepts')
    return concepts

def concept_indices(tokenizer,targets,concepts):
    ids=targets.tolist();prefix=[tokenizer.decode(ids[:i],skip_special_tokens=True) for i in range(len(ids)+1)]
    text=prefix[-1];ends=[len(x) for x in prefix]
    if any(not text.startswith(x) for x in prefix):raise MedicalAdmissionRejected('Nonmonotone token span decoding')
    out=[]
    for concept in concepts:
        if text.casefold().count(concept.casefold())!=1:raise MedicalAdmissionRejected('Ambiguous source concept span')
        start=text.casefold().index(concept.casefold());end=start+len(concept)
        indices=[i for i in range(len(ids)) if ends[i]<end and ends[i+1]>start]
        if not indices:raise MedicalAdmissionRejected('Empty concept span')
        out.append(indices)
    if set(out[0])&set(out[1]):raise MedicalAdmissionRejected('Concept spans share a token')
    return out

def contrasts(b,model,full,removed,indices):
    values=[]
    for f,r in zip(full,removed):
        z,y=b.logits(model,f);zr,yr=b.logits(model,r);assert torch.equal(y,yr)
        lp=F.log_softmax(z.double(),-1).gather(-1,y[:,None]).flatten()
        lr=F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).flatten()
        values.extend([(lp[j]-lr[j]).mean() for j in indices[len(values)//2]])
    return values

def direct_KLs(b,model,removed,anchors):
    with torch.no_grad():
        return [float(F.kl_div(F.log_softmax(b.logits(model,batch)[0].double(),-1),anchor,reduction='batchmean')) for batch,anchor in zip(removed,anchors)]

def validate_margins(values):
    margins=[float(x.detach()) if isinstance(x,torch.Tensor) else float(x) for x in values]
    if len(margins)!=10 or any(not math.isfinite(m) or m<=0 for m in margins):
        raise MedicalAdmissionRejected('All ten source evidence margins must be finite and strictly positive')
    return margins


def contrast_details(b,model,full,removed,indices):
    assert len(full)==len(removed)==len(indices)==5 and all(len(ix)==2 for ix in indices)
    values=[];complete_ce=0.
    for k,(f,r) in enumerate(zip(full,removed)):
        z,y=b.logits(model,f);zr,yr=b.logits(model,r);assert torch.equal(y,yr)
        lp=F.log_softmax(z.double(),-1).gather(-1,y[:,None]).flatten()
        lr=F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).flatten()
        complete_ce+=(.5 if k==0 else .125)*float(F.cross_entropy(z.double(),y))
        for j,ix in enumerate(indices[k]):
            a=float(lp[ix].mean());v=float(lr[ix].mean())
            # Match the frozen gate's subtraction-before-mean arithmetic exactly.
            margin=float((lp[ix]-lr[ix]).mean())
            values.append(dict(view=k,concept=j,full_logp=a,removed_logp=v,margin=margin,
                               finite=all(math.isfinite(x) for x in (a,v,margin))))
    from source_loss import EVIDENCE_MARGIN
    return dict(values=values,count=10,scalar_only=True,complete_CE=complete_ce,training_margin=EVIDENCE_MARGIN,
                training_target_attained=sum(v['finite'] and v['margin']>=EVIDENCE_MARGIN for v in values),
                evidence_hinge=sum((.5 if v['view']==0 else .125)*max(0.,EVIDENCE_MARGIN-v['margin'])/2 for v in values))
