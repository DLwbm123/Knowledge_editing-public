import torch
import torch.nn.functional as F
from runtime import b, ARM
SOFT = ARM in ('SH10','SH11')

def loss(runtime_model,full,removed,anchors,slot):
    value=0.;ce=0.;scope=0.
    for k in (0,1+(slot-1)%4):
        z,y=b.logits(runtime_model,full[k]);part=.5*F.cross_entropy(z.double(),y)
        value=value+part;ce+=float(part.detach())
        if removed and SOFT:
            zr,yr=b.logits(runtime_model,removed[k]);assert torch.equal(y,yr)
            part=.5*F.kl_div(F.log_softmax(zr.double(),-1),anchors[k],reduction='batchmean')
            value=value+part;scope+=float(part.detach())
    return value,ce,scope


def scope_anchors(model,removed):
    with torch.no_grad():return [b.logits(model,batch)[0].double().softmax(-1).detach() for batch in removed]
