"""Source-local adaptations; no medical-grounding claim and no query-answer inputs."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def initialize(module):
    # LLaVA loader globally disables Linear/LayerNorm.reset_parameters.
    # Initialize new modules explicitly; never touch pretrained base modules.
    import math
    for layer in module.modules():
        if isinstance(layer,nn.Linear):
            nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5))
            if layer.bias is not None:
                bound=1/math.sqrt(layer.weight.shape[1]);nn.init.uniform_(layer.bias,-bound,bound)
        elif isinstance(layer,nn.LayerNorm):
            if layer.weight is not None:nn.init.ones_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)


def radius(d_positive, d_negative, alpha=.2, definition='official_code'):
    if not 0 <= alpha <= 1: raise ValueError('alpha outside [0,1]')
    if definition == 'official_code': return alpha*d_positive+(1-alpha)*d_negative
    if definition == 'paper_equation3': return (1-alpha)*d_positive+alpha*d_negative
    raise ValueError('Unknown radius_definition')


class RadiusRouter(nn.Module):
    def __init__(self, anchors, positives, negatives, alpha=.2):
        super().__init__()
        if anchors.ndim != 2 or positives.shape != anchors.shape or negatives.shape != anchors.shape:
            raise ValueError('One declared positive and negative proxy per anchor required')
        self.register_buffer('anchors', anchors.detach().clone())
        self.register_buffer('radii', radius((positives-anchors).norm(dim=-1),
                                            (negatives-anchors).norm(dim=-1), alpha))
    def forward(self, query):
        distances=torch.cdist(query, self.anchors)
        return (distances <= self.radii).any(-1), distances


class LDKERouter(nn.Module):
    """Paper eq8–18, source-local adaptation; hyperparameters explicitly in config."""
    def __init__(self, width, bottleneck=32, embedding=64):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.up=nn.Linear(width,bottleneck,bias=False)
        self.down=nn.Linear(bottleneck,width,bias=False)
        self.emb=nn.Linear(width,embedding,bias=False)
        self.cls=nn.Linear(width,1,bias=False)
        initialize(self)
    def forward(self,x):
        r=x+self.down(F.relu(self.up(self.norm(x))))
        return F.normalize(self.emb(r),dim=-1),self.cls(r).squeeze(-1)
    def objective(self,anchor,positive,negative,margin=.1):
        e,le=self(anchor);p,lp=self(positive);n,ln=self(negative)
        ep=(e*p).sum(-1);en=(e*n).sum(-1);pn=(p*n).sum(-1)
        trip=F.relu(en-ep+margin)+F.relu(pn-ep+margin)
        absolute=F.relu(1-ep)+F.relu(en)
        logits=torch.cat((le,lp,ln)); labels=torch.cat((torch.ones_like(le),torch.ones_like(lp),torch.zeros_like(ln)))
        return trip.mean()+absolute.mean()+F.binary_cross_entropy_with_logits(logits,labels)
    def route(self,query,anchors):
        q,_=self(query);a,_=self(anchors);score=(q@a.T).max(-1).values
        return score >= .5,score


@dataclass(frozen=True)
class Evidence:
    visual: torch.Tensor
    global_context: torch.Tensor
    relation: torch.Tensor
    availability: bool
    provenance: str


class EvidenceProvider:
    """Frozen query-conditioned projected-token pooling; WEAK_EVIDENCE only."""
    @staticmethod
    def compress(x, width=64):
        if x.shape[-1] % width: raise ValueError('Feature width not divisible')
        return F.normalize(x.reshape(*x.shape[:-1],width,-1).mean(-1),dim=-1)
    def __call__(self,visual_tokens,text_tokens,visual_mask=None,text_mask=None):
        if visual_tokens.ndim != 3 or text_tokens.ndim != 3: raise ValueError('B,T,D tensors required')
        if visual_tokens.shape[0] != text_tokens.shape[0]: raise ValueError('Batch mismatch')
        vm=torch.ones(visual_tokens.shape[:2],dtype=torch.bool,device=visual_tokens.device) if visual_mask is None else visual_mask.bool()
        tm=torch.ones(text_tokens.shape[:2],dtype=torch.bool,device=text_tokens.device) if text_mask is None else text_mask.bool()
        if not vm.any(-1).all() or not tm.any(-1).all(): raise ValueError('Evidence unavailable')
        q=(text_tokens*tm.unsqueeze(-1)).sum(1)/tm.sum(1,keepdim=True)
        scores=(F.normalize(visual_tokens,dim=-1)*F.normalize(q,dim=-1).unsqueeze(1)).sum(-1)
        weights=scores.masked_fill(~vm,float('-inf')).softmax(-1)
        z=(visual_tokens*weights.unsqueeze(-1)).sum(1)
        glob=(visual_tokens*vm.unsqueeze(-1)).sum(1)/vm.sum(1,keepdim=True)
        return Evidence(self.compress(z).detach(),self.compress(glob).detach(),self.compress(q).detach(),True,'WEAK_EVIDENCE: frozen projected visual tokens; query text; no ROI')


def pair_features(query,anchor):
    return torch.cat((query.visual,query.relation,anchor.visual.expand_as(query.visual),anchor.relation.expand_as(query.relation)),dim=-1)


class EvidenceGate(nn.Module):
    def __init__(self,width=256):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,16),nn.Tanh(),nn.Linear(16,1));initialize(self)
    def forward(self,x): return self.net(x).squeeze(-1)
    def route(self,x): return (self(x)>=0).to(x.dtype)


class Conditioner(nn.Module):
    def __init__(self,width=256,bond=4,rho=.5):
        super().__init__();self.rho=rho
        self.net=nn.Sequential(nn.Linear(width,16),nn.Tanh(),nn.Linear(16,bond))
        initialize(self)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return 1+self.rho*torch.tanh(self.net(x))


class ConditionedExpert(nn.Module):
    """Central rank modulation via TT contractions, never materializes a dense matrix."""
    def __init__(self,writer):
        super().__init__();self.writer=writer;self.gate=None;self.gamma=None
    @contextmanager
    def request(self,gate,gamma):
        if self.gate is not None: raise RuntimeError('Nested or leaked condition')
        if gate.ndim!=1 or gamma.ndim!=2 or gamma.shape!=(gate.shape[0],4): raise ValueError('Condition dimensions')
        if not torch.isfinite(gamma).all() or not ((gamma>=.5)&(gamma<=1.5)).all(): raise ValueError('Unbounded gamma')
        if not ((gate==0)|(gate==1)).all(): raise ValueError('Binary fixed gate required')
        self.gate,self.gamma=gate,gamma
        try: yield self
        finally:self.gate=self.gamma=None
    def residual(self,x):
        if self.gate is None: raise RuntimeError('Condition missing')
        if x.shape[0]!=len(self.gate): raise ValueError('Condition batch mismatch')
        b,a=self.writer.factors();x=x.to(a.dtype)
        x=x/(x.square().mean(-1,keepdim=True).sqrt()+self.writer.epsilon)
        projected=x@a.T
        shape=(len(self.gate),)+(1,)*(x.ndim-2)
        return ((projected*self.gamma.reshape(*shape,4))@b.T)*self.gate.reshape(*shape,1)


def localize(pre,post,norm,unembedding,target_token,first_candidate):
    """Source-authorized first-token logit lens; residual before/after FFN."""
    if set(pre)!=set(post):raise ValueError('Missing layer states')
    scores={}
    with torch.no_grad():
        for layer in sorted(pre):
            if layer<first_candidate:continue
            left=unembedding(norm(pre[layer])).log_softmax(-1)[...,target_token]
            right=unembedding(norm(post[layer])).log_softmax(-1)[...,target_token]
            scores[layer]=float((right-left).mean())
    if not scores: raise ValueError('No candidate layers')
    return min(scores,key=lambda k:(-scores[k],k)),scores


def evidence_increment_loss(edited_pos,base_pos,edited_neg,base_neg,*,verified_medical_pair,margin=.1):
    if not verified_medical_pair: raise ValueError('True medical-negative support required; proxies forbidden')
    return F.relu(margin-(edited_pos-base_pos)+(edited_neg-base_neg))


def target_score(logits,labels,eos):
    y=labels[:,1:];mask=(y!=-100)&(y!=eos)
    if not mask.any():raise ValueError('No non-EOS answer tokens')
    return logits[:,:-1].log_softmax(-1)[mask].gather(-1,y[mask].unsqueeze(-1)).mean()
