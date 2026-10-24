"""Fixed token-position masks; no learned gate and no answer/reference access."""
from contextlib import contextmanager
import torch

ARMS=('ALL_TOKENS','PREFILL_ONLY','VISUAL_ONLY')

def layout(raw,expanded):
    ids=raw['input_ids'];assert ids.ndim==2 and ids.shape[0]==1
    positions=torch.where(ids[0]==-200)[0].tolist();assert len(positions)<=1
    n=expanded[4].shape[1]
    if positions:
        start=positions[0];count=n-ids.shape[1]+1
        assert count>0 and start+count<=n
    else:start=0;count=0;assert n==ids.shape[1]
    return dict(prefix_length=n,visual_start=start,visual_count=count,raw_length=ids.shape[1])

class TokenHook:
    def __init__(self,module,expert,arm):
        assert arm in ARMS
        self.module=module;self.expert=expert;self.arm=arm
        self.handle=None;self.mask=None;self.generating=False;self.current=None
        self.trace=[];self.last_generation_trace=()
    def attach(self):
        assert self.handle is None
        self.handle=self.module.register_forward_hook(self.forward)
    def detach(self):
        if self.handle is not None:self.handle.remove()
        self.handle=None;self.mask=None;self.generating=False
    def set_layout(self,value):
        assert not self.generating
        self.current=dict(value)
    def position_mask(self,n,device,offset=0):
        assert self.current is not None
        p=torch.arange(n,device=device)+offset
        if self.arm=='ALL_TOKENS':return torch.ones((1,n),dtype=torch.bool,device=device)
        if self.arm=='PREFILL_ONLY':return (p<self.current['prefix_length']).unsqueeze(0)
        start=self.current['visual_start'];return ((p>=start)&(p<start+self.current['visual_count'])).unsqueeze(0)
    def teacher(self,labels,attention=None):
        assert not self.generating and labels.shape[0]==1
        first=int(torch.where(labels[0]!=-100)[0][0]);assert first==self.current['prefix_length']
        a=self.current['visual_start'];b=a+self.current['visual_count'];assert (labels[:,a:b]==-100).all()
        self.mask=self.position_mask(labels.shape[1],labels.device)
        if attention is not None:self.mask=self.mask&attention.bool()
    @contextmanager
    def generation_request(self,attention=None):
        assert not self.generating and self.current is not None
        self.generating=True;self.trace=[];self.next_position=0
        try:yield
        finally:self.last_generation_trace=tuple(self.trace);self.generating=False;self.mask=None
    def forward(self,module,args,output):
        if not self.generating and self.mask is None:return output
        x=args[0];assert x.ndim==3 and x.shape[0]==1
        if self.generating:
            offset=self.next_position if x.shape[1]==1 and self.trace else 0
            mask=self.position_mask(x.shape[1],x.device,offset)
            self.next_position=offset+x.shape[1]
        else:mask=self.mask
        assert mask.shape==x.shape[:-1]
        if mask.any():
            residual=self.expert.residual(x).to(output.dtype)
            result=output+residual*mask.to(output.device).unsqueeze(-1)
            norm=float(residual[mask].detach().float().norm()) if self.generating else None
        else:result=output;norm=0.
        if self.generating:
            active=torch.where(mask[0])[0]
            self.trace.append(dict(sequence_length=x.shape[1],absolute_offset=offset,active_predictor_count=active.numel(),first_active_predictor=int(active[0]) if active.numel() else None,active_residual_norm=norm))
        return result
