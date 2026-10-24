"""Three fixed SOURCE writers; only target module and token mask vary."""
from contextlib import contextmanager
import io
import torch
from torch import nn

ARMS = ('A', 'B', 'C')
TARGETS = {'A': 'model.layers.21.mlp.down_proj', 'B': 'model.layers.21.mlp.down_proj',
           'C': 'model.layers.31.mlp.up_proj'}


class TT(nn.Module):
    def __init__(self, seed, up=False):
        super().__init__()
        shapes = [(1,128,8),(8,112,4),(4,64,8),(8,64,1)] if up else [(1,64,8),(8,64,4),(4,112,8),(8,128,1)]
        self.d_in, self.d_out = (4096,14336) if up else (14336,4096)
        self.epsilon = 1e-6
        rng = torch.Generator().manual_seed(seed)
        self.G1 = nn.Parameter(torch.zeros(shapes[0]))
        for name, shape in zip(('G2','G3','G4'), shapes[1:]):
            x = torch.randn(shape[1]*shape[2], shape[0], generator=rng)
            q, r = torch.linalg.qr(x, mode='reduced')
            assert torch.isfinite(r).all() and (torch.diag(r).abs() >= 1e-8).all()
            q = q * torch.where(torch.diag(r)<0, -torch.ones_like(torch.diag(r)), torch.ones_like(torch.diag(r)))
            setattr(self, name, nn.Parameter(q.T.reshape(shape).contiguous()))
        assert sum(v.numel() for v in self.parameters()) == 7168

    def factors(self):
        return (torch.einsum('ia,ajb->ijb', self.G1[0], self.G2).reshape(self.d_out,4),
                torch.einsum('aib,bj->aij', self.G3,self.G4[:,:,0]).reshape(4,self.d_in))

    def residual(self, activation):
        b, a = self.factors()
        x = activation.to(a.dtype)
        x = x / (x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)
        return (x @ a.T) @ b.T


class Hook:
    def __init__(self, module, expert, all_tokens):
        self.module, self.expert, self.all_tokens = module, expert, all_tokens
        self.mask = None
        self.generating = False
        self.attention = None
        self.trace = []
        self.last_generation_trace = ()
        self.handle = None

    def attach(self):
        assert self.handle is None
        self.handle = self.module.register_forward_hook(self.forward)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.mask = None
        self.generating = False

    def teacher(self, labels, attention):
        self.generating = False
        if self.all_tokens:
            self.mask = torch.ones_like(labels,dtype=torch.bool) if attention is None else attention.bool()
        else:
            self.mask = torch.zeros_like(labels,dtype=torch.bool)
            self.mask[:,:-1] = labels[:,1:] != -100

    @contextmanager
    def generation_request(self, attention=None):
        assert not self.generating
        self.generating, self.attention, self.trace = True, attention, []
        try:
            yield
        finally:
            self.last_generation_trace = tuple(self.trace)
            self.generating, self.mask, self.attention = False, None, None

    def forward(self, module, args, output):
        if not self.generating and self.mask is None:
            return output
        x = args[0]
        if self.generating:
            mask = torch.zeros(x.shape[:-1],dtype=torch.bool,device=x.device)
            if self.all_tokens:
                mask[:] = True
                if self.attention is not None and x.shape[-2] > 1:
                    assert self.attention.shape == mask.shape
                    mask = self.attention.bool()
            else:
                mask[:,-1] = True
        else:
            mask = self.mask
        assert mask.shape == x.shape[:-1] and mask.any()
        residual = self.expert.residual(x).to(output.dtype)
        if self.generating:
            active = torch.where(mask[0])[0]
            self.trace.append(dict(sequence_length=x.shape[-2],first_active_predictor=int(active[0]),
                                   active_predictor_count=int(active.numel()),
                                   active_residual_norm=float(residual[mask].detach().float().norm())))
        return output + residual * mask.to(output.device).unsqueeze(-1)


def snapshot(expert):
    return {k:v.detach().clone() for k,v in expert.state_dict().items()}


def next_step(r, before, gradient, scalar):
    # Reuse the V22 SOURCE proposal and canonicalization, including float operations.
    from path_v18 import proposal, construct
    keys = list(before)
    raw, updated, direction = proposal(r,before,gradient,scalar,keys)
    return construct(r,before,raw,direction,direction,1.,keys), updated


def map_change(before, after, math):
    b0,a0 = math.factors({k:v.detach().cpu().double() for k,v in before.items()})
    b1,a1 = math.factors({k:v.detach().cpu().double() for k,v in after.items()})
    b,a = torch.cat((b1,-b0),1),torch.cat((a1,a0),0)
    squared = ((b.T@b)*(a@a.T)).sum()
    assert torch.isfinite(squared) and squared >= -1e-10
    return max(0.,float(squared))**.5


def selfcheck(r, seed):
    original = r.p.tr.TT4(seed,8,8)
    current = TT(seed)
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in current.state_dict().items())
    torch.manual_seed(24)
    for expert in (current, TT(seed,True)):
        assert expert.G1.count_nonzero() == 0
        x = torch.randn(2,expert.d_in)
        assert expert.residual(x).count_nonzero() == 0
        with torch.no_grad():expert.G1.normal_(0,.02)
        before = snapshot(expert)
        expected = expert.residual(x).detach()
        expert.load_state_dict(r.mm.canonicalize(before)[0])
        torch.testing.assert_close(expert.residual(x),expected,rtol=1e-4,atol=1e-6)
        data=io.BytesIO();torch.save(snapshot(expert),data);data.seek(0)
        data.seek(0);loaded=torch.load(data,weights_only=True)
        assert all(torch.equal(v,loaded[k]) for k,v in expert.state_dict().items())
        scalar=dict(step=0,second_moments=[0.,0.])
        for _ in range(3):
            loss=expert.residual(x).square().mean()+expert.residual(x).mean()
            gs=torch.autograd.grad(loss,tuple(expert.parameters()))
            assert all(torch.isfinite(g).all() for g in gs)
            state,scalar=next_step(r,snapshot(expert),torch.cat([g.detach().cpu().double().flatten() for g in gs]),scalar)
            expert.load_state_dict(state)
    # Independent original TT class versus new A, three exact optimizer steps.
    current,original=TT(seed),r.p.tr.TT4(seed,8,8)
    initial=r.mm.canonicalize(snapshot(original))[0]
    current.load_state_dict(initial);original.load_state_dict(initial)
    scalars=[dict(step=0,second_moments=[0.,0.]) for _ in range(2)]
    x=torch.randn(3,14336);target=torch.randn(3,4096)
    for _ in range(3):
        for i,e in enumerate((current,original)):
            loss=(e.residual(x)-target).square().mean()
            gs=torch.autograd.grad(loss,tuple(e.parameters()))
            state,scalars[i]=next_step(r,snapshot(e),torch.cat([g.detach().cpu().double().flatten() for g in gs]),scalars[i]);e.load_state_dict(state)
        assert scalars[0]==scalars[1]
        assert all(torch.equal(v,original.state_dict()[k]) for k,v in current.state_dict().items())
    class FakeLayer(nn.Module):
        def forward(self,x):return x[...,:4096]
    layer=FakeLayer();x=torch.randn(1,6,14336);labels=torch.tensor([[-100,-100,-100,3,4,-100]])
    legacy=r.q.scoped.replay.MedTraceLayerHook(layer,current)
    revised=Hook(layer,current,False)
    answers=[]
    for hook in (legacy,revised):
        hook.attach()
        try:
            if hook is legacy:hook.set_teacher_routing(labels)
            else:hook.teacher(labels,torch.ones_like(labels))
            teacher=layer(x).detach()
            with hook.generation_request():
                prefill=layer(x[:,:3]).detach();decode=layer(x[:,3:4]).detach()
            answers.append((teacher,prefill,decode))
        finally:hook.detach()
    assert all(torch.equal(a,b) for a,b in zip(*answers))
    revised=Hook(layer,current,True);revised.teacher(labels,torch.tensor([[1,1,1,1,1,0]]))
    assert revised.mask.tolist()==[[True,True,True,True,True,False]]
    # Real Mistral CPU architecture, tiny random model: last-layer mask equivalence.
    from transformers import MistralConfig, MistralForCausalLM
    config=MistralConfig(vocab_size=37,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                         num_attention_heads=2,num_key_value_heads=2,head_dim=8,attention_dropout=0.,
                         max_position_embeddings=32)
    config._attn_implementation='eager'
    model=MistralForCausalLM(config).double().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    class Small(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.randn(32,16,dtype=torch.float64)*.03)
        def residual(self,x):return x@self.weight.T
    expert=Small();ids=torch.tensor([[1,3,5,7,9,2]]);labels=ids.clone();labels[:,:3]=-100
    attention=torch.ones_like(ids);logits=[];grads=[];cached=[]
    for all_tokens in (False,True):
        hook=Hook(model.model.layers[-1].mlp.up_proj,expert,all_tokens);hook.attach()
        try:
            hook.teacher(labels,attention);out=model(input_ids=ids,attention_mask=attention,labels=labels,use_cache=False)
            logits.append(out.logits[:,2:-1].detach());grads.append(torch.autograd.grad(out.loss,expert.weight)[0])
            with torch.no_grad(),hook.generation_request():
                out=model(input_ids=ids[:,:3],use_cache=True)
                first=out.logits[:,-1].detach()
                out=model(input_ids=ids[:,3:4],past_key_values=out.past_key_values,use_cache=True)
                cached.append((first,out.logits[:,-1].detach()))
        finally:hook.detach()
    for left,right in [(logits[0],logits[1]),(grads[0],grads[1]),*zip(cached[0],cached[1])]:
        torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-10)
    # Same masks in a non-final layer can influence later attention.
    predictions=[]
    for all_tokens in (False,True):
        hook=Hook(model.model.layers[0].mlp.up_proj,expert,all_tokens);hook.attach()
        try:
            with torch.no_grad(),hook.generation_request():predictions.append(model(input_ids=ids[:,:3]).logits[:,-1])
        finally:hook.detach()
    assert not torch.allclose(*predictions,rtol=1e-8,atol=1e-10)
    return dict(status='PASS',model_calls=0,real_pretrained_model_calls=0,
                synthetic_random_Mistral_only=True,parameters=7168,effective_rank=4,
                original_A_three_step_exact=True,canonical_and_save_load=True,
                last_layer_logits_gradients_KV_equivalent=True,nonfinal_scope_can_differ=True)
