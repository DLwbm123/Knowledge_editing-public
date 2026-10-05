"""Small dense oracle and parity against the exact pinned upstream functions."""
import ast
import os
from pathlib import Path
import torch
from .core import projection, optimizer_class, combine, factors


def main():
    upstream=Path(os.environ['UPSTREAM_DIR'])
    source=ast.parse((upstream/'easyeditor/models/crispedit/utils.py').read_text())
    names={'get_rank_and_threshold_by_energy_ratio','calculate_projection_cache_with_kfac'}
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==2
    reference={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-reference>','exec'),reference)
    torch.manual_seed(69);dtype=torch.float64
    x=torch.randn(8,5,dtype=dtype);y=torch.randn(8,3,dtype=dtype)
    a=x.T@x;b=y.T@y
    ours,_=projection(a,b,.9)
    theirs=reference['calculate_projection_cache_with_kfac'](a,b,.9)
    assert torch.equal(ours['M'],theirs['M'])
    grad=torch.randn(3,5,dtype=dtype)
    factored=ours['Ub']@((ours['Ub'].T@grad@ours['Ua'])*ours['M'].T)@ours['Ua'].T
    # Column-major vectorization makes vec(B X A^T)=(A kron B)vec(X).
    basis=torch.kron(ours['Ua'],ours['Ub'])
    dense=basis@torch.diag(ours['M'].flatten().to(dtype))@basis.T
    expected=(dense@grad.T.contiguous().flatten()).view(5,3).T
    torch.testing.assert_close(factored,expected,rtol=1e-10,atol=1e-11)
    p=torch.nn.Parameter(torch.randn(3,5,dtype=dtype));q=torch.nn.Parameter(p.detach().clone())
    opt=optimizer_class(upstream)([p],{p:ours},lr=5e-4)
    ordinary=torch.optim.Adam([q],lr=5e-4)
    for _ in range(3):
        p.grad=grad.clone();q.grad=factored.clone();opt.step();ordinary.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    old_m=opt.state[p]['exp_avg'].clone();opt.reset_cache({p:ours})
    torch.testing.assert_close(opt.state[p]['exp_avg'],
        ours['Ub']@((ours['Ub'].T@old_m@ours['Ua'])*ours['M'].T)@ours['Ua'].T)
    z=combine({1:dict(A=a,B=b,N=7)},{1:dict(A=2*a,B=3*b,N=2)})[1]
    torch.testing.assert_close(z['A'],11*a/9);torch.testing.assert_close(z['B'],13*b/9)
    assert z['N']==9
    # Equal eigenvalues exercise the upstream strict threshold convention.
    tied,_=projection(torch.eye(5,dtype=dtype),torch.eye(3,dtype=dtype),.9)
    assert not tied['M'].any()
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module()
            block=torch.nn.Module();block.mlp=torch.nn.Module()
            block.mlp.down_proj=torch.nn.Linear(5,3,bias=False)
            self.model.layers=torch.nn.ModuleList([block]);self.requires_grad_(False)
        def forward(self,inputs_embeds):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.model.layers[0].mlp.down_proj(inputs_embeds))
    tiny=Tiny();inputs=torch.randn(1,4,5);labels=torch.tensor([[-100,-100,2,1]])
    with torch.no_grad():
        z=tiny(inputs).logits[0,1:3];derivative=z.softmax(-1)
        derivative[torch.arange(2),torch.tensor([2,1])]-=1
    v=factors(tiny,[(dict(inputs_embeds=inputs),labels)],[0])[0]
    torch.testing.assert_close(v['A'],inputs[0,1:3].T@inputs[0,1:3]/2)
    torch.testing.assert_close(v['B'],derivative.T@derivative/2)
    assert v['N']==2 and not tiny.model.layers[0].mlp.down_proj._forward_hooks
    print('PASS: dense projection, upstream optimizer, weighted history, ties, masked summed-CE factors')


if __name__=='__main__':main()
