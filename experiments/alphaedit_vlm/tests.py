"""Small meaningful algebra/gradient/expanded-position checks; CPU only."""
import torch
from .core import single_update, answer_nll, prefix_inputs


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    u = torch.linalg.qr(torch.randn(11, 5, dtype=dtype)).Q
    k, r = torch.randn(11, dtype=dtype), torch.randn(7, dtype=dtype)
    p = u @ u.T
    actual = single_update(k, r, u, 10)
    official = torch.linalg.solve(p @ torch.outer(k,k) + 10*torch.eye(11,dtype=dtype),
                                  p @ torch.outer(k,r)).T
    assert torch.allclose(actual, official, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual @ (torch.eye(11,dtype=dtype)-p), torch.zeros(7,11,dtype=dtype),atol=1e-12)
    assert not single_update(k,r,u[:,:0],10).any()
    logits = torch.randn(1,9,13,requires_grad=True)
    labels = torch.tensor([[-100]*6+[2,3,4]])
    loss = answer_nll(logits,labels);loss.backward()
    assert not logits.grad[0,:5].any() and logits.grad[0,5:8].abs().sum()>0
    assert not logits.grad[0,8].any()
    prepared = ({'inputs_embeds':torch.zeros(1,9,8),'attention_mask':torch.ones(1,9),
                 'position_ids':None},labels)
    prefix, position = prefix_inputs(prepared)
    assert prefix['inputs_embeds'].shape[1]==6 and position==5
    print('PASS: official-solve equivalence, protected-space orthogonality, empty nullspace, causal answer/EOS loss, expanded prefix boundary')


if __name__=='__main__':
    main()
