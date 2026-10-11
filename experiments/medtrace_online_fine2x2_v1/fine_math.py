"""Fixed online source acceptance gates; no optimization or threshold search."""
import torch


def detail(values):
    return torch.tensor([-v['NLL_change'] for v in values],dtype=torch.float64),[torch.tensor(v['target_margin'],dtype=torch.float64) for v in values]


def constraints(before,raw):
    bg,bm=detail(before);rg,rm=detail(raw)
    assert bg.shape==rg.shape and len(bm)==len(rm)
    masks=[(a>0)|(b>0) for a,b in zip(bm,rm)]
    required_margin=[.9*torch.maximum(a,b).clamp_min(0) for a,b in zip(bm,rm)]
    return bg+.9*(rg-bg).clamp_min(0),required_margin,masks


def check(values,required,arm):
    gain,margins=detail(values);required_gain,required_margin,masks=required
    assert arm in ('N','T','NT')
    nll=bool(torch.isfinite(gain).all() and torch.all(gain>=required_gain-1e-8))
    token=all(bool(torch.isfinite(m).all() and torch.all(m[mask]>=r[mask]-1e-8)) for m,r,mask in zip(margins,required_margin,masks))
    return (nll if arm=='N' else token if arm=='T' else nll and token),dict(nll=nll,token=token)


def selfcheck():
    def row(g,m):return dict(NLL_change=-g,target_margin=m)
    before=[row(1.,[1.,-1.]),row(1.,[.5,-.5])];raw=[row(1.2,[1.2,.2]),row(.9,[.2,-.2])]
    req=constraints(before,raw)
    assert torch.allclose(req[0],torch.tensor([1.18,1.],dtype=torch.float64))
    assert not check(raw,req,'N')[0] and not check(raw,req,'T')[0]
    good=[row(1.18,[1.08,.18]),row(1.,[.45,-10.])]
    assert all(check(good,req,a)[0] for a in ('N','T','NT'))
    bad_token=[row(1.18,[.5,.18]),row(1.,[.45,-10.])]
    assert check(bad_token,req,'N')[0] and not check(bad_token,req,'T')[0] and not check(bad_token,req,'NT')[0]
    bad_nll=[row(1.17,[1.08,.18]),row(1.01,[.45,-10.])]
    assert not check(bad_nll,req,'N')[0] and check(bad_nll,req,'T')[0]
    return dict(status='PASS',no_group_compensation=True,independent_factor_gates=True,raw_not_assumed_feasible=True,wrong_tokens_excluded=True)

if __name__=='__main__':print(selfcheck())
