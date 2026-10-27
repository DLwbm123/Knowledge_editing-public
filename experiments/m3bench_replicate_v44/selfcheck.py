"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, adaptive_gradient, paired_gradient


def main():
    import torch
    rows=[dict(query_id=str(i),question='question '+str(i),reference='answer '+str(i),image_path='private',eval_group='SOURCE' if i<2 else 'HELDOUT') for i in range(3)]
    task=dict(train_rows=copy.deepcopy(rows[:2]),eval_rows=rows)
    result=source_rows(task)
    assert len(result)==4 and [r['reference'] for r in result]==['answer 0','answer 1']*2
    assert all('answer' not in r['question'] for r in result)
    for mode in ('heldout','changed_reference'):
        bad=copy.deepcopy(task)
        if mode=='heldout':bad['train_rows'][1]=bad['eval_rows'][2]
        else:bad['train_rows'][0]['reference']='leak'
        try:source_rows(bad)
        except AssertionError:pass
        else:raise AssertionError('Bad training source accepted')
    x=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True)
    losses=[((x-i)**2).mean() for i in range(4)]
    expected=torch.autograd.grad(torch.logsumexp(torch.stack(losses),0),x,retain_graph=True)[0]
    gs=[torch.autograd.grad(y,x,retain_graph=True)[0] for y in losses]
    actual,w=adaptive_gradient(gs,[float(y.detach()) for y in losses])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    equal,we=adaptive_gradient(gs,[2.]*4)
    torch.testing.assert_close(equal,torch.stack(gs).mean(0),rtol=1e-12,atol=1e-12)
    extreme,wx=adaptive_gradient(gs,[10000.,0.,-10000.,1.])
    assert torch.isfinite(extreme).all() and wx[0]==1
    for bad in ([1.,2.], [1.,float('nan'),2.,3.]):
        try:adaptive_gradient(gs,bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid losses accepted')
    for arm in ('COVER2','ADAPT2'):
        actual,w=paired_gradient(gs,[float(y.detach()) for y in losses],arm)
        expected=torch.stack(gs).mean(0) if arm=='COVER2' else adaptive_gradient(gs,[float(y.detach()) for y in losses])[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    return dict(paired_aggregators=True,status='PASS' ,source_only=True,heldout_rejected=True,changed_reference_rejected=True,logsumexp_gradient_exact=True,equal_loss_mean=True,extreme_loss_stable=True,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
