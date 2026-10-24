"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, normalized_gradient


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
    grads=[torch.tensor(v,dtype=torch.float64) for v in ([2.,0.],[-1.,0.],[0.,1.],[0.,1.])]
    actual,w=normalized_gradient(grads,[0.]*4)
    torch.testing.assert_close(actual,torch.tensor([0.,(5/16)**.5],dtype=torch.float64),rtol=1e-12,atol=1e-12)
    raw=sum(g*v for g,v in zip(grads,w));torch.testing.assert_close(actual.norm(),raw.norm(),rtol=1e-12,atol=1e-12)
    same=[torch.tensor(v,dtype=torch.float64) for v in ([1.,0.],[0.,1.],[-1.,0.],[0.,-1.])]
    equal,w=normalized_gradient(same,[1.,2.,3.,4.])
    torch.testing.assert_close(equal,sum(g*v for g,v in zip(same,w)),rtol=1e-12,atol=1e-12)
    cancellation,w=normalized_gradient(grads[:2]*2,[0.]*4)
    assert torch.equal(cancellation,torch.zeros_like(cancellation))
    zero,w=normalized_gradient([torch.zeros(2,dtype=torch.float64)]*4,[0.]*4)
    assert torch.equal(zero,torch.zeros_like(zero))
    extreme,w=normalized_gradient(grads,[10000.,0.,-10000.,1.])
    torch.testing.assert_close(extreme,grads[0],rtol=0,atol=0)
    for bad in ([1.,2.], [1.,float('nan'),2.,3.]):
        try:normalized_gradient(grads,bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid losses accepted')
    return dict(status='PASS',unequal_norm_closed_form=True,aggregate_norm_preserved_when_direction_nonzero=True,equal_norm_identity=True,exact_cancellation_zero=True,zero_gradient_safe=True,source_only=True,heldout_rejected=True,changed_reference_rejected=True,extreme_loss_stable=True,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
