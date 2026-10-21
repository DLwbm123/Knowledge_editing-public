"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, mean_gradient, swapped_rows, distribution_shift


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
    expected=torch.autograd.grad(torch.stack(losses).mean(),x,retain_graph=True)[0]
    gs=[torch.autograd.grad(y,x,retain_graph=True)[0] for y in losses]
    actual,w=mean_gradient(gs,[float(y.detach()) for y in losses])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    equal,we=mean_gradient(gs,[2.]*4)
    torch.testing.assert_close(equal,torch.stack(gs).mean(0),rtol=1e-12,atol=1e-12)
    extreme,wx=mean_gradient(gs,[10000.,0.,-10000.,1.])
    assert torch.isfinite(extreme).all() and torch.equal(wx,torch.full((4,),.25,dtype=torch.float64))
    for bad in ([1.,2.], [1.,float('nan'),2.,3.]):
        try:mean_gradient(gs,bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid losses accepted')
    tasks=[dict(key=str(i),eval_rows=[dict(query_id=str(i),question='q',reference='r',image_path='image'+str(i))]) for i in range(4)]
    for i,t in enumerate(tasks):
        swapped=swapped_rows(t,tasks)
        assert swapped[0]['image_path']=='image'+str((i+1)%4)
        assert all(swapped[0][k]==t['eval_rows'][0][k] for k in ('query_id','question','reference'))
        assert tasks[i]['eval_rows'][0]['image_path']=='image'+str(i)
    z=torch.tensor([[0.,1.],[1000.,-1000.]],dtype=torch.float64)
    identical=distribution_shift(z,z);assert identical['tv_mean']==0 and abs(identical['js_nats_mean'])<1e-12 and identical['argmax_changed']==0
    flipped=distribution_shift(z,z.flip(-1));reverse=distribution_shift(z.flip(-1),z)
    assert flipped['tv_mean']>0 and flipped['js_nats_mean']>0 and flipped['argmax_changed']==2
    assert abs(flipped['tv_mean']-reverse['tv_mean'])<1e-12 and abs(flipped['js_nats_mean']-reverse['js_nats_mean'])<1e-12
    return dict(distribution_null_symmetry_extreme=True,swap_only_image_fixed_cycle=True,status='PASS',source_only=True,heldout_rejected=True,changed_reference_rejected=True,mean_gradient_exact=True,equal_loss_mean=True,extreme_loss_stable=True,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
