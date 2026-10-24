"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, contrastive_rows, contrastive_gradient


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
    paired=contrastive_rows(task)
    assert len(paired)==8
    for i in range(4):
        assert paired[i+4]['question']==paired[i]['question']
        assert paired[i+4]['image_path']==paired[i]['image_path']
        assert paired[i+4]['reference']==paired[i^1]['reference']
        assert paired[i+4]['reference']!=paired[i]['reference']
    x=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True)
    losses=[((x-i/3)**2).mean() for i in range(8)]
    values=torch.stack(losses)
    target=torch.logsumexp(values[:4],0)+torch.nn.functional.softplus(values[:4]-values[4:]).mean()
    expected=torch.autograd.grad(target,x,retain_graph=True)[0]
    grads=[torch.autograd.grad(y,x,retain_graph=True)[0] for y in losses]
    actual,weights=contrastive_gradient(grads,[float(y.detach()) for y in losses])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert (weights[:4]>=0).all() and (weights[4:]<=0).all()
    extreme,w=contrastive_gradient(grads,[10000.,0.,1.,2.,0.,10000.,2.,1.]);assert torch.isfinite(extreme).all()
    for bad in ([1.]*4,[float('nan')]*8):
        try:contrastive_gradient(grads,bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid loss list accepted')
    bad=copy.deepcopy(task);bad['train_rows'][1]['reference']=bad['train_rows'][0]['reference']
    bad['eval_rows'][1]['reference']=bad['train_rows'][0]['reference']
    try:contrastive_rows(bad)
    except AssertionError:pass
    else:raise AssertionError('Identical contrast answers accepted')
    return dict(status='PASS',source_only=True,heldout_rejected=True,changed_reference_rejected=True,same_case_source_answer_swaps=True,combined_objective_autograd_exact=True,signed_weights=True,extreme_stable=True,identical_answers_rejected=True,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
