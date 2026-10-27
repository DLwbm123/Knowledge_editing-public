"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, mean_gradient, cycle_step


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
    gs=[torch.full((7168,),float(i),dtype=torch.float64) for i in range(4)]
    assert torch.equal(mean_gradient(gs),torch.full((7168,),1.5,dtype=torch.float64))
    try:mean_gradient(gs[:2])
    except AssertionError:pass
    else:raise AssertionError('Wrong row count accepted')
    from types import SimpleNamespace
    def rms(before,grads,scalar):
        return {k:v-.1*grads[k] for k,v in before.items()},dict(step=scalar['step']+1,second_moments=[0.,0.])
    r=SimpleNamespace(mm=SimpleNamespace(canonical_rms_step=rms,canonicalize=lambda state:(state,None)))
    before={'x':torch.tensor([1.,2.],dtype=torch.float64)};grad=torch.tensor([.1,.2],dtype=torch.float64)
    after,scalar=cycle_step(r,before,grad,dict(step=0,second_moments=[0.,0.]))
    torch.testing.assert_close(after['x'],before['x']-.025*grad,rtol=0,atol=1e-15)
    assert scalar['step']==1
    state={'x':torch.tensor([0.],dtype=torch.float64)};sc=dict(step=0,second_moments=[0.,0.])
    expected=0.
    for target in (.1,.2,.3,.4):
        gradient=state['x']-target
        state,sc=cycle_step(r,state,gradient,sc);expected=expected-.025*(expected-target)
    assert sc['step']==4 and abs(float(state['x'][0])-expected)<1e-15
    return dict(quarter_step_closed_form=True,sequential_state_and_clock=True,status='PASS' ,source_only=True,heldout_rejected=True,changed_reference_rejected=True,four_source_mean=True,parameters=7168,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
