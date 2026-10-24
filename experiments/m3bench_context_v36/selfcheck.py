"""Source admission rejects heldout rows and checks four-row averaging."""
import copy
import json
from protocol import source_rows, mean_gradient


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
    return dict(status='PASS',source_only=True,heldout_rejected=True,changed_reference_rejected=True,four_source_mean=True,parameters=7168,model_calls=0)

if __name__=='__main__':print(json.dumps(main()))
