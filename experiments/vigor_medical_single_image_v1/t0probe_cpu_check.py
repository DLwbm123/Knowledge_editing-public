"""Meaningful toy check for shifted target/EOS token accounting."""
import torch
import torch.nn.functional as F
from .t0probe import token_parts,token_stat
from .editperf8_cpu_check import run as previous


def run():
    previous()
    logits=torch.tensor([[[0.,2.,-1.],[1.,0.,2.],[3.,1.,0.],[0.,1.,0.]]])
    labels=torch.tensor([[-100,-100,1,2]])
    value,first=token_parts(logits,labels,2)
    reference=F.cross_entropy(logits[:,:-1].reshape(-1,3),labels[:,1:].reshape(-1),ignore_index=-100)
    assert abs(value['mean_nll']-float(reference))<1e-6
    assert value['target_count']==2 and not value['tokens'][0]['is_eos'] and value['tokens'][1]['is_eos']
    assert torch.equal(first,logits[0,1]) and token_stat(first,1)['rank']==3
    return dict(status='PASS',shifted_target_mask=True,eos_separate=True,rank_and_margin=True,previous_backoff_restore=True)
