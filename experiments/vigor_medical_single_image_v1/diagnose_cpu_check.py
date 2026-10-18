"""Check length-weighted factors and isolated one-step state restoration."""
import torch
from .cpu_check import Runtime, record
from .editor import calibrate
from .diagnose import candidate, length_audit, sum_target_factor, CASES


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    runtime=Runtime(torch.float16)
    expand=runtime._expand_multimodal
    def two_lengths(**kwargs):
        x,mask,pos,labels=expand(**kwargs)
        if int(kwargs['raw_input_ids'][0,-1])==3:
            labels[:,3]=3
        return x,mask,pos,labels
    runtime._expand_multimodal=two_lengths
    def forbidden(*args,**kwargs):
        raise AssertionError('Generation forbidden')
    runtime.generate=forbidden
    records=[record('train-1',.8,'1'),record('train-2',1.2,'3')]
    name='model.layers.0.mlp.down_proj'
    binding=dict(model='synthetic')
    calibration=calibrate(runtime,records,[name],train_ids=[r.record_id for r in records],
        excluded_ids=['trial'],binding=binding)
    lengths,positions=length_audit(runtime,records)
    assert lengths==[1,2]
    original=calibration['factors'][name]
    other=sum_target_factor(runtime,records,original,lengths,positions)
    assert torch.equal(original['a'],other['a'])
    ratio=other['b']/original['b']
    assert (ratio>=.999).all() and (ratio<=4.001).all() and float(ratio.max()-ratio.min())>.01
    base=runtime.model.layer.weight.detach().clone()
    for factor,d in [(original,.02),(original,.005),(other,.005)]:
        result=candidate(runtime,record('trial'),0,factor,d)
        assert torch.equal(base,runtime.model.layer.weight)
        assert result['fp32_update']['normalized_null_residual']<1e-4
        assert not runtime.model.layer._forward_hooks and not runtime.model.layer._forward_pre_hooks
    assert len(CASES)==4 and CASES[1][3]==CASES[2][3]==CASES[3][3]
    return dict(status='PASS',lengths=[1,2],factor_change='length_squared_weighting',
        base_restoration=True,generation_calls=0,judge_calls=0)
