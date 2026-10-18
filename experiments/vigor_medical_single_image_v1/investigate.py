"""Read-only repeatability probe for the historical first-step anchor."""
import json
from pathlib import Path
import torch
from . import diagnose as d
from .editor import editable_fp32,paired_batches,gradient_pair,direction,nll
from .prepare import write


def main(config):
    torch.use_deterministic_algorithms(True)
    root=Path(config['output'])
    def probe(runtime,record,index,factor,requested,historical=None):
        layer=runtime.get_module('model.layers.0.mlp.down_proj')
        pair=paired_batches(runtime,record)
        results=[]
        previous=None
        with editable_fp32(runtime.model,layer,readonly=True):
            a,b=(factor[k].to(layer.weight.device,torch.float32).clone() for k in ('a','b'))
            for i in range(4):
                torch.use_deterministic_algorithms(i>=2)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    values,grads=gradient_pair(runtime,layer,pair)
                _,geo=direction(*grads,a,b)
                row=dict(repeat=i,deterministic=i>=2,attention_backend=runtime.model.config._attn_implementation,
                    attention_ops=sorted(set(e.key for e in prof.key_averages() if 'attention' in e.key.lower())),losses=values,geometry=geo,
                    relative_E_difference=geo['E']/historical['E']-1)
                if previous is not None:
                    row['gradient_relative_l2']=[float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y)) for x,y in zip(grads,previous)]
                    row['gradient_equal']=[torch.equal(x,y) for x,y in zip(grads,previous)]
                previous=tuple(g.clone() for g in grads)
                results.append(row)
                write(root/'public/PROBE.json',dict(rows=results,torch=str(torch.__version__),
                    attention_class=type(runtime.get_module('model.layers.1.self_attn')).__name__,
                    tf32=torch.backends.cuda.matmul.allow_tf32,
                    fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    deterministic=torch.are_deterministic_algorithms_enabled()))
        raise RuntimeError('READ_ONLY_PROBE_FINISHED_NO_UPDATE')
    d.candidate=probe
    d.sum_target_factor=lambda *args:torch.load(config['existing_sum_factor'],weights_only=True)['factor']
    d.main(config)
