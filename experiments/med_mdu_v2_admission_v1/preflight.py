"""Teacher-forced preflight on a small private Stage-0 panel."""
import json
import os
from pathlib import Path
import sys

import torch


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def main(root):
    outputs = root.parent
    plan = read(root/'protocol.json')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(plan['gpu_limit_bytes'] /
        torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    parent = outputs/'medtrace-curriculum146-v1-20261017/run/private'
    stages = [outputs/name/'run/MDU' for name in (
        'med-mdu-boundary-v1-20261105', 'med-medinit-101-v1-20261104', 'med-mdu150-v1-20261105'
    )]
    sys.path[:0] = [str(p) for p in (
        stages[2]/'tools', outputs/'med-fixed-depth-v1-20261030/run/tools',
        parent/'tools', parent/'source', Path(os.environ['M3BENCH_LLAVA_SOURCE']))]
    from llava.model import LlavaMistralForCausalLM
    from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything
    from methods.medtrace.core import MedTraceLayerHook
    from scripts.editor_paperspec_formal import assert_official_llavamed_source, load_runtime
    from structures import TT4
    from train_pair import state_digest

    tasks = read(root/'panel.json')
    assert len(tasks)==4 and sum(len(x['rows']) for x in tasks)==40
    bank = load(outputs/'med150-v1-20261027/run/retained/MED150.pt')
    for item in tasks:
        src = item['source']
        k = src['edit_id']
        model = bank['states'][k]
        paths = [p for path in stages if (p:=path/'weights'/f"{item['owner']}_MDU.pt").is_file()]
        assert len(paths)==1
        mdu = load(paths[0])
        receipts={r['owner']:r for f in paths[0].parent.parent.glob('TRAIN_COMPLETE_*.json') for r in read(f)['results']}
        assert state_digest(model)==mdu['initial_digest']
        assert state_digest(mdu['state'])==receipts[item['owner']]['final_digest']
        item['states']={'MED':model,'MDU':mdu['state']}
        item['parity']=mdu['curve'][0]['ce']

    assert_official_llavamed_source()
    original = LlavaMistralForCausalLM.from_pretrained
    attribute = LlavaMistralForCausalLM.__dict__.get('from_pretrained')
    @classmethod
    def load_on_cpu(cls, name, *args, **kwargs):
        kwargs.update(low_cpu_mem_usage=True, device_map={'':'cpu'})
        return original(name,*args,**kwargs)
    LlavaMistralForCausalLM.from_pretrained = load_on_cpu
    try:
        runtime=load_runtime(parent/'cpu_gate','cuda:0')
    finally:
        if attribute is None:
            del LlavaMistralForCausalLM.from_pretrained
        else:
            LlavaMistralForCausalLM.from_pretrained=attribute
    runtime.model.eval().float().to('cuda:0')
    assert not any(p.requires_grad for p in runtime.model.parameters())
    runtime.run_root=root
    layer=runtime.get_module('model.layers.21.mlp.down_proj')
    prefill=(runtime.model.model.embed_tokens,runtime.model.model.mm_projector,
             runtime.model.get_vision_tower())
    eos=runtime.adapter.tokenizer.eos_token_id
    activations=[]
    def capture(_layer,args):
        activations.append(args[0][0,position].detach())
    tap=layer.register_forward_pre_hook(capture)
    calls=0
    print('MODEL_READY',flush=True)
    try:
        with (root/'metrics.jsonl').open('x',buffering=1) as out, torch.no_grad():
            for item in tasks:
                src=item['source']
                native=src['source_rows'][0]
                image=Path(native['image_path'])
                assert image.is_file()
                for index,(role,q,reference) in enumerate(item['rows']):
                    seed_everything(src['seed'])
                    r=EditorRecord(src['edit_id'],src['dataset'],q,reference,q,image,
                                   native['original_image_path'],item['owner'],'STAGE0_REFERENCE')
                    for m in prefill:m.half()
                    try:
                        batch=runtime.build_edit_batch(r)
                    finally:
                        for m in prefill:m.float()
                    assert batch.image_sha256==native['image_sha256']
                    assert batch.inputs_embeds.dtype==torch.float16
                    decode={k:v.float() if isinstance(v,torch.Tensor) and v.is_floating_point() else v
                            for k,v in batch.forward_kwargs().items() if k!='labels'}
                    labels=batch.labels[0,1:]
                    mask=labels!=-100
                    assert mask.any().item()
                    tokens=labels[mask]
                    pos=mask.nonzero()[0].item()
                    position=batch.key_token_index
                    assert pos==position and (tokens!=eos).any().item()
                    activations.clear()
                    data={}
                    for arm in ('Base','MED','MDU'):
                        hook=None
                        if arm!='Base':
                            expert=TT4(src['seed'],8,8)
                            expert.load_state_dict(item['states'][arm])
                            expert.to(layer.weight.device).eval().requires_grad_(False)
                            hook=MedTraceLayerHook(layer,expert)
                            hook.attach()
                            hook.set_teacher_routing(batch.labels)
                        try:
                            result=runtime.model(**decode)
                            calls+=1
                            assert calls<=plan['max_forwards']
                            logp=result.logits[0,:-1][mask].float().log_softmax(-1)
                            losses=-logp.gather(1,tokens[:,None]).squeeze(1)
                            active=tokens!=eos
                            vals={'nll':float(losses[active].mean()),'with_eos':float(losses.mean())}
                            if arm=='Base':
                                teacher=logp[active]
                            else:
                                b,a=expert.factors()
                                x=activations[-1].float()
                                x=x/(x.square().mean().sqrt()+expert.epsilon)
                                vals.update(projection=float((x@torch.linalg.qr(a.T).Q).norm()),
                                            residual=float(((x@a.T)@b.T).norm()),
                                            kl=float((teacher.exp()*(teacher-logp[active])).sum(-1).mean()))
                            data[arm]=vals
                            del result,logp
                        finally:
                            if hook:hook.detach()
                    assert len(activations)==3 and all(torch.equal(activations[0],x) for x in activations[1:])
                    if index in (0,1):
                        assert abs(data['MED']['with_eos']-item['parity'][index])<=5e-4
                    print(json.dumps({'owner':item['owner'],'view':role,'metrics':data},allow_nan=False),file=out)
    finally:
        tap.remove()
    assert calls==120
    with (root/'done.json').open('x') as f:
        json.dump({'events':40,'forwards':calls,'training':0,'generation':0,'score':0},f)


if __name__=='__main__':
    main(Path.cwd())
