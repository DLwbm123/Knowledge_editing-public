"""Verify the locked Lingshu architecture and run its ten-case GPU preflight."""
import json
import os
from pathlib import Path
import sys
import time

import torch


def read(path):
    return json.loads(Path(path).read_text())


def layer_check(model, index):
    layers = [(n,m) for n,m in model.named_modules() if n.endswith('.layers.18.mlp.down_proj')]
    name,layer = layers[0]
    assert len(layers)==1 and f'{name}.weight' in index
    assert (layer.in_features,layer.out_features)==(18944,3584)
    return layer


def main():
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from importlib.metadata import version
    from methods.medtrace.core import MedTraceLayerHook
    from experiments.med_mdu_v2_admission_v1.qwen_expert import QwenExpert

    mode,raw,root = sys.argv[1:]
    root = Path(root)
    model_path = Path(raw)
    config = AutoConfig.from_pretrained(model_path,local_files_only=True)
    index = read(model_path/'model.safetensors.index.json')['weight_map']
    info = dict(transformers=version('transformers'), layers=config.num_hidden_layers,
                hidden=config.hidden_size, intermediate=config.intermediate_size, parameters=8256)
    assert (info['transformers'],info['layers'],info['hidden'],info['intermediate'])==('4.51.3',28,3584,18944)
    expert = QwenExpert(20261127)
    assert sum(p.numel() for p in expert.parameters())==8256
    assert all(p.dtype==torch.float32 for p in expert.parameters())
    if mode=='meta':
        with torch.device('meta'):
            model = Qwen2_5_VLForConditionalGeneration(config)
        layer_check(model,index)
        print(json.dumps(info))
        return
    if mode!='smoke' or torch.cuda.device_count()!=1:
        raise ValueError('L0 smoke needs precisely one pinned device')
    cfg = read(root/'launch.json')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(1)
    device = 'cuda:0'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True,device_map={'':device}).eval().requires_grad_(False)
    layer = layer_check(model,index)
    expert.to(device)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256*28*28,
                                             max_pixels=1280*28*28,local_files_only=True)
    hook = MedTraceLayerHook(layer,expert)
    from PIL import Image
    rows = read(root/'examples.json')
    assert len(rows)==10 and len({x['image_sha256'] for x in rows})==10
    receipt = []
    stops = model.generation_config.eos_token_id
    stops = [stops] if isinstance(stops,int) else stops

    with torch.no_grad():
        for i,row in enumerate(rows):
            if time.time()>=cfg['deadline']-20:
                raise TimeoutError('L0 smoke deadline')
            path = Path(row['image_path'])
            if not path.is_file():
                path = Path(os.environ['FROZEN_IMAGE_ROOT'])/row['image_path'].split('/DataP/',1)[1]
            with Image.open(path) as source:
                image = source.convert('RGB')
            msg = [dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=row['question'])])]
            prompt = processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            inputs = processor(text=[prompt],images=[image],return_tensors='pt').to(device)
            if i==0:
                n = inputs['input_ids'].shape[1]
                eos = processor.tokenizer.eos_token_id
                tokens = processor.tokenizer(row['answer'],add_special_tokens=False).input_ids+[eos]
                target = torch.tensor([tokens],device=device)
                labels = torch.cat((torch.full_like(inputs['input_ids'],-100),target),-1)
                sequence = torch.cat((inputs['input_ids'],target),-1)
                teacher = dict(inputs,input_ids=sequence,attention_mask=torch.ones_like(sequence))
                unedited = model(**teacher,use_cache=False).logits
                hook.attach()
                try:
                    hook.set_teacher_routing(labels)
                    modified = model(**teacher,use_cache=False).logits
                    assert torch.equal(unedited,modified)
                    # The first active logit predicts answer token 0 at prompt position n-1.
                    expected = torch.where(labels[0,1:]!=-100)[0]
                    assert int(expected[0])==n-1 and int(expected[-1])==sequence.shape[1]-2
                    with torch.no_grad():
                        expert.G1.fill_(.01)
                    changed = model(**teacher,use_cache=False).logits
                    assert torch.equal(changed[:,:n-1],unedited[:,:n-1])
                    assert not torch.equal(changed[:,n-1],unedited[:,n-1])
                finally:
                    expert.G1.zero_()
                    hook.detach()
                del unedited,modified,changed
            result = model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=1024,
                                    repetition_penalty=1.05,temperature=None,top_p=None,top_k=None,use_cache=True)
            suffix = result[0,inputs['input_ids'].shape[1]:].tolist()
            text = processor.decode(suffix,skip_special_tokens=True)
            receipt.append(dict(tokens=len(suffix),eos=bool(suffix and suffix[-1] in stops),nonempty=bool(text.strip())))
            print('CHECK',i+1,len(suffix),receipt[-1]['eos'],flush=True)
    assert all(x['eos'] and x['nonempty'] and 0<x['tokens']<1024 for x in receipt)
    with (root/'done.json').open('x') as f:
        json.dump(dict(info,smoke=receipt,zero_residual_identical=True,first_answer_mask_valid=True),f)


if __name__=='__main__':
    main()
