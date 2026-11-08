"""Thirty-two clean-Base generations per process; private tensor diagnostics only."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


CFG=json.loads(Path(os.environ['DIAG_CONFIG']).read_text())
ROOT=Path(CFG['root']);ARM=os.environ['DIAG_ARM'];OUT=ROOT/ARM
CALLS=0


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def budget():
    assert time.time()<CFG['clock']['hard_end']-120 and not (ROOT/'STOP.json').exists()
    assert time.time()-json.loads((ROOT/'HEALTH.json').read_text())['time']<30
    assert CALLS<=32


def main():
    global CALLS
    OUT.mkdir();(OUT/'WORKER_ONCE').touch(exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original diagnostic clock')))
    signal.alarm(max(1,int(CFG['clock']['hard_end']-time.time()-120)))
    env=json.loads(Path(CFG['environment']).read_text());os.environ.update(env)
    os.environ.update(CFG['overrides']);parent=Path(env['PARENT_RUN']);frozen=Path(CFG['frozen_root'])
    sys.path[:0]=[str(frozen/'tools'),CFG['med150_tools'],str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
    import torch
    import numpy as np
    from types import SimpleNamespace
    from m3bench_repro.inference.llava_med import LlavaMedAdapter
    from scripts.editor_paperspec_formal import assert_authorized_device
    assert_authorized_device()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=torch.load(Path(env['RAW_MED_ROOT'])/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    if ARM=='A':
        import module_run as local
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        local.c.budget=budget
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'))
        # Match base.py's pre-loop initialization, including SDP backend flags.
        # The first executed64-call run omitted this; its deviation stays recorded.
        local.old.fp32(runtime)
        runtime.run_root=OUT/'runtime';runtime.run_root.mkdir()
        adapter=runtime.adapter;model=runtime.model
        configuration=json.loads((frozen/'BASE_CONFIGURATION.private.json').read_text())
        assert runtime.generation_config==configuration['generation']
        source=json.loads((frozen/'SOURCE_INPUTS.private.json').read_text());tasks={t['edit_id']:t for t in source['tasks']}
    else:
        adapter=LlavaMedAdapter(env['M3BENCH_MODEL_PATH'],env['M3BENCH_VISION_PATH'],device='cuda:0',load_mode='official_native');adapter.load()
        model=adapter.model
        for p in model.parameters():p.requires_grad_(False)
    model.eval()
    existing_hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
    assert existing_hooks==0,'Clean Base contains unexpected pre-existing forward hooks'
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(any(s in n.lower() for s in ['lora_','medtrace','grace','expert']) for n,_ in model.named_parameters())
    write(OUT/'CLEAN_BASE.private.json',dict(from_original_checkpoint=True,existing_forward_hooks=existing_hooks,
        loader_mode=adapter.load_mode,model_class=type(model).__name__,arm=ARM,
        environment_versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        original_parameter_count=sum(p.numel() for p in model.parameters())))
    (OUT/'tensors').mkdir();(OUT/'answers').mkdir()
    panel=json.loads((ROOT/'PANEL.private.json').read_text())['queries'];assert len(panel)==32
    for index,row in enumerate(panel):
        budget();captured={};handles=[]
        def save(name,value):
            if name not in captured:
                captured[name]=value.detach().cpu().clone() if isinstance(value,torch.Tensor) else None
        def visual(_m,_args,out):save('vision',out[0] if isinstance(out,(list,tuple)) else out)
        def projected(_m,_args,out):save('projector',out)
        def fused(_m,args,kwargs):save('fused',kwargs.get('inputs_embeds'));save('backbone_attention_mask',kwargs.get('attention_mask'))
        def position(_m,args,kwargs):save('position_ids',kwargs.get('position_ids'));save('decoder_attention_mask',kwargs.get('attention_mask'))
        def logits(_m,_args,out):save('first_logits',out.logits[:,-1,:].float())
        handles.extend([model.get_vision_tower().register_forward_hook(visual),model.get_model().mm_projector.register_forward_hook(projected),
            model.get_model().register_forward_pre_hook(fused,with_kwargs=True),
            model.get_model().layers[0].register_forward_pre_hook(position,with_kwargs=True),model.register_forward_hook(logits)])
        try:
            if ARM=='A':
                model.half()
                try:item=local.s.prepare(runtime,{'configuration_only':configuration},tasks[row['edit_id']],[row['query']])[0]
                finally:local.old.fp32(runtime)
                raw=item[2]
            else:
                from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN
                from llava.conversation import conv_templates
                from llava.mm_utils import tokenizer_image_token
                from m3bench_repro.inference.preprocessing import deterministic_process_image
                q=row['query'];marker=DEFAULT_IMAGE_TOKEN
                if model.config.mm_use_im_start_end:marker=DEFAULT_IM_START_TOKEN+marker+DEFAULT_IM_END_TOKEN
                conv=conv_templates['mistral_instruct'].copy();conv.append_message(conv.roles[0],marker+'\n'+q['question']);conv.append_message(conv.roles[1],None)
                prompt=conv.get_prompt();ids=tokenizer_image_token(prompt,adapter.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
                pixels,seed,digest=deterministic_process_image(q['image_path'],adapter.image_processor,model.config)
                assert digest==q['image_sha256'] and isinstance(pixels,torch.Tensor)
                raw=dict(prompt=prompt,input_ids=ids,images=pixels.half().cuda(),attention_mask=None,preprocessing_seed=seed)
            assert isinstance(raw['images'],torch.Tensor)
            if ARM=='A':assert raw['input_ids'][0].tolist()==row['historical_prompt_ids']
            save('pixels',raw['images']);save('prompt_ids',raw['input_ids']);save('raw_attention_mask',raw.get('attention_mask'))
            local.c.restore_rng(rng['rng']) if ARM=='A' else restore_rng(rng['rng'],torch)
            np.random.set_state(rng['numpy_rng'])
            marker=OUT/'answers'/f'{index:02d}.attempt';marker.touch(exist_ok=False)
            CALLS+=1;write(OUT/'PROGRESS.json',dict(time=time.time(),arm=ARM,generation_attempts=CALLS,completed=index))
            if ARM=='A':
                answer,_=local.s.fit.short.generate(runtime,None,raw,item[3]);tokens=list(answer.raw_token_ids);text=answer.decoded_text
            else:
                with torch.inference_mode():
                    generated=model.generate(raw['input_ids'],images=raw['images'],do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True)
                tokens=generated[0].tolist();text=adapter.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            assert 0<len(tokens)<=1024
            for name in ['pixels','prompt_ids','vision','projector','fused','first_logits']:assert name in captured and captured[name] is not None,name
            torch.save(captured,OUT/'tensors'/f'{index:02d}.pt')
            write(OUT/'answers'/f'{index:02d}.json',dict(index=index,edit_id=row['edit_id'],role=row['role'],arm=ARM,
                prompt=raw['prompt'],tokens=tokens,answer=text,EOS=tokens[-1]==adapter.tokenizer.eos_token_id,at_cap=len(tokens)==1024,
                preprocessing_seed=raw['preprocessing_seed'],parameter_dtypes=dict(__import__('collections').Counter(str(p.dtype) for p in model.parameters())),
                tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),min=float(v.min()),max=float(v.max())) if v is not None else None for k,v in captured.items()}))
        finally:
            for h in handles:h.remove()
        write(OUT/'PROGRESS.json',dict(time=time.time(),arm=ARM,generation_attempts=CALLS,completed=index+1))
    assert CALLS==32
    write(OUT/'COMPLETE.json',dict(arm=ARM,generation_attempts=CALLS,completed=32,updates=0,Judge_calls=0,time=time.time()))


def restore_rng(state,torch):
    import random
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu())
    if state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'].cpu())


if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'FAILURE.json',dict(time=time.time(),arm=ARM,generation_attempts=CALLS,error=traceback.format_exc()));raise
