"""Fixed image intervention and native-path audit; no parameter updates."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V81_ROOT'])
MODE = os.environ['DIAGNOSTIC_MODE']
if MODE == 'reference':
    sys.path.insert(0,str(ROOT/'pinned_dependencies'))
sys.path.insert(0,str(ROOT/'tools'))
import v81 as v


def native_inputs(model, tokenizer, processor, row, image=True):
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    question = row['question']
    if image:
        marker = DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        question = marker+'\n'+question
    conv = conv_templates['mistral_instruct'].copy()
    conv.append_message(conv.roles[0],question);conv.append_message(conv.roles[1],None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
    pixels = None
    if image:
        with Image.open(row['image_path']) as im:
            assert im.mode=='RGB'
            pixels = process_images([im],processor,model.config)[0].unsqueeze(0).half().cuda()
    return dict(input_ids=ids,images=pixels,prompt=prompt)


def reference_setup():
    import torch
    import transformers
    import tokenizers
    assert transformers.__version__=='4.36.2' and tokenizers.__version__=='0.15.2'
    sys.path.insert(0,os.environ['M3BENCH_LLAVA_SOURCE'])
    from llava.model.builder import load_pretrained_model
    from llava.model import LlavaMistralForCausalLM
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    assert not getattr(LlavaMistralForCausalLM,'_m3bench_transformers_compat_installed',False)
    import subprocess
    gpu=os.environ['GPU'];observed=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert observed[0]==v.read(ROOT/'HARDWARE.json')[gpu] and int(observed[1])>=45*1024
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False
    disable_torch_init()
    original=CLIPVisionTower.__init__
    def local(instance,_name,*args,**kwargs):original(instance,os.environ['M3BENCH_VISION_PATH'],*args,**kwargs)
    CLIPVisionTower.__init__=local
    try:tokenizer,model,processor,_=load_pretrained_model(os.environ['M3BENCH_MODEL_PATH'],None,get_model_name_from_path(os.environ['M3BENCH_MODEL_PATH']),device='cuda')
    finally:CLIPVisionTower.__init__=original
    model.eval();model.requires_grad_(False)
    def count(*_):
        v.budget();v.counts['forwards']+=1
        if not v.generating:v.counts['non_generation_forwards']+=1
    model.register_forward_pre_hook(count)
    argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()
    assert not any(s in argv for s in ['wangbomin','Knowledge_editing','medtrace'])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines()
    own=[s for s in apps if s.split(',')[0].strip()==str(os.getpid())]
    assert own and all(s.split(',')[1].strip()==observed[0] for s in own)
    v.write(ROOT/'PROCESS_AUDIT_reference_0.json',dict(argv=argv,gpu_processes=own,neutral=True))
    return torch,model,tokenizer,processor


class Observer:
    def __init__(self,torch,model):
        self.torch=torch;self.model=model;self.reset()
        self.handles=[model.get_vision_tower().register_forward_hook(self.vision),
            model.get_model().mm_projector.register_forward_hook(self.projector),
            model.register_forward_pre_hook(self.before,with_kwargs=True),
            model.register_forward_hook(self.after,with_kwargs=True)]
    def reset(self):
        self.vectors={};self.calls=dict(vision=0,projector=0);self.embeds=None;self.logits=None
    def vision(self,m,a,out):
        self.calls['vision']+=1;self.vectors['vision']=out.detach().float().cpu()
    def projector(self,m,a,out):
        self.calls['projector']+=1;self.vectors['projector']=out.detach().float().cpu()
    def before(self,m,args,kwargs):
        if self.embeds is None:
            assert kwargs.get('inputs_embeds') is not None,'Missing actual language-model input embeddings'
            self.embeds=kwargs['inputs_embeds'].detach().float().cpu()
    def after(self,m,args,kwargs,out):
        if self.logits is None:self.logits=out.logits[0,-1].detach().float().cpu()
    def report(self,ids,image):
        t=self.torch;tokens=ids[0].tolist();markers=tokens.count(-200)
        out=dict(calls=dict(self.calls),image_markers=markers,prompt_length=len(tokens),language_prefix_length=self.embeds.shape[1],
            finite_logits=bool(t.isfinite(self.logits).all()),first_token=int(self.logits.argmax()))
        assert out['finite_logits']
        if image:
            assert markers==1 and self.calls==dict(vision=1,projector=1)
            for key,value in self.vectors.items():
                assert bool(t.isfinite(value).all())
                out[key]=dict(shape=list(value.shape),norm=float(value.norm()))
            projected=self.vectors['projector'];start=tokens.index(-200);length=projected.shape[1]
            assert self.embeds.shape[1]==len(tokens)-1+length
            difference=(self.embeds[:,start:start+length]-projected).abs().max()
            out['injected_visual_block_max_abs_difference']=float(difference)
            assert difference==0,'Projected visual features were not injected unchanged'
        else:
            assert markers==0 and self.calls==dict(vision=0,projector=0)
            assert self.embeds.shape[1]==len(tokens)
        return out


def main():
    with (ROOT/f'WORKER_ONCE_{MODE}').open('x'):pass
    v.allocations=dict(v.limits['counts'])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original diagnostic deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.budget()
    if MODE=='current':
        import route_common as c
        v.setup();torch=v.torch;model=v.runtime.model;tokenizer=v.runtime.adapter.tokenizer;processor=v.runtime.adapter.image_processor
        routes=['C','N']
    else:
        torch,model,tokenizer,processor=reference_setup();routes=['R']
    import transformers,tokenizers
    v.write(ROOT/f'ENVIRONMENT_{MODE}.json',dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        forward_compat_installed=bool(getattr(type(model),'_m3bench_transformers_compat_installed',False)),
        model_class=type(model).__name__,image_aspect_ratio=getattr(model.config,'image_aspect_ratio',None),
        mm_vision_select_layer=getattr(model.config,'mm_vision_select_layer',None),generation_config=model.generation_config.to_dict()))
    cases=v.read(ROOT/'CASES_FROZEN.json');assert len(cases)==8
    prior_inputs={};all_logits={};pairs={}
    if MODE=='reference':prior_inputs=torch.load(ROOT/'C_INPUTS.pt',map_location='cpu',weights_only=True)
    observer=Observer(torch,model)
    for route in routes:
        if route!='C':model.half()
        route_logits={};feature_pairs={};pixel_pairs={};audits={}
        for index,row in enumerate(cases):
            key=str(row['qid']);group=row['source_group'];observer.reset();v.budget()
            dest=ROOT/'outputs'/route/(key+'.json');assert not dest.exists()
            v.write(ROOT/'attempts'/route/(key+'.json'),dict(time=time.time(),qid=row['qid']))
            if route=='C':
                item=v.prepare({},[row],False)[0];raw=item[2]
                prior_inputs[key]=dict(ids=raw['input_ids'].cpu(),pixels=raw['images'].detach().half().cpu())
                answer=c.generate(None,item);tokens=list(answer.raw_token_ids);decoded=answer.decoded_text
            else:
                raw=native_inputs(model,tokenizer,processor,row,True)
                assert torch.equal(raw['input_ids'].cpu(),prior_inputs[key]['ids']),'Prompt tokens differ from current path'
                pixels_match=torch.equal(raw['images'].cpu(),prior_inputs[key]['pixels'])
                # Record a preprocessing mismatch, but retain the full fixed matrix.
                audits[key]=dict(prompt_equal=True,pixels_equal=pixels_match,pixel_max_abs_difference=float((raw['images'].cpu()-prior_inputs[key]['pixels']).abs().max()))
                v.generating=True;v.counts['evaluation_generations']+=1
                with torch.inference_mode():out=model.generate(raw['input_ids'],images=raw['images'],do_sample=False,temperature=0,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)
                v.generating=False;tokens=out[0].cpu().tolist();decoded=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            trace=observer.report(raw['input_ids'],True)
            trace['processed_pixels']=dict(shape=list(raw['images'].shape),finite=bool(torch.isfinite(raw['images']).all()),norm=float(raw['images'].float().norm()))
            route_logits[key]=observer.logits
            pixel_pairs.setdefault(group,[]).append(raw['images'].detach().float().cpu())
            feature_pairs.setdefault(group,[]).append(observer.vectors['projector'])
            prior=v.read(row['previous_output']) if route=='C' else None
            v.write(dest,dict(query=row,route=route,condition='image',raw_answer=decoded,raw_token_ids=tokens,
                EOS=tokens[-1]==tokenizer.eos_token_id,at_cap=len(tokens)>=1024,trace=trace,
                V98_exact_tokens=(tokens==prior['raw_token_ids']) if prior else None,
                native_input_parity=audits.get(key)))
            v.progress(stage='diagnose',route=route,completed=index+1)
            if route=='C':del item
        for group in pixel_pairs:
            pixels=pixel_pairs[group];features=feature_pairs[group]
            assert len(pixels)==len(features)==2
            pairs[route+'_'+group]=dict(pixel_max_abs_difference=float((pixels[0]-pixels[1]).abs().max()),
                pixel_relative_L2=float((pixels[0]-pixels[1]).norm()/pixels[0].norm()),
                projector_max_abs_difference=float((features[0]-features[1]).abs().max()),
                projector_relative_L2=float((features[0]-features[1]).norm()/features[0].norm()))
        del pixel_pairs,feature_pairs
        for row in cases[::2]:
            group=row['source_group'];observer.reset();v.budget();raw=native_inputs(model,tokenizer,processor,row,False)
            dest=ROOT/'outputs'/route/(group+'_noimage.json');assert not dest.exists()
            v.write(ROOT/'attempts'/route/(group+'_noimage.json'),dict(time=time.time(),group=group))
            v.generating=True;v.counts['evaluation_generations']+=1
            kwargs=dict(images=None,do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True)
            if route=='C':kwargs.update(attention_mask=torch.ones_like(raw['input_ids']),pad_token_id=tokenizer.eos_token_id)
            with torch.inference_mode():out=model.generate(raw['input_ids'],**kwargs)
            v.generating=False;tokens=out[0].cpu().tolist();decoded=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            trace=observer.report(raw['input_ids'],False);route_logits[group+'_noimage']=observer.logits
            v.write(dest,dict(query=dict(question=row['question'],source_group=group),route=route,condition='noimage',raw_answer=decoded,raw_token_ids=tokens,
                EOS=tokens[-1]==tokenizer.eos_token_id,at_cap=len(tokens)>=1024,trace=trace))
        torch.save(route_logits,ROOT/(route+'_LOGITS.pt'));all_logits[route]=route_logits
        v.write(ROOT/(route+'_INPUT_PARITY.json'),audits)
        if route=='C':torch.save(prior_inputs,ROOT/'C_INPUTS.pt')
        v.progress(stage='route_complete',route=route,outputs=12)
    v.write(ROOT/f'PAIR_FEATURE_DIFFERENCES_{MODE}.json',pairs)
    assert v.counts['backwards']==v.counts['updates']==0
    v.write(ROOT/f'COMPLETE_{MODE}.json',dict(status='COMPLETE',counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as exc:
        code=1;v.write(ROOT/f'FAILURE_{MODE}.json',dict(error=repr(exc),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        if not (ROOT/'STOP').exists():v.write(ROOT/'STOP',dict(reason='DIAGNOSTIC_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code)
        v.write(ROOT/f'EXIT_{MODE}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
