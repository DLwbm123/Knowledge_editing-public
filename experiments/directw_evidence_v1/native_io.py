"""Native model IO only; clean reload must not import the editor or its state."""
import sys
import resource
import time
import unicodedata
from collections import Counter
import torch
from PIL import Image

WEIGHT = 'model.layers.21.mlp.down_proj.weight'


def minimum_target_margin(logits, labels):
    """Worst target-versus-best-other logit margin in row zero, including EOS."""
    if logits.ndim!=3 or labels.shape!=logits.shape[:2]:
        raise ValueError('aligned token labels required')
    mask=labels[0,1:]!=-100
    if not mask.any():raise ValueError('empty target token sequence')
    targets=labels[0,1:][mask];z=logits[0,:-1][mask].float()
    best,indices=z.topk(2,dim=-1)
    other=torch.where(indices[:,0]==targets,best[:,1],best[:,0])
    return (z.gather(1,targets[:,None])[:,0]-other).min()


def answer_metrics(prediction, reference):
    """Conservative lexical diagnostics, not semantic or clinical grading."""
    def normalize(text):
        return ' '.join(''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                               if not unicodedata.category(c).startswith('P')).split())
    p,r=normalize(prediction),normalize(reference)
    if not r:raise ValueError('empty normalized reference')
    pt,rt=p.split(),r.split();overlap=sum((Counter(pt)&Counter(rt)).values())
    return dict(normalized_exact_match=p==r,token_F1=2*overlap/(len(pt)+len(rt)))


def generation_snapshot(model, tokenizer, generation, reference, max_new_tokens):
    if not isinstance(max_new_tokens,int) or not 1<=max_new_tokens<=64:
        raise ValueError('generation diagnostic token cap must be 1..64')
    with torch.no_grad():
        out=model.generate(**dict(generation,max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,output_scores=True))
    # Native inputs_embeds generation can include a BOS prefix. The number of
    # decoding scores identifies the generated suffix without guessing a prefix.
    n=len(out.scores)
    if not 0<n<=max_new_tokens or out.sequences.shape[0]!=1 or out.sequences.shape[1]<n:
        raise RuntimeError('unexpected native generation output')
    tokens=out.sequences[0,-n:].detach().cpu().tolist()
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    eos=tokenizer.eos_token_id in tokens
    return dict(tokens=tokens,text=text,reference=reference,metrics=dict(
        **answer_metrics(text,reference),generated_tokens=n,eos_reached=eos,
        hit_token_cap=n==max_new_tokens and not eos))


def prefix_target_margin(model, generation, labels, trace=None):
    """Deployed minimum margin on independently supplied native target prefixes."""
    if labels.ndim!=2 or labels.shape[0]!=1:
        raise ValueError('single aligned target sequence required')
    targets=labels[0,1:][labels[0,1:]!=-100]
    if not 1<=len(targets)<=64:raise ValueError('target sequence must contain1..64tokens')
    margins=[]
    with torch.no_grad():
        for j,target in enumerate(targets):
            prefix=torch.cat((generation['inputs'],targets[:j][None,:]),dim=1)
            out=model.generate(**dict(generation,inputs=prefix,max_new_tokens=1,
                               return_dict_in_generate=True,output_scores=True))
            if len(out.scores)!=1:raise RuntimeError('one-step native scores required')
            z=out.scores[0][0].float()
            if not torch.isfinite(z).all():raise RuntimeError('nonfinite native prefix scores')
            values,indices=z.topk(2)
            other=torch.where(indices[0]==target,values[1],values[0])
            margins.append(z[target]-other)
    values=torch.stack(margins)
    if trace is not None:trace.append(values.detach().cpu().tolist())
    return values.min()


def token_path_audit(model, prepared, batch_logits, batch_labels):
    """Read-only comparison; prefix decoding scores never decide edit acceptance."""
    inputs,labels,generation,_=prepared
    mask=labels[0,1:]!=-100;targets=labels[0,1:][mask]
    batch_mask=batch_labels[0,1:]!=-100
    if not 1<=len(targets)<=64 or not torch.equal(targets,batch_labels[0,1:][batch_mask]):
        raise ValueError('bounded aligned target sequence required')
    def stats(z,target):
        z=z.float()
        if not torch.isfinite(z).all():raise RuntimeError('nonfinite token-path logits')
        values,indices=z.topk(2)
        other=values[1] if indices[0]==target else values[0]
        return dict(margin=float(z[target]-other),argmax_is_target=bool(z.argmax()==target),
                    target_ties=int((z==z[target]).sum()))
    records=[]
    with torch.no_grad():
        single=model(**inputs).logits[0,:-1][mask]
        batch=batch_logits[0,:-1][batch_mask]
        for j,target in enumerate(targets):
            prefix=torch.cat((generation['inputs'],targets[:j][None,:]),dim=1)
            kwargs=dict(generation,inputs=prefix,max_new_tokens=1,
                        return_dict_in_generate=True,output_scores=True)
            native=model.generate(**kwargs)
            masked=model.generate(**dict(kwargs,attention_mask=torch.ones_like(prefix,dtype=torch.bool)))
            if len(native.scores)!=1 or len(masked.scores)!=1:
                raise RuntimeError('one-step generation diagnostic required')
            z=native.scores[0][0];explicit=masked.scores[0][0]
            records.append(dict(position=j,batch_teacher=stats(batch[j],target),
                single_teacher=stats(single[j],target),prefix_generate=stats(z,target),
                prefix_generate_explicit_mask=stats(explicit,target),
                native_generated_target=bool(native.sequences[0,-1]==target),
                mask_generated_same=bool(native.sequences[0,-1]==masked.sequences[0,-1]),
                batch_single_max_logit_difference=float((batch[j]-single[j]).abs().max()),
                single_prefix_max_score_difference=float((single[j].float()-z.float()).abs().max()),
                implicit_explicit_mask_max_score_difference=float((z.float()-explicit.float()).abs().max())))
    return dict(target_positions=len(targets),positions=records,
        scope='Independent native one-step generation conditioned on each forced target prefix; not a claim about cached continuations after divergence')


def deployment_dtype(config):
    name=config.get('native_deployment_dtype','bfloat16')
    if name not in ('bfloat16','float32'):
        raise ValueError('unsupported native deployment precision')
    return getattr(torch,name)


def load(config):
    dtype=deployment_dtype(config)
    sys.path.insert(0, config['native_source'])
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
    cfg=LlavaMistralConfig.from_pretrained(config['model_path'],local_files_only=True)
    cfg.mm_vision_tower=config['vision_path']
    cfg.use_cache=False
    model=LlavaMistralForCausalLM.from_pretrained(config['model_path'],config=cfg,
        torch_dtype=torch.bfloat16,attn_implementation=config['attention_backend'],local_files_only=True)
    tower=model.get_vision_tower()
    if not tower.is_loaded:tower.load_model()
    # Preserve the original native mixed-precision loader by default. The control
    # promotes its loaded values, including any native vision-tower precision.
    model=(model.to('cuda') if dtype==torch.bfloat16 else model.to(device='cuda',dtype=dtype)).eval()
    model.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(config['model_path'],use_fast=False,local_files_only=True)
    return model,tokenizer,tower.image_processor


def prepare(model,tokenizer,processor,row,question_suffix=''):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token,process_images
    conv=conv_templates['mistral_instruct'].copy()
    conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+row['question']+question_suffix)
    conv.append_message(conv.roles[1],None)
    prefix=conv.get_prompt()
    prefix_ids=tokenizer_image_token(prefix,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
    # Prefix is fixed; concatenate explicit answer tokens and EOS without retokenizing it.
    answer_ids=tokenizer(row['answer'],add_special_tokens=False).input_ids
    if not answer_ids:raise ValueError('empty original answer tokens')
    ids=torch.cat((prefix_ids,torch.tensor(answer_ids+[tokenizer.eos_token_id])))[None,:].cuda()
    labels=ids.clone();labels[:,:len(prefix_ids)]=-100
    image=Image.open(row['image_path']).convert('RGB')
    images=process_images([image],processor,model.config)
    dtype=model.get_input_embeddings().weight.dtype
    images=[x.to('cuda',dtype=dtype) for x in images] if isinstance(images,list) else images.to('cuda',dtype=dtype)
    mask=torch.ones_like(ids,dtype=torch.bool)
    with torch.no_grad():
        expanded=model.prepare_inputs_labels_for_multimodal(ids,None,mask,None,labels,images,[image.size])
    input_ids,positions,padding,past,embeds,expanded_labels=expanded
    if past is not None or embeds is None:raise ValueError('unexpected multimodal expansion')
    inputs=dict(inputs_embeds=embeds.detach(),attention_mask=padding,position_ids=positions,use_cache=False,return_dict=True)
    generation=dict(inputs=prefix_ids[None,:].cuda(),images=images,image_sizes=[image.size],do_sample=False,
                    num_beams=1,max_new_tokens=8,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    template=dict(template_name='mistral_instruct',prefix=prefix,input_length=ids.shape[1],prefix_length=len(prefix_ids),
                  answer_length=len(answer_ids),expanded_length=embeds.shape[1],image_token_positions=(ids==IMAGE_TOKEN_INDEX).nonzero().tolist(),
                  expanded_labels=expanded_labels.detach().cpu().tolist(),padding_mask=padding.detach().cpu().tolist(),
                  eos_policy='EXCLUDE_CONTENT_SCORE',answer_tokenization='fixed native prefix + explicit answer tokens + EOS')
    return inputs,expanded_labels,generation,template


def clean_reload(config):
    """New process, ordinary native class only; no editor import, hooks or routing."""
    started=time.monotonic()
    model,tokenizer,processor=load(config)
    selected=dict(model.named_parameters())[WEIGHT]
    edited=torch.load(config['edited_matrix'],map_location='cpu',weights_only=True)
    if edited.shape!=selected.shape or edited.dtype!=selected.dtype:raise ValueError('export shape/dtype mismatch')
    with torch.no_grad():selected.copy_(edited.to(selected.device))
    inputs,labels,generation,template=prepare(model,tokenizer,processor,config['smoke_row'])
    with torch.no_grad():
        logits=model(**inputs).logits.float()
        tokens=model.generate(**generation)
    diagnostic=(generation_snapshot(model,tokenizer,generation,config['smoke_row']['answer'],
                config['generation_max_new_tokens']) if config.get('generation_max_new_tokens') else None)
    hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks) for m in model.modules())
    prohibited=[name for name,_ in model.named_parameters() if any(t in name.lower() for t in ('lora','adapter','tucker','router'))]
    torch.save(dict(logits=logits.cpu(),tokens=tokens.cpu(),generation_diagnostic=diagnostic,hooks=hooks,prohibited_parameter_names=prohibited,
                    editor_imported=any(k.endswith('.editor') for k in sys.modules),
                    wall_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),config['reload_result'])
