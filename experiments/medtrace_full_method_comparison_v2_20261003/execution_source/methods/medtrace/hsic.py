"""LOKI normalized-CCA HIB Top-1 adaptation; no labels or evaluation inputs.

Reference: neu-spiral/LOKI bbb5819, hsic.py and loki_main.py (MIT).
We use CPU float64 linear solves instead of the author's float32 inverse.
"""
from contextlib import contextmanager
import torch
from .core import MedTraceLayerHook

CANDIDATE_LAYERS=tuple(range(1,31))


def layer_path(layer_id):
    if type(layer_id) is not int or not 0<=layer_id<32:raise ValueError('Invalid 32-layer writer binding')
    return f'model.layers.{layer_id}.mlp.down_proj'


def centered_kernel(x):
    x=x.detach().to(device='cpu',dtype=torch.float64)
    if x.ndim!=2 or len(x)<4 or x.shape[1]<1 or not torch.isfinite(x).all():raise ValueError('HSIC needs >=4 finite feature observations')
    if torch.unique(x,dim=0).shape[0]!=len(x) or float((x-x.mean(0)).square().mean())<=1e-12:raise ValueError('UNSUPPORTED: duplicate/constant observations')
    d=(x.square().sum(1)[:,None]+x.square().sum(1)[None,:]-2*x@x.T).abs()
    k=torch.exp(-d/(2*x.shape[1])) # Author sigma=1, dimension-scaled Gaussian.
    h=torch.eye(len(x),dtype=x.dtype)-torch.ones((len(x),len(x)),dtype=x.dtype)/len(x)
    return k@h


def normalized_cca(x,y):
    if len(x)!=len(y):raise ValueError('Observation axes differ')
    a,b=centered_kernel(x),centered_kernel(y)
    eye=torch.eye(len(a),dtype=a.dtype)*1e-5*len(a)
    ra=torch.linalg.solve((a+eye).T,a.T).T
    rb=torch.linalg.solve((b+eye).T,b.T).T
    value=(ra*rb.T).sum()
    if not torch.isfinite(value):raise ValueError('Degenerate HSIC solve')
    return float(value)


def select_layer(features):
    # Inputs are five question-only frozen-Base observations, never target labels.
    if set(features)!={'input','final','layers','sample_ids'} or len(features['sample_ids'])!=5 or len(set(features['sample_ids']))!=5:
        raise ValueError('Expected native plus four distinct native-only questions')
    if set(features['layers'])!=set(CANDIDATE_LAYERS):raise ValueError('Candidate set changed')
    scores={};invalid={}
    for layer,x in features['layers'].items():
        if len(x)!=5:raise ValueError('Wrong observation axis')
        try:scores[layer]=.001*normalized_cca(x,features['final'])-.001*normalized_cca(features['input'],x)
        except ValueError as e:invalid[layer]=str(e)
    if not scores:raise ValueError('UNSUPPORTED: all candidate layers degenerate')
    return dict(layer_id=max(sorted(scores),key=scores.get),scores=scores,invalid=invalid,samples=5,
        interpretation='Exploratory LOKI-HIB adaptation; correlated prompt variants, not independent facts')


def collect_features(runtime,record,fit_questions):
    from scripts.medtrace.stage18_cfact import assert_base_off
    questions=[record.question]+fit_questions
    q=record.question
    expected=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}']
    if fit_questions!=expected:raise ValueError('Only frozen native-only wrappers allowed')
    assert_base_off(runtime);values={};counts=[]
    for question in questions:
        batch=runtime.build_question_batch(record,question=question);mask=batch.attention_mask
        if mask is None:mask=torch.ones(batch.inputs_embeds.shape[:2],device=batch.inputs_embeds.device,dtype=torch.bool)
        mask=mask.bool();one={};handles=[]
        def capture(name,is_output):
            def hook(_module,args,out):
                z=(out[0] if isinstance(out,tuple) else out) if is_output else args[0]
                if z.shape[:2]!=mask.shape or z.shape[0]!=1:raise ValueError('Prefill token mask mismatch')
                one[name]=z[mask].float().mean(0).detach().cpu()
            return hook
        modules=[('input','model.layers.0',False),('final','model.layers.31',True)]+[(i,layer_path(i),False) for i in CANDIDATE_LAYERS]
        try:
            for name,path,output in modules:handles.append(runtime.get_module(path).register_forward_hook(capture(name,output)))
            with torch.inference_mode():runtime.model(inputs_embeds=batch.inputs_embeds,attention_mask=batch.attention_mask,position_ids=batch.position_ids,use_cache=False)
        finally:
            for h in handles:h.remove()
        if set(one)!={x[0] for x in modules}:raise ValueError('Missing Base features')
        for k,v in one.items():values.setdefault(k,[]).append(v)
        counts.append(dict(total=int(mask.sum()),visual=batch.image_token_end-batch.image_token_start,
            text=int(mask.sum())-(batch.image_token_end-batch.image_token_start)))
    assert_base_off(runtime)
    return dict(input=torch.stack(values['input']),final=torch.stack(values['final']),
        layers={i:torch.stack(values[i]) for i in CANDIDATE_LAYERS},sample_ids=questions),counts


@contextmanager
def bound_hook(runtime,expert,binding):
    """Attach only the selected expert's saved layer; always restore OFF on exit."""
    if not binding.get('expert_id') or not binding.get('W0_id'):raise ValueError('Missing expert/initialization identity')
    path=layer_path(binding['layer_id'])
    for module in runtime.model.modules():
        if any(isinstance(getattr(fn,'__self__',None),MedTraceLayerHook) for fn in module._forward_hooks.values()):
            raise ValueError('Another writer hook remains attached')
    hook=MedTraceLayerHook(runtime.get_module(path),expert);hook.attach()
    try:yield hook
    finally:hook.detach()
