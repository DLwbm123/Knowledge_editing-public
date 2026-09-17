"""CPU formula, masking, multi-layer binding and restore acceptance."""
import io
from types import SimpleNamespace
import torch
from methods.medtrace.hsic import normalized_cca,select_layer,CANDIDATE_LAYERS,bound_hook,centered_kernel
from methods.medtrace import AsymmetricCPExpert

def main():
    torch.manual_seed(19001);x=torch.randn(5,8,dtype=torch.float64);y=torch.randn(5,6,dtype=torch.float64)
    a,b=centered_kernel(x),centered_kernel(y);eye=torch.eye(5,dtype=torch.float64)*5e-5
    expected=torch.sum((a@torch.inverse(a+eye))*(b@torch.inverse(b+eye)).T)
    assert abs(normalized_cca(x,y)-float(expected))<1e-10
    assert abs(normalized_cca(x,y)-normalized_cca(x[[2,0,4,1,3]],y[[2,0,4,1,3]]))<1e-10
    try:normalized_cca(torch.ones(5,3),y);raise AssertionError('Constant accepted')
    except ValueError:pass
    fs=dict(input=x,final=y,layers={i:x for i in CANDIDATE_LAYERS},sample_ids=list('abcde'))
    assert select_layer(fs)['layer_id']==min(CANDIDATE_LAYERS)
    try:select_layer(dict(fs,H_eval=[]));raise AssertionError('Eval field accepted')
    except ValueError:pass
    model=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False) for _ in range(32)]).requires_grad_(False)
    runtime=SimpleNamespace(model=model,get_module=lambda path:model[int(path.split('.')[2])])
    expert=AsymmetricCPExpert(4,4,2)
    with torch.no_grad():expert.rho.fill_(.4)
    inp=torch.randn(1,4,4);saved=[]
    for layer in [3,21,3]:
        binding=dict(layer_id=layer,expert_id='test',W0_id='same-edit-same-layer-'+str(layer));base=model[layer](inp)
        with bound_hook(runtime,expert,binding) as hook:
            assert torch.equal(model[layer](inp),base)
            with hook.generation_request():
                prefill=model[layer](inp)
                assert torch.equal(prefill[:,:-1],base[:,:-1]) and not torch.equal(prefill[:,-1],base[:,-1])
                cached=model[layer](inp[:,-1:]);assert not torch.equal(cached,model[layer].forward(inp[:,-1:]))
            assert not hook.enabled and torch.equal(model[layer](inp),base)
            saved.append(prefill.detach().clone())
        assert not model[layer]._forward_hooks
    assert torch.equal(saved[0],saved[2])
    buffer=io.BytesIO();torch.save(dict(binding=binding,expert=expert.state_dict()),buffer);buffer.seek(0)
    state=torch.load(buffer,weights_only=True);restored=AsymmetricCPExpert(4,4,2);restored.load_state_dict(state['expert'])
    with bound_hook(runtime,restored,state['binding']) as hook:
        with hook.generation_request():assert torch.equal(model[3](inp),saved[-1])
    try:
        with bound_hook(runtime,restored,state['binding']) as hook:
            with hook.generation_request():raise RuntimeError('intentional')
    except RuntimeError:pass
    assert all(not m._forward_hooks for m in model)
    print('PASS: reference formula, permutation, degeneracy, input boundary, tie, layer switching, OFF, prefill/cache, save/load, exception cleanup')



def check_shared_initialization():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from scripts.medtrace import stage19_train as m
    runtime=SimpleNamespace(model=torch.nn.Linear(4,4).requires_grad_(False),device='cpu')
    task=dict(canonical_edit_id='test',order=1,seed=123,native={'question':'q'},fit_questions=['a','b','c','d'],U_fit=[])
    cfg=dict(freeze_id='frozen',approved_training_freeze='frozen',approved_budget_seconds=60,train_seconds=60,runtime_lock={},code_commit='test')
    cfg['approved_task_bindings']={task['canonical_edit_id']:m.digest(task)}
    seen=[];initializations=[]
    def initialize(*args,**kwargs):
        initializations.append(kwargs['layer_path']);return AsymmetricCPExpert(4,4,4)
    def train(*args,**kwargs):
        expert=args[4];seen.append((kwargs['layer_id'],m.state_hash(expert)))
        with torch.no_grad():next(expert.parameters()).add_(.1)
        return expert
    with tempfile.TemporaryDirectory() as tmp,patch.object(m,'validate_task'),patch.object(m.stage15,'initialize',side_effect=initialize),patch.object(m,'teachers_for',return_value=[]),patch.object(m,'train',side_effect=train):
        result=m.train_triplet(runtime,Path(tmp),cfg,task,None,selection_mode='fixed21')
        assert len(initializations)==1 and initializations[0]=='model.layers.21.mlp.down_proj'
        assert len(seen)==3 and len(set(seen))==1 and len(result['branches'])==3
        try:m.train_triplet(runtime,Path(tmp),dict(cfg,approved_training_freeze='other'),task,None,selection_mode='fixed21');raise AssertionError('Unapproved freeze accepted')
        except ValueError:pass
    print('PASS: one same-layer W0 initialization, three identical starting states, distinct branch calls, approval binding')




def check_campaign_contract():
    from scripts.medtrace.stage19_campaign import validate,digest
    task=dict(canonical_edit_id='one');tasks=dict(tasks=[task]);tasks['freeze_id']=digest(tasks['tasks'])
    packet=dict(candidate_packages=[dict(candidate_id='one',training=task)],orders={k:['one'] for k in ('canonical','18001','18002')},early_anchors=['one'],prefixes=[1]);packet['freeze_id']=digest(packet)
    ext=dict(text_queries={});ext['freeze_id']=digest(ext)
    cfg=dict(mode='STAGE19_APPROVED_SEVEN',gpu='3',gpu_uuid='GPU-43e3d478-7979-ea29-8130-64a467b48a5c',freeze_id=tasks['freeze_id'],approved_training_freeze=tasks['freeze_id'],approved_evaluation_freeze=packet['freeze_id'],approved_extension_freeze=ext['freeze_id'],train_seconds=18000,approved_budget_seconds=18000,approved_task_bindings={'one':digest(task)})
    assert validate(cfg,packet,tasks,ext)=={'one':digest(task)}
    for changed in ({'gpu':'2'},{'approved_budget_seconds':0},{'approved_training_freeze':None},{'approved_task_bindings':{}},{'approved_extension_freeze':'different'}):
        try:validate(dict(cfg,**changed),packet,tasks,ext);raise AssertionError('Bad contract accepted')
        except ValueError:pass
    print('PASS: seven-config approval, cohort, GPU3, extension and budget contracts')

def check_source_screen():
    import tempfile
    from pathlib import Path
    from PIL import Image
    from scripts.medtrace.stage19_sources import near_duplicate_screen,extension_rephrase
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);im=Image.new('L',(17,16));im.putdata([x*15 for y in range(16) for x in range(17)])
        im.save(p/'a.png');im.save(p/'b.png');im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(p/'c.png')
        result=near_duplicate_screen({k:p/(k+'.png') for k in 'abc'})
        assert len(result['candidate_pairs'])==1 and result['candidate_pairs'][0]['distance']==0
    assert extension_rephrase('Does the picture contain liver?')=='Is liver visible in this image?'
    try:extension_rephrase('Unknown question');raise AssertionError('Unapproved template accepted')
    except ValueError:pass
    print('PASS: duplicate candidate screen and bounded extension templates')

if __name__=='__main__':
    main();check_shared_initialization();check_campaign_contract();check_source_screen()

