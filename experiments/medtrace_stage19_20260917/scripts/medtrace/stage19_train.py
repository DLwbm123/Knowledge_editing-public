"""Layer-bound three-branch training primitive; dispatch requires approved V3 budget.

The same-edit/same-layer initialization is shared; each branch gets a fresh
optimizer through stage18_cfact.train. No evaluation rows are accepted here.
"""
import copy
from pathlib import Path
import torch
from methods.medtrace.hsic import layer_path,collect_features,select_layer
from methods.medtrace.selective_write import LowRankExpert
from scripts.medtrace import stage15
from scripts.medtrace.stage18_cfact import BRANCHES,train,teachers_for,state_hash,assert_base_off
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.astra_judge_bundle import read,write_new


def train_triplet(runtime,root,cfg,task,record,*,selection_mode):
    validate_task(task);assert_base_off(runtime)
    if selection_mode not in ('fixed21','HSIC_Top1'):raise ValueError('Undeclared layer condition')
    if cfg.get('approved_training_freeze')!=cfg.get('freeze_id') or not cfg.get('approved_budget_seconds'):
        raise ValueError('V3 cohort and numeric budget approval required')
    if cfg['approved_budget_seconds']!=cfg['train_seconds']:raise ValueError('Budget changed')
    if cfg.get('approved_task_bindings',{}).get(task['canonical_edit_id'])!=digest(task):raise ValueError('Task is outside approved cohort')
    root=Path(root);directory=root/'private/edits'/f"e{task['order']:03d}";directory.mkdir(parents=True,exist_ok=True)
    point=directory/'LAYER_SELECTION.json';scope=dict(edit=task['canonical_edit_id'],mode=selection_mode,
        native=digest(task['native']),fit=digest(task['fit_questions']),runtime=cfg['runtime_lock'],code=cfg['code_commit'])
    if point.exists():
        selection=read(point)
        if selection['binding']!=scope:raise ValueError('Layer selection binding changed')
    else:
        if selection_mode=='HSIC_Top1':
            features,counts=collect_features(runtime,record,task['fit_questions']);selection=dict(select_layer(features),token_counts=counts)
        else:selection=dict(layer_id=21)
        selection['binding']=scope;write_new(point,selection)
    layer=selection['layer_id'];path=layer_path(layer)
    adapted=dict(canonical_edit_id=task['canonical_edit_id'],order=task['order'],seed=task['seed'],probes=[task['native']],U=task['U_fit'],fit_questions=task['fit_questions'])
    initialization_root=Path(cfg.get('initialization_root',root/'initialization'))/f'layer{layer}'
    (initialization_root/'private').mkdir(parents=True,exist_ok=True);(initialization_root/'public').mkdir(exist_ok=True)
    cp=stage15.initialize(runtime,initialization_root,cfg,adapted,record=record,seed_base=20260912,layer_path=path)
    template=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device);initial=copy.deepcopy(template.state_dict());w0=state_hash(template)
    x=torch.linspace(-1,1,cp.d_in,device=runtime.device).reshape(1,-1)
    if not torch.allclose(cp.residual(x),template.residual(x),rtol=2e-4,atol=2e-5):raise ValueError('CP conversion changed function')
    del cp,x
    init_binding=dict(edit=task['canonical_edit_id'],layer_id=layer,W0=w0,selection=digest(selection),code=cfg['code_commit'])
    initial_point=directory/'SHARED_LAYER_W0_BINDING.json'
    if initial_point.exists():
        if read(initial_point)!=init_binding:raise ValueError('Shared W0 changed')
    else:write_new(initial_point,init_binding)
    teachers=teachers_for(runtime,root,cfg,task,record);receipts=[]
    for branch in BRANCHES:
        template.load_state_dict(initial)
        expert=train(runtime,root,cfg,task,template,branch,record,teachers,w0,layer_id=layer)
        receipts.append(dict(branch=branch,layer_id=layer,W0=w0,path=str(directory/branch/'latest.pt')))
    assert_base_off(runtime)
    return dict(selection=selection,shared_initialization=init_binding,branches=receipts)
