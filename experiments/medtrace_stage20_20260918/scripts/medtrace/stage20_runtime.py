"""Stage20 consumers of the existing locked training implementation."""
from pathlib import Path
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest


def hsic_diagnostics(features,selection):
    import torch
    from methods.medtrace.hsic import normalized_cca,centered_kernel,layer_path
    rows=[]
    for layer,x in features['layers'].items():
        if layer in selection['invalid']:continue
        a=normalized_cca(x,features['final']);b=normalized_cca(features['input'],x)
        assert abs(.001*(a-b)-selection['scores'][layer])<1e-15
        k=centered_kernel(x);regularized=k+torch.eye(5,dtype=k.dtype)*5e-5
        eig=torch.linalg.eigvals(k)
        rows.append(dict(layer=layer,hook=layer_path(layer),final_cca=a,input_cca=b,score=selection['scores'][layer],
            regularized_condition=float(torch.linalg.cond(regularized)),eigenvalues_real=eig.real.tolist(),
            eigenvalues_max_imag=float(eig.imag.abs().max()),observation_shape=list(x.shape)))
    ranked=sorted(selection['scores'].values(),reverse=True)
    return dict(rows=rows,top1_top2_margin=ranked[0]-ranked[1] if len(ranked)>1 else None,
        observations=5,Base_OFF=True,uses_H_G_eval=False,correlated_wrappers_not_independent_facts=True)


def train_ablation(runtime,root,cfg,task,record,branch):
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace.stage18_cfact import train,teachers_for,state_hash,assert_base_off
    source=Path(cfg['main_run']);directory=source/'private/edits'/f"e{task['order']:03d}"
    state=torch.load(directory/'W0_R4.pt',map_location='cpu',weights_only=True)
    if state['task']!=digest(task) or state['code'] not in [cfg['code_commit']]+cfg.get('approved_predecessor_commits',[]):raise ValueError('Ablation W0 lineage changed')
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),task['seed'],rank=4).to(runtime.device)
    expert.load_state_dict(state['expert'])
    if state_hash(expert)!=state['W0']:raise ValueError('Ablation W0 is not identical')
    teachers=teachers_for(runtime,source,cfg,task,record)
    # NO_H has no G consumer. EXTRA retains the strict H/G token-stratum gate.
    mode='STAGE19_FASTTRACK_TWO_ARMS' if branch=='C_NO_H' else 'STAGE20_REGISTERED_ABLATION'
    train(runtime,root,dict(cfg,mode=mode),task,expert,branch,record,teachers,state['W0'],layer_id=state['layer_id'])
    del teachers,expert;assert_base_off(runtime)
    return dict(kind=branch,path=str(root/'private/edits'/f"e{task['order']:03d}"/branch/'latest.pt'),
        layer_id=state['layer_id'],W0=state['W0'],task=digest(task),code=cfg['code_commit'],selection=digest(read(directory/'SELECTION.json')))


def ablation_config(cfg):
    root=Path(cfg['run']);target=root/'private/ablation';(target/'private').mkdir(parents=True,exist_ok=True);(target/'public').mkdir(exist_ok=True)
    stream=read(root/'private/STREAM.json');stream=dict(stream,tasks=stream['tasks'][:11],new_rows=[],positive_rows=[])
    for name,value in [('STREAM.json',stream),('FRESH_BASE_OUTPUTS.json',read(root/'private/FRESH_BASE_OUTPUTS.json'))]:
        path=target/'private'/name
        if path.exists():
            if not cfg.get('resume_audit') or read(path)!=value:raise ValueError('Unapproved/changed ablation recovery input')
        else:write_new(path,value)
    return dict(cfg,run=str(target),main_run=str(root),stream_binding=digest(stream),ablation=True)
