"""One preselected edit: real frozen-Base HSIC and multi-layer hook mechanics only."""
import argparse,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_single import setup
from scripts.medtrace.stage17_prepare import digest

def run(cfg):
    if cfg['mode']!='STAGE19_HSIC_MECHANICS_ONLY' or cfg['gpu']!='3':raise ValueError('Scope changed')
    setup(cfg)
    import torch
    from dataclasses import replace
    from m3bench_repro.editors.llava_runtime import EditorRecord
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from scripts.medtrace.stage18_cfact import assert_base_off
    from scripts.medtrace import stage15
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.hsic import collect_features,select_layer,bound_hook
    root=Path(cfg['run']);t=read(root/'private/ONE_TASK.json')
    if digest(t)!=cfg['task_binding']:raise ValueError('Task changed')
    n=t['native'];record=EditorRecord(t['canonical_edit_id'],n['dataset'],n['question'],'','',Path(n['image_path']),n['image_path'],1,'MECHANICAL_ONLY','NO_GOLD')
    print('LOADING_RUNTIME',flush=True)
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])));runtime.run_root=root/'private/work';runtime.model.eval()
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()];start=time.time()
    assert_base_off(runtime)
    features,counts=collect_features(runtime,record,t['fit_questions']);selection=select_layer(features)
    torch.save(features,root/'private/HSIC_BASE_FEATURES.pt')
    base,raw,_,_=stage15.base_output(runtime,root,n,record)
    torch.manual_seed(19001);expert=AsymmetricCPExpert(14336,4096,4).to(runtime.device).requires_grad_(False)
    with torch.no_grad():expert.rho.fill_(.01)
    saved=dict(binding=dict(layer_id=3,expert_id='synthetic-mechanical-only',W0_id='not-trained'),expert={k:v.cpu() for k,v in expert.state_dict().items()})
    torch.save(saved,root/'private/SYNTHETIC_EXPERT.pt');loaded=torch.load(root/'private/SYNTHETIC_EXPERT.pt',map_location=runtime.device,weights_only=True);expert.load_state_dict(loaded['expert'])
    receipts=[];seen={}
    for layer in [3,21,3]:
        stage15.budget(root,cfg);binding=dict(loaded['binding'],layer_id=layer)
        with bound_hook(runtime,expert,binding) as hook:
            out=stage15.generate(runtime,raw,base['binding'],hook);trace=list(hook.last_generation_trace)
            if layer in seen and out['raw_token_ids']!=seen[layer]:raise ValueError('Restored layer generation drift')
            seen[layer]=out['raw_token_ids']
            if not trace or not any(r['active_residual_norm']>0 for r in trace) or hook.enabled:raise ValueError('No actual hook effect/cleanup')
        assert_base_off(runtime)
        off=stage15.generate(runtime,raw,base['binding'])
        if off['raw_token_ids']!=base['raw_token_ids']:raise ValueError('Base OFF changed')
        receipts.append(dict(layer_id=layer,prefill_length=trace[0]['sequence_length'],prefill_active=trace[0]['active_predictor_count'],
            generation_calls=len(trace),cached_calls=sum(r['sequence_length']==1 for r in trace),nonzero_effect=True,Base_OFF=True))
    if any(p._version!=v or p.data_ptr()!=ptr or p.requires_grad for p,v,ptr in frozen):raise ValueError('Base changed')
    write_new(root/'public/HSIC_GPU_MECHANICS.json',dict(status='PASS',selection=selection,token_counts=counts,
        samples=5,source_edits=1,feature_seconds_including_mechanics=time.time()-start,mechanics=receipts,
        trained_experts=0,writer_updates=0,new_method_scores=False,gpu_uuid=cfg['gpu_uuid'],code=cfg['code_commit'],
        interpretation='Synthetic writer mechanical qualification, one-edit Base HSIC; not a method comparison or selection distribution'))
    print('COMPLETE',flush=True)

if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
