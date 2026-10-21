"""S03 fixed matched-progress probe: unrounded FP32, rounded FP32 and native."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import torch
from .backoff16 import _attempt,_rollback
from .batch8 import baseline_check
from .diagnose import candidate,FIRST_D
from .direct_fisher16 import estimate,check_time
from .editor import paired_batches
from .finite_null import gates,native_losses
from .normmatch4 import control
from .precision_abc import compare
from .prepare import write

from .normpath8 import run_probe

from .nllmatch import reconstruct,probe_endpoint

SELECTED_ALPHA=.504169231828125
AUTH=dict(sample='S03',endpoints=['A8','D4_selected'],alpha=SELECTED_ALPHA,search=False,
    branches=['unrounded_editable_FP32','rounded_editable_FP32','native'],
    probe_candidates_per_endpoint=1,native_norm_ratio_applications=1,native_norm_tolerance=.01,
    main_metric='old',correction_metric='old',probe_metric='new',probe_corrections=0,
    calibration_samples=32,half_fallback=False,replace=False,generation=False,judge=False,full16=False)


def replay_selected(runtime,record,factor,snapshot,state,selected,previous,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair);state['replay']={}
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        losses=native_losses(runtime,pair);start=layer.weight.detach().clone()
        if losses!=state['endpoint_native_nll']:raise ValueError('Selected step start changed')
        v=_attempt(runtime,record,factor,pair,start,losses,state['baseline']['native'][1],selected['requested'],None,state['replay'],emit,None,
            correction_factor=None,separate_correction=True)
        control(state['replay'],selected['trial'],'selected.replay',state['checks'])
        if not v['accepted'] or abs(v['native_nll'][0]-previous['aligned_endpoint_native_nll'][0])>1e-5:raise ValueError('Selected endpoint changed')
        if abs(v['native_nll'][0]-2.6600356101989746)>.005:raise ValueError('NLL qualification changed')
        state['aligned_endpoint_native_nll']=native_losses(runtime,pair);emit()
        return layer.weight.detach().cpu().clone()
    finally:
        state['replay_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['replay_base_restored']:raise ValueError('Selected step base restoration failed')


def precision_probe(runtime,record,factor,diagonal,snapshot,state,target,history,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    state['probes']['matched']={};probe=state['probes']['matched']
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        endpoint=layer.weight.detach().clone();losses=native_losses(runtime,pair)
        if losses!=state.get('aligned_endpoint_native_nll',state['endpoint_native_nll']):raise ValueError('Precision endpoint changed')
        p=candidate(runtime,record,0,factor,FIRST_D/16,retain=True,solve_factor=diagonal,native_norm_target=target,rounded_fp32=True)
        probe['proposal']=p;probe['verdict']=gates(runtime,pair,factor,endpoint,losses,state['baseline']['native'][1])
        probe['matching_passed']=p['native_norm_matching']['passed']
        if probe['matching_passed']:
            compare(p,history['proposal'],'probe.control',state['checks'])
            if probe['verdict']['gates']!=history['verdict']['gates']:raise ValueError('Native precision control gate changed')
            probe['branches']=dict(A=dict(before=p['fp32_before'],after=p['fp32_after']),
                B=dict(before=p['fp32_before'],after=p['rounded_fp32_after']),C=dict(before=p['native_before'],after=p['native_after']))
            state['status']='COMPLETE_DIAGNOSTIC'
        else:state['status']='STOPPED_MATCH_FAILED'
        probe['endpoint_restored']=_rollback(layer,pair,runtime,endpoint,losses);emit()
        if not probe['endpoint_restored']:raise ValueError('Precision endpoint restoration failed')
        return probe
    finally:
        state['probe_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['probe_base_restored']:raise ValueError('Precision base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only frozen matched-progress precision readouts authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);started=time.time();deadline=config['deadline_epoch']
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
    states=[dict(arm=a,sample='S03',status='NOT_RUN',checks=[],probes={}) for a in ('A','D')]
    def emit():write(root/'public/ARMS.json',states)
    handle=None;calls=[0]
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime binding changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        read=lambda p:json.loads(Path(p).read_text())
        raw=read(root/'private/records.json');manifest=read(config['manifest']);pilot=read(config['record'])
        groups=['SLAKE:'+r['relative_image_path'].split('/')[0] for r in raw]
        excluded=set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups'])|{'SLAKE:'+pilot['relative_image_path'].split('/')[0]}
        if len(raw)!=8 or len(set(groups))!=8 or set(groups)&excluded:raise ValueError('Frozen isolation changed')
        cal=torch.load(config['calibration'],map_location='cpu',weights_only=True)
        if cal['binding']!=config['binding'] or cal['selected_layer']!='model.layers.0.mlp.down_proj':raise ValueError('Fisher binding changed')
        factor=cal['factors'][cal['selected_layer']]
        priors=read(config['normpath_result'])
        if priors['status']!='COMPLETE' or priors['denominator']!=3 or priors['completed_prefixes']!=3:
            raise ValueError('Previous three arms incomplete')
        histories={s['arm']:s for s in priors['arms']}
        oldD=read(config['normd_result'])['arms'][0]
        if oldD['arm']!='D' or not oldD['prefix_completed']:raise ValueError('D predecessor incomplete')
        target_nll=histories['A']['endpoint_native_nll'][0]
        probe_target=histories['A']['probes']['raw']['proposal']['quantization']['native_delta_l2']
        fifth_effective=oldD['steps'][4]['trial']['proposal']['effective_reduction']
        matched=read(config['nllbisect_result'])
        if matched['status']!='COMPLETE' or not matched['alignment_success']:raise ValueError('Matched predecessor incomplete')
        previous={s['arm']:s for s in matched['arms']}
        selected=previous['D']['search']['candidates'][-1]
        frozen=dict(sample='S03',alpha=SELECTED_ALPHA,target_nll=target_nll,nll_tolerance=.005,
            fifth_full_effective=fifth_effective,probe_native_norm=probe_target,
            D_endpoint_nll=previous['D']['aligned_endpoint_native_nll'][0])
        if frozen!=config['frozen_targets'] or selected['alpha']!=SELECTED_ALPHA:raise ValueError('Frozen selected endpoint changed')
        write(root/'public/FROZEN_TARGETS.json',frozen)
        write(root/'public/PRIORS_AD.json',dict(A=histories['A'],D=oldD))
        write(root/'public/PRIOR_MATCHED.json',matched)
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        def count(module,args):calls[0]+=1
        handle=runtime.model.register_forward_pre_hook(count)
        def forbidden(*args,**kwargs):raise PermissionError('Generation forbidden')
        runtime.generate=forbidden
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in gpu.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):raise ValueError('Process identity not neutral')
        train=read(config['records']);ids=[r['record_id'] for r in train]
        if len(ids)!=32 or len(set(ids))!=32 or ids!=manifest['calibration_ids'] or ids!=cal['calibration_ids']:raise ValueError('Calibration list changed')
        training=[EditorRecord(**dict(r,image_path=Path(r['image_path']))) for r in train]
        diagonal,metadata=estimate(runtime,training,factor,lambda e:write(root/'public/CALIBRATION_PROGRESS.json',e),min(deadline,time.time()+120))
        fisher_reference=read(config['new_fisher_summary'])
        for k in ('raw_trace','final_trace','multiplier'):
            if abs(metadata[k]/fisher_reference[k]-1)>1e-5:raise ValueError('Direct Fisher changed')
        metadata['prior_summary_match']=True;write(root/'public/DIRECT_FISHER.json',metadata)
        record=EditorRecord(**dict(raw[2],image_path=Path(raw[2]['image_path'])))
        write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',arm='A'))
        snapshotA=reconstruct(runtime,record,factor,histories['A'],8,states[0],emit,deadline)
        write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',arm='D'))
        snapshotD=reconstruct(runtime,record,factor,oldD,4,states[1],emit,deadline)
        snapshotD=replay_selected(runtime,record,factor,snapshotD,states[1],selected,previous['D'],emit,deadline)
        write(root/'public/CONTROL_GATE.json',dict(status='PASS',A_steps=8,D_steps=4,selected_step_replayed_once=True,no_search=True,both_before_precision=True))
        for state,snapshot in zip(states,(snapshotA,snapshotD)):
            write(root/'public/STATUS.json',dict(status='PRECISION',arm=state['arm']))
            probe=precision_probe(runtime,record,factor,diagonal,snapshot,state,probe_target,previous[state['arm']]['probes']['matched'],emit,deadline)
            if not probe['matching_passed']:break
        write(root/'public/RESULT.json',dict(status='COMPLETE',denominator=2,sample='S03',arms=states,
            matched_endpoints=True,probe_count=sum(len(s.get('probes',{})) for s in states),
            all_base_restored=all(s['base_restored'] for s in states),elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_forward_calls=calls[0],generation_calls=0,judge_calls=0,
            half_candidates=0,full16_runs=0))
        write(root/'public/STATUS.json',dict(status='COMPLETE',arms=2))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit()
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:
        if handle is not None:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls[0]))
