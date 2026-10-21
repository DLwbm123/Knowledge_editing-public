"""S03 fixed-direction mixed versus complete executed FP32 chain diagnostic."""
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
from .chainprecision_grid import read_grid
from .finite_null import gates,native_losses
from .normmatch4 import control
from .precision_abc import compare
from .prepare import write

from .normpath8 import run_probe

from .nllmatch import reconstruct,probe_endpoint

from .nllprecision import replay_selected,SELECTED_ALPHA

AUTH=dict(sample='S03',endpoints=['A8','D4_selected'],alpha=SELECTED_ALPHA,search=False,
    probe_metrics=['F1'],probe_candidates_per_metric_endpoint=1,max_probes=2,
    readonly_positions=[0,.5,1],readout_precisions=['mixed','chain_fp32'],max_null_gradients=4,
    native_norm_ratio_applications=1,native_norm_tolerance=.01,
    main_metric='old',correction_metric='old',probe_corrections=0,
    calibration_samples=32,half_fallback=False,replace=False,generation=False,judge=False,full16=False)


def metric_probe(runtime,record,factor,diagonal,snapshot,state,metric,target,history,emit,deadline,observer):
    if metric not in ('F0','F1'):raise PermissionError('Only two frozen existing metrics')
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    probe={};state['probes'][metric]=probe
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        endpoint=layer.weight.detach().clone();losses=native_losses(runtime,pair)
        if losses!=state.get('aligned_endpoint_native_nll',state['endpoint_native_nll']) or not torch.equal(layer.weight.cpu(),snapshot):raise ValueError('Metric probe start changed')
        p=candidate(runtime,record,0,factor,FIRST_D/16,retain=True,
            solve_factor=diagonal if metric=='F1' else None,native_norm_target=target,update_observer=observer)
        probe['proposal']=p;probe['verdict']=gates(runtime,pair,factor,endpoint,losses,state['baseline']['native'][1])
        probe['matching_passed']=p['native_norm_matching']['passed']
        if probe['matching_passed'] and history is not None:
            compare(p,history['proposal'],'probe.F1.control',state['checks'])
            if probe['verdict']['gates']!=history['verdict']['gates']:raise ValueError('F1 control gate changed')
        state['status']='COMPLETE_DIAGNOSTIC' if probe['matching_passed'] else 'STOPPED_MATCH_FAILED'
        probe['endpoint_restored']=_rollback(layer,pair,runtime,endpoint,losses);emit()
        if not probe['matching_passed']:raise ValueError('Single norm matching failed; no retry')
        if not probe['endpoint_restored']:raise ValueError('Metric probe endpoint restoration failed')
        return probe
    finally:
        state['probe_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['probe_base_restored']:raise ValueError('Metric probe base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only fixed matched-progress F0/F1 probes authorized')
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
        captured={}
        for metric in ('F1',):
            for state,snapshot in zip(states,(snapshotA,snapshotD)):
                write(root/'public/STATUS.json',dict(status='PROBING',arm=state['arm'],metric=metric))
                history=previous[state['arm']]['probes']['matched'] if metric=='F1' else None
                probe=metric_probe(runtime,record,factor,diagonal,snapshot,state,metric,probe_target,history,emit,deadline,lambda u:captured.update({state['arm']:u.cpu().clone()}))
                if not probe['matching_passed']:break
            if not probe['matching_passed']:break
            if metric=='F1':write(root/'public/PROBE_CONTROL_GATE.json',dict(status='PASS',both_F1_before_diagnostics=True))
        for state,snapshot in zip(states,(snapshotA,snapshotD)):
            write(root/'public/STATUS.json',dict(status='READONLY_GRID',arm=state['arm']))
            state['grid']={}
            read_grid(runtime,record,snapshot,captured[state['arm']],state['probes']['F1']['proposal'],deadline,lambda:emit(),output=state['grid'])
            emit()
        captured.clear()
        result_status='COMPLETE' if sum(len(s['probes']) for s in states)==2 and all(p['matching_passed'] for s in states for p in s['probes'].values()) else 'STOPPED_MATCH_FAILED'
        write(root/'public/RESULT.json',dict(status=result_status,denominator=2,sample='S03',arms=states,
            matched_endpoints=True,probe_count=sum(len(s.get('probes',{})) for s in states),
            all_base_restored=all(s['base_restored'] for s in states),elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_forward_calls=calls[0],generation_calls=0,judge_calls=0,
            half_candidates=0,full16_runs=0))
        write(root/'public/STATUS.json',dict(status=result_status,arms=2))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit()
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:
        if handle is not None:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls[0]))
