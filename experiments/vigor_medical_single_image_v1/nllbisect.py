"""S03 bounded native-NLL bisection with hard eight-candidate cap."""
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

UPPER=.608808129
AUTH=dict(sample='S03',arms=['A','D'],reference_A_steps=8,reference_D_steps=4,
    upper_alpha=UPPER,max_alignment_candidates=8,max_midpoints=7,nll_tolerance=.005,
    main_metric='old',correction_metric='old',probe_metric='new',probe_candidates_per_endpoint=1,
    probe_ratio_applications=1,probe_norm_tolerance=.01,max_corrections_per_candidate=1,
    probe_corrections=0,calibration_samples=32,half_fallback=False,replace=False,generation=False,judge=False,full16=False)


def search(runtime,record,factor,snapshot,state,target,fifth_effective,upper_reference,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    q=dict(accepted=False,target_nll=target,tolerance=.005,max_candidates=8,candidates=[])
    state['search']=q;low=0.;high=UPPER
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        start_losses=native_losses(runtime,pair)
        if start_losses!=state['endpoint_native_nll'] or not torch.equal(layer.weight.cpu(),snapshot):raise ValueError('Search start changed')
        start=layer.weight.detach().clone();low_loss=start_losses[0];high_loss=None
        if low_loss<=target:raise ValueError('Invalid lower bracket anchor')
        for i in range(8):
            check_time(deadline)
            alpha=UPPER if i==0 else (low+high)/2
            row=dict(index=i+1,alpha=alpha,requested=alpha*fifth_effective,low_alpha=low,high_alpha=high,
                low_nll=low_loss,high_nll=high_loss,start_native_nll=start_losses,trial={})
            q['candidates'].append(row);emit()
            accepted_snapshot=None
            try:
                if not torch.equal(layer.weight,start) or native_losses(runtime,pair)!=start_losses:raise ValueError('Search candidate start differs')
                v=_attempt(runtime,record,factor,pair,start,start_losses,state['baseline']['native'][1],row['requested'],None,
                    row['trial'],emit,None,correction_factor=None,separate_correction=True)
                loss=v['native_nll'][0];row.update(final_nll=loss,error=abs(loss-target),accepted=False)
                if i==0:
                    control(row['trial'],upper_reference,'search.upper',state['checks'])
                    q['upper_control_pass']=True
                if not v['accepted']:
                    row['decision']='STOP_GATE_REJECTED';state['status']='STOPPED_GATE_REJECTED'
                elif i==0 and not loss<target:
                    row['decision']='STOP_NO_BRACKET';state['status']='STOPPED_NO_BRACKET'
                elif i>0 and not high_loss-1e-5<=loss<=low_loss+1e-5:
                    row['decision']='STOP_NON_MONOTONIC';state['status']='STOPPED_NON_MONOTONIC'
                elif abs(loss-target)<=.005:
                    row['decision']='ACCEPT_FIRST_MATCH';row['accepted']=True;q.update(accepted=True,selected_index=i+1,selected_alpha=alpha)
                    state['aligned_endpoint_native_nll']=native_losses(runtime,pair);state['status']='ALIGNED'
                    accepted_snapshot=layer.weight.detach().cpu().clone()
                elif loss>target:
                    row['decision']='RAISE_LOWER_ALPHA';low=alpha;low_loss=loss
                else:
                    row['decision']='LOWER_UPPER_ALPHA';high=alpha;high_loss=loss
                emit()
            finally:
                row['rollback_pass']=_rollback(layer,pair,runtime,start,start_losses);emit()
                if not row['rollback_pass']:raise ValueError('Candidate D4 restoration failed')
            if accepted_snapshot is not None:return accepted_snapshot
            if row['decision'].startswith('STOP'):return None
        state['status']='STOPPED_CANDIDATE_CAP';emit()
        return None
    finally:
        state['search_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['search_base_restored']:raise ValueError('Search base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only bounded S03 NLL bisection authorized')
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
        frozen=dict(sample='S03',upper_alpha=UPPER,max_candidates=8,target_nll=target_nll,nll_tolerance=.005,
            fifth_full_effective=fifth_effective,probe_native_norm=probe_target)
        if frozen!=config['frozen_targets']:raise ValueError('Frozen alignment changed')
        write(root/'public/FROZEN_TARGETS.json',frozen)
        write(root/'public/PRIORS_AD.json',dict(A=histories['A'],D=oldD))
        one_shot=read(config['nllmatch_result'])
        if one_shot['status']!='COMPLETE' or one_shot['alignment_success']:raise ValueError('One-shot predecessor changed')
        upper_reference=one_shot['arms'][1]['alignment']['trial']
        write(root/'public/PRIOR_ONE_SHOT.json',one_shot)
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
        previous=read(config['new_fisher_summary'])
        for k in ('raw_trace','final_trace','multiplier'):
            if abs(metadata[k]/previous[k]-1)>1e-5:raise ValueError('Direct Fisher changed')
        metadata['prior_summary_match']=True;write(root/'public/DIRECT_FISHER.json',metadata)
        record=EditorRecord(**dict(raw[2],image_path=Path(raw[2]['image_path'])))
        write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',arm='A'))
        snapshotA=reconstruct(runtime,record,factor,histories['A'],8,states[0],emit,deadline)
        write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',arm='D'))
        snapshotD=reconstruct(runtime,record,factor,oldD,4,states[1],emit,deadline)
        write(root/'public/CONTROL_GATE.json',dict(status='PASS',A_steps=8,D_steps=4,both_before_alignment=True,full_D5_not_replayed=True))
        write(root/'public/STATUS.json',dict(status='ALIGNING',arm='D'))
        aligned=search(runtime,record,factor,snapshotD,states[1],target_nll,fifth_effective,upper_reference,emit,deadline)
        if aligned is not None:
            write(root/'public/STATUS.json',dict(status='PROBING',arm='A'))
            probeA=probe_endpoint(runtime,record,factor,diagonal,snapshotA,states[0],probe_target,emit,deadline,histories['A']['probes']['raw'])
            if probeA['matching_passed']:
                write(root/'public/STATUS.json',dict(status='PROBING',arm='D'))
                probe_endpoint(runtime,record,factor,diagonal,aligned,states[1],probe_target,emit,deadline)
            else:states[1]['probe_missing_reason']='REFERENCE_PROBE_MATCH_FAILED';emit()
        else:
            states[0]['probe_missing_reason']='D_ALIGNMENT_FAILED';states[1]['probe_missing_reason']='D_ALIGNMENT_FAILED';emit()
        write(root/'public/RESULT.json',dict(status='COMPLETE',denominator=2,sample='S03',arms=states,
            alignment_success=states[1]['search']['accepted'],probe_count=sum(len(s.get('probes',{})) for s in states),
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
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started))
