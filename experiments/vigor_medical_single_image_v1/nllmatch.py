"""S03 fixed one-shot NLL alignment: A8 versus D4 plus one shortened fifth step."""
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

ALPHA=.608808129
AUTH=dict(sample='S03',arms=['A','D'],reference_A_steps=8,reference_D_steps=4,
    alpha=ALPHA,alignment_candidates=1,nll_tolerance=.005,main_metric='old',correction_metric='old',
    probe_metric='new',probe_candidates_per_endpoint=1,probe_ratio_applications=1,probe_norm_tolerance=.01,
    max_corrections_per_candidate=1,probe_corrections=0,calibration_samples=32,
    tune=False,half_fallback=False,replace=False,generation=False,judge=False,full16=False)


def reconstruct(runtime,record,factor,history,count,state,emit,deadline):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        baseline=baseline_check(runtime,record,factor)
        compare(baseline,history['baseline'],'baseline',state['checks'])
        compare(baseline['anchor'],history['baseline']['anchor'],'anchor',state['checks'])
        state.update(baseline=baseline,steps=[],status='RECONSTRUCTING');emit()
        for previous in history['steps'][:count]:
            check_time(deadline)
            start=layer.weight.detach().clone();losses=native_losses(runtime,pair)
            row=dict(step=previous['step'],start_native_nll=losses,trial={});state['steps'].append(row)
            compare(dict(native_before=losses),dict(native_before=previous['start_native_nll']),f"step{row['step']}.start",state['checks'])
            m=previous['trial']['proposal'].get('native_norm_matching')
            v=_attempt(runtime,record,factor,pair,start,losses,baseline['native'][1],FIRST_D/16,None,row['trial'],emit,None,
                correction_factor=None,separate_correction=True,native_norm_target=m['target'] if m else None)
            control(row['trial'],previous['trial'],f"step{row['step']}",state['checks'])
            if not v['accepted']:raise ValueError('Frozen prefix changed')
            row['accepted']=True;emit()
        state['endpoint_native_nll']=native_losses(runtime,pair);state['status']='RECONSTRUCTED';emit()
        return layer.weight.detach().cpu().clone()
    finally:
        state['base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['base_restored']:raise ValueError('Reconstruction base restoration failed')


def align(runtime,record,factor,snapshot,state,target_nll,fifth_effective,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    a=dict(alpha=ALPHA,target_nll=target_nll,tolerance=.005,requested=ALPHA*fifth_effective,trial={})
    state['alignment']=a
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        losses=native_losses(runtime,pair)
        if losses!=state['endpoint_native_nll'] or not torch.equal(layer.weight.cpu(),snapshot):raise ValueError('Alignment start changed')
        start=layer.weight.detach().clone();a['start_native_nll']=losses;emit()
        v=_attempt(runtime,record,factor,pair,start,losses,state['baseline']['native'][1],a['requested'],None,a['trial'],emit,None,
            correction_factor=None,separate_correction=True)
        a['error']=abs(v['native_nll'][0]-target_nll);a['nll_match_pass']=a['error']<=.005
        a['accepted']=v['accepted'] and a['nll_match_pass']
        if not a['accepted']:
            a['rollback_pass']=_rollback(layer,pair,runtime,start,losses)
            a['after_rollback_native_nll']=native_losses(runtime,pair)
            if not a['rollback_pass']:raise ValueError('Alignment rejection restoration failed')
            state['status']='STOPPED_GATE_REJECTED' if not v['accepted'] else 'STOPPED_NLL_MISMATCH';emit()
            return None
        state['aligned_endpoint_native_nll']=native_losses(runtime,pair);state['status']='ALIGNED';emit()
        return layer.weight.detach().cpu().clone()
    finally:
        state['alignment_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['alignment_base_restored']:raise ValueError('Alignment base restoration failed')


def probe_endpoint(runtime,record,factor,diagonal,snapshot,state,target,emit,deadline,history=None):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        expected=state.get('aligned_endpoint_native_nll',state['endpoint_native_nll'])
        if native_losses(runtime,pair)!=expected or not torch.equal(layer.weight.cpu(),snapshot):raise ValueError('Probe start changed')
        p=run_probe(runtime,record,factor,diagonal,pair,state['baseline']['native'][1],state,'matched',emit,deadline,target=target)
        if history is not None:
            compare(p['proposal'],history['proposal'],'probe.reference',state['checks'])
            if p['verdict']['gates']!=history['verdict']['gates']:raise ValueError('Reference probe gate changed')
        state['status']='STOPPED_PROBE_MATCH_FAILED' if not p['matching_passed'] else ('COMPLETE' if p['verdict']['accepted'] else 'STOPPED_PROBE_REJECTED');emit()
        return p
    finally:
        state['probe_base_restored']=_rollback(layer,pair,runtime,base,base_losses);emit()
        if not state['probe_base_restored']:raise ValueError('Probe base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only frozen A8 versus one-shot D NLL alignment authorized')
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
        frozen=dict(sample='S03',alpha=ALPHA,target_nll=target_nll,nll_tolerance=.005,
            fifth_full_effective=fifth_effective,alignment_request=ALPHA*fifth_effective,
            probe_native_norm=probe_target)
        if frozen!=config['frozen_targets']:raise ValueError('Frozen alignment changed')
        write(root/'public/FROZEN_TARGETS.json',frozen)
        write(root/'public/PRIORS_AD.json',dict(A=histories['A'],D=oldD))
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
        aligned=align(runtime,record,factor,snapshotD,states[1],target_nll,fifth_effective,emit,deadline)
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
            alignment_success=states[1]['alignment']['accepted'],probe_count=sum(len(s.get('probes',{})) for s in states),
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
