"""S03 reverse D arm: old direction matched once to frozen large native step norms."""
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

AUTH=dict(sample='S03',arms=['D'],steps=8,main_metric='old',correction_metric='old',
    target_main_norm_source='previous_B',target_probe_norm_source='previous_A',probe_metric='new',
    seed_denominator=16,norm_ratio_applications=1,norm_match_tolerance=.01,
    max_corrections_per_step=1,probe_corrections=0,calibration_samples=32,
    half_fallback=False,tune=False,replace=False,generation=False,judge=False,full16=False)


def verify_reference(runtime,record,factor,history,state,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);losses=native_losses(runtime,pair);state['checks']=[]
    try:
        baseline=baseline_check(runtime,record,factor);state['baseline']=baseline
        compare(baseline,history['baseline'],'baseline',state['checks'])
        compare(baseline['anchor'],history['baseline']['anchor'],'anchor',state['checks'])
        state['trial']={}
        v=_attempt(runtime,record,factor,pair,base,losses,baseline['native'][1],FIRST_D/16,None,state['trial'],emit)
        control(state['trial'],history['steps'][0]['trial'],'reference.first',state['checks'])
        if not v['accepted']:raise ValueError('Frozen reference no longer accepted')
        state['status']='PASS'
    finally:
        state['base_restored']=_rollback(layer,pair,runtime,base,losses);emit()
        if not state['base_restored']:raise ValueError('Reference base restoration failed')


def run_arm(runtime,record,factor,diagonal,state,emit,deadline,targets,probe_target,history=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        baseline=baseline_check(runtime,record,factor)
        if history is not None:
            compare(baseline,history['baseline'],'baseline',state['checks'])
            compare(baseline['anchor'],history['baseline']['anchor'],'anchor',state['checks'])
        state.update(baseline=baseline,steps=[],probes={},status='RUNNING');emit()
        for i in range(8):
            check_time(deadline)
            start=layer.weight.detach().clone();losses=native_losses(runtime,pair)
            row=dict(step=i+1,start_native_nll=losses,trial={});state['steps'].append(row)
            verdict=_attempt(runtime,record,factor,pair,start,losses,baseline['native'][1],FIRST_D/16,None,
                row['trial'],emit,None,
                correction_factor=None,separate_correction=True,
                native_norm_target=targets[i])
            row['accepted']=verdict['accepted']
            if not verdict['accepted']:
                row['rollback_pass']=_rollback(layer,pair,runtime,start,losses)
                if not row['rollback_pass']:raise ValueError('Rejected full-step restoration failed')
                state['status']='STOPPED_MATCH_FAILED' if row['trial']['status']=='MATCH_FAILED' else 'STOPPED_REJECTED'
                emit();break
            row['status']='ACCEPTED';emit()
        accepted=[t for t in state['steps'] if t['accepted']]
        state['accepted_steps']=len(accepted);state['prefix_completed']=len(accepted)==8
        state['endpoint_native_nll']=native_losses(runtime,pair)
        state['accepted_requested_reduction']=sum(t['trial']['proposal']['requested_reduction'] for t in accepted)
        state['accepted_effective_reduction']=sum(t['trial']['proposal']['effective_reduction'] for t in accepted)
        if state['prefix_completed']:
            probe=run_probe(runtime,record,factor,diagonal,pair,baseline['native'][1],state,'matched',emit,deadline,target=probe_target)
            state['matched_probe_key']='matched';state['probe_norm_match_passed']=probe['matching_passed']
            state['probe_accepted']=probe['verdict']['accepted'] and probe['matching_passed']
            state['status']='STOPPED_PROBE_MATCH_FAILED' if not probe['matching_passed'] else ('COMPLETE' if state['probe_accepted'] else 'STOPPED_PROBE_REJECTED')
        else:
            state['probe_norm_match_passed']=False;state['probe_missing_reason']='INCOMPLETE_EIGHT_STEP_PREFIX'
        emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        state['base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses
        emit()
        if not state['base_restored'] or layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Arm base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only S03 reverse D arm authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);started=time.time();deadline=config['deadline_epoch']
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
    states=[dict(arm='D',sample='S03',status='NOT_RUN',checks=[])]
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
        targets=[t['trial']['proposal']['quantization']['native_delta_l2'] for t in histories['B']['steps']]
        probe_target=histories['A']['probes']['raw']['proposal']['quantization']['native_delta_l2']
        frozen=dict(sample='S03',step_native_norms=targets,probe_native_norm=probe_target,source='previous normpath armB steps / armA probe')
        if len(targets)!=8 or frozen!=config['frozen_targets']:raise ValueError('Frozen targets changed')
        write(root/'public/FROZEN_TARGETS.json',frozen)
        write(root/'public/PRIORS_ABC.json',priors)
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
        reference={}
        verify_reference(runtime,record,factor,histories['A'],reference,lambda:write(root/'public/REFERENCE.json',reference),deadline)
        write(root/'public/CONTROL_GATE.json',dict(status='PASS',reference='A baseline and first original full step',before_D=True,full_ABC_replayed=False))
        write(root/'public/STATUS.json',dict(status='RUNNING',arm='D'))
        run_arm(runtime,record,factor,diagonal,states[0],emit,deadline,targets,probe_target)
        write(root/'public/RESULT.json',dict(status='COMPLETE',denominator=1,sample='S03',arms=states,
            completed_prefixes=sum(s['prefix_completed'] for s in states),probe_count=sum(len(s.get('probes',{})) for s in states),
            all_base_restored=all(s['base_restored'] for s in states),elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_forward_calls=calls[0],generation_calls=0,judge_calls=0,
            half_candidates=0,full16_runs=0))
        write(root/'public/STATUS.json',dict(status='COMPLETE',arms=1))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit()
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:
        if handle is not None:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started))
