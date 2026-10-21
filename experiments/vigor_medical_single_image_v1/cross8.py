"""S03 eight-step factorial: independently assign main and correction metrics."""
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

ARMS=('00','11','10','01')
AUTH=dict(sample='S03',steps=8,arms=list(ARMS),calibration_samples=32,main_denominator=16,
    probe_denominator=16,probe_metric='new',probe_corrections=0,max_corrections_per_step=1,
    half_fallback=False,tune=False,replace=False,generation=False,judge=False,full16=False)


def run_arm(runtime,record,factor,diagonal,state,emit,deadline,history=None,probe_history=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    main_metric=diagonal if state['arm'][0]=='1' else None
    correction_metric=diagonal if state['arm'][1]=='1' else None
    try:
        baseline=baseline_check(runtime,record,factor)
        if history is not None:
            compare(baseline,history['baseline'],'baseline',state['checks'])
            compare(baseline['anchor'],history['baseline']['anchor'],'anchor',state['checks'])
        state['baseline']=baseline;state['steps']=[];state['status']='RUNNING';emit()
        for i in range(8):
            check_time(deadline)
            start=layer.weight.detach().clone();losses=native_losses(runtime,pair)
            row=dict(step=i+1,start_native_nll=losses,trial={});state['steps'].append(row)
            verdict=_attempt(runtime,record,factor,pair,start,losses,baseline['native'][1],FIRST_D/16,None,
                row['trial'],emit,main_metric,correction_factor=correction_metric,separate_correction=True)
            if history is not None:
                previous=history['method']['steps'][i]
                if previous['selected_attempt']!='primary':raise ValueError('Historical prefix used a half step')
                compare(dict(native_before=losses),dict(native_before=previous['start_native_nll']),f'step{i+1}.start',state['checks'])
                control(row['trial'],previous['primary'],f'step{i+1}',state['checks'])
            row['accepted']=verdict['accepted']
            if not verdict['accepted']:
                row['rollback_pass']=_rollback(layer,pair,runtime,start,losses)
                if not row['rollback_pass']:raise ValueError('Rejected full-step restoration failed')
                state['status']='STOPPED_REJECTED';emit();break
            row['status']='ACCEPTED';emit()
        accepted=[r for r in state['steps'] if r['accepted']]
        state['accepted_steps']=len(accepted);state['prefix_completed']=len(accepted)==8
        state['endpoint_native_nll']=native_losses(runtime,pair)
        state['accepted_requested_reduction']=sum(r['trial']['proposal']['requested_reduction'] for r in accepted)
        state['accepted_effective_reduction']=sum(r['trial']['proposal']['effective_reduction'] for r in accepted)
        state['all_main_unclipped']=all(not r['trial']['proposal']['clipped'] for r in state['steps'])
        if state['prefix_completed']:
            check_time(deadline)
            endpoint=layer.weight.detach().clone();endpoint_losses=native_losses(runtime,pair)
            state['probe']={};emit()
            try:
                p=candidate(runtime,record,0,factor,FIRST_D/16,retain=True,solve_factor=diagonal)
                state['probe']['proposal']=p
                state['probe']['verdict']=gates(runtime,pair,factor,endpoint,endpoint_losses,baseline['native'][1])
                if probe_history is not None:
                    compare(p,probe_history['proposal'],'probe',state['checks'])
                    if state['probe']['verdict']['gates']!=probe_history['proposal_verdict']['gates']:
                        raise ValueError('Historical probe gate changed')
            finally:
                state['probe']['endpoint_restored']=_rollback(layer,pair,runtime,endpoint,endpoint_losses)
                if not state['probe']['endpoint_restored']:raise ValueError('Probe did not restore endpoint')
            state['status']='COMPLETE'
        else:
            state['probe']=dict(status='NOT_RUN',reason='INCOMPLETE_EIGHT_STEP_PREFIX')
        emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        state['base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses
        emit()
        if not state['base_restored'] or layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Arm base restoration failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only S03 eight-step cross authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);started=time.time();deadline=config['deadline_epoch']
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
    states=[dict(arm=a,sample='S03',status='NOT_RUN',checks=[]) for a in ARMS]
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
        old=read(config['old_result'])['samples'][2];new=read(config['new_result'])['samples'][2]
        if old['sample']!='S03' or new['sample']!='S03':raise ValueError('Sample alias changed')
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
        probe00=read(config['normmatch_result'])['points'][1]['arms']['B']
        for index,state in enumerate(states):
            check_time(deadline)
            write(root/'public/STATUS.json',dict(status='RUNNING',arm=state['arm']))
            history=old if state['arm']=='00' else new if state['arm']=='11' else None
            probe_history=probe00 if state['arm']=='00' else new['method']['steps'][8]['primary'] if state['arm']=='11' else None
            run_arm(runtime,record,factor,diagonal,state,emit,deadline,history,probe_history)
            if index==1:
                if not all(s['prefix_completed'] for s in states[:2]):raise ValueError('Control prefixes incomplete')
                write(root/'public/CONTROL_GATE.json',dict(status='PASS',controls=['00','11'],both_before_mixed=True))
        write(root/'public/RESULT.json',dict(status='COMPLETE',denominator=4,sample='S03',arms=states,
            completed_prefixes=sum(s['prefix_completed'] for s in states),probe_count=sum('proposal' in s['probe'] for s in states),
            all_base_restored=all(s['base_restored'] for s in states),elapsed_seconds=time.time()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_forward_calls=calls[0],generation_calls=0,judge_calls=0,
            half_candidates=0,full16_runs=0))
        write(root/'public/STATUS.json',dict(status='COMPLETE',arms=4))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit()
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:
        if handle is not None:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started))
