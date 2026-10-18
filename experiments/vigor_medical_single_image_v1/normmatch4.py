"""Four same-start controls; one native-norm ratio candidate and no scalar search."""
import json
import math
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
from .diagnose import FIRST_D
from .direct_fisher16 import estimate,check_time
from .editor import paired_batches
from .finite_null import native_losses
from .precision_abc import compare
from .prepare import write

POINTS=(('P1','S02',15,'old'),('P2','S03',9,'old'),('P3','S05',6,'old'),('P4','S02',6,'new'))
AUTH=dict(points=[list(p) for p in POINTS],arms=['A','B','C'],main_candidates=12,
    calibration_samples=32,max_corrections_per_candidate=1,norm_match_relative_tolerance=.01,
    ratio_applications=1,tune=False,replace=False,generation=False,judge=False,full16=False)


def ratio(a,b):
    if not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0:
        raise ValueError('Undefined one-shot norm ratio')
    return a/b


def control(trial,old,label,checks):
    compare(trial['proposal'],old['proposal'],label+'.proposal',checks)
    if bool(trial['correction'])!=bool(old['correction']):raise ValueError('Correction trigger changed')
    if trial['correction']:
        compare(trial['correction'],old['correction'],label+'.correction',checks)
        if trial['correction']['status']!=old['correction']['status']:raise ValueError('Correction status changed')
    for key in ('proposal_verdict','final_verdict'):
        if trial[key]['gates']!=old[key]['gates']:raise ValueError('Control gate changed')


def reconstruct(runtime,record,factor,diagonal,old,state,emit,deadline):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    checks=state['checks'];state['prefix']=[]
    try:
        baseline=baseline_check(runtime,record,factor)
        compare(baseline,old['baseline'],'baseline',checks)
        compare(baseline['anchor'],old['baseline']['anchor'],'anchor',checks)
        state['baseline']=baseline
        for previous in old['method']['steps'][:state['step']-1]:
            check_time(deadline)
            start=layer.weight.detach().clone();losses=native_losses(runtime,pair)
            compare(dict(native_before=losses),dict(native_before=previous['start_native_nll']),
                f"step{previous['step']}.start",checks)
            row=dict(step=previous['step'],primary={},half=None);state['prefix'].append(row)
            verdict=_attempt(runtime,record,factor,pair,start,losses,baseline['native'][1],
                FIRST_D/16,None,row['primary'],emit,diagonal)
            control(row['primary'],previous['primary'],f"step{previous['step']}.primary",checks)
            if not verdict['accepted']:
                row['primary']['rollback_pass']=_rollback(layer,pair,runtime,start,losses)
                if not row['primary']['rollback_pass']:raise ValueError('Prefix rollback failed')
                check_time(deadline);row['half']={}
                verdict=_attempt(runtime,record,factor,pair,start,losses,baseline['native'][1],
                    FIRST_D/32,None,row['half'],emit,diagonal)
                control(row['half'],previous['half'],f"step{previous['step']}.half",checks)
            if not verdict['accepted']:raise ValueError('Frozen prefix no longer accepted')
            row['final_verdict']=verdict;emit()
        losses=native_losses(runtime,pair)
        compare(dict(native_before=losses),dict(native_before=old['method']['steps'][state['step']-1]['start_native_nll']),
            'target.start',checks)
        state['start_native_nll']=losses;state['reconstruction']='PASS';emit()
        return layer.weight.detach().cpu().clone()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        state['reconstruction_base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses
        emit()
        if not state['reconstruction_base_restored']:raise ValueError('Prefix base restoration failed')


def evaluate(runtime,record,factor,diagonal,snapshot,state,arm,requested,emit,deadline):
    check_time(deadline)
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    trial={};state['arms'][arm]=trial
    try:
        with torch.no_grad():layer.weight.copy_(snapshot)
        losses=native_losses(runtime,pair)
        if losses!=state['start_native_nll'] or not torch.equal(layer.weight.cpu(),snapshot):
            raise ValueError('Candidates do not share the exact same start')
        trial['same_start_verified']=True
        _attempt(runtime,record,factor,pair,layer.weight.detach().clone(),losses,state['baseline']['native'][1],
            requested,None,trial,emit,None if arm=='A' else diagonal)
        return trial
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        trial['base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses
        if layer._forward_hooks or layer._forward_pre_hooks or not trial['base_restored']:
            raise ValueError('Candidate base restoration failed')
        emit()


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only four frozen ABC points authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);started=time.time();deadline=config['deadline_epoch']
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
    states=[dict(point=p,sample=s,step=i,source=v,status='NOT_RUN',checks=[],arms={}) for p,s,i,v in POINTS]
    def emit():write(root/'public/POINTS.json',states)
    handle=None;calls=[0]
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime source binding changed')
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
        histories={k:read(config[k+'_result'])['samples'] for k in ('old','new')}
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        def count(module,args):calls[0]+=1
        handle=runtime.model.register_forward_pre_hook(count)
        def forbidden(*args,**kwargs):raise PermissionError('Generation forbidden')
        runtime.generate=forbidden
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in gpu.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):
            raise ValueError('Process identity is not neutral')
        train=read(config['records']);ids=[r['record_id'] for r in train]
        if len(ids)!=32 or len(set(ids))!=32 or ids!=manifest['calibration_ids'] or ids!=cal['calibration_ids']:
            raise ValueError('Original training calibration changed')
        training=[EditorRecord(**dict(r,image_path=Path(r['image_path']))) for r in train]
        diagonal,metadata=estimate(runtime,training,factor,lambda e:write(root/'public/CALIBRATION_PROGRESS.json',e),min(deadline,time.time()+120))
        previous=read(config['new_fisher_summary'])
        for key in ('raw_trace','final_trace','multiplier'):
            if abs(metadata[key]/previous[key]-1)>1e-5:raise ValueError('Direct Fisher reconstruction changed')
        metadata['prior_summary_relative_tolerance']=1e-5;metadata['prior_summary_match']=True
        write(root/'public/DIRECT_FISHER.json',metadata)
        snapshots=[];records=[];olds=[]
        for state in states:
            state['status']='RECONSTRUCTING';write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',point=state['point']))
            index=int(state['sample'][1:])-1
            record=EditorRecord(**dict(raw[index],image_path=Path(raw[index]['image_path'])))
            old=histories[state['source']][index];records.append(record);olds.append(old)
            snapshots.append(reconstruct(runtime,record,factor,diagonal if state['source']=='new' else None,old,state,emit,deadline))
        write(root/'public/RECONSTRUCTION_GATE.json',dict(status='PASS',all_four_before_evaluation=True))
        for state,record,snapshot,old in zip(states,records,snapshots,olds):
            state['status']='EVALUATING_AB';write(root/'public/STATUS.json',dict(status='EVALUATING_AB',point=state['point']))
            for arm in ('A','B'):evaluate(runtime,record,factor,diagonal,snapshot,state,arm,FIRST_D/16,emit,deadline)
            arm='A' if state['source']=='old' else 'B'
            control(state['arms'][arm],old['method']['steps'][state['step']-1]['primary'],'target.control',state['checks'])
            state['target_control']='PASS';emit()
        write(root/'public/CONTROL_GATE.json',dict(status='PASS',all_four_before_C=True))
        for state,record,snapshot in zip(states,records,snapshots):
            state['status']='EVALUATING_C';write(root/'public/STATUS.json',dict(status='EVALUATING_C',point=state['point']))
            a,b=(state['arms'][k]['proposal']['quantization']['native_delta_l2'] for k in ('A','B'))
            scale=ratio(a,b);requested=state['arms']['B']['proposal']['effective_reduction']*scale
            state['matching']=dict(reference_native_norm=a,unscaled_B_native_norm=b,ratio=scale,applications=1,relative_tolerance=.01)
            c=evaluate(runtime,record,factor,diagonal,snapshot,state,'C',requested,emit,deadline)
            error=abs(c['proposal']['quantization']['native_delta_l2']/a-1)
            state['matching'].update(relative_error=error,passed=error<=.01)
            state['status']='COMPLETE';emit()
        write(root/'public/RESULT.json',dict(status='COMPLETE',denominator=4,points=states,
            matched=sum(s['matching']['passed'] for s in states),new_main_candidates=12,
            new_corrections=sum(bool(t.get('correction')) for s in states for t in s['arms'].values()),
            all_base_restored=all(s['reconstruction_base_restored'] and all(t['base_restored'] for t in s['arms'].values()) for s in states),
            elapsed_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_forward_calls=calls[0],
            generation_calls=0,judge_calls=0,full16_runs=0))
        write(root/'public/STATUS.json',dict(status='COMPLETE',points=4))
    except BaseException as error:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit()
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(error).__name__))
        raise
    finally:
        if handle is not None:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started))
