"""Four fixed proposals, at most one Fisher-minimal null correction per proposal."""
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
import torch.nn.functional as F
from .prepare import write
from .diagnose import candidate,DETERMINISTIC_ANCHOR,FIRST_D
from .editor import direction,editable_fp32,finite,gradient_pair,nll,paired_batches,update_metrics

NULL_BUDGET=.01
TRUST_BUDGET=.001
EDIT_THRESHOLD=4.295644


def native_losses(runtime,pair):
    with torch.no_grad():return [float(nll(runtime.model,b)) for b in pair]


def secant_multiplier(rho,eta):
    """One interpolation of quantized linear effects; no extrapolation or search."""
    if not all(math.isfinite(x) and x>0 for x in (rho,eta)):return None
    if rho==1:return 1.0
    second=1/rho
    if not math.isfinite(second) or second==1 or not min(rho,eta)<=1<=max(rho,eta):return None
    slope=(eta-rho)/(second-1)
    if not math.isfinite(slope) or slope<=0:return None
    alpha=1+(1-rho)/slope
    return alpha if math.isfinite(alpha) and alpha>0 and min(1,second)<=alpha<=max(1,second) else None


def correction(runtime,record,factor,base_null, *, evaluate_rounded=False, compensate=False, solve_factor=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj')
    original=layer.weight.detach().clone()
    pair=paired_batches(runtime,record)
    before=native_losses(runtime,pair)
    residual=before[1]-base_null
    a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    with editable_fp32(runtime.model,layer):
        fp_before,gradients=gradient_pair(runtime,layer,pair)
        # Swap the existing projection roles: correct null while protecting visual to first order.
        tangent,geometry=direction(gradients[1],gradients[0],a,b,diagonal=solve_factor)
        if tangent is None:
            return dict(status='DEGENERATE',native_before=before,native_after=before,residual=residual)
        delta=finite(-residual/geometry['E']*tangent)
        with torch.no_grad():layer.weight.add_(delta)
        actual=layer.weight.detach()-original.float()
        native_delta=layer.weight.detach().half().float()-original.float()
        norm=float(torch.linalg.vector_norm(actual))
        fp_metrics=update_metrics(actual,*gradients,a,b)
        native_metrics=update_metrics(native_delta,*gradients,a,b)
        rounding=float(torch.linalg.vector_norm(native_delta-actual))
        quant=dict(absolute_rounding_l2=rounding,relative_rounding_l2=rounding/norm if norm else None,
            delta_cosine=float(F.cosine_similarity(actual.flatten(),native_delta.flatten(),dim=0)),
            fp32_delta_l2=norm,native_delta_l2=float(torch.linalg.vector_norm(native_delta)),
            changed_elements=int(torch.count_nonzero(native_delta)))
        extra={}
        if compensate:
            rho=native_metrics['null_linear_change']/(-residual) if residual else float('nan')
            extra['compensation']=dict(rho=rho if math.isfinite(rho) else None,
                original_fp32_update=fp_metrics,original_native_update=native_metrics,original_quantization=quant)
            if not math.isfinite(rho) or rho<=0 or not math.isfinite(1/rho):
                with torch.no_grad():layer.weight.copy_(original.float())
                return dict(status='INVALID_COMPENSATION',native_before=before,native_after=before,**extra)
            extra['compensation']['multiplier']=1/rho
            # Rebuild from the same pre-correction weight, never add to the rounded control.
            with torch.no_grad():layer.weight.copy_(finite(original.float()+delta/rho))
            actual=layer.weight.detach()-original.float()
            native_delta=layer.weight.detach().half().float()-original.float()
            norm=float(torch.linalg.vector_norm(actual))
            fp_metrics=update_metrics(actual,*gradients,a,b)
            native_metrics=update_metrics(native_delta,*gradients,a,b)
            rounding=float(torch.linalg.vector_norm(native_delta-actual))
            quant=dict(absolute_rounding_l2=rounding,relative_rounding_l2=rounding/norm if norm else None,
                delta_cosine=float(F.cosine_similarity(actual.flatten(),native_delta.flatten(),dim=0)),
                fp32_delta_l2=norm,native_delta_l2=float(torch.linalg.vector_norm(native_delta)),
                changed_elements=int(torch.count_nonzero(native_delta)))
            extra['compensation']['realized_effect_ratio_after']=native_metrics['null_linear_change']/(-residual)
        if compensate=='secant':
            eta=native_metrics['null_linear_change']/(-residual)
            alpha=secant_multiplier(rho,eta)
            extra['compensation']['secant']=dict(first_scale=1,first_effect=rho,
                second_scale=1/rho,second_effect=eta if math.isfinite(eta) else None,
                multiplier=alpha,valid=alpha is not None)
            if alpha is None:
                with torch.no_grad():layer.weight.copy_(original.float())
                return dict(status='INVALID_SECANT',native_before=before,native_after=before,**extra)
            extra['compensation']['inverse_rho_multiplier']=1/rho
            extra['compensation']['multiplier']=alpha
            with torch.no_grad():layer.weight.copy_(finite(original.float()+delta*alpha))
            actual=layer.weight.detach()-original.float()
            native_delta=layer.weight.detach().half().float()-original.float()
            norm=float(torch.linalg.vector_norm(actual))
            fp_metrics=update_metrics(actual,*gradients,a,b)
            native_metrics=update_metrics(native_delta,*gradients,a,b)
            rounding=float(torch.linalg.vector_norm(native_delta-actual))
            quant=dict(absolute_rounding_l2=rounding,relative_rounding_l2=rounding/norm if norm else None,
                delta_cosine=float(F.cosine_similarity(actual.flatten(),native_delta.flatten(),dim=0)),
                fp32_delta_l2=norm,native_delta_l2=float(torch.linalg.vector_norm(native_delta)),
                changed_elements=int(torch.count_nonzero(native_delta)))
            extra['compensation']['realized_effect_ratio_after']=native_metrics['null_linear_change']/(-residual)
        fp_after=native_losses(runtime,pair)
        if evaluate_rounded:
            unrounded=layer.weight.detach().clone()
            rounded=unrounded.half()
            try:
                with torch.no_grad():layer.weight.copy_(rounded.float())
                extra['rounded_fp32_after']=native_losses(runtime,pair)
                extra['rounded_weights_equal']=torch.equal(layer.weight,rounded.float())
            finally:
                with torch.no_grad():layer.weight.copy_(unrounded)
    after=native_losses(runtime,pair)
    if evaluate_rounded:
        extra['native_weights_equal_rounded']=torch.equal(layer.weight,rounded)
    return dict(status='APPLIED',residual=residual,target_native_null=base_null,
        native_before=before,native_after=after,fp32_before=fp_before,fp32_after=fp_after,
        null_correction_energy=geometry['E'],protected_visual_directional_derivative=geometry['null_directional_derivative'],
        fp32_update=fp_metrics,native_update=native_metrics,quantization=quant,**extra)


def gates(runtime,pair,factor,start_weight,start_losses,base_null):
    layer=runtime.get_module('model.layers.0.mlp.down_proj')
    losses=native_losses(runtime,pair)
    delta=layer.weight.detach().float()-start_weight.float()
    a,b=(factor[k].to(layer.weight.device).double() for k in ('a','b'))
    cost=float((delta.double().square()*b[:,None]*a[None,:]).sum())/2
    drift=losses[1]-base_null
    verdict=dict(null=abs(drift)<=NULL_BUDGET,visual=losses[0]<start_losses[0],
        trust=math.isfinite(cost) and cost<=TRUST_BUDGET)
    return dict(native_nll=losses,cumulative_null_change=drift,combined_native_fisher_cost=cost,
        gates=verdict,accepted=all(verdict.values()))


def trajectory(runtime,record,factor,emit,anchor=None,first_reference=None, *, edit_threshold=EDIT_THRESHOLD, compensate=False, steps=4, prefix_reference=None, deadline_epoch=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj')
    original=layer.weight.detach().clone();pair=paired_batches(runtime,record)
    baseline=native_losses(runtime,pair);rows=[];accepted=[]
    try:
        for i in range(steps):
            if deadline_epoch is not None and time.time()>=deadline_epoch:
                raise TimeoutError("Frozen trajectory time budget exhausted")
            start_weight=layer.weight.detach().clone();start_losses=native_losses(runtime,pair)
            row=dict(step=i+1,start_native_nll=start_losses,status='PROPOSING',correction=None)
            rows.append(row);emit(rows)
            row['proposal']=candidate(runtime,record,0,factor,FIRST_D/16,anchor if i==0 else None,retain=True)
            row['proposal_verdict']=gates(runtime,pair,factor,start_weight,start_losses,baseline[1])
            row['status']='PROPOSED';emit(rows)
            if i==0 and first_reference is not None:
                if any(abs(a-b)>1e-5 for a,b in zip(row['proposal']['native_after'],first_reference)):
                    raise ValueError('First proposal reference mismatch')
            if not row['proposal_verdict']['gates']['null']:
                row['correction']=correction(runtime,record,factor,baseline[1],compensate=compensate)
                row['status']='CORRECTED';emit(rows)
            verdict=gates(runtime,pair,factor,start_weight,start_losses,baseline[1])
            if row['correction'] is not None and row['correction']['status']!='APPLIED':
                verdict['accepted']=False
                verdict['gates']['correction_nondegenerate']=False
            row['final_verdict']=verdict
            if prefix_reference is not None and i<len(prefix_reference):
                from .precision_abc import compare
                previous=prefix_reference[i];checks=[]
                compare(row['proposal'],previous['proposal'],'prefix.proposal',checks)
                if bool(row['correction'])!=bool(previous['correction']):
                    raise ValueError('Prefix correction decision changed')
                if row['correction']:
                    compare(row['correction'],previous['correction'],'prefix.correction',checks)
                compare(dict(native_after=verdict['native_nll']),
                    dict(native_after=previous['final_verdict']['native_nll']),'prefix.final',checks)
                if verdict['accepted']!=previous['final_verdict']['accepted']:
                    raise ValueError('Prefix acceptance changed')
                row['prefix_checks']=checks
            if verdict['accepted']:
                row['status']='ACCEPTED'
                accepted.append(dict(step=i+1,native_nll=verdict['native_nll'],
                    cumulative_visual_gain=baseline[0]-verdict['native_nll'][0],
                    cumulative_null_change=verdict['cumulative_null_change']))
                emit(rows)
            else:
                with torch.no_grad():layer.weight.copy_(start_weight)
                row['rollback_pass']=torch.equal(layer.weight,start_weight) and native_losses(runtime,pair)==start_losses
                row['status']='REJECTED_AND_STOPPED';emit(rows)
                if not row['rollback_pass']:raise ValueError('Rejected step rollback failed')
                break
        final=native_losses(runtime,pair)
        complete=len(accepted)==steps
        edit=final[0]<=edit_threshold and final[0]<baseline[0]
        null=bool(accepted) and all(abs(s['cumulative_null_change'])<=NULL_BUDGET for s in accepted)
        return dict(status='COMPLETE' if complete else 'STOPPED_REJECTED',baseline=baseline,
            steps=rows,accepted_trajectory=accepted,attempted_steps=len(rows),accepted_steps=len(accepted),
            correction_attempts=sum(s['correction'] is not None for s in rows),four_steps_completed=len(accepted)>=4,trajectory_completed=complete,requested_steps=steps,
            final_accepted_native_nll=final,edit_threshold_met=edit,null_gate_on_accepted_states=null,
            joint_gate=complete and edit and null,generation_calls=0,judge_calls=0)
    finally:
        with torch.no_grad():layer.weight.copy_(original)
        if not torch.equal(layer.weight,original) or native_losses(runtime,pair)!=baseline or layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Final base restoration failed')


def main(config):
    if config['authorization']!=dict(steps=4,max_corrections_per_step=1,tune=False,expand=False,generation=False,judge=False):
        raise PermissionError('Only the frozen four-step trajectory is authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);start=time.time()
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=start),f)
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime source binding changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        read=lambda p:json.loads(Path(p).read_text())
        raw=read(config['record']);manifest=read(config['manifest'])
        group='SLAKE:'+raw['relative_image_path'].split('/')[0]
        if group in set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups']):
            raise ValueError('Training sample overlap')
        record=EditorRecord(**dict(raw,image_path=Path(raw['image_path'])))
        calibration=torch.load(config['calibration'],map_location='cpu',weights_only=True)
        if calibration['binding']!=config['binding'] or calibration['selected_layer']!='model.layers.0.mlp.down_proj':
            raise ValueError('Calibration binding changed')
        old=read(config['multistep_result'])
        if old['status']!='COMPLETE' or len(old['steps'])!=4:raise ValueError('Missing old four-step comparison')
        reference=read(config['four_candidate_result'])
        if reference['status']!='COMPLETE':raise ValueError('Missing completed control')
        single=next(c for c in reference['candidates'] if c['name']=='sixteenth')
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        def forbidden(*args,**kwargs):raise PermissionError('Generation forbidden')
        runtime.generate=forbidden
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in gpu.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):
            raise ValueError('Process command is not neutral')
        write(root/'public/STATUS.json',dict(status='RUNNING'))
        def emit(rows):write(root/'public/CANDIDATES.json',rows)
        result=trajectory(runtime,record,calibration['factors']['model.layers.0.mlp.down_proj'],emit,
            anchor=DETERMINISTIC_ANCHOR,first_reference=single['native_after'])
        result.update(base_restored=True,elapsed_seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),first_step_reference_pass=True,
            deterministic_anchor=DETERMINISTIC_ANCHOR)
        write(root/'public/RESULT.json',result)
        write(root/'public/STATUS.json',dict(status=result['status'],accepted_steps=result['accepted_steps'],joint_gate=result['joint_gate']))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-start))
