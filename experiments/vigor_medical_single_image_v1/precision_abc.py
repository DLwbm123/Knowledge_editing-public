"""Frozen three-point precision attribution; no new optimization steps."""
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
from .batch8 import baseline_check
from .diagnose import candidate,FIRST_D
from .editor import paired_batches
from .finite_null import correction,gates,native_losses,NULL_BUDGET,TRUST_BUDGET
from .prepare import write

POINTS=(('S01',4),('S03',3),('S06',3))
AUTH=dict(points=[list(p) for p in POINTS],extra_steps=0,extra_corrections=0,tune=False,replace=False,generation=False,judge=False)


def compare(current,old,label,checks):
    """Original NLL absolute / energy relative tolerances, with explicit receipts."""
    for key in ('native_before','native_after','fp32_before','fp32_after','native','fp32'):
        if key in old:
            diff=max(abs(a-b) for a,b in zip(current[key],old[key]))
            checks.append(dict(field=label+'.'+key,absolute_difference=diff,tolerance=1e-5))
            if not math.isfinite(diff) or diff>1e-5:raise ValueError('Reconstruction NLL mismatch: '+label+'.'+key)
    for key in ('E','null_correction_energy'):
        if key in old:
            diff=abs(current[key]/old[key]-1)
            checks.append(dict(field=label+'.'+key,relative_difference=diff,tolerance=1e-5))
            if not math.isfinite(diff) or diff>1e-5:raise ValueError('Reconstruction energy mismatch: '+label+'.'+key)


def reconstruct(runtime,record,factor,old,target,state,emit):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);losses=native_losses(runtime,pair)
    checks=state['checks'];rows=state['prefix']
    try:
        baseline=baseline_check(runtime,record,factor)
        compare(baseline,old['baseline'],'baseline',checks)
        compare(baseline['anchor'],old['baseline']['anchor'],'baseline.anchor',checks)
        state['baseline']=baseline;emit()
        if len(old['method']['steps'])!=target:raise ValueError('Frozen failure position changed')
        for previous in old['method']['steps']:
            i=previous['step'];start_weight=layer.weight.detach().clone();start_losses=native_losses(runtime,pair)
            compare({'native_before':start_losses},{'native_before':previous['start_native_nll']},f'step{i}.start',checks)
            proposal=candidate(runtime,record,0,factor,FIRST_D/16,baseline['anchor'] if i==1 else None,retain=True)
            compare(proposal,previous['proposal'],f'step{i}.proposal',checks)
            verdict=gates(runtime,pair,factor,start_weight,start_losses,baseline['native'][1])
            if verdict['gates']!=previous['proposal_verdict']['gates']:raise ValueError('Proposal gate changed')
            row=dict(step=i,proposal=proposal,proposal_verdict=verdict);rows.append(row);emit()
            if i==target:
                if previous['status']!='REJECTED_AND_STOPPED' or verdict['gates']['null'] or not previous['correction']:
                    raise ValueError('Point is not the frozen rejected correction')
                state['reconstruction']='PASS';emit()
                return dict(weight=layer.weight.detach().cpu().clone(),start_weight=start_weight.cpu(),start_losses=start_losses,
                    start_fp32=proposal['fp32_before'],baseline=baseline)
            if previous['correction']:
                row['correction']=correction(runtime,record,factor,baseline['native'][1])
                compare(row['correction'],previous['correction'],f'step{i}.correction',checks)
            row['final_verdict']=gates(runtime,pair,factor,start_weight,start_losses,baseline['native'][1])
            if not row['final_verdict']['accepted'] or row['final_verdict']['gates']!=previous['final_verdict']['gates']:
                raise ValueError('Accepted prefix changed')
            emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        state['reconstruction_base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==losses
        emit()
        if not state['reconstruction_base_restored']:raise ValueError('Reconstruction base restore failed')


def attribution(c,baseline,start_fp32,start_native,cost):
    results={}
    for name,key,base,start in [('A','fp32_after',baseline['fp32'],start_fp32),
        ('B','rounded_fp32_after',baseline['fp32'],start_fp32),('C','native_after',baseline['native'],start_native)]:
        after=c[key];drift=after[1]-base[1]
        verdict=dict(null=abs(drift)<=NULL_BUDGET,visual=after[0]<start[0],trust=math.isfinite(cost) and cost<=TRUST_BUDGET)
        results[name]=dict(nll=after,initial_nll=base,step_start_nll=start,cumulative_null_change=drift,
            null_margin=NULL_BUDGET-abs(drift),visual_gain_from_step_start=start[0]-after[0],
            combined_native_fisher_cost=cost,gates=verdict,all_gates=all(verdict.values()),original_method=(name=='C'))
    a,b,c0=(results[k]['cumulative_null_change'] for k in 'ABC')
    raw_compute=results['C']['nll'][1]-results['B']['nll'][1]
    return dict(branches=results,weight_rounding_null_effect=b-a,
        computation_null_effect_adjusted=c0-b,computation_null_effect_raw=raw_compute,
        initial_precision_null_offset=baseline['native'][1]-baseline['fp32'][1],
        rounding_margin_consumed=abs(b)-abs(a),computation_margin_consumed=abs(c0)-abs(b),
        total_null_difference=c0-a)


def evaluate(runtime,record,factor,snapshot,old,state,emit):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        with torch.no_grad():layer.weight.copy_(snapshot['weight'])
        c=correction(runtime,record,factor,snapshot['baseline']['native'][1],evaluate_rounded=True)
        state['correction']=c;emit()
        if not c.get('rounded_weights_equal') or not c.get('native_weights_equal_rounded'):
            raise ValueError('B/C weight equality failed')
        compare(c,old['method']['steps'][-1]['correction'],'target.correction',state['checks'])
        verdict=gates(runtime,pair,factor,snapshot['start_weight'].to(layer.weight.device),snapshot['start_losses'],snapshot['baseline']['native'][1])
        if verdict['gates']!=old['method']['steps'][-1]['final_verdict']['gates']:
            raise ValueError('Target native gate changed')
        state['native_verdict']=verdict
        state['attribution']=attribution(c,snapshot['baseline'],snapshot['start_fp32'],snapshot['start_losses'],verdict['combined_native_fisher_cost'])
        state['status']='COMPLETE';emit()
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        state['base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses and not layer._forward_hooks and not layer._forward_pre_hooks
        emit()
        if not state['base_restored']:raise ValueError('ABC base restore failed')


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Only three frozen ABC points authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);start=time.time()
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=start),f)
    states=[dict(sample=s,step=i,status='NOT_RUN',checks=[],prefix=[]) for s,i in POINTS]
    def emit():write(root/'public/POINTS.json',states)
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime binding changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        read=lambda p:json.loads(Path(p).read_text())
        parent=Path(config['batch8_root']);raw=read(parent/'private/records.json');manifest=read(config['manifest']);pilot=read(config['record'])
        groups=['SLAKE:'+r['relative_image_path'].split('/')[0] for r in raw]
        excluded=set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups'])|{'SLAKE:'+pilot['relative_image_path'].split('/')[0]}
        if len(raw)!=8 or len(set(groups))!=8 or set(groups)&excluded:raise ValueError('Original isolation failed')
        calibration=torch.load(config['calibration'],map_location='cpu',weights_only=True)
        if calibration['binding']!=config['binding'] or calibration['selected_layer']!='model.layers.0.mlp.down_proj':raise ValueError('Fisher binding changed')
        factor=calibration['factors']['model.layers.0.mlp.down_proj']
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        def forbidden(*args,**kwargs):raise PermissionError('Generation forbidden')
        runtime.generate=forbidden
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in gpu.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):raise ValueError('Process command not neutral')
        snapshots=[];records=[];olds=[]
        for state in states:
            state['status']='RECONSTRUCTING';emit()
            write(root/'public/STATUS.json',dict(status='RECONSTRUCTING',sample=state['sample']))
            raw_record=raw[int(state['sample'][1:])-1]
            record=EditorRecord(**dict(raw_record,image_path=Path(raw_record['image_path'])))
            old=read(parent/'public'/f"{state['sample']}.json")
            records.append(record);olds.append(old)
            snapshots.append(reconstruct(runtime,record,factor,old,state['step'],state,emit))
        write(root/'public/RECONSTRUCTION_GATE.json',dict(status='PASS',points=3,all_before_ABC=True,NLL_absolute_tolerance=1e-5,E_relative_tolerance=1e-5))
        for state,record,snapshot,old in zip(states,records,snapshots,olds):
            state['status']='EVALUATING_ABC';emit()
            write(root/'public/STATUS.json',dict(status='EVALUATING_ABC',sample=state['sample']))
            evaluate(runtime,record,factor,snapshot,old,state,emit)
        write(root/'public/RESULT.json',dict(status='COMPLETE',points=states,generation_calls=0,judge_calls=0,
            extra_optimization_steps=0,extra_corrections=0,all_base_restored=all(s['base_restored'] for s in states),
            elapsed_seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        write(root/'public/STATUS.json',dict(status='COMPLETE',points=3))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        emit();write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-start))
