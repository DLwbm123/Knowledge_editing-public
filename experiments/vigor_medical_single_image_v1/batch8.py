"""Eight frozen image groups: paired reference and one-correction trajectory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import torch
from .diagnose import candidate,FIRST_D,DegenerateCandidateError
from .editor import direction,editable_fp32,gradient_pair,paired_batches
from .finite_null import trajectory,native_losses
from .prepare import write
from .batch8_selection import select_records




def baseline_check(runtime,record,factor):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');pair=paired_batches(runtime,record)
    native=native_losses(runtime,pair)
    rows=[]
    a,b=(factor[k].to(layer.weight.device) for k in ('a','b'))
    with editable_fp32(runtime.model,layer,readonly=True):
        for _ in range(2):
            losses,gradients=gradient_pair(runtime,layer,pair)
            tangent,geometry=direction(*gradients,a,b)
            if tangent is None:raise DegenerateCandidateError('Sample baseline direction is degenerate')
            rows.append(dict(visual_nll=losses[0],null_nll=losses[1],E=geometry['E']))
    if native_losses(runtime,pair)!=native:raise ValueError('Baseline native state changed')
    delta={k:abs(rows[1][k]-rows[0][k]) for k in ('visual_nll','null_nll')}
    erel=abs(rows[1]['E']/rows[0]['E']-1)
    if max(delta.values())>1e-5 or erel>1e-5:raise ValueError('Per-sample deterministic baseline mismatch')
    return dict(native=native,fp32=[rows[0]['visual_nll'],rows[0]['null_nll']],anchor=rows[0],
        repeats=rows,E_relative_difference=erel,NLL_absolute_differences=delta,status='PASS')


def stage_diagnostics(value,baseline):
    if 'fp32_after' not in value:return dict(available=False,reason=value.get('status','MISSING'))
    fp=value['fp32_after'][1]-baseline['fp32'][1]
    native=value['native_after'][1]-baseline['native'][1]
    return dict(available=True,fp32_cumulative_null_change=fp,native_cumulative_null_change=native,
        fp32_null_within_budget=abs(fp)<=.01,native_null_within_budget=abs(native)<=.01,
        null_budget_decision_flip=(abs(fp)<=.01)!=(abs(native)<=.01))


def sample_metrics(result):
    method=result.get('method')
    if not method:
        rows=result.get('method_candidates',[])
        accepted=[r['final_verdict'] for r in rows if r.get('status')=='ACCEPTED']
        maxdrift=max((abs(r['cumulative_null_change']) for r in accepted),default=None)
        return dict(joint_pass=False,accepted_steps=len(accepted),
            correction_attempts=sum(r.get('correction') is not None for r in rows)+
                sum(t.get('correction') is not None for r in rows for t in (r.get('primary',{}),r.get('half') or {})),
            max_abs_accepted_null=maxdrift,boundary_margin=.01-maxdrift if maxdrift is not None else None,
            near_boundary_pass=False,failures=[result.get('status','INCOMPLETE')])
    accepted=method['accepted_trajectory']
    maxdrift=max((abs(r['cumulative_null_change']) for r in accepted),default=None)
    failures=[]
    if not method.get('trajectory_completed',method['four_steps_completed']):
        last=method['steps'][-1]
        failures.extend(k for k,v in last.get('final_verdict',{}).get('gates',{}).items() if not v)
        if not failures:failures.append('INCOMPLETE_REQUESTED_STEPS')
    if not method['edit_threshold_met']:failures.append('EDIT_REFERENCE_NOT_MET')
    if not method['null_gate_on_accepted_states']:failures.append('NO_VALID_NULL_ACCEPTED_PATH')
    margin=.01-maxdrift if maxdrift is not None else None
    return dict(joint_pass=method['joint_gate'],accepted_steps=method['accepted_steps'],
        correction_attempts=method['correction_attempts'],max_abs_accepted_null=maxdrift,
        boundary_margin=margin,near_boundary_pass=method['joint_gate'] and margin is not None and margin<.001,
        failures=failures)


def run_sample(runtime,record,factor,state,emit, *, compensate=False,history=None,steps=4,prefix_reference=None,deadline_epoch=None,trajectory_fn=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone()
    pair=paired_batches(runtime,record);base_losses=native_losses(runtime,pair)
    try:
        state['phase']='BASELINE';emit(state)
        baseline=baseline_check(runtime,record,factor);state['baseline']=baseline;emit(state)
        state['phase']='REFERENCE';emit(state)
        reference=candidate(runtime,record,0,factor,FIRST_D/4,baseline['anchor'])
        state['reference']=reference
        if history is not None:
            from .precision_abc import compare
            checks=[]
            compare(baseline,history['baseline'],'baseline',checks)
            compare(baseline['anchor'],history['baseline']['anchor'],'anchor',checks)
            compare(reference,history['reference'],'reference',checks)
            state['historical_checks']=checks
        state['reference_diagnostics']=stage_diagnostics(reference,baseline)
        state['reference_improved_visual']=reference['native_after'][0]<base_losses[0]
        state['phase']='METHOD';emit(state)
        def progress(rows):
            state['method_candidates']=rows
            # Diagnostics never affect acceptance: each path uses its own initial baseline.
            for row in rows:
                trials=(row['primary'],row.get('half') or {}) if 'primary' in row else (row,)
                for trial in trials:
                    if 'proposal' in trial:trial['proposal_diagnostics']=stage_diagnostics(trial['proposal'],baseline)
                    if trial.get('correction'):trial['correction_diagnostics']=stage_diagnostics(trial['correction'],baseline)
            emit(state)
        run_trajectory=trajectory if trajectory_fn is None else trajectory_fn
        result=run_trajectory(runtime,record,factor,progress,anchor=baseline['anchor'],
            edit_threshold=reference['native_after'][0],compensate=compensate,steps=steps,prefix_reference=prefix_reference,deadline_epoch=deadline_epoch)
        state['method']=result;state.pop('method_candidates',None)
        state['status']='EVALUATED';state['phase']='DONE';state['metrics']=sample_metrics(state)
        return state
    finally:
        with torch.no_grad():layer.weight.copy_(base)
        restored=torch.equal(layer.weight,base) and native_losses(runtime,pair)==base_losses
        restored=restored and not layer._forward_hooks and not layer._forward_pre_hooks
        state['base_restored']=restored;emit(state)
        if not restored:raise ValueError('Cross-sample base restoration failed; stop entire batch')


def main(config, *, trajectory_fn=None,authorization=None):
    steps=config.get('trajectory_steps',4)
    if steps not in (4,16):raise PermissionError('Unfrozen horizon')
    expected=dict(samples=8,steps=steps,max_corrections_per_step=1,reference_steps=1,
        tune=False,replace=False,expand=False,generation=False,judge=False)
    compensate=config.get('amplitude_compensation',False)
    if compensate=='secant':
        if steps!=16:raise PermissionError('Only the frozen secant sixteen-step horizon is authorized')
        expected['variant']='secant_stress16'
    elif compensate:expected['variant']='one_shot_amplitude' if steps==4 else 'one_shot_amplitude_stress16'
    if steps==16 and not compensate:raise PermissionError('Stress horizon requires compensated rule')
    if authorization is not None:expected=authorization
    if config['authorization']!=expected:
        raise PermissionError('Only the frozen eight-sample batch is authorized')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True)
    root=Path(config['output']);start=time.time()
    with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=start),f)
    states=[dict(sample=f'S{i+1:02}',status='NOT_RUN',joint_pass=False) for i in range(8)]
    active=None;model_forward_count=[0];forward_handle=None
    try:
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:
            raise ValueError('Runtime binding changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        read=lambda p:json.loads(Path(p).read_text())
        history=read(config['historical_batch'])['samples'] if compensate else None
        prefix=read(config['prefix_result'])['samples'] if steps==16 and compensate!='secant' else None
        raw=read(root/'private/records.json');manifest=read(config['manifest']);pilot=read(config['record'])
        if len(raw)!=8 or len({r['record_id'] for r in raw})!=8:raise ValueError('Frozen sample count changed')
        groups=['SLAKE:'+r['relative_image_path'].split('/')[0] for r in raw]
        forbidden_groups=set(manifest['calibration_source_groups'])|set(manifest['excluded_source_groups'])|{'SLAKE:'+pilot['relative_image_path'].split('/')[0]}
        if len(set(groups))!=8 or set(groups)&forbidden_groups:raise ValueError('Frozen sample isolation failed')
        calibration=torch.load(config['calibration'],map_location='cpu',weights_only=True)
        if calibration['binding']!=config['binding'] or calibration['selected_layer']!='model.layers.0.mlp.down_proj':
            raise ValueError('Fisher binding changed')
        factor=calibration['factors']['model.layers.0.mlp.down_proj']
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        if config.get('count_model_calls',False):
            def count_forward(module,args):model_forward_count[0]+=1
            forward_handle=runtime.model.register_forward_pre_hook(count_forward)
        def forbidden(*args,**kwargs):raise PermissionError('Generation forbidden')
        runtime.generate=forbidden
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        own=[r for r in gpu.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        if len(own)!=1 or any(w in (argv+own[0]).lower() for w in ['wangbomin','knowledge_editing','vigor']):
            raise ValueError('Process command is not neutral')
        def emit(state):
            write(root/'public'/f"{state['sample']}.json",state)
            write(root/'public/SAMPLES.json',states)
        for index,row in enumerate(raw):
            active=states[index];active['status']='RUNNING'
            write(root/'public/STATUS.json',dict(status='RUNNING',sample=active['sample'],completed=index,total=8))
            record=EditorRecord(**dict(row,image_path=Path(row['image_path'])))
            try:
                run_sample(runtime,record,factor,active,emit,compensate=compensate,history=history[index] if history else None,steps=steps,
                    prefix_reference=prefix[index]['method']['steps'] if prefix else None,
                    deadline_epoch=min(start+900,config['deadline_epoch']) if steps==16 else None,trajectory_fn=trajectory_fn)
            except DegenerateCandidateError:
                active['status']='DEGENERATE_DIRECTION';active['metrics']=sample_metrics(active)
            active['joint_pass']=active.get('metrics',{}).get('joint_pass',False);emit(active)
        result=dict(status='COMPLETE',denominator=8,passed=sum(s['joint_pass'] for s in states),samples=states,
            generation_calls=0,judge_calls=0,elapsed_seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_base_restored=all(s.get('base_restored',False) for s in states))
        result['pass_rate']=result['passed']/8
        result['amplitude_compensation']=compensate
        result['model_forward_calls']=model_forward_count[0] if forward_handle is not None else None
        write(root/'public/RESULT.json',result)
        write(root/'public/STATUS.json',dict(status='COMPLETE',completed=8,total=8,passed=result['passed']))
    except BaseException as e:
        if active is not None:
            active['status']='ENGINEERING_FAILURE';active['error_type']=type(e).__name__;active['joint_pass']=False
        write(root/'public/SAMPLES.json',states)
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__,denominator=8))
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        raise
    finally:
        if forward_handle is not None:forward_handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-start))
