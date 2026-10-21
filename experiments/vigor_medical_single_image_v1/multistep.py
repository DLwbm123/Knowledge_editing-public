"""Four fixed native-writeback steps on one training item, without generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import torch
from .diagnose import candidate, DETERMINISTIC_ANCHOR, FIRST_D
from .editor import paired_batches,nll
from .prepare import write


def trajectory(runtime,record,factor,emit,anchor=None,first_reference=None):
    layer=runtime.get_module('model.layers.0.mlp.down_proj')
    original=layer.weight.detach().clone()
    pair=paired_batches(runtime,record)
    def losses():
        with torch.no_grad():
            return [float(nll(runtime.model,b)) for b in pair]
    baseline=losses()
    rows=[]
    try:
        for i in range(4):
            value=candidate(runtime,record,0,factor,FIRST_D/16,anchor if i==0 else None,retain=True)
            row=dict(step=i+1,**value)
            row['cumulative_visual_gain']=baseline[0]-value['native_after'][0]
            row['cumulative_null_change']=value['native_after'][1]-baseline[1]
            rows.append(row)
            emit(rows)
            if i==0 and first_reference is not None:
                if any(abs(a-b)>1e-5 for a,b in zip(value['native_after'],first_reference)):
                    raise ValueError('First-step native reference mismatch; do not continue')
        result=dict(status='COMPLETE',baseline=baseline,steps=rows,
            edit_gate=rows[-1]['native_after'][0]<=4.295644,
            null_gate=all(abs(r['cumulative_null_change'])<=.01 for r in rows),
            matched_prediction_budget=not any(r['clipped'] for r in rows),
            total_predicted_reduction=sum(r['effective_reduction'] for r in rows),
            generation_calls=0,judge_calls=0)
        result['joint_gate']=result['edit_gate'] and result['null_gate']
        return result
    finally:
        with torch.no_grad():layer.weight.copy_(original)
        if not torch.equal(layer.weight,original) or losses()!=baseline or layer._forward_hooks or layer._forward_pre_hooks:
            raise ValueError('Trajectory base restoration failed')


def main(config):
    if config['authorization']!=dict(steps=4,tune=False,expand=False,generation=False,judge=False):
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
        def emit(rows):write(root/'public/TRAJECTORY.json',rows)
        result=trajectory(runtime,record,calibration['factors']['model.layers.0.mlp.down_proj'],emit,
            anchor=DETERMINISTIC_ANCHOR,first_reference=single['native_after'])
        result.update(base_restored=True,elapsed_seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),first_step_reference_pass=True,
            deterministic_anchor=DETERMINISTIC_ANCHOR)
        write(root/'public/RESULT.json',result)
        write(root/'public/STATUS.json',dict(status='COMPLETE',complete=4,joint_gate=result['joint_gate']))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc())
        write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__))
        raise
    finally:write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-start))
