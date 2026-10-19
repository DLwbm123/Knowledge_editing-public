"""Frozen first-eight benchmark independent edits and formal generation."""
import json,os,sys,time,traceback,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from .prepare import write
from .batch8 import baseline_check
from .backoff16 import trajectory
from .diagnose import candidate,FIRST_D,DegenerateCandidateError
from .finite_null import native_losses
from .editor import paired_batches

AUTH=dict(cohort='benchmark_first8',samples=8,steps=16,metric='F0',secant=True,unique_half=True,independent=True,
    max_generation=250,max_new_judge=246,gpu_seconds=3600,no_tuning=True)


def main(config):
    if config['authorization']!=AUTH:raise PermissionError('Frozen formal evaluation only')
    root=Path(config['output']);started=time.time();deadline=config['deadline_epoch'];calls=[197];generation=[5];states=[];runtime=None;handle=None
    def emit():write(root/'public/SAMPLES.json',states)
    def check():
        if time.time()>=deadline:raise TimeoutError('Generation/edit time budget')
    def generate(row,alias,phase):
        check()
        path=root/'private/outputs'/phase/(alias+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        marker=path.with_suffix('.STARTED.json')
        if path.exists() or marker.exists():
            allowed={'base':('Q001','Q002','Q003'),'base_repeat':('Q001','Q002')}
            if phase not in allowed or alias not in allowed[phase] or not path.exists() or not marker.exists():raise FileExistsError('No generation replay')
            prior=json.loads(path.read_text())
            if prior['status']!='COMPLETE':raise ValueError('Prior output not complete')
            return prior
        if generation[0]>=250:raise RuntimeError('Generation ceiling')
        raw=runtime.adapter.prepare_inputs(Path(row['image_path']),row['question'],None)
        bound=bindings[row['opaque_Base_id']]
        if raw['image_sha256']!=bound['image_sha256'] or raw['input_ids'].tolist()!=[bound['prompt_ids']] or raw['attention_mask'].tolist()!=[bound['attention_mask']]:raise ValueError('Frozen generation input changed')
        write(marker,dict(alias=alias,phase=phase,epoch=time.time()));generation[0]+=1
        write(root/'public/COUNTS.json',dict(generation_attempts=generation[0],model_forward_calls=calls[0]))
        try:
            with torch.inference_mode():value=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            result=dict(status='COMPLETE',raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids),token_cap_hit=len(value.raw_token_ids)>=runtime.generation_config['max_new_tokens'])
        except BaseException as e:
            write(path,dict(status='FAILED',error_type=type(e).__name__));raise
        write(path,result);return result
    try:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Missing deterministic workspace')
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0,config['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['source_root'],text=True).strip()!=config['source_commit']:raise ValueError('Runtime source changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        manifest=json.loads((root/'private/MANIFEST.json').read_text());bindings=json.loads((root/'private/BINDINGS.json').read_text());queries=manifest['queries'];tasks=manifest['tasks']
        runtime=load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
        if runtime.generation_config!=next(iter(bindings.values()))['generation']:raise ValueError('Generation configuration changed')
        def count(m,args):calls[0]+=1
        handle=runtime.model.register_forward_pre_hook(count)
        layer=runtime.get_module('model.layers.0.mlp.down_proj')
        if layer.weight.dtype!=torch.float16 or layer._forward_hooks or layer._forward_pre_hooks:raise ValueError('Native FP16 qualification failed')
        cal=torch.load(config['calibration'],map_location='cpu',weights_only=True);factor=cal['factors']['model.layers.0.mlp.down_proj']
        if cal['binding']!=config['binding']:raise ValueError('Old F0 binding changed')
        write(root/'public/STATUS.json',dict(status='BASE_GENERATION'))
        state_changes={}
        for i,(alias,row) in enumerate(queries.items()):
            result=generate(row,alias,'base');old=bindings[row['opaque_Base_id']]['output']
            drift=result['raw_token_ids']!=old['raw_generated_token_ids'] or result['raw_answer']!=old['model_answer_raw']
            if drift:
                changed=state_changes.setdefault('historical_output_changes',[]);changed.append(alias)
                write(root/'public/BASE_CHANGES.json',state_changes)
            if i<2:
                repeated=generate(row,alias,'base_repeat')
                if repeated!=result:raise ValueError('Cached native generation repeat mismatch')
            if alias=='Q003':
                repeated=generate(row,alias,'resume_qualification')
                write(root/'public/RESUME_QUALIFICATION.json',dict(status='PASS' if repeated==result else 'FAILED',query=alias,one_attempt_only=True,text_equal=repeated['raw_answer']==result['raw_answer'],token_equal=repeated['raw_token_ids']==result['raw_token_ids']))
                if repeated!=result:raise ValueError('Q003 current Base repeat mismatch; stop without retry')
        write(root/'public/BASE_QUALIFICATION.json',dict(status='PASS',unique_inputs=len(queries),repeated_inputs=2,exact_historical_outputs=False,current_Base_pairing_authorized=True))
        for index,task in enumerate(tasks):
            check();state=dict(sample=task['alias'],status='RUNNING',steps_requested=16);states.append(state);emit()
            write(root/'public/STATUS.json',dict(status='EDITING',sample=task['alias'],completed=index))
            source=task['record'];record=EditorRecord(**dict(source,image_path=Path(source['image_path'])))
            base=layer.weight.detach().clone();pair=paired_batches(runtime,record);before=native_losses(runtime,pair);endpoint=[]
            try:
                baseline=baseline_check(runtime,record,factor);state['baseline']=baseline;reference=candidate(runtime,record,0,factor,FIRST_D/4,baseline['anchor']);state['reference']=reference;emit()
                def progress(rows):state['candidates']=rows;emit()
                result=trajectory(runtime,record,factor,progress,anchor=baseline['anchor'],edit_threshold=reference['native_after'][0],deadline_epoch=deadline,endpoint_observer=lambda w,r:endpoint.append(w.cpu().clone()))
                state['method']=result;state.pop('candidates',None);state['edit_gate_passed']=result['joint_gate'];emit()
                if not endpoint:raise ValueError('Endpoint not captured')
                with torch.no_grad():layer.weight.copy_(endpoint[0])
                for entry in task['panel']:
                    result=generate(queries[entry['query']],entry['query'],task['alias'])
                    if index==0 and entry['task']=='T0':
                        repeat=generate(queries[entry['query']],entry['query'],'edited_repeat')
                        if repeat!=result:raise ValueError('Edited native generation repeat mismatch')
                state['status']='EVALUATED';state['generated_inputs']=len(task['panel']);emit()
            except DegenerateCandidateError:
                state['status']='DEGENERATE_EDIT';state['edit_gate_passed']=False;emit()
            finally:
                with torch.no_grad():layer.weight.copy_(base)
                state['base_restored']=torch.equal(layer.weight,base) and native_losses(runtime,pair)==before
                endpoint.clear();emit()
                if not state['base_restored']:raise ValueError('Independent edit base restoration failed')
        write(root/'public/RESULT.json',dict(status='GENERATION_COMPLETE',denominator=8,samples=states,generation_calls=generation[0],model_forward_calls=calls[0],all_base_restored=all(s.get('base_restored') for s in states),judge_calls=0))
        write(root/'public/STATUS.json',dict(status='GENERATION_COMPLETE',completed=len(states)))
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());emit();write(root/'public/STATUS.json',dict(status='FAILED',error_type=type(e).__name__,generation_attempts=generation[0],model_forward_calls=calls[0]));raise
    finally:
        if handle:handle.remove()
        write(root/'ENDED.json',dict(pid=os.getpid(),epoch=time.time(),resident_seconds=time.time()-started,generation_attempts=generation[0],model_forward_calls=calls[0],peak_allocated_bytes=torch.cuda.max_memory_allocated() if runtime is not None else None))
