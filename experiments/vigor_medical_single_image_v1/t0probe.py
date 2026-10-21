"""E01 frozen reconstruction and two-endpoint token/prefill/first-step audit."""
import json,os,sys,time,traceback,subprocess,math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .prepare import write
from .batch8 import baseline_check
from .backoff16 import trajectory
from .diagnose import candidate,FIRST_D
from .editor import paired_batches
from .finite_null import native_losses
from .precision_abc import compare

AUTH=dict(sample='E01',steps=16,metric='F0',secant=True,unique_half=True,forward_max=600,seconds=900,observation_forward_max=16,generation_max=2,judge=0,retry=0)


def token_stat(logits,target):
    v=logits.detach().float().cpu();logp=F.log_softmax(v,dim=-1);best=int(v.argmax())
    return dict(nll=float(-logp[target]),probability=float(logp[target].exp()),rank=int((v>v[target]).sum())+1,
        top1_matches_target=best==target,top1_margin_over_target=float(v[best]-v[target]))


def token_parts(logits,labels,eos):
    y=labels[:,1:];z=logits[:,:-1].float();mask=y!=-100;ids=y[mask].tolist();v=z[mask]
    rows=[dict(position=i+1,is_eos=t==eos,**token_stat(a,t)) for i,(a,t) in enumerate(zip(v,ids))]
    return dict(mean_nll=sum(r['nll'] for r in rows)/len(rows),tokens=rows,content_mean_nll=sum(r['nll'] for r in rows if not r['is_eos'])/sum(not r['is_eos'] for r in rows),eos_nll=sum(r['nll'] for r in rows if r['is_eos']),target_count=len(rows)),v[0].detach().float().cpu()


def main(c):
    if c['authorization']!=AUTH:raise PermissionError('Frozen E01 only')
    root=Path(c['output']);started=time.time();rt=None;layer=None;base=None;pair=None;before=None;h=None
    calls=0;observed=0;generations=0;restoring=False;state=dict(status='LOADING',checks=[],endpoints={});captured=[]
    def emit():write(root/'public/RESULT.json',state)
    def check():
        if time.time()>=c['deadline_epoch']:raise TimeoutError('Work time ceiling')
    def count(m,args):
        nonlocal calls,observed
        if calls>=600 or (not restoring and (calls>=580 or time.time()>=c['deadline_epoch'])):raise TimeoutError('Forward/time ceiling; restoration reserve')
        if not restoring and state['status'].startswith('OBSERVING_'):
            if observed>=16:raise RuntimeError('Observation ceiling')
            observed+=1
        calls+=1
    def eq(a,b,label):
        ok=a==b;state['checks'].append(dict(field=label,passed=ok));emit()
        if not ok:raise ValueError('Frozen structural mismatch: '+label)
    def near(a,b,label):
        diff=abs(a-b);state['checks'].append(dict(field=label,absolute_difference=diff,tolerance=1e-5));emit()
        if not math.isfinite(diff) or diff>1e-5:raise ValueError('Frozen scalar mismatch: '+label)
    try:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=started),f)
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Workspace missing')
        torch.use_deterministic_algorithms(True);sys.path.insert(0,c['source_root'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()!=c['source_commit']:raise ValueError('Runtime source changed')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        prior=Path(c['prior']);manifest=json.loads((prior/'private/MANIFEST.json').read_text());binding=json.loads((prior/'private/BINDINGS.json').read_text())
        old=json.loads((prior/'public/SAMPLES.json').read_text())[0];t=manifest['tasks'][0];eq(t['alias'],'E01','fixed_first_edit')
        row=manifest['queries'][next(e['query'] for e in t['panel'] if e['task']=='T0')];record=EditorRecord(**dict(t['record'],image_path=Path(t['record']['image_path'])))
        eq(record.question,row['question'],'source_question');eq(record.target,row['reference'],'source_target')
        rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));h=rt.model.register_forward_pre_hook(count)
        flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];generation_state=rt.model.generation_config.to_dict()
        layer=rt.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone();eq(str(layer.weight.dtype),'torch.float16','native_dtype')
        pair=paired_batches(rt,record);before=native_losses(rt,pair)
        cal=torch.load(c['calibration'],map_location='cpu',weights_only=True);eq(cal['binding'],c['binding'],'F0_binding');factor=cal['factors']['model.layers.0.mlp.down_proj']
        baseline=baseline_check(rt,record,factor);state['baseline']=baseline;emit();compare(baseline,old['baseline'],'baseline',state['checks']);compare(baseline['anchor'],old['baseline']['anchor'],'anchor',state['checks']);emit()
        ref=candidate(rt,record,0,factor,FIRST_D/4,baseline['anchor']);state['reference']=ref;emit();compare(ref,old['reference'],'reference',state['checks']);emit()
        raw=rt.adapter.prepare_inputs(record.image_path,record.question,None);bound=binding[row['opaque_Base_id']]
        eq(raw['input_ids'].tolist(),[bound['prompt_ids']],'generation_prompt');eq(raw['attention_mask'].tolist(),[bound['attention_mask']],'generation_mask');eq(raw['image_sha256'],bound['image_sha256'],'generation_image')
        def observe(name,expected):
            nonlocal observed,generations
            check();state['status']='OBSERVING_'+name;emit();w=layer.weight.detach().clone();n0=calls
            data={};state['endpoints'][name]=data
            with torch.inference_mode():
                out=rt.model(**pair[0].forward_kwargs());parts,teacher_first=token_parts(out.logits,pair[0].labels,rt.adapter.tokenizer.eos_token_id);del out
                data['teacher_forced']=parts;emit();near(parts['mean_nll'],expected[0],name+'.teacher_nll')
                full=pair[0].raw_input_ids;prefix=raw['input_ids'];eq(full[:,:prefix.shape[1]].tolist(),prefix.tolist(),name+'.train_generation_prefix')
                kwargs=dict(input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],images=raw['images'],use_cache=False,return_dict=True)
                out=rt.model(**kwargs);prefill=out.logits[0,-1].detach().float().cpu();del out
                target=pair[0].target_token_ids[0];data['prefix_only']=token_stat(prefill,target);emit()
                # Capture the realized first model forward, before generation processors.
                got=[]
                def capture(m,args,output):got.append(output.logits[0,-1].detach().float().cpu())
                hook=rt.model.register_forward_hook(capture)
                try:
                    if generations>=2:raise RuntimeError('Single-token generation cap')
                    generations+=1;data['generation_started']=True;emit()
                    kw={k:v for k,v in rt.generation_config.items() if v is not None and k in {'do_sample','num_beams','use_cache','eos_token_id','pad_token_id'}}
                    kw.update(max_new_tokens=1,return_dict_in_generate=True,output_scores=True)
                    if kw.get('pad_token_id') is None:kw['pad_token_id']=rt.adapter.tokenizer.eos_token_id
                    generated=rt.model.generate(raw['input_ids'],attention_mask=raw['attention_mask'],images=raw['images'],**kw)
                finally:hook.remove()
                ids=generated.sequences[0].detach().cpu().tolist();processed=generated.scores[0][0].detach().float().cpu()
                data['generation_forward_count']=len(got);data['generation_sequence_length']=len(ids);emit();eq(len(got),1,name+'.generation_one_forward')
                chosen=ids[-1];data['official_generation_raw']=token_stat(got[0],target);data['official_generation_scores']=token_stat(processed,target)
                data.update(teacher_vs_prefill_max_logit_difference=float((teacher_first-prefill).abs().max()),prefill_vs_generation_max_logit_difference=float((prefill-got[0]).abs().max()),generation_raw_vs_processed_max_logit_difference=float((got[0]-processed).abs().max()),teacher_prefill_top1_same=int(teacher_first.argmax())==int(prefill.argmax()),prefill_generation_top1_same=int(prefill.argmax())==int(got[0].argmax()),selected_is_processed_argmax=chosen==int(processed.argmax()),selected_matches_target=chosen==target)
                previous=json.loads((prior/'private/outputs'/('base' if name=='BASE' else 'E01')/'Q001.json').read_text())
                data['selected_matches_historical_first']=chosen==previous['raw_token_ids'][0]
                private=dict(selected_ids=ids,target_ids=list(pair[0].target_token_ids),decoded_single_token=rt.adapter.tokenizer.decode(ids,skip_special_tokens=True).strip())
                write(root/'private'/('OBS_'+name+'.json'),private);emit()
                after=native_losses(rt,pair);data['native_losses_after']=after;data['weight_unchanged']=torch.equal(w,layer.weight);emit()
                near(after[0],expected[0],name+'.after_visual');near(after[1],expected[1],name+'.after_null');eq(data['weight_unchanged'],True,name+'.weight_unchanged');eq(data['selected_matches_historical_first'],True,name+'.history_first_token');eq(data['selected_is_processed_argmax'],True,name+'.greedy_contract')
            data['observation_forwards']=calls-n0;state['status']='OBSERVATION_DONE';emit()
            if observed>16:raise RuntimeError('Observation forward ceiling')
        observe('BASE',before)
        state['status']='RECONSTRUCTING';emit();checked=set()
        def progress(rows):
            state['reconstruction_steps']=rows;emit()
            for row0 in rows:
                i=row0['step']
                if row0['status']!='ACCEPTED' or i in checked:continue
                previous=old['method']['steps'][i-1]
                eq(row0['selected_attempt'],previous['selected_attempt'],f'step{i}.selection')
                for arm in ['primary','half']:
                    a=row0[arm];b=previous[arm];eq(a is None,b is None,f'step{i}.{arm}.present')
                    if a is None:continue
                    compare(a['proposal'],b['proposal'],f'step{i}.{arm}.proposal',state['checks'])
                    eq(a['correction'] is None,b['correction'] is None,f'step{i}.{arm}.correction_presence')
                    if a['correction'] is not None:compare(a['correction'],b['correction'],f'step{i}.{arm}.correction',state['checks'])
                    eq(a['final_verdict']['gates'],b['final_verdict']['gates'],f'step{i}.{arm}.gates')
                checked.add(i);emit()
        def endpoint(w,result):
            state['method']=result;emit();eq(len(checked),16,'all_16_steps_checked');eq(result['joint_gate'],True,'original_joint_gate');eq(result['half_proposals'],4,'four_frozen_halves');eq(result['correction_attempts'],20,'twenty_frozen_secants')
            for a,b in zip(result['final_accepted_native_nll'],old['method']['final_accepted_native_nll']):near(a,b,'endpoint.native_nll')
            observe('POST',result['final_accepted_native_nll'])
        trajectory(rt,record,factor,progress,anchor=baseline['anchor'],edit_threshold=ref['native_after'][0],deadline_epoch=c['deadline_epoch'],endpoint_observer=endpoint)
        state['status']='COMPLETE';emit()
    except BaseException as e:
        (root/'private/FAILURE.txt').write_text(traceback.format_exc());state.update(status='FAILED_STOPPED_NO_RETRY',error_type=type(e).__name__);emit()
    finally:
        restoring=True
        try:
            if layer is not None and base is not None:
                with torch.no_grad():layer.weight.copy_(base)
                state['base_weight_restored']=torch.equal(layer.weight,base)
                state['base_losses_after_restore']=native_losses(rt,pair)
                state['base_losses_restored']=state['base_losses_after_restore']==before
                state['hooks_restored']=not layer._forward_hooks and not layer._forward_pre_hooks
                state['flags_restored']=flags==[p.requires_grad for p in rt.model.parameters()] and modes==[m.training for m in rt.model.modules()]
                state['generation_config_restored']=generation_state==rt.model.generation_config.to_dict()
                if not all(state[k] for k in ['base_weight_restored','base_losses_restored','hooks_restored','flags_restored','generation_config_restored']):state['status']='RESTORE_FAILED_STOPPED'
        except BaseException:
            state['status']='RESTORE_FAILED_STOPPED';(root/'private/RESTORE_FAILURE.txt').write_text(traceback.format_exc())
        if h:h.remove()
        state.update(model_forward_calls=calls,observation_forward_calls=observed,generation_calls=generations,judge_calls=0);emit()
        write(root/'ENDED.json',dict(epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls,observation_forward_calls=observed,generation_calls=generations,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None,status=state['status']))
