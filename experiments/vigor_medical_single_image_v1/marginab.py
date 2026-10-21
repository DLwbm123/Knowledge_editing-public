"""E01 frozen endpoint A/B: NLL versus margin, one native F0 cost match."""
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

AUTH=dict(sample='E01',endpoint_steps=16,arms=['A','B'],metric='F0',secant=True,matching_applications=1,matching_tolerance=.01,forward_max=600,seconds=900,generation_max=2,half_rescue=False,judge=0,retry=0)


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
    calls=0;generations=0;restoring=False;state=dict(status='LOADING',checks=[],arms={});captured=[]
    def emit():write(root/'public/RESULT.json',state)
    def check():
        if time.time()>=c['deadline_epoch']:raise TimeoutError('Work time ceiling')
    def count(m,args):
        nonlocal calls
        if calls>=600 or (not restoring and (calls>=580 or time.time()>=c['deadline_epoch'])):raise TimeoutError('Forward/time ceiling; restoration reserve')
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
            nonlocal generations
            from .marginab_ops import read_margin,propose
            from .finite_null import gates,correction
            state['method']=result;emit();eq(len(checked),16,'all_16_steps_checked');eq(result['joint_gate'],True,'original_joint_gate');eq(result['half_proposals'],4,'four_frozen_halves');eq(result['correction_attempts'],20,'twenty_frozen_secants')
            for a,b in zip(result['final_accepted_native_nll'],old['method']['final_accepted_native_nll']):near(a,b,'endpoint.native_nll')
            endpoint_weight=layer.weight.detach().clone();start_losses=native_losses(rt,pair)
            margin0,identity=read_margin(rt,pair[0]);state['endpoint_margin']=margin0;emit();eq(identity[0],0,'frozen_first_unfavored_position');near(margin0['margin'],-5.8671875,'historical_endpoint_margin')
            write(root/'private/IDENTITY.json',dict(position=identity[0],target=identity[1],competitor=identity[2]))
            target_cost=None
            for arm in ['A','B']:
                check();state['status']='ARM_'+arm;entry={};state['arms'][arm]=entry;emit()
                try:
                    eq(torch.equal(layer.weight,endpoint_weight),True,arm+'.common_start_weight')
                    if arm=='A':
                        entry['proposal']=candidate(rt,record,0,factor,FIRST_D/16,retain=True)
                        target_cost=entry['proposal']['native_update']['fisher_cost'];entry['matching_reference_cost']=target_cost;emit()
                        if not math.isfinite(target_cost) or target_cost<=0:raise ValueError('Invalid A native F0 cost')
                    else:
                        def save(value):entry['proposal']=value;emit()
                        entry['proposal']=propose(rt,layer,pair,factor,identity,target_cost,save);emit()
                    entry['main_margin'],_=read_margin(rt,pair[0],identity[0],identity[2]);entry['main_gate']=gates(rt,pair,factor,endpoint_weight,start_losses,before[1]);emit()
                    entry['correction']=None
                    if not entry['main_gate']['gates']['null']:
                        entry['correction']=correction(rt,record,factor,before[1],compensate='secant');emit()
                    entry['final_gate']=gates(rt,pair,factor,endpoint_weight,start_losses,before[1]);entry['final_margin'],_=read_margin(rt,pair[0],identity[0],identity[2])
                    entry['margin_improved']=entry['final_margin']['margin']>margin0['margin']
                    entry['original_gate_accepted']=entry['final_gate']['accepted'] and (entry['correction'] is None or entry['correction']['status']=='APPLIED')
                    entry['margin_augmented_accepted']=entry['original_gate_accepted'] and entry['margin_improved'];emit()
                    if generations>=2:raise RuntimeError('Generation cap')
                    generations+=1;entry['generation_started']=True;emit()
                    got=[]
                    def capture(m,args,output):got.append(output.logits[0,-1].detach().float().cpu())
                    hook=rt.model.register_forward_hook(capture)
                    try:
                        kw={k:v for k,v in rt.generation_config.items() if v is not None and k in {'do_sample','num_beams','use_cache','eos_token_id','pad_token_id'}}
                        kw.update(max_new_tokens=1,return_dict_in_generate=True,output_scores=True)
                        if kw.get('pad_token_id') is None:kw['pad_token_id']=rt.adapter.tokenizer.eos_token_id
                        with torch.inference_mode():out=rt.model.generate(raw['input_ids'],attention_mask=raw['attention_mask'],images=raw['images'],**kw)
                    finally:hook.remove()
                    ids=out.sequences[0].detach().cpu().tolist();v=out.scores[0][0].detach().float().cpu();target=identity[1]
                    entry['generation']=dict(forwards=len(got),sequence_length=len(ids),target_stats=token_stat(v,target),selected_matches_target=ids[-1]==target,selected_is_processed_argmax=ids[-1]==int(v.argmax()),raw_vs_processed_max_difference=float((got[0]-v).abs().max()))
                    write(root/'private'/('GEN_'+arm+'.json'),dict(ids=ids,target=target));emit();eq(len(got),1,arm+'.generation_one_forward');eq(entry['generation']['selected_is_processed_argmax'],True,arm+'.greedy_contract')
                    near(entry['generation']['target_stats']['nll'],entry['final_margin']['nll'],arm+'.teacher_generate_target_nll')
                    entry['local_flip']=entry['final_margin']['margin']>0 and entry['generation']['selected_matches_target'];entry['safe_local_flip']=entry['original_gate_accepted'] and entry['local_flip'];emit()
                finally:
                    with torch.no_grad():layer.weight.copy_(endpoint_weight)
                    entry['endpoint_restored']=torch.equal(layer.weight,endpoint_weight) and native_losses(rt,pair)==start_losses;emit()
                    if not entry['endpoint_restored']:raise ValueError('Arm endpoint restoration failed')

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
        state.update(model_forward_calls=calls,generation_calls=generations,judge_calls=0);emit()
        write(root/'ENDED.json',dict(epoch=time.time(),resident_seconds=time.time()-started,model_forward_calls=calls,generation_calls=generations,peak_allocated_bytes=torch.cuda.max_memory_allocated() if rt else None,status=state['status']))
