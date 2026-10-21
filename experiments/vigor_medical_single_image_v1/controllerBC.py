"""E01 from-base B/C comparison with one shared bounded response controller."""
import os,json,sys,time,subprocess,traceback,math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .prepare import write
from .editor import editable_fp32,direction,nll,finite,paired_batches
from .finite_null import native_losses,gates,correction
from .marginab_ops import read_margin,fisher_cost
from .native_null_center import center
from .native_control import SCALES,admission,response,next_radius

AUTH=dict(sample='E01',arms=['B','C'],independent_base=True,steps_max=8,trials_per_step=3,scales=list(SCALES),initial_radius=1.,cost_cap=5.368434395105091e-13,metric='F0',native_null_gate=.01,trust=.001,response_min=.1,progress_floor=1e-5,arm_edit_forward_max=180,arm_generation_forward_max=96,arm_secant_max=8,forward_max=600,seconds=900,generation_max=2,judge=0,checkpoint=0,retry=0)

class EngineeringStop(RuntimeError):pass
class ArmBudget(RuntimeError):pass

def main(c):
    if c['authorization']!=AUTH:raise PermissionError('Frozen comparison only')
    root=Path(c['output']);start=time.time();rt=None;layer=None;base=None;pair=None;before=None;hook=None
    calls=0;arm_calls=0;generation_calls=0;generating=False;restoring=False;active=None;state=dict(status='LOADING',arms=[],checks=[])
    def emit():
        write(root/'public/RESULT.json',state)
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>52428800:raise EngineeringStop('Disk ceiling')
    def check(ok,label):
        state['checks'].append(dict(field=label,passed=bool(ok)));emit()
        if not ok:raise EngineeringStop(label)
    def near(a,b,label):
        v=abs(a-b);state['checks'].append(dict(field=label,absolute_difference=v,tolerance=1e-5));emit()
        if not math.isfinite(v) or v>1e-5:raise EngineeringStop(label)
    def count(model,args):
        nonlocal calls,arm_calls,generation_calls
        if calls>=600 or (not restoring and (calls>=580 or time.time()>=c['deadline_epoch'])):raise EngineeringStop('Global forward/time ceiling')
        if active is not None and not restoring:
            if generating:
                if generation_calls>=96:raise ArmBudget('Generation forward cap')
                generation_calls+=1
            else:
                if arm_calls>=180:raise ArmBudget('Editing forward cap')
                arm_calls+=1
        calls+=1
    try:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),epoch=start),f)
        check(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','workspace')
        torch.use_deterministic_algorithms(True);sys.path.insert(0,c['source_root'])
        check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['source_root'],text=True).strip()==c['source_commit'],'runtime_source')
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import EditorRecord
        prior=Path(c['prior']);manifest=json.loads((prior/'private/MANIFEST.json').read_text());bindings=json.loads((prior/'private/BINDINGS.json').read_text())
        t=manifest['tasks'][0];check(t['alias']=='E01','E01')
        row=manifest['queries'][next(e['query'] for e in t['panel'] if e['task']=='T0')];record=EditorRecord(**dict(t['record'],image_path=Path(t['record']['image_path'])))
        check(record.target==row['reference'] and record.question==row['question'],'source_binding')
        rt=load_real_runtime(SimpleNamespace(cpu_gate=Path(c['cpu_gate'])));hook=rt.model.register_forward_pre_hook(count)
        layer=rt.get_module('model.layers.0.mlp.down_proj');base=layer.weight.detach().clone();check(base.dtype==torch.float16,'native_half')
        flags=[p.requires_grad for p in rt.model.parameters()];modes=[m.training for m in rt.model.modules()];genconfig=rt.model.generation_config.to_dict()
        pair=paired_batches(rt,record);before=native_losses(rt,pair);m0,identity=read_margin(rt,pair[0]);check(identity[0]==0,'first_source_target')
        old=json.loads((prior/'public/SAMPLES.json').read_text())[0]
        for x,y in zip(before,old['baseline']['native']):near(x,y,'original_base_loss')
        near(m0['mean_nll'],before[0],'base_repeat');state['base']=dict(native_nll=before,margin=m0);emit()
        cal=torch.load(c['calibration'],map_location='cpu',weights_only=True);check(cal['binding']==c['binding'],'old_F0_binding');factor=cal['factors']['model.layers.0.mlp.down_proj'];a,b=(factor[k].to(base.device) for k in ['a','b'])
        raw=rt.adapter.prepare_inputs(record.image_path,record.question,None);bound=bindings[row['opaque_Base_id']]
        check(raw['input_ids'].tolist()==[bound['prompt_ids']] and raw['attention_mask'].tolist()==[bound['attention_mask']] and raw['image_sha256']==bound['image_sha256'],'official_prompt_binding')
        check(rt.generation_config==bound['generation'],'frozen_generation_config')
        for name in ['B','C']:
            active=name;arm_calls=0;generation_calls=0;secants=0;radius=1.;accepted=0;last=before;lastm=m0;current_identity=identity
            entry=dict(name=name,status='EDITING',steps=[],generation=None,accepted_steps=0);state['arms'].append(entry);state['status']='RUNNING_'+name;emit()
            with torch.no_grad():layer.weight.copy_(base)
            try:
                for step in range(8):
                    original=layer.weight.detach().clone();start_losses=last;start_m=lastm;local_identity=current_identity
                    r=dict(step=step+1,radius=radius,start_native_nll=last,start_margin=lastm['margin'],trials=[],accepted=False);entry['steps'].append(r);emit()
                    with editable_fp32(rt.model,layer,readonly=True):
                        out=rt.model(**pair[0].forward_kwargs());logits=out.logits.float();_,target,j,absolute=local_identity;v=logits[0,absolute]
                        loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),pair[0].labels[:,1:].reshape(-1),ignore_index=-100)
                        lm=v[target]-v[j];gm=finite(torch.autograd.grad(lm,layer.weight,retain_graph=True)[0].detach());gn=finite(torch.autograd.grad(loss,layer.weight)[0].detach());del out,logits,v,loss,lm
                        ln=nll(rt.model,pair[1]);g0=finite(torch.autograd.grad(ln,layer.weight)[0].detach());del ln
                        qn,geometry=direction(gn,g0,a,b)
                        r['geometry']=geometry;emit()
                        if qn is None:entry['status']='STOPPED_DEGENERATE_DIRECTION';break
                        delta=finite(-math.sqrt(2*AUTH['cost_cap']/geometry['E'])*qn)
                    check(layer.weight.dtype==base.dtype and torch.equal(layer.weight,original),name+'.gradient_restore')
                    seen=[];chosen=False
                    for scale in SCALES:
                        used=radius*scale;proposal=(original.float()+used*delta).half()
                        trial=dict(scale=used,status='FORMED',center=None);r['trials'].append(trial);emit()
                        if name=='C':
                            q=proposal.float()-original.float();mask=(q!=0)&(gn*q<0)&(gm*q>0);proposal=torch.where(mask,proposal,original)
                            q=proposal.float()-original.float();bias=last[1]-before[1]+float((g0.double()*q.double()).sum())
                            if abs(bias)>1e-10:proposal,trial['center']=center(original,proposal,gm,g0,last[1]-before[1])
                            else:trial['center']=dict(status='ALREADY_PREDICTED_CENTERED')
                            if proposal is None:trial['status']='CONSTRUCTION_REJECTED';emit();continue
                        if torch.equal(proposal,original) or any(torch.equal(proposal,p) for p in seen):
                            trial['status']='ZERO_OR_DUPLICATE_NO_FORWARD';emit();continue
                        seen.append(proposal.clone());q=proposal.float()-original.float();cost=fisher_cost(q,a,b);linear=[float((g.double()*q.double()).sum()) for g in [gn,gm,g0]]
                        trial.update(cost=cost,linear=linear,predicted_original_base_null=last[1]-before[1]+linear[2]);emit()
                        if not admission(cost,linear,AUTH['cost_cap']*used*used,last[1]-before[1]):trial['status']='NATIVE_ADMISSION_REJECTED';emit();continue
                        with torch.no_grad():layer.weight.copy_(proposal)
                        check(torch.equal(layer.weight,proposal),name+'.native_commit')
                        verdict=gates(rt,pair,factor,original,last,before[1]);mm,ii=read_margin(rt,pair[0],identity[0],local_identity[2])
                        trial.update(main_gate=verdict,main_margin=mm,actual_response=response(last[0],verdict['native_nll'][0],lastm['margin'],mm['margin'],linear));emit();near(mm['mean_nll'],verdict['native_nll'][0],name+'.main_repeat')
                        if not verdict['gates']['trust']:entry['status']='STOPPED_TRUST';break
                        if not trial['actual_response']['passed']:
                            with torch.no_grad():layer.weight.copy_(original)
                            check(torch.equal(layer.weight,original),name+'.rejected_trial_restore');trial['status']='ACTUAL_RESPONSE_REJECTED';emit();continue
                        trial['correction']=None
                        if not verdict['gates']['null']:
                            if secants>=8:entry['status']='STOPPED_SECANT_CAP';break
                            secants+=1;trial['correction']=correction(rt,record,factor,before[1],compensate='secant')
                            final=gates(rt,pair,factor,original,last,before[1]);fm,fi=read_margin(rt,pair[0],identity[0],local_identity[2])
                        else:final,fm,fi=verdict,mm,ii
                        final_response=response(last[0],final['native_nll'][0],lastm['margin'],fm['margin'],linear)
                        trial.update(final_gate=final,final_margin=fm,final_response=final_response);emit();near(fm['mean_nll'],final['native_nll'][0],name+'.final_repeat')
                        if not final['accepted'] or not final_response['passed'] or (trial['correction'] is not None and trial['correction']['status']!='APPLIED'):
                            entry['status']='STOPPED_POST_CORRECTION';break
                        trial['status']='ACCEPTED';r['accepted']=True;accepted+=1;last=final['native_nll'];lastm=fm;current_identity=fi;radius=next_radius(used,final_response);chosen=True
                        entry['accepted_steps']=accepted;entry['last_accepted']=dict(native_nll=last,margin=lastm,cumulative_F0_cost=fisher_cost(layer.weight.detach().float()-base.float(),a,b));emit();break
                    del seen
                    if not chosen:
                        with torch.no_grad():layer.weight.copy_(original)
                        last=start_losses;lastm=start_m;current_identity=local_identity
                        if entry['status']=='EDITING':entry['status']='STOPPED_NO_ELIGIBLE_TRIAL'
                        emit();break
                else:entry['status']='STEP_CAP_REACHED'
            except ArmBudget:
                with torch.no_grad():layer.weight.copy_(original)
                last=start_losses;lastm=start_m
                entry['status']='STOPPED_EDIT_FORWARD_CAP';emit()
            # Generate exactly once from the last accepted endpoint; preserve incomplete calls.
            entry.setdefault('last_accepted',dict(native_nll=last,margin=lastm,cumulative_F0_cost=fisher_cost(layer.weight.detach().float()-base.float(),a,b)))
            entry['editing_forwards']=arm_calls;entry['secant_calls']=secants;emit()
            generating=True;captured=[]
            def capture(model,args,out):captured.append(int(out.logits[0,-1].argmax()))
            gh=rt.model.register_forward_hook(capture)
            write(root/('private/'+name+'_GENERATION_STARTED.json'),dict(epoch=time.time()))
            try:
                with torch.inference_mode():value=rt.adapter.generate_prepared_with_result(raw,rt.generation_config)
                private=dict(status='COMPLETE',raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids));write(root/('private/'+name+'_GENERATION.json'),private)
                text=value.decoded_text.strip();target_text=record.target.strip()
                entry['generation']=dict(status='COMPLETE',raw_token_count=len(value.raw_token_ids),source_target_exact_match=text.casefold()==target_text.casefold(),semantic_T0_score=None,source_target_exact_match_is_not_semantic_score=True,cap_hit=len(value.raw_token_ids)>=rt.generation_config['max_new_tokens'])
            except ArmBudget:
                write(root/('private/'+name+'_GENERATION.json'),dict(status='FORWARD_BUDGET_STOP',partial_greedy_ids=captured));entry['generation']=dict(status='FORWARD_BUDGET_STOP',partial_forward_count=len(captured),semantic_T0_score=None)
            finally:gh.remove();generating=False
            entry['generation_forwards']=generation_calls;restoring=True
            with torch.no_grad():layer.weight.copy_(base)
            restored=native_losses(rt,pair);entry['base_restored']=torch.equal(layer.weight,base) and restored==before;emit();check(entry['base_restored'],name+'.base_restore');restoring=False;active=None
        state['status']='COMPLETE';emit()
    except BaseException as e:
        write(root/'private/FAILURE.json',dict(type=type(e).__name__,traceback=traceback.format_exc()));state.update(status='FAILED_STOPPED_NO_RETRY',error_type=type(e).__name__);emit()
    finally:
        restoring=True;active=None
        if layer is not None and base is not None:
            try:
                with torch.no_grad():layer.weight.data=base.clone()
                state['restoration']=dict(base_weight=torch.equal(layer.weight,base),base_losses=native_losses(rt,pair)==before,dtype=layer.weight.dtype==base.dtype,hooks=not layer._forward_hooks and not layer._forward_pre_hooks,flags=flags==[p.requires_grad for p in rt.model.parameters()] and modes==[m.training for m in rt.model.modules()],generation_config=genconfig==rt.model.generation_config.to_dict())
                if not all(state['restoration'].values()):state['status']='RESTORE_FAILED_STOPPED'
            except BaseException:state['status']='RESTORE_FAILED_STOPPED'
        if hook is not None:hook.remove()
        state['model_forward_calls']=calls;emit();write(root/'ENDED.json',dict(epoch=time.time(),resident_seconds=time.time()-start,model_forward_calls=calls,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,status=state['status']))
