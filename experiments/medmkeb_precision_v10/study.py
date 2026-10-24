"""Bounded31-step cumulative feasibility; first transitions reuse the frozen V7 audit."""
import os,json,time,copy,math
from pathlib import Path
import torch
from prefix_cache import PrefixCache
from qp import project,selfcheck as qp_check
from constraints import build,selfcheck as constraint_check
from readout import read as readout,selfcheck as readout_check
from path_math import direct,capped,distance,selfcheck as path_check
from engine import proposal,construct,match,violation,same,first_feasible,selfcheck as engine_check

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);prior=Path(os.environ['V7_ROOT'])/str(partition);old8=Path(os.environ['V8_ROOT'])/str(partition);read=lambda p:json.loads(Path(p).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def save(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def budget():
        import shutil
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert counts['forwards']<=1200 and counts['backwards']==0 and counts['candidate_checks']<=20 and counts['updates']==0 and counts['logical_transitions']==0 and counts['generations']==0
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',dict(QP=qp_check(),constraints=constraint_check(),engine=engine_check(),path=path_check(),readout=readout_check()))
    task=read(shared/'DATA_ADMISSION.json')['tasks'][partition];pool=read(shared/'SUPPORT_FROZEN.json');assert len(pool)==9
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json')
    rows=[task['source']]
    for c in pool:
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            for text in (c['row']['reference'],c['wrong_choice']):rows.append(dict(c['row'],query_id='S'+str(len(rows)-1),reference=form.format(text.rstrip('.'))))
    runtime.model.half();items=r.s.prepare(runtime,bindings,template,rows,True);r.old.fp32(runtime);batches=[]
    for index,(row,value,raw,expanded,binding) in enumerate(items):
        if index:
            kwargs,labels,tokens=value;start=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist();assert full[:len(start)]==start
            content=tuple(full[len(start):]);assert tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
            positions=torch.where(labels[0]!=-100)[0];labels=labels.clone();labels[0,positions[len(content):]]=-100;value=(dict(kwargs,labels=labels),labels,content)
        else:
            base_binding=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(base_binding[k]==binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        old_binding=read(prior/'bindings'/(row['query_id']+'.json'));new_binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']);assert all(new_binding[k]==old_binding[k] for k in old_binding if k!='image_path'),'V7 input binding drift';batches.append(r.old.batch_on(runtime,value));write(out/'bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    before=r.old.snapshot(expert);keys=list(before);params=dict(expert.named_parameters());assert set(keys)==set(params)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    previous=Path(os.environ['V9_ROOT'])/str(partition);bundle=torch.load(previous/'AUDIT_INPUTS.pt',map_location='cpu',weights_only=True);assert read(previous/'EXIT.json')['code']==0
    expert.load_state_dict(bundle['state']);assert all(torch.equal(expert.state_dict()[k].cpu(),bundle['state'][k]) for k in keys)
    before=r.old.snapshot(expert);delta=bundle['projected_direction'];d0=bundle['source_direction'];v9baseline=read(previous/'BASELINE_AUDIT.json')['private_metrics']['no_grad'];anchor=read(shared/'REFERENCE_POOL.json')['nll']
    current_batch=None;captured=None;capture_error=0.
    def capture(_model,_args,output):
        nonlocal captured,capture_error
        assert current_batch is not None
        f32,f64=readout(output.logits,current_batch.labels);captured=(f32,f64);assert output.loss is not None
        error=abs(f32-float(output.loss.detach()));capture_error=max(capture_error,error);assert error<=1e-6,'Same-logit loss capture mismatch'
    tap=runtime.model.register_forward_hook(capture)
    def evaluate(state):
        nonlocal current_batch,captured
        expert.load_state_dict(state);a=[];b=[]
        with torch.no_grad():
            for batch in batches:
                budget();current_batch=batch;captured=None;hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);assert captured is not None
                a.append(float(loss));b.append(captured[1])
        current_batch=None
        return dict(CE32=dict(source_loss=a[0],support_nll=a[1:]),CE64=dict(source_loss=b[0],support_nll=b[1:]))
    raw,_,rebuilt=proposal(r,before,bundle['gradients'][0],bundle['scalar'],keys);assert torch.equal(rebuilt,d0),'Stored source proposal drift';full_source=construct(r,before,raw,d0,d0,1.,keys);full_norm=r.fm.map_norm(before,full_source)
    states=[('ZERO_BASE',0.,{k:v.to(before[k]) for k,v in base.items()}),('STATE20',0.,before)]
    for scale in [2.**-j for j in range(8)]:
        candidate,actual_scale=capped(before,delta,scale,keys,full_norm,r.fm.map_norm);states.append(('DIRECT',scale,candidate))
    summaries=[];all_values={};hook.attach()
    try:
        with PrefixCache(runtime,batches):
            for precision in ('TT32','TT64'):
                expert.to(dtype=torch.float32 if precision=='TT32' else torch.float64);phase={}
                for name,scale,state in states:
                    value=evaluate(state);counts['candidate_checks']+=1
                    if precision=='TT32':
                        old=dict(source_loss=read(prior/'candidates/SOURCE_1.0.json')['private_metrics']['source_loss']+read(prior/'candidates/SOURCE_1.0.json')['summary']['source_gain'],support_nll=anchor) if name=='ZERO_BASE' else v9baseline if name=='STATE20' else read(previous/('candidates/DIRECT_'+str(scale)+'_False.json'))['private_metrics']
                        assert distance(value['CE32'],old)<=1e-6,'Original FP32 metric reproduction failed'
                    phase[name+'_'+str(scale)]=value;write(out/'metrics'/(precision+'_'+name+'_'+str(scale)+'.json'),value)
                    write(out/'PROGRESS.json',dict(precision=precision,state=name,scale=scale,counts=counts));print(task['key'],precision,name,scale,flush=True)
                all_values[precision]=phase
                for readmode in ('CE32','CE64'):
                    anchor_mode=phase['ZERO_BASE_0.0'][readmode];current=phase['STATE20_0.0'][readmode]
                    for name,scale,state in states[2:]:
                        value=phase[name+'_'+str(scale)][readmode];actual_delta=torch.cat([(state[k]-before[k]).detach().cpu().double().reshape(-1) for k in keys]);prediction=bundle['gradients']@actual_delta
                        observed=torch.tensor([value['source_loss']-current['source_loss']]+[a-b for a,b in zip(value['support_nll'],current['support_nll'])],dtype=torch.float64)
                        summaries.append(dict(precision=precision,readout=readmode,scale=scale,max_absolute_NLL_linear_error=float((observed-prediction).abs().max()),source_gain=current['source_loss']-value['source_loss'],max_support_regression=violation(anchor_mode['support_nll'],value['support_nll']),current_max_support_regression=violation(anchor_mode['support_nll'],current['support_nll']),diagnostic_only=True))
            write(out/'AUDIT_COMPLETE.json',dict(status='COMPLETE',task=task['key'],summaries=summaries,original_FP32_reproduction=True,loaded_state_exact=True,capture_max_error=capture_error,TT64_changes_Transformer_precision=False,counts=counts,semantic_effectiveness_tested=False,primary_criterion_unchanged=True))
    finally:hook.detach();tap.remove()
