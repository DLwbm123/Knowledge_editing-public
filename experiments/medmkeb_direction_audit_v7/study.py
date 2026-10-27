"""One fresh gradient audit per task, finite candidate checks and no training loop."""
import os,json,time,math,copy
from pathlib import Path
import torch
from prefix_cache import PrefixCache
from qp import project,selfcheck

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);read=lambda p:json.loads(Path(p).read_text())
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
    def budget():
        import shutil
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1))
        assert counts['forwards']<=1500 and counts['backwards']<=60 and counts['candidate_checks']<=24 and counts['updates']==counts['generations']==0
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',selfcheck())
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
            prior=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(prior[k]==binding[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        batches.append(r.old.batch_on(runtime,value));write(out/'bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base)
    before=r.old.snapshot(expert);keys=list(before);params=dict(expert.named_parameters());assert set(keys)==set(params)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
    try:
        with PrefixCache(runtime,batches):
            losses=[];gradients=[]
            for batch in batches:
                budget();hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);assert torch.isfinite(loss)
                values=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1
                gradients.append(torch.cat([(g if g is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,g in zip(keys,values)]));losses.append(float(loss.detach()))
            reference=read(shared/'REFERENCE_POOL.json');difference=max(abs(a-b) for a,b in zip(losses[1:],reference['nll']));assert difference<=1e-6,'Initial support drift'
            initial_metrics=dict(source_loss=losses[0],support_nll=losses[1:],margins=[losses[j+2]-losses[j+1] for j in range(0,54,2)])
            local={k:v.detach().clone().requires_grad_(True) for k,v in before.items()};offset=0
            for k in keys:
                n=local[k].numel();local[k].grad=gradients[0][offset:offset+n].reshape(local[k].shape).to(local[k]);offset+=n
            torch.nn.utils.clip_grad_norm_(list(local.values()),1.)
            raw_source,_=r.mm.canonical_rms_step(before,{k:local[k].grad.detach() for k in keys},dict(step=0,second_moments=[0.,0.]))
            direction=torch.cat([(raw_source[k]-before[k]).detach().cpu().double().reshape(-1) for k in keys])
            matrix=torch.stack([g for j in range(27) for g in (gradients[1+2*j],gradients[1+2*j]-gradients[2+2*j])]);bounds=torch.full((54,),1e-4,dtype=torch.float64)
            projected,qp=project(direction,matrix,bounds);assert qp['status']=='PASS','Fixed finite solver did not converge'
            assert float((matrix@projected-bounds).max())<=1e-7
            def construct(delta,scale):
                if torch.equal(delta,direction):raw=raw_source
                else:
                    raw={};offset=0
                    for k in keys:
                        n=before[k].numel();raw[k]=before[k]+delta[offset:offset+n].reshape(before[k].shape).to(before[k]);offset+=n
                value={k:before[k]+scale*(raw[k]-before[k]) for k in keys}
                return r.mm.canonicalize(value)[0]
            zero,_=project(direction,torch.empty((0,direction.numel())),torch.empty(0));a=construct(direction,1.);b=construct(zero,1.);assert all(torch.equal(a[k],b[k]) for k in keys),'Zero-constraint state identity failed'
            full_source=a;full_norm=r.fm.map_norm(before,full_source);assert full_norm>0
            torch.save(dict(matrix=matrix,bounds=bounds,source_direction=direction,source_gradient=gradients[0],projected_direction=projected),out/'QP_INPUTS.pt')
            write(out/'ENGINEERING_ADMISSION.json',dict(status='PASS',support_initial_difference=difference,zero_constraint_state_identity=True,QP=qp,parameter_frame_metric=True,operator_invariance_claim=False,counts=copy.deepcopy(counts)))
            cache=[({k:v.detach().cpu().clone() for k,v in before.items()},initial_metrics)]
            def metrics(state):
                cpu={k:v.detach().cpu() for k,v in state.items()}
                for old,value in cache:
                    if all(torch.equal(old[k],cpu[k]) for k in keys):return value,True
                expert.load_state_dict(state);values=[]
                with torch.no_grad():
                    for batch in batches:budget();hook.set_teacher_routing(batch.labels);values.append(float(runtime.compute_loss(batch)))
                assert all(math.isfinite(x) for x in values)
                value=dict(source_loss=values[0],support_nll=values[1:],margins=[values[j+2]-values[j+1] for j in range(0,54,2)]);cache.append(({k:v.clone() for k,v in cpu.items()},value));return value,False
            def record(kind,scale,state):
                budget();counts['candidate_checks']+=1;value,reused=metrics(state);norm=r.fm.map_norm(before,state)
                violation=max(max(a-b for a,b in zip(initial_metrics['margins'],value['margins'])),max(value['support_nll'][j]-initial_metrics['support_nll'][j] for j in range(0,54,2)))
                row=dict(kind=kind,scale=scale,source_gain=losses[0]-value['source_loss'],actual_operator_norm=norm,max_support_regression=violation,actual_feasible=violation<=1e-4,reused_exact_state=reused,counts=copy.deepcopy(counts));write(out/'candidates'/(kind+'_'+str(scale)+'.json'),dict(summary=row,private_metrics=value));return row
            def match_down(state,target):
                if target==0:return {k:v.clone() for k,v in before.items()}
                return r.mm.canonicalize(r.match_proposal(before,state,target)[0])[0]
            rows=[]
            for scale in [2.**-j for j in range(8)]:
                rows.append(record('SOURCE',scale,construct(direction,scale)))
                candidate=construct(projected,scale);norm=r.fm.map_norm(before,candidate)
                if norm>full_norm:candidate=match_down(candidate,full_norm)
                protected=record('PROJECT',scale,candidate);assert protected['actual_operator_norm']<=full_norm*1.002+1e-10;rows.append(protected)
                matched=match_down(full_source,protected['actual_operator_norm']);control=record('NORM_SOURCE',scale,matched)
                assert abs(control['actual_operator_norm']-protected['actual_operator_norm'])<=1e-10+.002*protected['actual_operator_norm'];rows.append(control)
                write(out/'PROGRESS.json',dict(task=task['key'],scale=scale,candidates=len(rows),counts=counts));print(task['key'],scale,len(rows),flush=True)
            source_gain=rows[0]['source_gain'];qualified=[x['scale'] for x in rows if x['kind']=='PROJECT' and x['actual_feasible'] and x['actual_operator_norm']>0 and source_gain>0 and x['source_gain']>=max(1e-4,.1*source_gain)]
            write(out/'AUDIT_COMPLETE.json',dict(status='COMPLETE',task=task['key'],engineering='PASS',candidate_qualified=bool(qualified),qualified_scales=qualified,criterion='Actual support<=1e-4, nonzero norm<=SOURCE bound, gain>=max(1e-4,10% full SOURCE gain); not semantic effectiveness',full_source_gain=source_gain,candidates=rows,qp=qp,counts=counts,formal_training_started=False,independent_confirmation=False))
    finally:hook.detach()
