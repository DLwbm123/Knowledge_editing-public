"""Log-bound source completeness, prefix and local feasible-direction diagnosis."""
import json,math
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
from .editor import nll,finite,editable_fp32
from .coefficient_runtime import compose,native_weight,coefficient_forward
from .contrast_basis import solve as rebuild
from .brc_pair import joint_correction,tangent_direction
from .dual_solver import fisher_cost
from .factF3_ops import bind_groups


def directions(gs,gn,gm):
    def unit(x):
        norm=x.norm()
        if not torch.isfinite(norm) or float(norm)==0:raise ValueError('Degenerate diagnostic direction')
        return x/norm
    ce=tangent_direction(gs,gn).double()
    mn=tangent_direction(-gm,gn).double()
    g=gs.double();h=gn.double();g=g-h*(g.flatten()@h.flatten())/h.square().sum()
    both=mn.clone()
    if float((both*g).sum())>0 and float(g.square().sum())>0:
        both-=g*(both*g).sum()/g.square().sum()
    return [unit(x).float() for x in (ce,mn,both)]


def run(runtime,layer,record,pair,factor,config,budget,root):
    root=Path(root);pub=root/'public';pub.mkdir();private=root/'private'
    binding=json.loads((private/'diagnostic_binding.json').read_text())
    original=layer.weight.detach().clone();flags=[(p,p.requires_grad) for p in runtime.model.parameters()];modes=[(m,m.training) for m in runtime.model.modules()]
    a,b=(factor[k].to(original.device).float() for k in ('a','b'))
    baseline=None
    def save(name,x):(pub/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    def event(x):
        with (pub/'REBUILD_EVENTS.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    try:
        old=json.loads((Path(__file__).resolve().parents[2]/'reports/vigor_contrastBasis_v1_20261021/FREEZE.json').read_text())
        old['arms']=['CONTRAST4']
        rebuilt=rebuild(runtime,layer,pair,factor,old,dict(source_bound=True,input_bound=True,F0_bound=True,KV_smoke_pass=True,resource_guard_ready=True),event,budget=budget)
        baseline=rebuilt['baseline'];arm=rebuilt['arms']['CONTRAST4'];expected=binding['endpoint_expected']
        for left,right in zip(arm['finals'],expected['finals']):
            if any(abs(x-y)>1e-6 for x,y in zip(left['native_nll'],right['native_nll'])) or abs(left['fisher_cost']-right['fisher_cost'])>1e-12:raise ValueError('Reconstruction differs from frozen endpoint')
        if abs(arm['native_distance']-expected['native_distance'])>1e-8:raise ValueError('Reconstruction distance differs')
        u,v=arm['basis_private'];z=arm['coefficients_private'][0];scale=math.sqrt(2*old['trust_budget'])
        endpoint=compose(original,u,z*scale,v).half();del rebuilt,arm,z
        packets={}
        for pi,prefix in enumerate(binding['prefixes']):
            for ci,completion in enumerate(binding['completions']):
                text=prefix+completion;batch=runtime.build_edit_batch(replace(record,target=text))
                group,meta=bind_groups(runtime,batch,dict(source_target=text,groups=[dict(start=len(prefix),end=len(text))]))
                if pi==1:
                    ids=batch.labels[:,1:];ids=ids[ids!=-100].tolist();count=min(group[0])
                    if count==0 or ids[:count]!=binding["saved_generated_ids"][:count]:raise ValueError("Actual prefix token binding differs")
                facts=[]
                if ci==0:
                    start=len(prefix)
                    for fact in binding['facts']:
                        pos=text.index(fact,start);fg,_=bind_groups(runtime,batch,dict(source_target=text,groups=[dict(start=pos,end=pos+len(fact))]));facts.append(fg[0]);start=pos+len(fact)
                packets[pi,ci]=(batch,group[0],facts)
        def measure(key,details=False):
            batch,content,facts=packets[key];out=runtime.model(**batch.forward_kwargs()).logits[:,:-1].float();labels=batch.labels[:,1:]
            valid=labels!=-100;ce=F.cross_entropy(out.reshape(-1,out.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(labels)[valid]
            selected=ce[content];loss=finite(selected.mean())
            if not details:return loss
            top=(out.argmax(-1)[valid]==labels[valid]);return dict(mean_logp=-float(loss),sum_logp=-float(selected.sum()),content_tokens=len(content),facts=[dict(nll=float(ce[g].mean()),top1_hits=int(top[g].sum()),tokens=len(g)) for g in facts])
        def margin():return measure((1,1))-measure((1,0))
        table={}
        for name,w in [('W0',original),('ENDPOINT0',endpoint)]:
            with native_weight(runtime.model,layer,w),torch.no_grad():
                rows=[[measure((pi,ci),True) for ci in range(3)] for pi in range(2)]
            table[name]=dict(prefixes=rows,margins=[row[0]['mean_logp']-row[1]['mean_logp'] for row in rows])
        save('PREFIX_FACTS.json',table)
        with native_weight(runtime.model,layer,endpoint):
            with editable_fp32(runtime.model,layer,readonly=True):
                gm=-budget.grad(measure((1,0)),(layer.weight,),1)[0].detach()
                gm+=budget.grad(measure((1,1)),(layer.weight,),1)[0].detach();finite(gm)
            qu=torch.linalg.qr(u,mode='reduced').Q;qv=torch.linalg.qr(v,mode='reduced').Q
            retained=qu.T@gm@qv
            retention=float(retained.double().norm()/gm.double().norm())
            mapped=scale*(u.T@gm@v);del gm,qu,qv,retained
            zero=torch.zeros((4,4),device=original.device,requires_grad=True)
            def grad_one(fn):
                with coefficient_forward(runtime.model,layer,u,zero*scale,v):return finite(budget.grad(fn(),(zero,),1)[0].detach())
            gm=-grad_one(lambda:measure((1,0)))+grad_one(lambda:measure((1,1)))
            gs=grad_one(lambda:nll(runtime.model,pair[0]));gn=grad_one(lambda:nll(runtime.model,pair[1]))
            mapping_error=float((mapped-gm).norm()/gm.norm().clamp_min(1e-20));ds=directions(gs,gn,gm)
            geometry=dict(full_gradient_fixed_span_retention=retention,full_to_coefficient_relative_error=mapping_error,source_null_cosine=float((gs*gn).sum()/gs.norm()/gn.norm()),margin_source_cosine=float((gm*gs).sum()/gm.norm()/gs.norm()),margin_null_cosine=float((gm*gn).sum()/gm.norm()/gn.norm()),margin_gradient_norm=float(gm.norm()),directional_predictions=[dict(direction=name,margin_derivative=float((gm*d).sum()),source_derivative=float((gs*d).sum()),null_derivative=float((gn*d).sum())) for name,d in zip(config['diagnostic_directions'],ds)])
            if mapping_error>1e-3:raise ValueError('Full/coefficient gradient mismatch')
        save('GEOMETRY.json',geometry)
        mean0=expected['finals'][0]['native_nll'][0];m0=table['ENDPOINT0']['margins'][1]
        def native_eval(d,with_margin=True):
            w=compose(endpoint,u,d*scale,v).half()
            with native_weight(runtime.model,layer,w),torch.no_grad():
                vals=[float(nll(runtime.model,bb)) for bb in pair];m=float(margin()) if with_margin else None
            cost=fisher_cost(w.float()-original.float(),a,b)
            passed=vals[0]<=mean0 and abs(vals[1]-baseline[1])<=.01 and cost<=.001
            return dict(native_nll=vals,fisher_cost=cost,margin=m,source_pass=vals[0]<=mean0,null_pass=abs(vals[1]-baseline[1])<=.01,trust_pass=cost<=.001,all_gates=passed)
        probes=[]
        for name,d in zip(config['diagnostic_directions'],ds):
            candidate=d*config['local_radius'];raw=native_eval(candidate);proposed=raw;trace=[]
            for j in range(config['repair_steps']):
                rs=max(0.,proposed['native_nll'][0]-mean0);rn=proposed['native_nll'][1]-baseline[1]
                if rs==0 and abs(rn)<=.01:break
                point=candidate.detach().requires_grad_(True)
                with native_weight(runtime.model,layer,endpoint):
                    with coefficient_forward(runtime.model,layer,u,point*scale,v):g=budget.grad(nll(runtime.model,pair[0]),(point,),1)[0].detach()
                    with coefficient_forward(runtime.model,layer,u,point*scale,v):h=budget.grad(nll(runtime.model,pair[1]),(point,),1)[0].detach()
                delta=joint_correction(g,h,rs,rn,config['repair_radius'])
                if delta is None:trace.append(dict(status='DEGENERATE_OR_RADIUS'));break
                candidate=(point.detach()+delta).detach();proposed=native_eval(candidate,False);trace.append(proposed)
            final=native_eval(candidate) if trace and proposed.get('margin') is None else proposed
            probes.append(dict(direction=name,raw=raw,corrections=trace,final=final,actual_margin_gain=final['margin']-m0,feasible_margin_improvement=final['all_gates'] and final['margin']-m0>1e-5))
        save('LOCAL_PROBES.json',probes)
        save('RESULT.json',dict(status='COMPLETE_DIAGNOSTIC_ONLY',reconstructed_endpoint_matched=True,new_generations=0,new_judge_calls=0,local_feasible_margin_directions=sum(x['feasible_margin_improvement'] for x in probes),geometry=geometry))
    except BaseException as e:
        save('FAILURE.json',dict(status='FAILED_NO_RETRY',exception_type=type(e).__name__,message=str(e)));raise
    finally:
        budget.restoring=True
        with torch.no_grad():layer.weight.copy_(original)
        for p,f in flags:p.requires_grad_(f)
        for m,mode in modes:m.training=mode
        if baseline is not None:
            with torch.no_grad():actual=[float(nll(runtime.model,b)) for b in pair]
            if any(abs(x-y)>1e-5 for x,y in zip(actual,baseline)):raise RuntimeError('Diagnostic baseline restoration')
        if not torch.equal(layer.weight,original):raise RuntimeError('Diagnostic weights restoration')
        save('COSTS.json',dict(forwards=budget.forwards,backward_API=budget.backwards,model_gradient_paths=budget.gradient_paths,restored=True))
