"""Fixed versus movable rank-four search under the same native protection gates."""
import json
import math
import time
import torch
from .brc_pair import joint_correction
from .coefficient_runtime import native_weight, validate
from .contrast_basis import residual_gradient
from .dual_execution import ExecutionBudget
from .dual_solver import fisher_cost, source_basis
from .editor import editable_fp32, finite, nll


class ArmBudgetExhausted(Exception):
    pass


class ReachBudget(ExecutionBudget):
    arm_start = None

    def before_forward(self, model, args):
        if not self.restoring and self.arm_start is not None and self.forwards-self.arm_start >= self.config['arm_edit_forward_max']:
            raise ArmBudgetExhausted()
        super().before_forward(model, args)


def dense(state):
    u,s,v = state
    return (u*s)@v.T


def project(g,u,v,fixed):
    if fixed:
        return u@(u.T@g@v)@v.T
    left = u@(u.T@g)
    return left+(g@v-u@(u.T@g@v))@v.T


def retract(y,u,v,rank,fixed):
    if fixed:
        l,s,r = torch.linalg.svd(u.T@y@v,full_matrices=False)
        return u@l[:,:rank],s[:rank],v@r[:rank].T
    # Every trial is current rank-r plus a tangent at (u,v), hence rank <= 2r.
    a=y.T@u
    b=y@v-u@(u.T@y@v)
    ql,rl=torch.linalg.qr(torch.cat((u,b),dim=1),mode='reduced')
    qr,rr=torch.linalg.qr(torch.cat((a,v),dim=1),mode='reduced')
    l,s,r=torch.linalg.svd(rl@rr.T,full_matrices=False)
    return ql@l[:,:rank],s[:rank],qr@r[:rank].T


def run(runtime,layer,record,pair,factor,config,budget,root,*,replay_plan=None):
    validate(runtime.model,layer)
    model=runtime.model
    base=layer.weight.detach().clone()
    modes=[(m,m.training) for m in model.modules()]
    flags=[(p,p.requires_grad) for p in model.parameters()]
    pub=root/'public';pub.mkdir()
    private=root/'private'
    manifest=json.loads((private/'binding.json').read_text())
    result=dict(status='RUNNING',arms={})
    baseline=None
    def save(name,value):
        with (pub/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    def event(value):
        value.update(forwards=budget.forwards,backward_API=budget.backwards)
        with (pub/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>config['disk_bytes_max']-16*1024**2:
            raise OSError('Storage reserve exhausted')
    a,b=(factor[k].to(base.device).float() for k in ('a','b'))
    sa,sb=a.sqrt(),b.sqrt()
    scale=math.sqrt(2*config['trust_budget'])
    def weight(state):
        return finite(base.float()+scale*dense(state)/sb[:,None]/sa[None,:]).to(base.dtype)
    def gradients(state=None):
        with native_weight(model,layer,base if state is None else weight(state)):
            with editable_fp32(model,layer,readonly=True):
                gs=finite(budget.grad(nll(model,pair[0]),(layer.weight,),1)[0].detach())
                gn=finite(budget.grad(nll(model,pair[1]),(layer.weight,),1)[0].detach())
        return gs,gn
    def evaluate(state):
        w=weight(state)
        with native_weight(model,layer,w),torch.no_grad():
            losses=[float(nll(model,batch)) for batch in pair]
        cost=fisher_cost(w.float()-base.float(),a,b)
        return dict(native_nll=losses,fisher_cost=cost,changed=bool(torch.count_nonzero(w-base)),protection_feasible=abs(losses[1]-baseline[1])<=config['null_budget'] and cost<=config['trust_budget'])
    def white_gradients(state):
        gs,gn=gradients(state)
        return finite(scale*gs/sb[:,None]/sa[None,:]),finite(scale*gn/sb[:,None]/sa[None,:])
    try:
        model.eval()
        with torch.no_grad():baseline=[float(nll(model,batch)) for batch in pair]
        gs,gn=gradients()
        residual,ratio=residual_gradient(gs,gn,1e-6)
        u,v=source_basis(residual,a,b,config)
        u0=torch.linalg.qr(sb[:,None]*u,mode='reduced')[0]
        v0=torch.linalg.qr(sa[:,None]*v,mode='reduced')[0]
        del gs,gn,residual,u,v
        for trial in range(config['seed_trials']):
            seed=(u0,torch.full((config['rank'],),config['seed_amplitude']*.5**trial,device=base.device),v0)
            initial=evaluate(seed)
            event(dict(status='SEED',trial=trial,metrics=initial))
            if initial['changed'] and initial['protection_feasible']:break
        else:raise RuntimeError('No admissible common rank-four seed')
        result.update(baseline=baseline,seed=initial,seed_trial=trial,residual_ratio=ratio)
        states={}
        arms=config['arms'] if replay_plan is None else ['ADAPT']
        for arm in arms:
            fixed=arm=='FIXED';state=seed;current=initial;accepted=0;trace=[];milestone=None
            phase_start=budget.forwards
            budget.arm_start=phase_start if replay_plan is None else None;start_b=budget.backwards
            stop='STEP_SAFETY_NOT_CONVERGED'
            try:
                for step in range(config['steps_safety_max'] if replay_plan is None else len(replay_plan)):
                    if current['native_nll'][0]<=config['source_target']:
                        stop='SOURCE_TARGET_REACHED';break
                    gs,gn=white_gradients(state)
                    pu,pv=(u0,v0) if fixed else (state[0],state[2])
                    ps,pn=project(gs,pu,pv,fixed),project(gn,pu,pv,fixed)
                    d=-ps
                    n2=float(pn.double().square().sum())
                    if n2>0:d=d-pn*(float((pn.double()*d.double()).sum())/n2)
                    ratio=float(d.double().norm()/gs.double().norm())
                    d=d/max(float(d.norm()),1e-30)
                    del gs,gn,ps,pn
                    ok=False
                    trials=range(config['backtracks']) if replay_plan is None else [replay_plan[step]['trial']]
                    for trial in trials:
                        candidate=retract(dense(state)+config['lr']*.5**trial*d,pu,pv,config['rank'],fixed)
                        proposed=evaluate(candidate)
                        for repair in range(config['repair_steps']):
                            nr=proposed['native_nll'][1]-baseline[1]
                            sr=max(0.,proposed['native_nll'][0]-current['native_nll'][0])
                            if abs(nr)<=config['null_budget'] and sr==0:break
                            cg,ch=white_gradients(candidate)
                            cu,cv=(u0,v0) if fixed else (candidate[0],candidate[2])
                            cg,ch=project(cg,cu,cv,fixed),project(ch,cu,cv,fixed)
                            correction=joint_correction(cg,ch,sr,nr,config['repair_radius'])
                            del cg,ch
                            if correction is None:break
                            candidate=retract(dense(candidate)+correction,cu,cv,config['rank'],fixed)
                            del correction
                            proposed=evaluate(candidate)
                        ok=proposed['protection_feasible'] and proposed['native_nll'][0]<current['native_nll'][0]
                        event(dict(status='ACCEPT' if ok else 'REJECT',arm=arm,step=step+1,trial=trial,projected_gradient_ratio=ratio,metrics=proposed))
                        if replay_plan is not None:
                            expected=replay_plan[step]['metrics']
                            if not ok or any(abs(x-y)>1e-5 for x,y in zip(proposed['native_nll'],expected['native_nll'])) or abs(proposed['fisher_cost']-expected['fisher_cost'])>1e-6*max(expected['fisher_cost'],1e-30):
                                raise RuntimeError('Accepted path reconstruction mismatch')
                        if ok:break
                    del d
                    if not ok:
                        stop='SEARCH_STALLED';break
                    gain=(current['native_nll'][0]-proposed['native_nll'][0])/max(current['native_nll'][0],1e-30)
                    state,current=candidate,proposed;accepted+=1
                    trace.append(dict(accepted=accepted,source_relative_gain=gain,projected_gradient_ratio=ratio,arm_forwards=budget.forwards-phase_start))
                    if accepted==4:milestone=(state,current)
                    if ratio<=config['stationarity_ratio'] and len(trace)>=config['stationarity_window'] and all(x['source_relative_gain']<=config['relative_progress'] for x in trace[-config['stationarity_window']:]):
                        stop='LOCAL_STATIONARITY_PROXY';break
                if current['native_nll'][0]<=config['source_target']:stop='SOURCE_TARGET_REACHED'
            except ArmBudgetExhausted:
                stop='BUDGET_NOT_CONVERGED'
            finally:
                arm_f=budget.forwards-phase_start;budget.arm_start=None
            states[arm]=(milestone,(state,current))
            result['arms'][arm]=dict(accepted=accepted,stop=stop,edit_forwards=arm_f,backward_API=budget.backwards-start_b,trace=trace,stage4=milestone[1] if milestone else None,final=current,generations=[])
            event(dict(status='ARM_COMPLETE',arm=arm,stop=stop,accepted=accepted))
        # All optimization is sealed before the first complete generation.
        for arm in arms:
            for i,item in enumerate(states[arm]):
                if replay_plan is not None and i==0:
                    continue  # The failed milestone request is never replayed.
                alias=f'{arm}_{i}'
                if item is None:
                    packet=dict(alias=alias,status='STAGE_NOT_REACHED')
                elif i==1 and result['arms'][arm]['accepted']==4:
                    packet=json.loads((private/f'{arm}_0_GENERATION.json').read_text());packet['alias']=alias
                    result['arms'][arm]['generations'].append(dict(endpoint=i,reused_identical_current_state=True,generation_forwards=0))
                else:
                    with (private/f'{alias}_RESERVED.json').open('x') as f:json.dump(dict(epoch=time.time()),f)
                    budget.begin_generation()
                    try:
                        with native_weight(model,layer,weight(item[0])),torch.inference_mode():
                            raw=runtime.adapter.prepare_inputs(record.image_path,record.question,None)
                            if raw['input_ids'].tolist()!=manifest['input']['prompt_ids'] or raw['attention_mask'].tolist()!=manifest['input']['attention_mask'] or raw['image_sha256']!=manifest['input']['existing_image_binding']:
                                raise PermissionError('Generation input binding mismatch')
                            output=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        ids=list(output.raw_token_ids);eos=model.generation_config.eos_token_id
                        eos=eos if isinstance(eos,(list,tuple)) else [eos]
                        complete=bool(ids) and ids[-1] in eos and len(ids)<=config['max_new_tokens']
                        packet=dict(alias=alias,status='COMPLETE' if complete else 'INCOMPLETE_NO_RETRY',question=record.question,gold_answer=record.target,raw_base_answer=output.decoded_text,raw_token_ids=ids)
                        result['arms'][arm]['generations'].append(dict(endpoint=i,generation_forwards=budget.request_forwards,generated_tokens=len(ids),complete=complete))
                    finally:budget.generating=False
                with (private/f'{alias}_GENERATION.json').open('x') as f:json.dump(packet,f)
        result['status']='COMPLETE'
    except BaseException as error:
        result.update(status='FAILED_NO_RETRY',exception_type=type(error).__name__)
        raise
    finally:
        budget.generating=False;budget.restoring=True;budget.arm_start=None
        try:
            with torch.no_grad():
                layer.weight.copy_(base)
                if baseline is not None:
                    actual=[float(nll(model,batch)) for batch in pair]
                    if any(abs(x-y)>1e-5 for x,y in zip(actual,baseline)):raise RuntimeError('Base restoration loss mismatch')
            result['restored']=torch.equal(layer.weight,base)
            event(dict(status='RESTORED'))
        finally:
            for p,f in flags:p.requires_grad_(f)
            for m,mode in modes:m.training=mode
            save('NUMERIC_TERMINAL.json',result)
            (private/'NUMERIC_TERMINAL.json').write_text(json.dumps(result))
            save('COSTS.json',dict(forwards=budget.forwards,backward_API=budget.backwards,model_gradient_paths=budget.gradient_paths,generation_requests=budget.requests,elapsed_seconds=time.monotonic()-budget.started))
