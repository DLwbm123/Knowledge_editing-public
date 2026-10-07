"""Fixed TT88 continuation; no routing/loss search. Uses frozen PR31 runtime."""
import os,sys,time,copy,random
from pathlib import Path
from dataclasses import replace
RUN=Path(os.environ['RUN_ROOT']);BASE=Path(os.environ['BASE_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import common
import importlib.util
sys.path.insert(0,str(RUN/'private/tools'))
def tool(name):
    spec=importlib.util.spec_from_file_location('closeout_'+name,RUN/'private/tools'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
train=tool('train');sys.modules['train']=train
worker=tool('worker')
import torch
from methods.medtrace.core import MedTraceLayerHook


def hinge(x,b):return torch.relu(x-b)

def cpu_check():
    x=torch.tensor(2.,requires_grad=True);assert hinge(x,2.).item()==0;hinge(x,1.).backward();assert x.grad.item()==1
    x=torch.tensor(1.,requires_grad=True);hinge(x,2.).backward();assert x.grad.item()==0
    from structures import TT4
    e=TT4(1,8,8);assert sum(p.numel() for p in e.parameters())==7168
    common.write(RUN/'public/CPU_TEST.json',dict(status='PASS',hinge_zero_trigger_inactive_gradient=True,TT_parameters=7168))


def hold_baselines(runtime,t,init):
    e=train.clone(init,t['seed'],runtime.device);h=MedTraceLayerHook(runtime.get_module(common.LAYER),e);h.attach()
    rec=common.record(t);batches=[runtime.build_edit_batch(rec)]+[runtime.build_edit_batch(replace(rec,question=q)) for q in t['fit_questions']]
    try:
        with common.diagnostic_scope(h),torch.no_grad():
            vals=[];errors=[]
            for b in batches:
                h.set_teacher_routing(b.labels);a=float(runtime.compute_loss(b));c=float(runtime.compute_loss(b));vals.append(a);errors.append(abs(a-c))
        assert max(errors)==0.,'Nonzero repeat error: freeze tolerance before any training'
        return vals
    finally:h.detach()


def update_for(mode,baselines,fit_order):
    original=train.update;step=[0]
    def update(runtime,hook,e,opt,native,fit,teacher):
        if mode=='CE_U_MULTI':return original(runtime,hook,e,opt,native,fit,teacher)
        opt.zero_grad(set_to_none=True);before={k:v.detach().clone() for k,v in e.state_dict().items()};terms={};fwd=bwd=0
        if mode=='HOLD_U':
            for name,batch,i in [('native',native,0),('fit',fit,fit_order[(int(next(iter(opt.state.values()))['step']) if opt.state else 0)%4])]:
                hook.set_teacher_routing(batch.labels);ce=runtime.compute_loss(batch);hold=hinge(ce,baselines[i]);(.5*hold).backward();fwd+=1;bwd+=1
                terms[name]=dict(unweighted=float(ce.detach()),weighted=.5*float(hold.detach()),baseline=baselines[i],hinge=float(hold.detach()),active=bool(hold.detach()>0),tokens=len(batch.target_token_ids))
        kwargs,labels,mask,logp,row,_=teacher;hook.set_teacher_routing(labels);kl=train.full_vocab_kl(runtime.model(**kwargs).logits[mask],logp);(.01*kl).backward();fwd+=1;bwd+=1
        terms['U']=dict(unweighted=float(kl.detach()),weighted=.01*float(kl.detach()),tokens=int(mask.sum()),source=common.digest(row['source_group']))
        norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.);assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters());assert not any(p.grad is not None for p in runtime.model.parameters())
        post=float(train.grad(e).norm());opt.step();step[0]+=1
        return dict(terms=terms,preclip_norm=float(norm),postclip_norm=post,core_updates={k:float((v-before[k]).norm()) for k,v in e.state_dict().items()},learning_rates=[g['lr'] for g in opt.param_groups],Adam_steps=[int(s['step']) for s in opt.state.values()],forwards=fwd,backwards=bwd,train_tokens=sum(x['tokens'] for x in terms.values()),hinge_active=sum(x.get('active',False) for x in terms.values()))
    return update


def native_check(runtime,t,init,teachers,baseline):
    rec=common.record(t);native=runtime.build_edit_batch(rec);fit=runtime.build_edit_batch(replace(rec,question=t['fit_questions'][0]))
    e=train.clone(init,t['seed'],runtime.device);h=MedTraceLayerHook(runtime.get_module(common.LAYER),e);h.attach()
    try:
        with common.diagnostic_scope(h):
            h.set_teacher_routing(native.labels);ce=runtime.compute_loss(native);assert hinge(ce,baseline[0]).item()==0
            triggered=hinge(ce,baseline[0]-.01);assert triggered.item()>0;triggered.backward();assert train.grad(e).norm()>0
            assert not any(p.grad is not None for p in runtime.model.parameters())
            opt=train.optimizer_for(e,runtime.model);item=update_for('U_ONLY',baseline,[1,2,3,4])(runtime,h,e,opt,native,fit,teachers[0]);assert item['forwards']==item['backwards']==1
            assert teachers[0][2].sum()==len(teachers[0][-1]['tokens']) and native.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id
        common.write(RUN/'public/NATIVE_TEST_'+os.environ['PARTITION']+'.json',dict(status='PASS',hinge_initial_zero=True,hinge_trigger_gradient=True,TT_gradient=True,Base_gradient=False,teacher_all_editing_off=True,token_mask_EOS=True,parameters=7168,epsilon=0.,extra_forwards=2,extra_backward=2))
    finally:h.detach()


def main():
    gpu=int(os.environ['GPU']);part=int(os.environ['PARTITION']);ts=[t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'];selected=ts[:8]
    with common.lease(gpu):
        runtime,bindings=common.load(gpu);roles=common.read(BASE/'private/U_ROLES.json');train.assert_base_off(runtime)
        teachers=train.teachers_for(runtime,roles['FIT']);check=train.teachers_for(runtime,roles['CAL']+roles['CHECK']);initial_update=train.update
        first=selected[part];init0=torch.load(worker.w0_path(first,0),map_location='cpu',weights_only=True)['expert'];bl=hold_baselines(runtime,first,init0);native_check(runtime,first,init0,teachers,bl)
        for t in selected[part::2]:
            w0=worker.w0_path(t,0);init=torch.load(w0,map_location='cpu',weights_only=True)['expert'];baseline=hold_baselines(runtime,t,init)
            common.write(RUN/'private'/('HOLD_BASELINE_'+t['anonymous_edit']+'.json'),dict(CE=baseline,epsilon=0.,repeat_error=0.,extra_forwards=10))
            for slot in range(2):
                for arm in ['CE_U_MULTI','U_ONLY','HOLD_U']:
                    if arm=='CE_U_MULTI' and slot==0:continue
                    common.budget();ct=worker.train_task(t,slot,roles['FIT']);ct['bindings']['closeout_mode']=arm
                    fit=list(range(1,5));random.Random(ct['seed']).shuffle(fit);train.update=update_for(arm,baseline,fit)
                    directory=RUN/'private/edits'/t['anonymous_edit']/('s'+str(slot))/arm
                    train.continuation(runtime,ct,common.record(t),init,directory,teachers,160,[160]);train.update=initial_update
                    common.write(RUN/'private'/('UNIT_'+t['anonymous_edit']+'_'+str(slot)+'_'+arm+'.json'),dict(status='TRAINED',updates=160))
            print('EDIT_TRAIN_COMPLETE',t['order'],flush=True)
        common.write(RUN/'private'/('TRAIN_PART_'+str(part)+'.json'),dict(status='COMPLETE',epoch=time.time()))
    common.write(RUN/'public/PROGRESS.json',dict(status='CORE_TRAINING_PART_FINISHED',part=part,whole_task_complete=False))

if __name__=='__main__':
    if os.environ.get('ACTION')=='cpu':cpu_check()
    else:
        try:main()
        except BaseException as e:
            import traceback
            common.write(RUN/'private'/('FAILURE_CORE_'+os.environ.get('PARTITION','x')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
