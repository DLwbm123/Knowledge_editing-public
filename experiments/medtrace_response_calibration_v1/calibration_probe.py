"""Frozen PR47 directions: original arithmetic versus FP32 on identical embeddings."""
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import fisher_probe as prior
import calibration_math as math

q,c,p,RUN = prior.q,prior.c,prior.p,prior.RUN
PARENT = Path(os.environ['CALIBRATION_PARENT'])
SCALES = (.25,.5,1.)
ARMS = ('RAW','BOUNDED')
PRECISIONS = ('ORIGINAL_FP16','MATH_FP16','MATH_FP32')


def package(task):
    return RUN/'private/candidates'/f"{task['order']}.pt"


def plan():
    basis,held=q.split()
    assert len(basis)==61 and len(held)==96 and len(q.d.selected())==8
    assert c.read(PARENT/'public/RESULTS.json')['decision']=='NO_LOCAL_SUPPORT'
    assert sum(len(c.read(x['response_path'])['R0']['raw_token_ids']) for x in basis)==1514
    assert all(package(t).is_file() for t in q.d.selected()),'Reuse completed original candidates only'
    lock=dict(experts=8,basis_questions=61,edit_questions=5,heldout_measured=False,
        scales=[0.,*SCALES],precisions=PRECISIONS,arms=ARMS,new_candidates=0,
        candidate_reconstructions=0,inherited_fixed_candidates=8,forward_calls=13279,backward_passes=4232,
        reconstruction_forwards=0,reconstruction_backwards=0,
        mechanical_forwards=31,mechanical_backwards=8,
        diagnostic_forwards=13248,diagnostic_backwards=4224,
        JVP_method='autograd.functional.jvp double backward, no random probes',
        primary_gate_unchanged='PR47 NO_LOCAL_SUPPORT; calibration is not a method success test',
        new_generations=0,new_Judge=0,temporary_candidate_packages=8,
        role_binding=c.digest(basis),code=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
    c.write(RUN/'private/CALIBRATION_LOCK.json',lock)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',selfcheck=math.selfcheck(),
        **{k:v for k,v in lock.items() if k not in ('role_binding','code')}))
    p.done('PLAN_COMPLETE')


def reconstruct(runtime,task):
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert not package(task).exists(),'No implicit candidate reconstruction retry'
    seed_everything(task['seed'])
    expert=p.expert(q.d.start_state(task),task['seed'],runtime.device)
    before={k:v.detach().clone() for k,v in expert.state_dict().items()}
    hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    try:
        matrix,backwards,basis_audit=prior.basis_gradients(runtime,hook,q.split()[0],task,dict(expert.named_parameters()))
        native=[runtime.build_edit_batch(replace(c.record(task),question=x)) for x in [task['native']['question']]+task['fit_questions']]
        seed_everything(task['seed']);opt=p.optimizer(expert,runtime.model);opt.zero_grad(set_to_none=True)
        for batch in native[:2]:
            hook.set_teacher_routing(batch.labels);(.5*runtime.compute_loss(batch)).backward()
        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
        assert torch.isfinite(norm)
        opt.step();q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW')
        raw={k:v.detach().clone() for k,v in expert.state_dict().items()}
        delta,audit,extra_f,extra_b=q.candidate_direction(runtime,hook,expert,native,before,raw,matrix)
        bounded=q.add(before,delta)
        old=c.read(PARENT/'private/results'/f"{task['order']}.json")['geometry']
        for key in ('objective_RAW','objective_BOUNDED','RAW_map_norm','BOUNDED_map_norm','raw_linear_edit_progress','actual_linear_edit_progress'):
            assert abs(audit[key]-old[key])<=1e-14+1e-8*abs(old[key]),(key,audit[key],old[key])
        states={'BASE':before,'RAW':raw,'BOUNDED':bounded}
        assert all(torch.equal(state[k],before[k]) for state in states.values() for k in ('G2','G3','G4'))
        predictions={arm:(.5*(matrix@(q.flatten(states[arm])-q.flatten(before))).square().reshape(61,32).sum(-1)*61).tolist() for arm in ARMS}
        value=dict(states={a:{k:v.detach().cpu() for k,v in state.items()} for a,state in states.items()},
            sketch_predictions=predictions,geometry=audit,expert_order=task['order'],
            forward_calls=61+2+extra_f,backward_passes=backwards+2+extra_b,
            parent_geometry_reproduced=True,basis_audit=basis_audit)
        c.save(package(task),value)
        c.write(RUN/'private/reconstruction'/f"{task['order']}.json",{k:v for k,v in value.items() if k not in ('states','sketch_predictions')})
        expert.load_state_dict(before)
        assert all(torch.equal(v,expert.state_dict()[k]) for k,v in before.items())
    finally:
        hook.detach()


def require_memory(gpu):
    import subprocess
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=48000,'FP32/JVP preflight requires 48 GiB-class free memory including reserve'


def prepare_worker():
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part]
    require_memory(gpu)
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for task in q.d.selected()[part::len(q.GPUS)]:reconstruct(runtime,task)
    p.done('PREPARED_'+str(part))


def cpu_batches(runtime,task,mechanical):
    basis=q.split()[0]
    rows=[('BASIS',i,q.protection_batch(runtime,row,task)) for i,row in enumerate(basis[:1] if mechanical else basis)]
    if not mechanical:
        rows += [('EDIT',i,runtime.build_edit_batch(replace(c.record(task),question=x))) for i,x in enumerate([task['native']['question']]+task['fit_questions'])]
    return [(role,index,{k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in batch.forward_kwargs().items()},batch.labels.cpu()) for role,index,batch in rows]


def measure(runtime,task,mechanical=False):
    saved=torch.load(package(task),map_location='cpu',weights_only=True)
    expert=p.expert(saved['states']['BASE'],task['seed'],runtime.device).requires_grad_(False)
    hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    original=expert.G1
    base=original.detach().clone()
    changes={arm:saved['states'][arm]['G1'].to(base)-base for arm in ARMS}
    dtypes=[(tensor,tensor.dtype) for tensor in list(runtime.model.parameters())+list(runtime.model.buffers()) if tensor.is_floating_point()]
    tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    sdp=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),
         torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled())
    native_references={}
    rows=[];counts=dict(forward_calls=0,backward_passes=0,zero_Base_checks=0,repeat_checks=0,JVP_zero_checks=0,parent_response_checks=0)
    parent={(x['role'],x['index']):x for x in c.read(PARENT/'private/results'/f"{task['order']}.json")['diagnostics']}
    try:
        batches=cpu_batches(runtime,task,mechanical)
        for precision in PRECISIONS:
            if precision!='ORIGINAL_FP16':
                torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if precision=='MATH_FP32':runtime.model.float()
            dtype=next(runtime.model.parameters()).dtype
            assert dtype==(torch.float32 if precision=='MATH_FP32' else torch.float16)
            for batch_index,(role,index,cpu,labels) in enumerate(batches):
                c.budget();labels=labels.to(runtime.device);mask=labels[:,1:]!=-100
                target=labels[:,1:][mask].cpu()
                kwargs={k:(v.to(device=runtime.device,dtype=dtype if v.is_floating_point() else v.dtype) if isinstance(v,torch.Tensor) else v) for k,v in cpu.items()}
                kwargs['labels']=None
                hook.set_teacher_routing(labels)
                def forward(alpha,arm='RAW'):
                    expert._parameters['G1']=base+alpha*changes[arm]
                    counts['forward_calls']+=1
                    return runtime.model(**kwargs).logits[:,:-1][mask].float()
                with torch.no_grad():reference=forward(0.).detach().cpu().double()
                if batch_index==0:
                    with torch.no_grad():repeat=forward(0.).detach().cpu().double()
                    assert torch.equal(reference,repeat);counts['repeat_checks']+=1
                    hook.enabled=False
                    try:
                        with torch.no_grad():unedited=forward(0.).detach().cpu().double()
                    finally:hook.enabled=True
                    assert torch.equal(reference,unedited);counts['zero_Base_checks']+=1
                if precision=='ORIGINAL_FP16':native_references[(role,index)]=reference
                baseline_shift=q.compare(native_references[(role,index)],reference,target)['KL']
                for arm in ARMS:
                    tangent=None
                    if precision!='ORIGINAL_FP16':
                        alpha=torch.zeros((),device=runtime.device,dtype=torch.float32)
                        with torch.enable_grad():
                            z,tangent=torch.autograd.functional.jvp(lambda a:forward(a,arm),alpha,torch.ones_like(alpha),strict=True)
                        counts['backward_passes']+=2
                        assert torch.equal(reference,z.detach().cpu().double())
                        assert torch.isfinite(tangent).all();counts['JVP_zero_checks']+=1
                        tangent=tangent.detach().cpu().double();del z
                    for scale in SCALES:
                        with torch.no_grad():candidate=forward(scale,arm).detach().cpu().double()
                        response=q.compare(reference,candidate,target)
                        row=dict(precision=precision,role=role,index=index,arm=arm,scale=scale,
                            **response,**math.response_metrics(reference,candidate,tangent,scale),
                            baseline_KL_from_native=baseline_shift,
                            sketch_quadratic=saved['sketch_predictions'][arm][index]*scale**2 if role=='BASIS' else None)
                        if precision=='ORIGINAL_FP16' and scale==1.:
                            old=parent[('BASIS_FIT' if role=='BASIS' else 'EDIT',index)]['candidates'][arm]
                            for key in ('KL','NLL_change','lost_correct_tokens'):
                                assert abs(row[key]-old[key])<=1e-12,('Parent response changed',role,index,arm,key,row[key],old[key])
                            counts['parent_response_checks']+=1
                        rows.append(row)
                    del tangent,candidate
                if (batch_index+1)%12==0:print('CALIBRATION',task['order'],precision,batch_index+1,flush=True)
        assert not any(v.grad is not None for v in runtime.model.parameters())
    finally:
        expert._parameters['G1']=original
        hook.detach()
        for tensor,dtype in dtypes:
            if tensor.dtype!=dtype:tensor.data=tensor.data.to(dtype)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32
        for setter,value in zip((torch.backends.cuda.enable_flash_sdp,torch.backends.cuda.enable_mem_efficient_sdp,
                                  torch.backends.cuda.enable_math_sdp,torch.backends.cuda.enable_cudnn_sdp),sdp):setter(value)
    assert all(torch.equal(expert.state_dict()[k].cpu(),v) for k,v in saved['states']['BASE'].items())
    expected=1 if mechanical else 66
    assert counts['forward_calls']==25*expected+6 and counts['backward_passes']==8*expected
    assert counts['parent_response_checks']==2*expected
    return dict(status='COMPLETE',expert_order=task['order'],mechanical_only=mechanical,
        rows=rows,counts=counts,state_restored_exact=True,
        original_dtypes_restored=all(t.dtype==dtype for t,dtype in dtypes),
        float32_scope='MATH_FP16/MATH_FP32 share math SDPA and disabled TF32; same FP16 weight values and frozen embeddings; native FP16 has no JVP',
        original_TF32=list(tf32),original_SDP=list(sdp),
        zero_new_optimization=True)


def mechanical():
    require_memory(2)
    with p.lease(2):
        runtime,_=c.load(2);result=measure(runtime,q.d.selected()[0],True)
        c.write(RUN/'private/MECHANICAL.json',result)
    p.done('MECHANICAL_COMPLETE')


def worker():
    assert (RUN/'private/MECHANICAL_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];require_memory(gpu)
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for task in q.d.selected()[part::len(q.GPUS)]:
            destination=RUN/'private/results'/f"{task['order']}.json"
            assert not destination.exists(),'No implicit repeat'
            c.write(destination,measure(runtime,task))
    p.done('WORKER_'+str(part))


def controller():
    import pipeline
    assert all(package(t).is_file() for t in q.d.selected()),'Recovery must reuse original candidates'
    pipeline.wait([pipeline.launch('calibration_probe.py','calibration_mechanical',2)])
    pipeline.wait([pipeline.launch('calibration_probe.py','calibration_worker',gpu,i) for i,gpu in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('calibration_report.py','calibration_report')])


if __name__=='__main__':
    try:
        {'calibration_plan':plan,'calibration_prepare':prepare_worker,'calibration_mechanical':mechanical,
         'calibration_worker':worker,'calibration_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),implicit_retry=False))
        raise
