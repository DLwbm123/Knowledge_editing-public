"""Read-only precision diagnosis; no native qualification or edit is implied."""
from __future__ import annotations
import json,resource,signal,time,traceback
from pathlib import Path
import torch
from .gate import require_external_approval
from .native_io import load,prepare,WEIGHT
from .editor import MatrixRuntime,mean_answer_logprob
from .numerics import FrozenGGN,cg


def diagnose(config,approval,trusted):
    require_external_approval('NATIVE_SMOKE',config['bindings'],approval,config,trusted_authorization=trusted)
    root=Path(config['run_root']);report=dict(result_kind='MECHANICAL_NATIVE_VALIDATION',native_smoke_status='NOT_QUALIFIED',
        precision_mode='FP32_FUNCTIONAL_ARITHMETIC_DIAGNOSTIC',deployment_dtype='torch.bfloat16',GGN_calls=0,
        candidate_writes=0,paid_judge_calls=0,real_pilot_started=False,sequential_started=False)
    rt=None;start=time.monotonic()
    def save():
        (root/'PRECISION_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def timeout(*_):raise TimeoutError('remaining original smoke budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(config['native_wall_seconds_cap']-30)
    torch.manual_seed(20260930);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        model,tokenizer,processor=load(config)
        rt=MatrixRuntime(model,WEIGHT)
        if list(rt.weight.shape)!=[4096,14336] or rt.weight.dtype!=torch.bfloat16:raise RuntimeError('binding mismatch')
        inputs,labels,_,_=prepare(model,tokenizer,processor,config['smoke_row'])
        mask=(labels[:,1:]!=-100)&(labels[:,1:]!=tokenizer.eos_token_id);positions=mask.nonzero()[:,1]
        score=lambda z:mean_answer_logprob(z.float(),labels,eos_id=tokenizer.eos_token_id).mean()
        with torch.no_grad():deployed=rt.normal_logits(inputs).float()
        base=rt.base.detach().requires_grad_(True)
        native_grad=torch.autograd.grad(score(rt.logits(base,inputs)),base)[0].detach()
        index=int(native_grad.abs().reshape(-1).argmax())
        raw=rt.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        point=rt.base.float();variable=point.detach().requires_grad_(True)
        smooth=raw(variable);a=torch.autograd.grad(score(smooth),variable)[0].detach()
        difference=(smooth.detach()-deployed).abs()
        report['deployment_comparison']=dict(max_logit_difference=float(difference.max()),
            mean_logit_difference=float(difference.mean()),argmax_equal_fraction=float((smooth.detach().argmax(-1)==deployed.argmax(-1)).float().mean()),
            answer_predictor_argmax_equal=bool(torch.equal(smooth.detach()[:,positions].argmax(-1),deployed[:,positions].argmax(-1))),
            frozen_max_logit_tolerance=.001,within_frozen_parity_tolerance=bool(difference.max()<=.001))
        report['FP32_gradient']=dict(norm=float(a.norm()),finite=bool(torch.isfinite(a).all()),dtype=str(a.dtype),index=index)
        epsilon=max(float(point.reshape(-1)[index].abs())/16,.001)
        direction=torch.zeros_like(point);direction.reshape(-1)[index]=1
        with torch.no_grad():
            plus=score(raw(point+epsilon*direction));minus=score(raw(point-epsilon*direction));fd=float((plus-minus)/(2*epsilon))
        analytic=float(a.reshape(-1)[index]);error=abs(fd-analytic)/max(abs(analytic),1e-8)
        report['finite_difference']=dict(epsilon=epsilon,analytic=analytic,central_difference=fd,relative_error=error,
            frozen_relative_tolerance=.2,passed=error<=.2,definition='smooth FP32 functional calculation, not the rounded BF16 deployment map')
        logits=lambda w:raw(w)[:,positions,:]
        F=FrozenGGN(logits,point,torch.full((1,len(positions)),1/len(positions),device=point.device))
        def matvec(v):
            if report['GGN_calls']>=config['maximum_GGN_calls']:raise TimeoutError('remaining GGN call budget exhausted')
            report['GGN_calls']+=1
            return F(v.reshape(point.shape)).reshape(v.shape)
        v=torch.zeros_like(point);v.reshape(-1)[(index+14336)%point.numel()]=1
        torch.cuda.synchronize();t=time.monotonic();fu=matvec(direction);torch.cuda.synchronize()
        seconds=time.monotonic()-t;fv=matvec(v);again=matvec(direction)
        uv=float((direction*fv).sum());vu=float((v*fu).sum());quad=float((direction*fu).sum())
        report['GGN']=dict(symmetry_absolute_error=abs(uv-vu),u_Fv=uv,v_Fu=vu,PSD_quadratic=quad,
            repeated_max_difference=float((fu-again).abs().max()),GGN_norm=float(fu.norm()),seconds_per_matvec=seconds,
            symmetry_pass=abs(uv-vu)<=1e-5*max(1,abs(uv),abs(vu)),PSD_pass=quad>=-1e-6,
            finite=bool(torch.isfinite(fu).all() and torch.isfinite(fv).all()),approximation_used=False)
        solve=cg(lambda x:matvec(x)+config['damping']*x,a,rtol=1e-4,max_iter=4)
        report['CG']=dict(status=solve.status,relative_residual=solve.relative_residual,iterations=solve.iterations,calls=solve.calls,
            recursive_relative_residuals=solve.recursive_relative_residuals,condition_surrogate=solve.condition_surrogate,
            frozen_rtol=1e-4,frozen_max_iter=4,damping=config['damping'])
        rt.audit(rt.base_state,rt.hooks())
        report['deployment_state_unchanged']=torch.equal(rt.weight,rt.base) and rt.weight.dtype==torch.bfloat16
        report['status']='DIAGNOSTIC_COMPLETE'
    except BaseException as error:
        report.update(status='DIAGNOSTIC_FAILED',failure=dict(type=type(error).__name__,reason=str(error)))
        (root/'FAILURE.private.txt').write_text(traceback.format_exc())
    finally:
        if rt is not None:report['final_Base_W_bitwise_equal']=torch.equal(rt.weight,rt.base)
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save()
    return report
