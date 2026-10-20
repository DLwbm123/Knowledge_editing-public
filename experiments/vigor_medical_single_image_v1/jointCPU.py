"""One frozen synthetic CPU qualification; never loads project data or a VLM."""
import inspect
import json
import subprocess
import time
from pathlib import Path


def qualify():
    import math
    import resource
    import signal
    import time
    from contextlib import contextmanager
    import torch
    import torch.nn.functional as F
    started = time.time(); torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU only')
    counts = dict(forward=0, gradient_paths=0, backward_api=0)
    checks = []; arms = []
    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:raise ValueError(name)
    def forward(weight, x, y, gradient=False, return_logits=False):
        counts['forward'] += 1; counts['gradient_paths'] += int(gradient)
        if counts['forward'] > 800 or counts['gradient_paths'] > 600:raise RuntimeError('Model batch budget')
        logits=F.linear(x.to(weight.dtype), weight)
        loss=F.cross_entropy(logits.float(), y)
        return (loss,logits) if return_logits else loss
    def backward(loss):
        counts['backward_api'] += 1
        if counts['backward_api'] > 800:raise RuntimeError('Backward API budget')
        loss.backward()
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(3,4,dtype=torch.float16))
    model=Base(); w0=model.weight.detach().clone(); x=torch.eye(4)[:2]; y=torch.tensor([1,2]); xn=torch.eye(4)[2:3]; yn=torch.tensor([0])
    original=(model.training,model.weight.requires_grad,model.weight.dtype)
    def restored():
        return torch.equal(model.weight,w0) and (model.training,model.weight.requires_grad,model.weight.dtype)==original and model.weight.grad is None and not model._forward_hooks and not model._forward_pre_hooks
    @contextmanager
    def installed(weight):
        if not torch.isfinite(weight).all():raise ValueError('nonfinite write')
        try:
            with torch.no_grad():model.weight.copy_(weight.to(torch.float16))
            yield model.weight
        finally:
            with torch.no_grad():model.weight.copy_(w0)
    with torch.no_grad():lv0=float(forward(w0,x,y));ln0=float(forward(w0,xn,yn))
    gweight=w0.double().requires_grad_(); loss=forward(gweight,x.double(),y,True); backward(loss)
    g=gweight.grad.detach().double();u,s,vh=torch.linalg.svd(g,full_matrices=False);check('source_gradient_rank2',int((s>1e-10*s[0]).sum())==2)
    u=u[:,:2];v=vh[:2].T
    for j in range(2):
        if u[torch.argmax(u[:,j].abs()),j] < 0:u[:,j]*=-1;v[:,j]*=-1
    check('orthogonal_basis',float((u.T@u-torch.eye(2,dtype=torch.float64)).norm())<1e-8 and float((v.T@v-torch.eye(2,dtype=torch.float64)).norm())<1e-8)
    check('gradient_reconstruction',float((u@torch.diag(s[:2])@v.T-g).norm())<1e-8)
    a=torch.tensor([[.1,-.1],[.2,.1]],dtype=torch.float64,requires_grad=True)
    # Use double logits directly for the derivative fixture, avoiding an intentional float-logit cast.
    counts['forward']+=1;counts['gradient_paths']+=1
    z=F.cross_entropy(F.linear(x.double(),w0.double()+u@a@v.T),y);counts['backward_api']+=1
    ag=torch.autograd.grad(z,a)[0];fd=torch.empty_like(a)
    for i in range(2):
        for j in range(2):
            delta=torch.zeros_like(a);delta[i,j]=1e-6
            vals=[]
            for sign in [-1,1]:
                counts['forward']+=1;vals.append(float(F.cross_entropy(F.linear(x.double(),w0.double()+u@(a.detach()+sign*delta)@v.T),y)))
            fd[i,j]=(vals[1]-vals[0])/2e-6
    error=float((ag-fd).norm()/fd.norm());check('coefficient_autograd_finite_difference',error<=1e-5 and float(ag.norm())>0)
    u=u.float();v=v.float()
    def matrix(coef):return w0.float()+u@coef@v.T
    def pair_state(weights):
        return not torch.equal(*weights),float((weights[0].float()-weights[1].float()).norm())
    def gates(native, null_input=xn, null_labels=yn):
        with installed(native) as weight:
            with torch.no_grad():
                visual_loss,logits=forward(weight,x,y,return_logits=True)
                visual=float(visual_loss);null=float(forward(weight,null_input,null_labels));prediction=logits.argmax(-1)
                cost=float(.5e-4*(weight.float()-w0.float()).square().sum())
        return dict(visual=visual,null=null,drift=null-ln0,cost=cost,source_correct=bool(torch.equal(prediction,y)),
                    visual_gate=visual<lv0,null_gate=abs(null-ln0)<=.01,trust_gate=cost<=.001)
    for coupled in [False,True]:
        params=[torch.nn.Parameter(torch.zeros(2,2)),torch.nn.Parameter(torch.zeros(2,2))]
        with torch.no_grad():params[0][0,0]=.05;params[1][0,0]=-.05
        optimizer=torch.optim.Adam(params,lr=.02,betas=(.9,.999),eps=1e-8,weight_decay=0)
        trajectory=[]
        for step in range(64):
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for coef in params:
                visual=forward(matrix(coef),x,y,True);losses.append(float(visual.detach()))
                backward(visual+10*torch.relu(visual-lv0+1e-5).square())
                null=forward(matrix(coef),xn,yn,True)
                backward(10*torch.relu((null-ln0).abs()-.01).square())
            penalty=sum(10*torch.relu(.5e-4*(matrix(coef)-w0.float()).square().sum()-.001).square() for coef in params)
            if coupled:penalty=penalty+10*torch.relu(.25**2-(params[0]-params[1]).square().sum()).square()
            backward(penalty);optimizer.step()
            check('finite_coefficients',all(torch.isfinite(coef).all() for coef in params))
            trajectory.append(dict(step=step+1,source_before=losses,coefficient_distance=float((params[0]-params[1]).detach().norm())))
        check('64_steps_and_optimizer_isolation',all(int(optimizer.state[p]['step'])==64 for p in params) and optimizer.state[params[0]]['exp_avg'].data_ptr()!=optimizer.state[params[1]]['exp_avg'].data_ptr())
        native=[matrix(coef).detach().half() for coef in params];endpoints=[gates(w) for w in native]
        distinct,distance=pair_state(native)
        passes=all(all(row[k] for k in ['source_correct','visual_gate','null_gate','trust_gate']) for row in endpoints)
        arms.append(dict(coupled=coupled,steps=64,endpoints=endpoints,native_distinct=distinct,native_distance=distance,
                         diversity_pass=distance>=.25,endpoint_pair_pass=passes and distinct and (not coupled or distance>=.25),trajectory=trajectory))
        optimizer.zero_grad(set_to_none=True);check('base_after_arm',restored())
    # The registered 1e-8 case uses FP64 before casting, so the continuous difference exists.
    check('registered_1e8_collapse',not torch.equal(torch.ones(3,4,dtype=torch.float64),torch.ones(3,4,dtype=torch.float64)+1e-8) and torch.equal(torch.ones(3,4,dtype=torch.float64).half(),(torch.ones(3,4,dtype=torch.float64)+1e-8).half()))
    check('exact_native_duplicate_rejected',pair_state([native[0],native[0].clone()])==(False,0.))
    stressed=gates(native[0],x,y);check('same_input_stress_null_reject',stressed['null_gate'] is False)
    try:
        with installed(torch.ones(3,4,dtype=torch.float16)):raise RuntimeError('injected')
    except RuntimeError:pass
    check('exception_full_restore',restored())
    try:
        with installed(torch.full((3,4),float('nan'))):pass
    except ValueError:pass
    else:raise AssertionError('nonfinite must fail')
    check('normal_and_invalid_full_restore',restored())
    return dict(status='QUALIFIED_SYNTHETIC_ONLY' if arms[1]['endpoint_pair_pass'] and all(c['passed'] for c in checks) else 'NOT_QUALIFIED',
        checks=checks,arms=arms,finite_difference_relative_error=error,counters=counts,
        seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        synthetic_separable_fixture=True,null_sensitive_real_VLM_validation=False,
        VLM=0,GPU=0,Judge=0,source_permission_resolved=False,medical_protection_claim=False,base_restored=restored())


def main():
    root=Path(__file__).resolve().parents[2];out=root/'reports/vigor_jointCPU_v1_20261020';freeze=json.loads((out/'FREEZE.json').read_text())
    if time.time()-freeze['epoch']>3600:raise TimeoutError('Original clock')
    if (out/'RESULT.json').exists() or (out/'LAUNCH_RESERVED.json').exists():raise FileExistsError('No replay')
    (out/'LAUNCH_RESERVED.json').write_text(json.dumps(dict(epoch=time.time(),CPU_only=True))+'\n')
    code=inspect.getsource(qualify)+'\nprint(json.dumps(qualify()))'
    wrapper="import sys,json,signal;signal.signal(signal.SIGALRM,lambda *a: (_ for _ in ()).throw(TimeoutError('CPU hard guard')));signal.alarm(45);exec(json.loads(sys.stdin.readline()))"
    interpreter=json.loads((root/'.local_run/jointCPU_context.json').read_text())['python']
    command="env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "+__import__('shlex').quote(interpreter)+" -c "+__import__('shlex').quote(wrapper)
    run=subprocess.run(['ssh','-o','BatchMode=yes','pro5000',command],input=json.dumps('import json\n'+code)+'\n',capture_output=True,text=True,timeout=60)
    if run.returncode:
        (root/'.local_run/jointCPU_failure.txt').write_text(run.stderr)
        (out/'FAILURE.json').write_text(json.dumps(dict(status='FAILED_PRESERVED',returncode=run.returncode,VLM=0,GPU=0,Judge=0))+'\n')
        raise RuntimeError('CPU qualification failed, private failure retained; no retry')
    value=json.loads(run.stdout);(out/'RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','finite_difference_relative_error','counters','seconds','VLM','GPU','Judge']}))
    print(json.dumps([dict(coupled=a['coupled'],pair=a['endpoint_pair_pass'],distance=a['native_distance'],endpoints=a['endpoints']) for a in value['arms']]))


if __name__=='__main__':main()
