"""All actual response-token constraints; fixed diagnostic, no rank tuning."""
import os
import time
import traceback
import torch
import torch.nn.functional as F
import qualified_probe as prior
q, c, p, RUN = prior.q, prior.c, prior.p, prior.RUN


def token_losses(logits, labels):
    mask = labels[:,1:] != -100
    return F.cross_entropy(logits[:,:-1][mask].float(),labels[:,1:][mask],reduction='none')


def selfcheck():
    logits=torch.tensor([[[2.,-1.],[0.,1.],[3.,-2.]]],requires_grad=True)
    labels=torch.tensor([[-100,0,1]])
    losses=token_losses(logits,labels)
    expected=F.cross_entropy(logits[:,:-1].reshape(-1,2),labels[:,1:].reshape(-1))
    grads=[torch.autograd.grad(v,logits,retain_graph=True)[0] for v in losses]
    mean_grad=torch.autograd.grad(expected,logits)[0]
    assert torch.allclose(losses.mean(),expected) and torch.allclose(torch.stack(grads).mean(0),mean_grad)


def basis_gradients(runtime, hook, basis, task, params):
    rows=[];parity=[]
    parameters=[params[k] for k in q.KEYS]
    for i,row in enumerate(basis):
        c.budget();batch=q.protection_batch(runtime,row,task);hook.set_teacher_routing(batch.labels)
        out=runtime.model(**batch.forward_kwargs());losses=token_losses(out.logits,batch.labels)
        assert len(losses)==len(batch.target_token_ids)
        assert torch.allclose(losses.mean(),out.loss.float(),atol=1e-6,rtol=1e-5)
        start=len(rows)
        for loss in losses:
            grad=torch.autograd.grad(loss,parameters,retain_graph=True)
            rows.append(torch.cat([g.detach().cpu().double().flatten() for g in grad]))
        original=torch.autograd.grad(out.loss,parameters)
        original=torch.cat([g.detach().cpu().double().flatten() for g in original])
        average=torch.stack(rows[start:]).mean(0)
        error=float((average-original).norm()/(original.norm()+1e-30))
        assert error<=1e-3, ('Mean token gradient does not match sequence gradient',error)
        parity.append(error)
        del batch,out,losses,original,average,grad,loss
        if (i+1)%12==0:print('TOKEN_BASIS',task['order'],i+1,len(rows),flush=True)
    assert len(rows)==1514
    matrix=torch.stack(rows);assert torch.isfinite(matrix).all()
    return matrix,1514+61,dict(rows=1514,questions=61,mean_gradient_checks=61,
        maximum_mean_gradient_relative_error=max(parity),all_actual_tokens_including_EOS=True)


q.basis_gradients=basis_gradients


def plan():
    selfcheck();prior.plan()
    basis,_=prior.split()
    total=sum(len(c.read(row['response_path'])['R0']['raw_token_ids']) for row in basis)
    assert total==1514
    lock=c.read(RUN/'private/PROBE_LOCK.json')
    lock.update(basis_token_constraints=1514,backward_calls=12616,
        mean_gradient_parity_checks=488,target='EVERY_ACTUAL_QUALIFIED_BASE_RESPONSE_TOKEN_NLL')
    c.write(RUN/'private/PROBE_LOCK.json',lock)
    a=c.read(RUN/'public/ADMISSION.json');a.update({k:v for k,v in lock.items() if k not in ('role_binding','code')})
    c.write(RUN/'public/ADMISSION.json',a)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('token_probe.py','token_worker',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('token_report.py','token_report')])


if __name__=='__main__':
    try:{'token_plan':plan,'token_worker':q.worker,'token_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
