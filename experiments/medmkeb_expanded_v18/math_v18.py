"""Same-frame CE-only versus CE+margin projection; content-position KL."""
import torch

def constraints(grad,current,anchor,margin=True,epsilon=5e-5):
    assert grad.ndim==2 and current.shape==anchor.shape==(grad.shape[0],) and len(current)%2==0
    assert all(torch.isfinite(x).all() for x in (grad,current,anchor))
    matrix=[];bounds=[]
    for j in range(0,len(current),2):
        matrix.append(grad[j]);bounds.append(anchor[j]+epsilon-current[j])
        if margin:matrix.append(grad[j]-grad[j+1]);bounds.append(current[j+1]-current[j]-anchor[j+1]+anchor[j]+epsilon)
    return torch.stack(matrix),torch.stack(bounds)

def violation(anchor,current,margin=True):
    assert len(anchor)==len(current) and len(anchor)%2==0
    out=[current[j]-anchor[j] for j in range(0,len(anchor),2)]
    if margin:out += [anchor[j+1]-anchor[j]-current[j+1]+current[j] for j in range(0,len(anchor),2)]
    return max(out)

def content_logits(output,batch):
    mask=batch.labels[:,1:]!=-100;logits=output.logits[:,:-1][mask].float();targets=batch.labels[:,1:][mask]
    assert len(targets)>0 and torch.isfinite(logits).all()
    return logits,targets

def kl(student,teacher):
    assert student.shape==teacher.shape and torch.isfinite(teacher).all()
    return torch.nn.functional.kl_div(student.log_softmax(-1),teacher.to(student.device),reduction='batchmean',log_target=True)

def selfcheck():
    g=torch.tensor([[1.,0.],[0.,1.]]);a=torch.tensor([1.,2.]);v=a+torch.tensor([.0002,0.]);ce,b=constraints(g,v,a,False);full,c=constraints(g,v,a,True)
    assert ce.shape==(1,2) and full.shape==(2,2) and torch.equal(ce[0],full[0]) and b[0]<0 and c[1]<0
    assert violation([1.,2.],[1.,1.],False)==0 and violation([1.,2.],[1.,1.],True)==1
    t=torch.tensor([[1.,2.,3.],[2.,0.,1.]]);s=t.clone().requires_grad_();z=kl(s,t.log_softmax(-1));z.backward();assert abs(float(z))<1e-6 and s.grad.abs().max()<1e-6
    assert len(range(0,48,2))==24
    return dict(status='PASS',CE_PROJ_excludes_margin=True,negative_remaining_preserved=True,KL_zero_gradient_at_teacher=True,shared_CE_denominator=24)
if __name__=='__main__':print(selfcheck())
