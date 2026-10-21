"""One sequential native Base pass; each source window is retired before the next."""
import os,time,traceback,math
import torch
from runtime import R,c,b,load_learner,track,backward,lease,peak_memory,WINDOW,CONTROL
from core import Memory
from access import install
from teacher_batch import teacher_batch

def main():
    with lease():
        assert b.available(2);reserve=torch.empty(43*1024**3,dtype=torch.uint8,device='cuda');del reserve
        torch.cuda.empty_cache=lambda:None
        model=load_learner();hooks=track(model)
        try:
            install(R,WINDOW,os.environ['DATA_ROOT']+'/knowledge_editing/data/m3bench')
            weight=model.get_module(c.LAYER).weight;weight.requires_grad_(True);memory=Memory(weight,'N');normalized=Memory(weight,'N');margin_diagnostics=[]
            for index in range(1,9):
                c.write(CONTROL/'NEED_SOURCE.json',dict(index=index))
                from runtime import wait
                wait(WINDOW/'current.json');packet=c.read(WINDOW/'current.json');assert packet['index']==index
                row=packet['row'];expected=packet['expected'];key='CALIBRATION_'+str(index);query=dict(row,opaque_Base_id=key)
                output=b.generate(model,{key:expected},query,R/'private/calibration_outputs'/(str(index)+'.json'),dict(stage='NATIVE_BASE_CALIBRATION',index=index))
                assert output['R0']==packet['historical_native'],'Fresh native Base output mismatch; no substitution/rejudge'
                tokens=output['R0']['raw_token_ids'];batch=teacher_batch(model,row,tokens,expected)
                gen=torch.Generator(device=model.device);gen.manual_seed(20260912+200000+index)
                rows=torch.zeros((2,weight.numel()),dtype=torch.float64);normal_rows=torch.zeros_like(rows)
                for j in range(2):
                    z,y=b.logits(model,batch);assert y.tolist()==tokens
                    p=z.detach().softmax(-1);top=z.detach().double().softmax(-1).topk(2,dim=-1).values;mu=top[:,0]-top[:,1]
                    assert torch.isfinite(mu).all() and bool((mu>0).all()),'BASE_DECISION_MARGIN_NOT_ADMITTED'
                    if j==0:margin_diagnostics.append(dict(source_position=index,tokens=len(y),minimum_gap=float(mu.min()),maximum_gap=float(mu.max())))
                    root=p.sqrt();noise=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=gen)
                    u=root*noise-p*(root*noise).sum(-1,keepdim=True)
                    scale=math.sqrt(1/(8*len(y)))/math.sqrt(2)
                    rows[j]=backward((z*u).sum()*scale,weight).flatten().double().cpu()
                    zn,yn=b.logits(model,batch);assert torch.equal(yn,y) and torch.equal(zn.detach(),z.detach())
                    normal_rows[j]=backward((zn.double()*(u/mu[:,None])).sum()*scale,weight).flatten().double().cpu()
                    assert torch.isfinite(rows[j]).all() and torch.isfinite(normal_rows[j]).all(),'BASE_NATIVE_VJP_NOT_FINITE'
                memory.consolidate(rows,'Base');normalized.consolidate(normal_rows,'Base')
                del rows,normal_rows,batch,packet,row,expected,query,output,tokens,z,zn,yn,y,p,root,noise,u,mu,top
                c.write(CONTROL/(str(index)+'.DONE.json'),dict(index=index,source_buffers_retired=True))
                wait(CONTROL/(str(index)+'.RETIRED.json'))
            weight.requires_grad_(False);b.save(R/'private/calibration/Base_plain.pt',memory.state());b.save(R/'private/calibration/Base_normalized.pt',normalized.state())
            peak_memory();c.write(R/'public/BASE_CALIBRATION.json',dict(status='PASS',native_outputs_exact=8,rows_per_factor=16,backwards=32,margin_diagnostics=margin_diagnostics,no_margin_floor=True,one_sequential_pass=True,current_source_buffers_retired=True,no_teacher_replay=True))
        finally:hooks.close();b.counts()
    c.write(R/'private/EXIT_calibration.json',dict(exit_code=0,epoch=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        b.counts();c.write(R/'private/failures'/('calibration_'+str(os.getpid())+'.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));raise
