"""Fixed 32-probe full-vocabulary local KL metric; same bounded candidate."""
import os,time,traceback
import torch
import bounded_probe as prior
import fisher_math
q,c,p,RUN=prior.q,prior.c,prior.p,prior.RUN
PROBES=32
q.DIAGNOSE_BASIS=True


def basis_gradients(runtime,hook,basis,task,params):
    rows=[];maximum_centering=0.;tokens=0
    parameters=[params[k] for k in q.KEYS]
    for index,row in enumerate(basis):
        c.budget();batch=q.protection_batch(runtime,row,task);hook.set_teacher_routing(batch.labels)
        out=runtime.model(**batch.forward_kwargs())
        mask=batch.labels[:,1:]!=-100;logits=out.logits[:,:-1][mask].float()
        n=len(logits);tokens+=n;assert n==len(batch.target_token_ids)
        probability=logits.detach().softmax(-1)
        generator=torch.Generator(device=logits.device).manual_seed(20261009+index)
        for sample in range(PROBES):
            signs=torch.randint(0,2,logits.shape,device=logits.device,dtype=torch.int8,generator=generator).float()*2-1
            vector=fisher_math.categorical_probe(probability,signs)
            centering=float(vector.sum(-1).abs().max());maximum_centering=max(maximum_centering,centering)
            assert centering<=2e-6
            value=(logits*vector).sum()/(len(basis)*PROBES*n)**.5
            gradient=torch.autograd.grad(value,parameters,retain_graph=sample+1<PROBES)
            rows.append(torch.cat([g.detach().cpu().double().flatten() for g in gradient]))
        del batch,out,logits,probability,signs,vector,value,gradient
        if (index+1)%12==0:print('FISHER_BASIS',task['order'],index+1,len(rows),flush=True)
    assert tokens==1514 and len(rows)==1952
    matrix=torch.stack(rows);assert torch.isfinite(matrix).all()
    return matrix,1952,dict(rows=1952,questions=61,tokens=1514,probes_per_question=32,
        probe_seed_rule='20261009+original_BASIS_question_index',probe_resampling=False,
        maximum_centering_error=maximum_centering,mean_gradient_checks=0,
        full_vocabulary=True,metric='FINITE_FISHER_SKETCH_NOT_EXACT_KL',
        zero_rows=int((matrix.norm(dim=1)==0).sum()))


q.basis_gradients=basis_gradients
prior.metric_rows=lambda matrix:matrix[:,:512]
original_direction=prior.candidate_direction


def candidate_direction(*args):
    value,audit,forwards,backwards=original_direction(*args)
    audit.update(algorithm='BOUNDED_FULL_VOCABULARY_FISHER_SKETCH',
        proxy_is_KL_curvature_estimate=True,probes_per_question=32)
    return value,audit,forwards,backwards


q.candidate_direction=candidate_direction


def plan():
    prior.plan();fisher_math.selfcheck()
    lock=c.read(RUN/'private/PROBE_LOCK.json')
    lock.pop('basis_token_constraints',None)
    lock.update(algorithm='BOUNDED_FISHER_RESPONSE',forward_calls=6544,backward_calls=15672,mean_gradient_parity_checks=0,
        basis_fisher_rows=1952,probes_per_question=32,source_response_tokens=1514,
        target='FULL_VOCABULARY_BASE_KL_LOCAL_FISHER',response_metric='FIXED_RADEMACHER_FISHER_SKETCH')
    c.write(RUN/'private/PROBE_LOCK.json',lock)
    a={k:v for k,v in lock.items() if k not in ('role_binding','code')}
    a.update(status='PASS',fisher_selfcheck=fisher_math.selfcheck(),constrained_solver_selfcheck=prior.math.selfcheck())
    c.write(RUN/'public/ADMISSION.json',a)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('fisher_probe.py','fisher_worker',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('fisher_report.py','fisher_report')])


if __name__=='__main__':
    try:{'fisher_plan':plan,'fisher_worker':q.worker,'fisher_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
