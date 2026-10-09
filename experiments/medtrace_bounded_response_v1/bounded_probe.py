"""A finite constrained-response candidate, not a protected replay training loss."""
import os,time,traceback
import torch
import token_probe as token
import bounded_math as math
q,c,p,RUN=token.q,token.c,token.p,token.RUN
q.CANDIDATE_ARM='BOUNDED'


def candidate_direction(runtime,hook,expert,native,before,raw,matrix):
    assert torch.count_nonzero(before['G1'])==0
    assert all(torch.equal(before[k],raw[k]) for k in ('G2','G3','G4'))
    assert torch.count_nonzero(matrix[:,512:])==0
    expert.load_state_dict(before)
    params=[dict(expert.named_parameters())[k] for k in q.KEYS]
    gradients=[]
    for batch in native:
        hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch)
        grad=torch.autograd.grad(loss,params)
        gradients.append(torch.cat([g.detach().cpu().double().flatten() for g in grad]))
    g=.5*gradients[0]+.125*sum(gradients[1:])
    assert torch.count_nonzero(g[512:])==0
    factor=math.metric_factor(before);inverse=torch.linalg.inv(factor)
    raw_delta=q.flatten(raw)-q.flatten(before)
    raw_white=(raw_delta[:512].reshape(64,8)@factor).flatten()
    radius=float(raw_white.norm());actual_radius=q.fm.map_norm(before,raw)
    assert abs(radius-actual_radius)<=1e-10+1e-6*radius
    gain=-(g[:512].reshape(64,8)@inverse.T).flatten()
    gain_norm=float(gain.norm());gain=gain/gain_norm
    raw_progress=float(gain@raw_white)*gain_norm
    assert raw_progress>1e-8, 'No positive RAW linear editing progress'
    rows=matrix[:,:512].clone();offset=0
    basis,_=token.prior.split()
    for row in basis:
        n=len(c.read(row['response_path'])['R0']['raw_token_ids'])
        rows[offset:offset+n]/=(len(basis)*n)**.5;offset+=n
    assert offset==1514
    white_rows=(rows.reshape(1514,64,8)@inverse.T).reshape(1514,512)
    required=.9*raw_progress/(gain_norm*radius)
    value,audit=math.solve(white_rows*radius,gain,required)
    delta=torch.zeros_like(raw_delta)
    delta[:512]=(value.reshape(64,8)*radius@inverse).flatten()
    state=q.add(before,delta);actual=q.flatten(state)-q.flatten(before)
    actual_progress=float(-g@actual);actual_norm=q.fm.map_norm(before,state)
    assert actual_progress>=.9*raw_progress*(1-1e-6)
    assert actual_norm<=actual_radius*(1+1e-6)+1e-10
    audit.update(algorithm='MINIMUM_UNNORMALIZED_TOKEN_RESPONSE_WITH_EDIT_GAIN_AND_MAP_NORM_CONSTRAINTS',
        raw_linear_edit_progress=raw_progress,actual_linear_edit_progress=actual_progress,
        RAW_map_norm=actual_radius,BOUNDED_map_norm=actual_norm,
        objective_RAW=float((rows@raw_delta[:512]).square().sum()),
        objective_BOUNDED=float((rows@actual[:512]).square().sum()),
        source_questions=61,source_tokens=1514,proxy_is_full_KL=False)
    return delta,audit,5,5


q.candidate_direction=candidate_direction


def plan():
    token.plan();math.selfcheck()
    import bounded_report
    bounded_report.selfcheck()
    lock=c.read(RUN/'private/PROBE_LOCK.json')
    lock.update(forward_calls=4592,backward_calls=13144,algorithm='BOUNDED_RESPONSE',
        edit_gain_fraction=.9,map_norm_cap='RAW',response_metric='QUESTION_BALANCED_RAW_TOKEN_JACOBIAN_SQUARED',
        arms=['RAW','BOUNDED','MATCHED_RAW'])
    c.write(RUN/'private/PROBE_LOCK.json',lock)
    a=c.read(RUN/'public/ADMISSION.json');a.update({k:v for k,v in lock.items() if k not in ('role_binding','code')})
    a['constrained_solver_selfcheck']=math.selfcheck();c.write(RUN/'public/ADMISSION.json',a)


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('bounded_probe.py','bounded_worker',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('bounded_report.py','bounded_report')])


if __name__=='__main__':
    try:{'bounded_plan':plan,'bounded_worker':q.worker,'bounded_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
