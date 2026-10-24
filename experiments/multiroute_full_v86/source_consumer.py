"""Registered fixed-source comparison; no method training or free generation."""
import os,time,traceback
from pathlib import Path
import v81 as v


def main():
    consumed=v.read(v.ROOT/'MAIN_COMPLETE.json')['counts'];v.allocations={k:v.limits['counts'][k]-consumed[k] for k in consumed}
    v.setup();torch=v.torch;r=v.r;dpc=Path(os.environ['DPC_ROOT'])
    assert v.read(dpc/'MAIN_COMPLETE.json')['outputs']==620
    tasks=v.read(v.ROOT/'TASKS.json');records=[]
    for task in tasks:
        owner=task['order'];items=v.prepare(task,v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)],True)
        batches=[r.old.batch_on(v.runtime,x[1]) for x in items];local=[]
        with v.cache_class(v.runtime,batches):
            gradients={}
            for arm in ('S0','FREE_FROB','DPC'):
                path=(v.ROOT/'candidates/S0'/f'{owner}.pt') if arm=='S0' else (dpc/'candidates'/arm/f'{owner}.pt')
                state=r.mm.canonicalize(r.p.load_state(path)['state'])[0];expert=r.p.expert(state,task['seed'],v.runtime.device)
                hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach();rows=[]
                try:
                    for view,b in enumerate(batches):
                        expert.zero_grad(set_to_none=True);hook.set_teacher_routing(b.labels)
                        loss=v.runtime.compute_loss(b);assert torch.isfinite(loss);loss.backward();v.counts['backwards']+=1
                        vector=torch.cat([p.grad.detach().flatten() for _,p in expert.named_parameters()]).double().cpu()
                        assert len(vector)==7168 and torch.isfinite(vector).all();rows.append(vector)
                        local.append(dict(owner=owner,view=view,arm=arm,source_CE=float(loss.detach()),TT_gradient_norm=float(vector.norm())))
                finally:hook.detach()
                gradients[arm]=rows;del expert
            cosines={}
            for arm in ('FREE_FROB','DPC'):
                cosines[arm]=[float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)) for a,b in zip(gradients['S0'],gradients[arm])]
            records.append(dict(owner=owner,source_rows=local,S0_gradient_cosines=cosines))
        v.progress(owner=owner,stage='REGISTERED_SOURCE_CONSUMER')
    assert len(records)==8 and v.counts['backwards']==120 and v.counts['updates']==v.counts['evaluation_generations']==v.counts['teacher_generations']==0
    v.write(v.ROOT/'SOURCE_CONSUMER_COMPLETE.json',dict(status='PASS',source_inputs=40,arms=['S0','FREE_FROB','DPC'],source_NN_backwards=120,
        no_training=True,no_free_generation=True,counts=dict(v.counts),time=time.time(),records=records,
        interpretation='Reference-trajectory CE and Canonicalized TT-coordinate gradients (floating-point recanonicalization may change rounding); not output-Jacobian equality, not generalization or clinical protection.'))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;v.write(v.ROOT/'FAILURE_source_compare_0.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        v.write(v.ROOT/'STOP',dict(reason='SOURCE_CONSUMER_FAILURE',time=time.time()));raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/'EXIT_source_compare_0.json',dict(exit_code=code,counts=v.counts,time=time.time()))
        v.write(v.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
