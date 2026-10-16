"""Disjoint bank partitions on three GPUs, preserving the running single lane."""
import os
from pathlib import Path
import time
import traceback
import cm_runtime146 as w

r,c,RUN,protocol=w.r,w.c,w.RUN,w.protocol


def assigned(rows, part):
    return [v for i,v in enumerate(rows) if i%3==part and v['mode'] in ('bank','insertion')]


def parity(runtime, bindings):
    import torch
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    import bank146
    gpu=int(os.environ['GPU']);part=int(os.environ['PARTITION'])
    rows=c.read(RUN/'private/CM_PARALLEL_FIXTURES1.json')
    byorder={t['order']:t for t in bank146.tasks()}
    for row in rows:
        task=byorder[row['owner']];query=dict(row['query'],audit_role=row['query'].get('audit_role','EVAL'))
        expected=c.read(w.worker.dest(row))
        runtime.model.half()
        saved=r.p.load_state(Path(c.read(RUN/'private/ROUTER_BINDINGS146.json')[str(task['order'])]))
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=saved['entries']),device=runtime.device)
        layer=runtime.target_lock['balancedit']['targets'][0]
        batch=runtime.build_question_batch(r.s.record(task,query))
        with torch.inference_mode():
            value=runtime.extract_layer_input_key(batch,module_path=layer,pooling='mean')
        route=decision_as_json(router.route(value))
        assert route==expected['route']
        item=r.s.prepare(runtime,bindings,task,[query])[0]
        r.old.fp32(runtime);hook=None
        if route['activated']:
            state=r.p.load_state(w.worker.point(task,row['method']))['states']['FINAL']
            expert=r.p.expert(state,task['seed'],runtime.device).requires_grad_(False)
            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
        out=RUN/f"private/parallel_parity/{gpu}/{row['method']}.json"
        assert not out.exists()
        try:r.s.emit(runtime,hook,task,row['method'],item,r.COUNTS,dest=out,route=route)
        finally:
            if hook is not None:hook.detach()
        actual=c.read(out)
        assert actual['R0']==expected['R0'] and actual['binding']==expected['binding']
    assert r.COUNTS['generation_attempts']==2 and r.COUNTS['forwards']<=4096
    c.write(RUN/f'private/CM_PARALLEL_ENGINEERING_COUNTS_{part}.json',dict(r.COUNTS))
    for key in r.COUNTS:r.COUNTS[key]=0
    r.save_counts();runtime.model.half()
    c.write(RUN/f'private/CM_PARALLEL_PARITY_{part}.json',dict(status='PASS',GPU=gpu,exact_tokens_binding_route=True,epoch=time.time()))
    while not all((RUN/f'private/CM_PARALLEL_PARITY_{p}.json').exists() for p in range(3)):
        assert not list((RUN/'private').glob('FAILURE_cm_parallel*.json'))
        protocol.project_budget(RUN);time.sleep(2)


def main():
    gate=protocol.admit(RUN,check_weights=True);part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU'])
    assert part in (0,1,2) and gpu==part+3
    assert c.read(RUN/'private/CM_PARALLEL_CPU_CHECK1.json')['status']=='PASS'
    with (RUN/f'private/CM_PARALLEL_GPU_ONCE_{part}').open('x'):pass
    rows=c.read(RUN/'private/CM_NEW_PLAN1.json');jobs=assigned(rows,part)
    assert all(not w.worker.dest(row).exists() for row in jobs)
    parent=Path(gate['parent'])
    w.worker.point=lambda task,arm:parent/f"private/candidates/{arm}/{task['order']}.pt"
    w.worker.configure();r.NATIVE_FORWARD_CAP=protocol.NON_GENERATION_FORWARDS
    original=r.count
    def count(*args):
        original(*args);protocol.project_budget(RUN)
        assert r.COUNTS['generation_attempts']<=len(jobs)
        assert r.COUNTS['updates']==r.COUNTS['backwards']==r.COUNTS['operator_backwards']==r.COUNTS['local_solves']==0
    r.count=count
    import bank146
    bank146.evaluate(292,3,lambda:protocol.admit(RUN),'CM_FROZEN1.json','CM_SCOPE1.json','CM_SOURCE1.json','CM_NEW_PLAN1.json',parity)
    assert r.COUNTS['generation_attempts']==len(jobs)
    c.write(RUN/f'private/CM_PARALLEL_COMPLETE_{part}.json',dict(status='COMPLETE',outputs=len(jobs),epoch=time.time()))


if __name__=='__main__':
    code=0;part=os.environ['PARTITION']
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/f'private/FAILURE_cm_parallel_{part}.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),automatic_retry=False));raise
    finally:
        r.save_counts();c.write(RUN/f'private/CM_PARALLEL_EXIT_{part}.json',dict(exit_code=code,epoch=time.time()))
