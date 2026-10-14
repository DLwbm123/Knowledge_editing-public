"""Serialize existing source/evaluation operators; retain exact boundary state and clocks."""
import os
from pathlib import Path
from contextlib import nullcontext
import time
import traceback
os.environ['PYTORCH_CUDA_ALLOC_CONF']='backend:native'
import torch
import runtime
from runtime import R, ARM, CONTROL, c, b


def build_source(source):
    begin=source.index('            if active.exists():')
    end=source.index('            for owner in range(start_owner,4):',begin)
    restore='''            requested=int(os.environ['SOURCE_OWNER'])
            assert 1<=requested<=3
            if active.exists():
                state=validate_state(checkpoint.read(active))
                assert state['boundary'] and state['owner']==requested-1
                assert (R/'private/units'/ARM/(str(requested-1)+'.json')).exists()
                with torch.no_grad():weight.copy_(state['native'].to(runtime.device))
                master=state['master'].to(runtime.device);risk.restore(state['risk'],runtime.device)
                c.restore_rng(state['rng']);n=state['numpy_rng']
                np.random.set_state((n[0],n[1].numpy().astype('uint32'),n[2],n[3],n[4]))
                completed=state['completed_slots'];del state
            else:assert requested==1
            start_owner=requested;resume=None
'''
    source=source[:begin]+restore+source[end:]
    source=source.replace('            for owner in range(start_owner,4):','            for owner in range(start_owner,start_owner+1):')
    begin=source.index("                wait(CONTROL/(str(owner)+'.EVAL_DONE.json'))")
    end=source.index("            b.save(R/'private/final'",begin)
    source=source[:begin]+source[end:]
    source=source.replace("            b.save(R/'private/final'/ARM/'native.pt'", "            if requested<3:return\n            b.save(R/'private/final'/ARM/'native.pt'",1)
    return source


def learner_source():
    import learner
    old=learner.old
    source=build_source(Path(old.__file__).read_text())
    tree={'__name__':'serialized_base_learner'};exec(compile(source,'serialized_base_learner','exec'),tree)
    for name in ('Risk','native_candidate','probe_rows','optimize','validate_state'):tree[name]=getattr(old,name)
    tree['main']()


def evaluator_source():
    import evaluator
    owner=int(os.environ['SOURCE_OWNER'])
    assert 1<=owner<=3 and (R/'private/control'/ARM/'EVALUATE.json').exists()
    model,bindings=c.load(6);hooks=runtime.track(model)
    try:
        native=torch.load(CONTROL/'native.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():model.get_module(c.LAYER).weight.copy_(native['native'].to(model.device))
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
        evaluator.panel(model,bindings,tasks,ledger,'insertion_R0',owner,owner)
        if owner in (1,3):evaluator.panel(model,bindings,tasks,ledger,'bank_R0',0,owner)
        if owner==3:evaluator.panel(model,bindings,tasks,ledger,'protect100_R0',0,100)
        runtime.peak_memory();c.write(CONTROL/(str(owner)+'.EVAL_DONE.json'),dict(owner=owner,outputs_durable=True))
    finally:hooks.close();b.counts()


def main():
    # Keep the same 45GiB admission/owned ceiling; reserve caching space while this sole GPU operator runs.
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    with runtime.lease():
        gpu=6;assert b.available(gpu)
        reserve=torch.empty(43*1024**3,dtype=torch.uint8,device='cuda');del reserve
        runtime.lease=lambda:nullcontext()
        # The previous operators clear idle cache at boundaries; retain our admitted cache until process exit.
        torch.cuda.empty_cache=lambda:None
        runtime.peak_memory()
        role=os.environ['SOURCE_ROLE']
        if role=='learn':learner_source()
        elif role=='evaluate':evaluator_source()
        else:
            import evaluator
            (evaluator.reference if role=='reference' else evaluator.reload)()
        runtime.peak_memory()
    c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(epoch=time.time(),traceback=traceback.format_exc(),counts=b.COUNTS));raise
