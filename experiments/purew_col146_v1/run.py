"""Finite C/O/L singles; reuse proven native runtime/generation and atomic state IO.

This module cannot authorize itself: controller admission checks frozen data,
publication and predecessor completion before invoking an arm.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import common as c
import base_run as b
from norm_math import update
from validate import resume_check
TOP=c.RUN
ARM=os.environ['COL_ARM']
assert ARM in ('O','L')
R=TOP/'private/arms'/ARM
b.R=R;b.ARM=ARM
b.ATTEMPT=TOP/'private/attempts'/f'{ARM}_{os.environ["ACTION"]}_{os.getpid()}.json'
b.CALL_LOG=TOP/'private/calls'/f'{ARM}_{os.environ["ACTION"]}_{os.getpid()}.log'
c.LAYER='model.layers.'+('31' if ARM=='L' else '21')+'.mlp.down_proj'

def cpu_state(s):
    return None if s is None else {k:v.detach().cpu() if hasattr(v,'detach') else v for k,v in s.items()}

def trajectory(runtime,bindings,tasks,weight,original,original_schema,frozen):
    import torch
    import torch.nn.functional as F
    mode='single_R0';active=R/'private/active/single_R0.pt'
    schedule=c.read(TOP/'private/C_NORM_SCHEDULE.json')
    state=torch.load(active,map_location='cpu',weights_only=True) if active.exists() else None
    lock=c.digest(c.read(TOP/'private/FROZEN_PROTOCOL.json'))
    units={int(p.stem):c.read(p)['step'] for p in (R/'private/units/single_R0').glob('*.json')}
    resume_check(state,units)
    if state:
        assert state['lock']==lock and state['arm']==ARM and state['mode']==mode
        c.restore_rng(state['rng'])
        import numpy as np
        nr=state['numpy_rng'];np.random.set_state((nr[0],nr[1].numpy().astype('uint32'),nr[2],nr[3],nr[4]))
        with torch.no_grad():weight.copy_(state['native'].to(runtime.device))
        c.write(R/'private/steps'/mode/str(state['owner'])/(str(state['step'])+'.json'),state['event'])
    master=state['master'].to(runtime.device) if state else original.float().clone()
    opt={k:v.to(runtime.device) if hasattr(v,'to') else v for k,v in state['optimizer'].items()} if state and state['optimizer'] is not None else None
    completed=state['committed'] if state else 0
    for i,t in enumerate(tasks,1):
        done=R/'private/units'/mode/(str(i)+'.json')
        if done.exists():
            assert c.read(done)['step']==160
            continue
        assert state is None or state['owner']<=i
        first=state['step'] if state and state['owner']==i else 0
        if not first:
            from m3bench_repro.editors.llava_runtime import seed_everything
            seed_everything(t['seed']);master=original.float().clone();opt=None
            with torch.no_grad():weight.copy_(master.half())
        native_batch=b.batch_cpu(runtime,t)
        fits=[b.batch_cpu(runtime,t,q) for q in t['fit_questions']]
        weight.requires_grad_(True)
        for step in range(first+1,161):
            started=time.perf_counter()
            c.budget();gradient=torch.zeros_like(master);losses=[]
            for batch,factor in ((native_batch,.5),(fits[(step-1)%4],.5)):
                weight.grad=None;z,y=b.logits(runtime,batch);loss=F.cross_entropy(z,y)
                b.COUNTS['backwards']+=1;b.call('B');b.counts()
                (128*factor*loss).backward()
                assert weight.grad is not None
                gradient.add_(weight.grad.float()/128);losses.append(float(loss.detach()));del z,loss
            before=weight.detach().float()
            target=schedule[i-1]['proposal_norm'][step-1]
            master,opt,event=update(gradient,master,target,'adam_direction' if ARM=='O' else 'sgd',opt)
            native=master.half()
            event.update(owner=i,step=step,mode=mode,arm=ARM,source_losses=losses,
                actual_native_norm=float((native.float()-before).norm()),changed_elements=int((native!=weight.detach()).sum()),
                C_native_norm=schedule[i-1]['actual_native_norm'][step-1],
                native_master_discrepancy=float((native.float()-master).norm()))
            event['optimization_seconds']=time.perf_counter()-started
            with torch.no_grad():weight.copy_(native)
            weight.grad=None;completed+=1
            import numpy as np
            nr=np.random.get_state()
            state=dict(lock=lock,arm=ARM,mode=mode,owner=i,step=step,committed=completed,
                master=master.detach().cpu(),native=native.detach().cpu(),optimizer=cpu_state(opt),rng=c.rng(),event=event,
                numpy_rng=(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4]))
            b.save(active,state);b.COUNTS['committed_updates']+=1;b.call('C')
            c.write(R/'private/steps'/mode/str(i)/(str(step)+'.json'),event);b.counts()
            if step%20==0:
                b.guard(runtime,weight,original_schema,frozen)
                b.write_progress(status='TRAINING',arm=ARM,owner=i,step=step,committed=completed)
                print('TRAIN',ARM,i,step,flush=True)
        weight.requires_grad_(False);b.guard(runtime,weight,original_schema,frozen)
        b.panel(runtime,bindings,[t],mode,i,1)
        c.write(done,dict(status='COMPLETE',step=160,committed=completed))
    assert completed==23360
    b.save(R/'private/final.pt',dict(native=weight.detach().cpu(),schema=original_schema,
        parameters=sum(p.numel() for p in runtime.model.parameters()),lock=lock))
    c.write(R/'private/single_R0_COMPLETE.json',dict(status='COMPLETE',committed=completed))
    c.write(R/'private/single_R0_CONSUMED.json',dict(path=str(active),bytes=active.stat().st_size,
        reason='All 146 single panels durable; final native retained only for fresh reload',retained_copy=False))
    active.unlink()

def gpu():
    import torch
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    with c.lease(5):
        runtime,bindings=c.load(5)
        def forward(*_):b.COUNTS['forwards']+=1;b.call('F')
        h=runtime.model.register_forward_pre_hook(forward)
        try:
            weight=runtime.get_module(c.LAYER).weight;original=weight.detach().clone()
            shape=b.schema(runtime.model);frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            def proposal(g, master):
                target=c.read(TOP/'private/C_NORM_SCHEDULE.json')[0]['proposal_norm'][0]
                after,_,event=update(g,master,target,'adam_direction' if ARM=='O' else 'sgd')
                return after,event
            b.proposals=proposal
            if not (R/'public/REAL_VLM_ADMISSION.json').exists():
                # Native Base check/true W gradient write at the arm's actual layer.
                b.admission(runtime,bindings,tasks,weight,shape,frozen)
            assert c.read(R/'public/REAL_VLM_ADMISSION.json')['status']=='PASS'
            if not (R/'private/single_R0_COMPLETE.json').exists():trajectory(runtime,bindings,tasks,weight,original,shape,frozen)
        finally:h.remove();b.counts()

def reload():
    import torch
    with c.lease(5):
        runtime,bindings=c.load(5)
        def forward(*_):b.COUNTS['forwards']+=1;b.call('F')
        h=runtime.model.register_forward_pre_hook(forward)
        try:
            saved=torch.load(R/'private/final.pt',map_location='cpu',weights_only=True)
            assert b.schema(runtime.model)==saved['schema'] and sum(p.numel() for p in runtime.model.parameters())==saved['parameters']
            weight=runtime.get_module(c.LAYER).weight
            with torch.no_grad():weight.copy_(saved['native'].to(runtime.device))
            assert not runtime.get_module(c.LAYER)._forward_hooks and not runtime.get_module(c.LAYER)._forward_pre_hooks
            assert not any(p.requires_grad for p in runtime.model.parameters())
            t=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][-1]
            ids=[t['edit_id'],next(e['all_probe_query_ids'][0] for e in t['events'] if e['task']=='T2G')]
            ledger=c.read(R/'private/EVAL_LEDGER.json')
            for q in ids:
                d=b.generate(runtime,bindings,ledger['queries'][q],R/'private/reload'/(q+'.json'),dict(stage='FRESH_NATIVE_RELOAD',arm=ARM))
                assert d['R0']==c.read(R/'private/outputs/single_R0/146/1'/(q+'.json'))['R0']
            size=(R/'private/final.pt').stat().st_size
            c.write(R/'public/NATIVE_RELOAD.json',dict(status='PASS',native_schema_exact=True,added_parameters=0,two_outputs_exact=True,serialized_native_bytes=size))
            c.write(R/'private/FINAL_CONSUMED.json',dict(bytes=size,reason='Fresh native reload verified and all panels durable',retained_copy=False))
            (R/'private/final.pt').unlink()
        finally:h.remove();b.counts()

if __name__=='__main__':
    try:
        {'gpu':gpu,'reload':reload}[os.environ['ACTION']]()
        c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))
    except Exception:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(traceback=traceback.format_exc(),counts=b.COUNTS,epoch=time.time()));raise
