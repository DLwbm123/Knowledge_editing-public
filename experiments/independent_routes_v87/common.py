"""Shared FP32 source cache/offload and independently bounded route experiments."""
import os
from pathlib import Path
import signal
import time
import traceback
import v81 as v


def prepare():
    with (v.ROOT/'WORKER_ONCE').open('x'):pass
    v.allocations=dict(v.limits['counts'])
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original route deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup()
    tasks=v.read(v.ROOT/'TASKS.json');items={};batches=[];by_owner={}
    for task in tasks:
        owner=task['order'];items[owner]=v.prepare(task,v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)],True)
        bs=[v.r.old.batch_on(v.runtime,x[1]) for x in items[owner]]
        by_owner[owner]=bs;batches+=bs
    return tasks,items,by_owner,batches


class CachedSource:
    def __init__(self,batches):self.batches=batches
    def __enter__(self):
        import prefix_cache
        self.cache=v.cache_class(v.runtime,self.batches);self.cache.__enter__()
        self.data={};module=v.runtime.get_module(v.r.c.LAYER)
        for batch in self.batches:
            captured=[]
            def capture(_module,args):captured.append(args[0].detach().clone())
            handle=module.register_forward_pre_hook(capture)
            try:
                with v.torch.no_grad():v.runtime.compute_loss(batch)
            finally:handle.remove()
            mask=v.torch.cat([batch.labels[:,1:]!=-100,v.torch.zeros_like(batch.labels[:,:1],dtype=v.torch.bool)],1)
            first=int(v.torch.where(batch.labels[0]!=-100)[0][0])-1
            self.data[id(batch)]=dict(x=captured[0][mask].float(),key=captured[0][0,first].float(),mask=mask)
        # Frozen source prefixes are already materialized; CUDA layers are not called on cache hits.
        for layer in v.runtime.model.model.layers[:21]:layer.to('cpu')
        self.cache.weights=[prefix_cache.signature(p) for p in self.cache.parameters]
        v.torch.cuda.empty_cache()
        v.write(v.ROOT/'CACHE_OFFLOAD_READY.json',dict(status='PASS',inputs=len(self.batches),
            prefix_layers_cpu=21,FP32=True,peak_cuda_bytes=v.torch.cuda.max_memory_allocated(),time=time.time()))
        return self
    def __exit__(self,*args):self.cache.__exit__(*args)


def restore():
    import subprocess
    # Keep V86's newly spawned evaluation load ahead of this independent consumer.
    parent=Path(os.environ['ACTIVE_V86'])
    while True:
        v.budget()
        moving=any(next(layer.parameters()).device.type=='cpu' for layer in v.runtime.model.model.layers[:21])
        free=int(subprocess.check_output(['nvidia-smi','-i',os.environ['GPU'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        registered=(parent/'ALL_CANDIDATES_FROZEN.json').exists()
        evaluators=list(parent.glob('PROCESS_AUDIT_evaluate_*.json'))
        pending_handoff=registered and len(evaluators)<4 and not (parent/'STOP').exists()
        if not pending_handoff and (not moving or free>=26*1024):break
        v.progress(stage='WAITING_SAFE_FULL_MODEL_RESTORE',free_MiB=free);time.sleep(10)
    for layer in v.runtime.model.model.layers[:21]:layer.to(v.runtime.device)


def generate(hook,item):
    torch=v.torch;runtime=v.runtime
    model=runtime.llava_model();original=model.prepare_inputs_labels_for_multimodal;used=[0]
    def prepared(*args,**kwargs):
        if args[0] is not None and torch.equal(args[0],item[2]['input_ids']):
            assert args[3] is None and args[4] is None;used[0]+=1;assert used[0]==1
            return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype) if isinstance(x,torch.Tensor) else x for x in item[3])
        return original(*args,**kwargs)
    model.prepare_inputs_labels_for_multimodal=prepared
    from contextlib import nullcontext
    try:
        v.generating=True;v.counts['evaluation_generations']+=1
        with torch.inference_mode(),(hook.generation_request() if hook else nullcontext()):answer=runtime.adapter.generate_prepared_with_result(item[2],runtime.generation_config)
        v.generating=False
    finally:model.prepare_inputs_labels_for_multimodal=original
    assert used[0]==1 and answer.raw_token_ids
    return answer


def evaluate(tasks,arms,loader):
    torch=v.torch
    restore()
    for task in tasks:
        owner=task['order'];plans=[p for p in v.read(v.ROOT/'EVAL_PLAN.json') if p['owner']==owner]
        routes=v.read(v.ROOT/'ROUTE_BINDINGS_FROZEN.json')[str(owner)]
        items=v.prepare(task,[p['query'] for p in plans])
        for arm in ['Base']+list(arms):
            expert=loader(arm,owner) if arm!='Base' else None
            if owner==1 and expert is not None:
                native=v.prepare(task,[v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)][0]],True)[0]
                batch=v.r.old.batch_on(v.runtime,native[1])
                if hasattr(expert,'request'):expert.request(int(v.torch.where(batch.labels[0]!=-100)[0][0])-1)
                mechanical=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);mechanical.attach()
                try:v.write(v.ROOT/'mechanical'/f'{arm}.json',v.r.s.fit.short.mechanical(v.runtime,mechanical,batch))
                finally:mechanical.detach()
            for index,(plan,item,route) in enumerate(zip(plans,items,routes)):
                assert route['question']==item[4]['question'] and route['query_id']==plan['query']['query_id'] and route['image_sha256']==item[4]['image_sha256']
                hook=None
                if expert is not None and route['effective_expert'] is not None:
                    if hasattr(expert,'request'):expert.request(None)
                    hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
                try:answer=generate(hook,item)
                finally:
                    if hook:hook.detach()
                tokens=list(answer.raw_token_ids)
                v.write(v.ROOT/'outputs'/arm/str(owner)/f'{index}.json',dict(owner=owner,arm=arm,query=plan['query'],binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,
                    EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,original_route_preserved=True))
                v.progress(stage='evaluate',owner=owner,arm=arm,index=index)


def ce_train(expert,hook,batches,steps=160):
    torch=v.torch
    groups=[{'params':[p for k,p in expert.named_parameters() if k in ('G1','G2')],'lr':1e-3},
            {'params':[p for k,p in expert.named_parameters() if k in ('G3','G4')],'lr':1e-4}]
    opt=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for view in (0,1+step%4):
            b=batches[view];hook.set_teacher_routing(b.labels);loss=v.runtime.compute_loss(b)
            assert torch.isfinite(loss);(.5*loss).backward();v.counts['backwards']+=1
        torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);opt.step();v.counts['updates']+=1


def finish(arms):
    import review
    expected=155*(len(arms)+1)
    assert len(list((v.ROOT/'outputs').glob('*/*/*.json')))==expected
    v.write(v.ROOT/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=expected,counts=v.counts,time=time.time()))
    review.evaluation_pack(v.ROOT,expected)


def run(main):
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/'FAILURE_worker.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='INDEPENDENT_ROUTE_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/'EXIT_worker.json',dict(exit_code=code,counts=v.counts,time=time.time()))
        v.write(v.ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
