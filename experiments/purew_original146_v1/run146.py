"""One source-only native-W arm: independent singles and a real continuous trajectory."""
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
import common as c
from contract import ARM, PREFIXES, proposals, selfcheck, base_completion

R = c.RUN
COUNTS = dict(forwards=0, backwards=0, committed_updates=0, generation_attempts=0)
ATTEMPT = R/'private/attempts'/f'{os.environ["ACTION"]}_{os.getpid()}.json'
CALL_LOG = R/'private/calls'/f'{os.environ["ACTION"]}_{os.getpid()}.log'


def call(kind):
    CALL_LOG.parent.mkdir(parents=True,exist_ok=True)
    with CALL_LOG.open('a') as f:
        f.write(kind+'\n')


def counts():
    c.write(ATTEMPT, dict(COUNTS, pid=os.getpid(), action=os.environ['ACTION']))
    rows = [c.read(p) for p in ATTEMPT.parent.glob('*.json')]
    p = c.read(R/'private/FROZEN_PROTOCOL.json')['budgets']
    assert sum(x['backwards'] for x in rows) <= p['maximum_backwards']
    assert sum(x['generation_attempts'] for x in rows) <= p['maximum_generations']
    assert sum(x['forwards'] for x in rows) <= p['maximum_forward_calls']


def write_progress(**d):
    c.write(R/'public/PROGRESS.json', dict(d, epoch=time.time()))


def available(gpu):
    c.budget()
    u, free = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
              '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert gpu == 5 and u == c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5']
    return int(free) >= 45*1024


c.available = available


def schema(model):
    return [(n, list(x.shape), str(x.dtype)) for n, x in model.state_dict().items()]


def save(path, state):
    c.save(path, state)
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def batch_cpu(runtime, task, question=None):
    record = c.record(task)
    if question is not None:
        record = replace(record, question=question)
    batch = runtime.build_edit_batch(record)
    kwargs = {k:v.detach().cpu() if hasattr(v, 'detach') else v for k,v in batch.forward_kwargs().items()}
    assert kwargs['inputs_embeds'].dtype == __import__('torch').float16
    return kwargs, batch.labels.detach().cpu()


def logits(runtime, batch):
    kwargs, labels = batch
    kwargs = {k:v.to(runtime.device) if hasattr(v, 'to') else v for k,v in kwargs.items()}
    labels = labels.to(runtime.device)[:, 1:]
    mask = labels != -100
    assert mask.any()
    output = runtime.model(**kwargs).logits[:, :-1][mask].float()
    assert __import__('torch').isfinite(output).all()
    return output, labels[mask]


def generate(runtime, bindings, row, dest, phase):
    import torch
    if dest.exists():
        d = c.read(dest)
        assert d['binding']['phase'] == phase and d['binding']['input'] == row
        return d
    marker = dest.with_suffix('.STARTED.json')
    assert not marker.exists(), 'Generation attempted without durable output; no implicit replay'
    c.budget()
    raw = runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])), row['question'], None)
    base = bindings[row['opaque_Base_id']]
    assert raw['image_sha256'] == base['image_sha256']
    assert raw['input_ids'].tolist() == [base['prompt_ids']]
    assert raw['attention_mask'].tolist() == [base['attention_mask']]
    c.write(marker, dict(phase=phase, epoch=time.time(), attempt=1))
    COUNTS['generation_attempts'] += 1
    call('G')
    counts()
    started = time.time()
    with torch.inference_mode():
        g = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
    d = dict(binding=dict(input=row, judge_input=base, phase=phase),
             R0=dict(raw_answer=g.decoded_text, raw_token_ids=list(g.raw_token_ids)),
             seconds=time.time()-started, added_parameters=0, native_inference=True)
    c.write(dest, d)
    counts()
    return d


def panel(runtime, bindings, tasks, mode, owner, prefix):
    ledger = c.read(R/'private/EVAL_LEDGER.json')
    qids = list(dict.fromkeys(q for t in tasks for e in t['events'] for q in e['all_probe_query_ids']))
    if mode == 'insertion_R0':
        qids = [tasks[-1]['edit_id']]
    phase = dict(arm=ARM, mode=mode, owner=owner, prefix=prefix,
                 protocol=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')),
                 code=c.read(R/'private/GPU_SOURCE_VERSION.json')['commit'])
    folder = R/'private/outputs'/mode/str(owner)/str(prefix)
    for qid in qids:
        dest = folder/(qid+'.json')
        generate(runtime, bindings, ledger['queries'][qid], dest, phase)
    c.write(folder/'COMPLETE.json', dict(phase=phase, outputs=len(qids)))


def guard(runtime, weight, original_schema, frozen):
    assert schema(runtime.model) == original_schema
    assert frozen == {n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
    assert not runtime.get_module(c.LAYER)._forward_hooks and not runtime.get_module(c.LAYER)._forward_pre_hooks


def admission(runtime, bindings, tasks, weight, original_schema, frozen):
    import torch
    import torch.nn.functional as F
    saved = weight.detach().clone()
    assert weight.dtype == torch.float16
    for t in (tasks[0], tasks[49], tasks[145]):
        d = generate(runtime, bindings, t['native'], R/'private/mechanical'/(str(t['order'])+'.json'), dict(stage='BASE_IDENTITY'))
        assert d['R0'] == base_completion(bindings[t['native']['opaque_Base_id']]['output']), 'Native Base identity differs; do not reuse baselines'
    batch = batch_cpu(runtime, tasks[0])
    features = {}
    h = runtime.get_module(c.LAYER).register_forward_pre_hook(lambda m,a:features.update(value=a[0].detach().clone()))
    try:
        with torch.no_grad():
            logits(runtime, batch)
        before = features['value']
        weight.requires_grad_(True)
        z, y = logits(runtime, batch)
        loss = F.cross_entropy(z,y)
        COUNTS['backwards'] += 1
        call('B')
        counts()
        (128*loss).backward()
        g = weight.grad.float()/128
        master, event = proposals(g, saved.float())
        with torch.no_grad():
            weight.copy_(master.half())
            logits(runtime,batch)
        changed = int((weight!=saved).sum())
        assert changed > 0 and torch.equal(before, features['value'])
        event.update(actual_native_norm=float((weight.float()-saved.float()).norm()), changed_elements=changed)
    finally:
        h.remove()
        with torch.no_grad():
            weight.copy_(saved)
        weight.grad = None
        weight.requires_grad_(False)
    guard(runtime,weight,original_schema,frozen)
    result = dict(status='PASS', three_Base_identities_exact=True, upstream_feature_exact=True,
                  real_native_W_gradient_write=True, added_parameters=0,
                  parameters=sum(p.numel() for p in runtime.model.parameters()), event=event)
    c.write(R/'public/REAL_VLM_ADMISSION.json',result)
    counts()


def trajectory(runtime, bindings, tasks, weight, original, original_schema, frozen, mode):
    import torch
    import torch.nn.functional as F
    active = R/'private/active'/f'{mode}.pt'
    native_batches = [batch_cpu(runtime,t) for t in tasks]
    state = torch.load(active,map_location='cpu',weights_only=True) if active.exists() else None
    if state:
        assert state['lock'] == c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')) and state['mode'] == mode
        c.restore_rng(state['rng'])
        import numpy as np
        nr=state['numpy_rng']
        np.random.set_state((nr[0],nr[1].numpy().astype('uint32'),nr[2],nr[3],nr[4]))
        with torch.no_grad():
            weight.copy_(state['native'].to(runtime.device))
        if state.get('event'):
            c.write(R/'private/steps'/mode/str(state['owner'])/(str(state['step'])+'.json'),state['event'])
    master = state['master'].to(runtime.device) if state else original.float().clone()
    completed = state['committed'] if state else 0
    for i,t in enumerate(tasks,1):
        complete = R/'private/units'/mode/(str(i)+'.json')
        if complete.exists():
            assert c.read(complete)['step']==160
            continue
        assert state is None or state['owner']<=i, 'Missing completed earlier unit'
        step0 = state['step'] if state and state['owner']==i else 0
        if not step0:
            from m3bench_repro.editors.llava_runtime import seed_everything
            seed_everything(t['seed'])
            if mode=='single_R0':
                master = original.float().clone()
            with torch.no_grad():
                weight.copy_(master.half())
        fits = [batch_cpu(runtime,t,q) for q in t['fit_questions']]
        weight.requires_grad_(True)
        for step in range(step0+1,161):
            c.budget()
            chosen = [(native_batches[i-1],.5),(fits[(step-1)%4],.5)]
            if mode=='bank_R0' and i>1:
                chosen.append((native_batches[(step-1)%(i-1)],.5))
            gradient = torch.zeros_like(master)
            losses = []
            for batch,factor in chosen:
                weight.grad = None
                z,y = logits(runtime,batch)
                loss = F.cross_entropy(z,y)
                COUNTS['backwards'] += 1
                call('B')
                counts()
                (128*factor*loss).backward()
                assert weight.grad is not None
                gradient.add_(weight.grad.float()/128)
                losses.append(float(loss.detach()))
                del z,loss
            before = weight.detach().float()
            master, event = proposals(gradient,master)
            native = master.half()
            event.update(owner=i,step=step,mode=mode,source_losses=losses,
                         actual_native_norm=float((native.float()-before).norm()),
                         changed_elements=int((native!=weight.detach()).sum()))
            with torch.no_grad():
                weight.copy_(native)
            weight.grad = None
            completed += 1
            state = dict(lock=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')),mode=mode,
                         owner=i,step=step,committed=completed,master=master.detach().cpu(),
                         native=native.detach().cpu(),rng=c.rng(),event=event)
            import numpy as np
            nr=np.random.get_state()
            state['numpy_rng']=(nr[0],torch.from_numpy(nr[1].astype('int64')),nr[2],nr[3],nr[4])
            save(active,state)
            COUNTS['committed_updates'] += 1
            call('C')
            c.write(R/'private/steps'/mode/str(i)/(str(step)+'.json'),event)
            counts()
            if step%20==0:
                guard(runtime,weight,original_schema,frozen)
                write_progress(status='TRAINING',mode=mode,owner=i,step=step,committed=completed)
                print('TRAIN',mode,i,step,flush=True)
        weight.requires_grad_(False)
        guard(runtime,weight,original_schema,frozen)
        if mode=='single_R0':
            panel(runtime,bindings,[t],mode,i,1)
        else:
            panel(runtime,bindings,tasks[:i],'insertion_R0',i,i)
            if i in PREFIXES:
                panel(runtime,bindings,tasks[:i],mode,0,i)
        c.write(complete,dict(status='COMPLETE',step=160,committed=completed))
    assert completed==23360
    if mode=='bank_R0':
        save(R/'private/final.pt',dict(native=weight.detach().cpu(),schema=original_schema,
             parameters=sum(p.numel() for p in runtime.model.parameters()),lock=state['lock']))
    c.write(R/'private'/f'{mode}_COMPLETE.json',dict(status='COMPLETE',committed=completed))
    c.write(R/'private'/f'{mode}_CONSUMED.json',dict(path=str(active),bytes=active.stat().st_size,
            reason='All registered unit/prefix outputs durable; final native capture retained separately',retained_copy=mode=='bank_R0'))
    active.unlink()


def gpu_main():
    import torch
    tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    with c.lease(5):
        runtime, bindings = c.load(5)
        def forward(*_):
            COUNTS['forwards']+=1
            call('F')
        h = runtime.model.register_forward_pre_hook(forward)
        try:
            weight = runtime.get_module(c.LAYER).weight
            original = weight.detach().clone()
            original_schema = schema(runtime.model)
            frozen = {n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            if not (R/'public/REAL_VLM_ADMISSION.json').exists():
                admission(runtime,bindings,tasks,weight,original_schema,frozen)
            assert c.read(R/'public/REAL_VLM_ADMISSION.json')['status']=='PASS'
            for mode in ('single_R0','bank_R0'):
                if not (R/'private'/f'{mode}_COMPLETE.json').exists():
                    with torch.no_grad():
                        weight.copy_(original)
                    trajectory(runtime,bindings,tasks,weight,original,original_schema,frozen,mode)
            c.write(R/'private/TRAINING_GENERATION_COMPLETE.json',dict(status='COMPLETE',updates=46720))
        finally:
            h.remove()
            counts()


def reload_main():
    import torch
    import statistics
    with c.lease(5):
        runtime,bindings = c.load(5)
        def forward(*_):
            COUNTS['forwards']+=1
            call('F')
        h=runtime.model.register_forward_pre_hook(forward)
        saved = torch.load(R/'private/final.pt',map_location='cpu',weights_only=True)
        assert schema(runtime.model)==saved['schema'] and sum(p.numel() for p in runtime.model.parameters())==saved['parameters']
        weight = runtime.get_module(c.LAYER).weight
        with torch.no_grad():
            weight.copy_(saved['native'].to(runtime.device))
        assert not any(p.requires_grad for p in runtime.model.parameters())
        assert not runtime.get_module(c.LAYER)._forward_hooks and not runtime.get_module(c.LAYER)._forward_pre_hooks
        tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
        rows=[]
        ledger=c.read(R/'private/EVAL_LEDGER.json')
        for t in (tasks[0],tasks[49],tasks[99],tasks[145]):
            qids=[t['edit_id'], next(e['all_probe_query_ids'][0] for e in t['events'] if e['task']=='T2G')]
            rows.extend(ledger['queries'][q] for q in qids)
        for row in rows[:2]:
            d=generate(runtime,bindings,row,R/'private/reload'/(row['query_id']+'.json'),dict(stage='FRESH_NATIVE_RELOAD'))
            original=c.read(R/'private/outputs/bank_R0/0/146'/(row['query_id']+'.json'))
            assert d['R0']==original['R0']
        events=[]
        for index,row in enumerate(rows):
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            for repeat in range(4):
                COUNTS['generation_attempts']+=1;call('G');counts()
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.inference_mode():
                    g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                torch.cuda.synchronize()
                events.append(dict(query_slot=index,repeat=repeat,warmup=repeat==0,seconds=time.perf_counter()-started,tokens=len(g.raw_token_ids),route_seconds=0))
        result=dict(status='PASS',exact_fresh_native_reload=True,added_parameters=0,original_schema=True,
                    serialized_native_bytes=(R/'private/final.pt').stat().st_size,
                    GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),events=events,
                    mean_query_medians_seconds=statistics.mean(statistics.median(x['seconds'] for x in events if x['query_slot']==i and not x['warmup']) for i in range(8)),
                    baseline_matched_latency='NA; old weights consumed; do not reconstruct')
        c.write(R/'public/NATIVE_RELOAD_PERFORMANCE.json',result)
        c.write(R/'private/FINAL_CONSUMED.json',dict(bytes=(R/'private/final.pt').stat().st_size,reason='Fresh native reload/timing and bound efficacy outputs durable',retained_copy=False))
        (R/'private/final.pt').unlink()
        h.remove()
        counts()


def controller():
    selfcheck()
    write_progress(status='GPU_MECHANICAL_PENDING' if (R/'private/BASELINE_REUSE_ADMISSION.json').exists() else 'BASELINE_REUSE_AUDIT_RUNNING')
    if not (R/'private/BASELINE_REUSE_ADMISSION.json').exists():
        import admit146
        admit146.main()
    for action in ('gpu','reload'):
        done=R/'private'/('TRAINING_GENERATION_COMPLETE.json' if action=='gpu' else 'FINAL_CONSUMED.json')
        if done.exists():
            continue
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env=dict(os.environ,ACTION=action,TASK_FILE=str(R/'private/tools/run146.py'))
        child=subprocess.Popen([os.environ['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs'/f'{action}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        start=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],entry=entry,epoch=time.time())
        c.write(R/'private'/f'START_{action}.json',start)
        assert not any(x in subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True) for x in ('wangbomin','Knowledge_editing','purew'))
        assert child.wait()==0, f'{action} failed; inspect saved state, never blindly restart'
    c.write(R/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',updates=46720,epoch=time.time()))
    write_progress(status='GENERATION_COMPLETE_SCORING_PENDING')


if __name__=='__main__':
    try:
        {'controller':controller,'gpu':gpu_main,'reload':reload_main}[os.environ['ACTION']]()
        c.write(R/'private'/f'EXIT_{os.environ["ACTION"]}.json',dict(exit_code=0,epoch=time.time()))
    except Exception:
        counts()
        c.write(R/'private/failures'/f'{os.environ["ACTION"]}_{os.getpid()}.json',dict(traceback=traceback.format_exc(),epoch=time.time(),counts=COUNTS))
        raise
