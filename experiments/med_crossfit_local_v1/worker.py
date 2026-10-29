"""Bounded one-step cross-view diagnostic, without persistent expert updates."""
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ['DIAG_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
COUNTS = dict(forwards=0, backwards=0, pairs=0)


def write(name, value):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def selective(path):
    """Decode only the four original training task fields and source rows."""
    import stream_extract as s
    def skip(f, first):
        if first == b'"': s.string(f)
        elif first in (b'{', b'['): s.skip_compound(f, first)
        else: s.scalar(f, first)
    def value(f, first):
        start = f.tell() - 1
        skip(f, first)
        end = f.tell()
        if end-start > 1048576: raise ValueError('field too large')
        f.seek(start); data = json.loads(f.read(end-start)); f.seek(end)
        return data
    def obj(f, keep):
        result = {}
        while True:
            b = s.nonspace(f)
            if b == b'}': return result
            if b != b'"': raise ValueError('expected field')
            key = s.string(f, True)
            if s.nonspace(f) != b':': raise ValueError('expected colon')
            first = s.nonspace(f)
            if key in keep:
                if key in result: raise ValueError('duplicate field')
                result[key] = value(f, first)
            else: skip(f, first)
            b = s.nonspace(f)
            if b == b'}': return result
            if b != b',': raise ValueError('expected comma')
    tasks = []
    with open(path, 'rb') as f:
        if s.nonspace(f) != b'{': raise ValueError('expected object')
        while True:
            if s.nonspace(f) != b'"': raise ValueError('expected field')
            key = s.string(f, True)
            if s.nonspace(f) != b':': raise ValueError('expected colon')
            first = s.nonspace(f)
            if key != 'tasks': skip(f, first)
            else:
                if tasks or first != b'[': raise ValueError('bad tasks')
                while True:
                    first = s.nonspace(f)
                    if first == b']': break
                    if first != b'{': raise ValueError('bad row')
                    entry = {}
                    while True:
                        if s.nonspace(f) != b'"': raise ValueError('bad key')
                        field = s.string(f, True)
                        if s.nonspace(f) != b':': raise ValueError('bad colon')
                        first = s.nonspace(f)
                        if field == 'task':
                            if first != b'{' or field in entry: raise ValueError('bad task')
                            entry[field] = obj(f, {'edit_id', 'order', 'native', 'fit_questions', 'seed'})
                        elif field == 'source_rows':
                            if field in entry: raise ValueError('duplicate source rows')
                            entry[field] = value(f, first)
                        else: skip(f, first)
                        b = s.nonspace(f)
                        if b == b'}': break
                        if b != b',': raise ValueError('bad row comma')
                    tasks.append(entry)
                    b = s.nonspace(f)
                    if b == b']': break
                    if b != b',': raise ValueError('bad array comma')
            b = s.nonspace(f)
            if b == b'}': break
            if b != b',': raise ValueError('bad root comma')
    if [e['task']['order'] for e in tasks] != [147, 148, 149, 150]: raise ValueError('wrong fixed cohort')
    if any(len(e['source_rows']) != 5 or len(e['task']['fit_questions']) != 4 for e in tasks): raise ValueError('wrong view count')
    return tasks


def directions(theta, loss_s, loss_f, alpha):
    import torch
    keys = list(theta)
    gs = torch.autograd.grad(loss_s(theta), tuple(theta.values()))
    gf = torch.autograd.grad(loss_f(theta), tuple(theta.values()))
    def shifted(g): return {k: (theta[k] - alpha*v).detach().requires_grad_() for k, v in zip(keys, g)}
    def grad(loss, state): return torch.autograd.grad(loss(state), tuple(state.values()))
    ss, ff = shifted(gs), shifted(gf)
    own_s, own_f = grad(loss_s, ss), grad(loss_f, ff)
    cross_f, cross_s = grad(loss_f, ss), grad(loss_s, ff)
    avg = lambda a,b: {k: (x+y).detach()*.5 for k,x,y in zip(keys,a,b)}
    return {'ordinary': avg(gs,gf), 'self': avg(own_s,own_f), 'cross': avg(cross_f,cross_s)}


def selfcheck():
    import torch
    random.seed(7); before = random.getstate()
    p = {'x': torch.tensor([0.3, -0.2], dtype=torch.double, requires_grad=True)}
    a = lambda x: ((x['x']-torch.tensor([1.,2.],dtype=torch.double))**2).sum()
    b = lambda x: ((2*x['x']-torch.tensor([-1.,3.],dtype=torch.double))**2).sum()
    d = directions(p,a,b,0.)
    assert all(torch.allclose(d['ordinary']['x'],d[k]['x'],atol=1e-12,rtol=0) for k in ('self','cross'))
    d = directions(p,a,b,.1)
    assert not torch.allclose(d['cross']['x'],d['self']['x'])
    assert torch.equal(p['x'].detach(),torch.tensor([.3,-.2],dtype=torch.double)) and random.getstate()==before
    return {'status':'PASS','zero_inner_parity':True,'root_unchanged':True,'self_cross_distinct':True}


def main():
    import torch
    import numpy as np
    from types import SimpleNamespace
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import seed_everything
    sys.path.insert(0, str(ROOT/'tools'))
    import module_run as r
    write('SELF_CHECK.json', selfcheck())
    tasks = selective(os.environ['TRAIN_INPUTS'])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval(); r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):
        COUNTS['forwards']+=1
        if COUNTS['forwards']>1600: raise RuntimeError('forward limit')
    counter=runtime.model.register_forward_pre_hook(count)
    rows = []
    for entry in tasks:
        task = entry['task']; source = entry['source_rows']
        assert [x['audit_role'] for x in source] == ['NATIVE','FIT','FIT','FIT','FIT']
        runtime.model.half()
        try: batches=[r.old.batch_on(runtime,r.old.cpu(runtime.build_edit_batch(r.s.record(task,row)))) for row in source]
        finally: r.old.fp32(runtime)
        for offset in (0,1):
            seed_everything(task['seed']+offset)
            initial=r.p.tr.TT4(task['seed']+offset,8,8).state_dict()
            expert=r.p.expert(initial,task['seed']+offset,runtime.device)
            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
            try:
                for b in batches:
                    hook.set_teacher_routing(b.labels)
                    mask=torch.zeros_like(b.labels,dtype=torch.bool);mask[:,:-1]=b.labels[:,1:]!=-100
                    assert torch.equal(hook.token_mask,mask)
                with PrefixCache(runtime,batches):
                    root={k:v.detach().clone().requires_grad_() for k,v in expert.named_parameters()}
                    before={k:v.detach().clone() for k,v in root.items()}
                    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone(); py_rng=random.getstate();np_rng=copy.deepcopy(np.random.get_state())
                    def loss_for(index):
                        def loss(state):
                            from torch.func import functional_call
                            # Functional expert parameters are installed temporarily without changing Base.
                            original=hook.expert
                            class ResidualForward(torch.nn.Module):
                                def __init__(self, expert): super().__init__(); self.expert = expert
                                def forward(self, x): return self.expert.residual(x)
                            class Functional:
                                def residual(self, x):
                                    return functional_call(ResidualForward(original), {'expert.'+k:v for k,v in state.items()}, (x,))
                            hook.expert=Functional()
                            try:
                                hook.set_teacher_routing(batches[index].labels)
                                return runtime.compute_loss(batches[index])
                            finally: hook.expert=original
                        return loss
                    funcs=[loss_for(i) for i in range(5)]
                    def measured(state):
                        with torch.no_grad(): return [float(f(state).detach()) for f in funcs]
                    base=measured(root);repeat=measured(root)
                    if max(abs(a-b) for a,b in zip(base,repeat))>1e-5: raise RuntimeError('repeat noise exceeds tolerance')
                    for j in range(1,5):
                        d=directions(root,funcs[0],funcs[j],1e-4); COUNTS['backwards']+=6
                        assert COUNTS['backwards']<=1000
                        norms={k:math.sqrt(sum(float(v.double().square().sum()) for v in g.values())) for k,g in d.items()}
                        if any(not math.isfinite(v) or v<=0 for v in norms.values()): raise RuntimeError('zero or nonfinite direction')
                        updates={}; sensitivity={}
                        for name,g in d.items():
                            for equal in (False,True):
                                scale=1e-4*(norms['ordinary']/norms[name] if equal else 1.)
                                state={k:(root[k]-scale*g[k]).detach() for k in root}
                                vals=measured(state)
                                data={'other_fit':sum(base[k]-vals[k] for k in range(1,5) if k!=j)/3,
                                      'native':base[0]-vals[0], 'current_fit':base[j]-vals[j]}
                                (sensitivity if equal else updates)[name]=data
                        rows.append({'edit':task['order'],'seed_offset':offset,'fit':j,'baseline_nll':base,
                                     'ordinary_norm':norms['ordinary'],'norms':norms,'updates':updates,'equal_norm':sensitivity})
                        COUNTS['pairs']+=1
                    assert all(torch.equal(root[k].detach(),before[k]) for k in root)
                    assert all(torch.equal(v.detach(),initial[k].to(v)) for k,v in expert.named_parameters())
                    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()) and random.getstate()==py_rng
                    assert np.array_equal(np_rng[1],np.random.get_state()[1])
            finally: hook.detach()
    assert COUNTS['pairs']==32
    counter.remove()
    write('RESULTS_PRIVATE.json',{'rows':rows,'counts':COUNTS})
    write('COMPLETE.json',{'status':'COMPLETE','pairs':32,'counts':COUNTS})


if __name__=='__main__':
    code=0
    try: main()
    except BaseException:
        code=1;write('FAILURE.json',{'error':traceback.format_exc(),'counts':COUNTS});raise
    finally:write('WORKER_EXIT.json',{'code':code,'counts':COUNTS,'time':time.time()})
