"""One fixed four-edit, three-arm experiment using the original M13P training step."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT, PARENT = Path(os.environ['EXPERIMENT_ROOT']), Path(os.environ['PARENT_RUN'])
COUNTS = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0, LM_input_positions=0)
LOCK = json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
CLOCK = json.loads((ROOT/'CLOCK.json').read_text())


def write(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def budget():
    assert time.time() < CLOCK['hard_end'] - 30 and not (ROOT/'STOP.json').exists()
    assert time.time() - json.loads((ROOT/'GUARD_HEALTH.json').read_text())['time'] < 45
    for k, limit in LOCK['budgets'].items():
        if k in COUNTS:
            assert COUNTS[k] <= limit, (k, COUNTS[k], limit)


def save(path, value):
    import torch
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    with (ROOT/'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen wall limit')))
    signal.alarm(max(1, int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0] = [str(ROOT/'tools'), str(PARENT/'private/tools'),
                    str(PARENT/'private/source'), str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import curvature
    import fisher_math
    import training_math
    import trajectory
    from prefix_cache import PrefixCache
    from m3bench_repro.editors.llava_runtime import seed_everything
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace

    r.c.budget = budget
    torch.set_num_threads(1)
    torch.manual_seed(LOCK['seed'])
    torch.cuda.manual_seed_all(LOCK['seed'])
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.51)
    inputs = json.loads((ROOT/'INPUTS_FROZEN.json').read_text())
    assert r.c.digest(inputs) == LOCK['input_binding']
    write('ALGEBRA_CHECK.json', curvature.selfcheck())
    # Same original optimizer and cache, versus the new factor-container dispatch.
    legacy_check = r.p.load_state(PARENT/'private/OPERATOR_FISHER.pt')
    C_check = r.p.load_state(PARENT/'private/candidates/C'/f"{LOCK['owners'][0]}.pt")['states']['FINAL']
    C_check = {k:v.cpu() for k,v in C_check.items()}
    generator = torch.Generator().manual_seed(132)
    gradients_check = {k:torch.randn(v.shape,generator=generator) for k,v in C_check.items()}
    original_penalty_check = r.mm.operator_penalty
    scalar_check = dict(step=0,second_moments=[0.,0.])
    first, scalar_first, _ = training_math.step(r,C_check,gradients_check,legacy_check,.001,
                                                scalar_check,.001,arm='M13P')
    r.mm.operator_penalty = lambda state,cache: curvature.penalty(state,cache,r.mm.factors)
    try:
        second, scalar_second, _ = training_math.step(r,C_check,gradients_check,dict(parts=[legacy_check]),
                                                     .001,scalar_check,.001,arm='M13P')
    finally:
        r.mm.operator_penalty = original_penalty_check
    assert scalar_first == scalar_second
    for k in first:
        torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
    write('LEGACY_STEP_PARITY.json',dict(status='PASS',nonzero_operator=True,all_cores_exact=True))
    del legacy_check,C_check,first,second
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings = r.c.read(PARENT/'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']

    def count(_module, args, kwargs):
        budget()
        COUNTS['forwards'] += 1
        x = kwargs.get('inputs_embeds', kwargs.get('input_ids'))
        if x is None and args:
            x = args[0]
        if isinstance(x, torch.Tensor):
            COUNTS['LM_input_positions'] += x.shape[0] * x.shape[1]

    fh = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    legacy = r.p.load_state(PARENT/'private/OPERATOR_FISHER.pt')
    basis, _ = r.q.split()
    assert len(basis) == len(legacy['token_counts']) == 61
    xpieces, ypieces = list(legacy['x'].split(legacy['token_counts'])), list(legacy['y'].split(legacy['token_counts']))
    fallback = curvature.concatenate([dict(x=xpieces[v['index']],y=ypieces[v['index']])
                                      for v in inputs['unmapped']])
    state = r.mm.canonicalize(r.p.tr.TT4(LOCK['seed'],8,8).state_dict())[0]
    expert = r.p.expert(state, LOCK['seed'], runtime.device)
    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    hook.attach()
    full_blocks, med_blocks, gen_blocks, audits = [], [], [], []
    try:
        for entry in inputs['mapping']:
            index, row = entry['index'], entry['row']
            assert row == basis[index]
            aliases = [a for group in entry['classes'] for a in group]
            hook.enabled = False
            runtime.model.half()
            frozen = []
            try:
                for alias in aliases:
                    tokens = runtime.adapter.tokenizer.encode(alias, add_special_tokens=False)
                    tokens += [runtime.adapter.tokenizer.eos_token_id]
                    kwargs, labels, mask, binding = r.p.tr.teacher_batch(runtime,row,tokens)
                    kwargs['labels'] = labels
                    value = SimpleNamespace(labels=labels,target_token_ids=tuple(tokens),forward_kwargs=lambda k=kwargs:k)
                    frozen.append(r.old.cpu(value))
            finally:
                r.old.fp32(runtime)
                hook.enabled = True
            paths, scores = [], []
            for alias_index, value in enumerate(frozen):
                batch = r.old.batch_on(runtime,value)
                hook.set_teacher_routing(batch.labels)
                support = hook.token_mask
                expected = torch.zeros_like(batch.labels,dtype=torch.bool)
                expected[:,:-1] = batch.labels[:,1:] != -100
                assert torch.equal(support,expected)
                assert int(support.sum()) == len(batch.target_token_ids)
                captured = {}
                def capture(_module, args, output):
                    x = args[0].detach()[support]
                    captured.update(x=x/(x.square().mean(-1,keepdim=True).sqrt()+expert.epsilon),output=output)
                tap = runtime.get_module(r.c.LAYER).register_forward_hook(capture)
                try:
                    kwargs = dict(batch.forward_kwargs()); kwargs.pop('labels')
                    output = runtime.model(**kwargs)
                    mask = batch.labels[:,1:] != -100
                    logits = output.logits[:,:-1][mask].float()
                    targets = batch.labels[:,1:][mask]
                    assert targets[-1] == runtime.adapter.tokenizer.eos_token_id
                    score = logits.log_softmax(-1).gather(1,targets[:,None]).mean()
                    parameters = list(expert.parameters())
                    gs = torch.autograd.grad(score,[captured['output']]+parameters,retain_graph=True)
                    COUNTS['backwards'] += 1
                    x, y = captured['x'], gs[0].detach()[support]
                    b, a = expert.factors()
                    proxy = ((x @ a.T)*(y @ b)).sum()
                    pulled = torch.autograd.grad(proxy,parameters)
                    for direct, composed in zip(gs[1:],pulled):
                        torch.testing.assert_close(direct,composed,rtol=2e-4,atol=2e-7)
                    score_gradient = gs[1].detach().clone()
                    probability = logits.detach().softmax(-1)
                    rng = torch.Generator(device=runtime.device).manual_seed(20261009+index*4+alias_index)
                    probes = []
                    for s in range(8):
                        budget()
                        signs = torch.randint(0,2,logits.shape,device=runtime.device,dtype=torch.int8,generator=rng).float()*2-1
                        vector = fisher_math.categorical_probe(probability,signs)
                        probe = (logits*vector).sum()/math.sqrt(61*32*len(logits))
                        adjoint, = torch.autograd.grad(probe,captured['output'],retain_graph=s!=7)
                        COUNTS['backwards'] += 1
                        probes.append(adjoint.detach()[support].cpu())
                    paths.append(dict(x=x.cpu(),score=y.cpu(),full=torch.stack(probes,1)))
                    scores.append(float(score.detach()))
                finally:
                    tap.remove()
                if not audits and alias_index == 0:
                    direction = score_gradient / score_gradient.norm()
                    before = r.old.snapshot(expert)
                    values = []
                    with torch.no_grad():
                        for sign in (-1,1):
                            changed = {k:v.clone() for k,v in before.items()}
                            changed['G1'] += sign*1e-3*direction
                            expert.load_state_dict(changed)
                            result = runtime.model(**kwargs)
                            z = result.logits[:,:-1][mask].float().log_softmax(-1)
                            values.append(float(z.gather(1,targets[:,None]).mean()))
                    expert.load_state_dict(before)
                    observed = (values[1]-values[0])/.002
                    expected_derivative = float(score_gradient.norm())
                    assert abs(observed-expected_derivative) <= 2e-4+.03*expected_derivative
                    write('ACTUAL_MASK_DERIVATIVE_CHECK.json',dict(status='PASS',finite_difference=observed,
                        analytic=expected_derivative,step=1e-3,actual_injection_mask_exact=True,
                        predictor_positions=int(support.sum()),EOS_scored=True,last_prompt_predictor_selected=True))
                del output,logits,score,gs,proxy,pulled,adjoint,probability,probes,captured
            try:
                full,med,gen,audit = curvature.anchor(paths,scores,index,LOCK['GEN_seed'])
            except curvature.DegenerateCurvature as error:
                # A declared numerical fallback changes all three arms together.
                common = dict(x=xpieces[index],y=ypieces[index])
                fallback_parts = list(zip(fallback['x'].split(fallback['token_counts']),fallback['y'].split(fallback['token_counts'])))
                fallback = curvature.concatenate([dict(x=x,y=y) for x,y in fallback_parts]+[common])
                audits.append(dict(index=index,status='COMMON_LEGACY_FALLBACK',reason=str(error)))
            else:
                full_blocks.extend(full);med_blocks.append(med);gen_blocks.append(gen)
                audits.append(dict(audit,status='MAPPED'))
            write('PROGRESS.json',dict(phase='CURVATURE',complete=len(audits),total=len(inputs['mapping']),
                                       counts=COUNTS,time=time.time()))
            print('CURVATURE',len(audits),flush=True)
        assert med_blocks and len(full_blocks)==4*len(med_blocks)
        assert any(abs(v.get('MED_GEN_operator_cosine',1)) < .999 for v in audits)
        caches = {name:dict(parts=[fallback,curvature.concatenate(blocks)])
                  for name,blocks in [('FULL_CTX',full_blocks),('MED',med_blocks),('GEN',gen_blocks)]}
        save('curvature/COMMON_FALLBACK.pt',fallback)
        for name,cache in caches.items():
            save('curvature/'+name+'.pt',dict(mapped_part=cache['parts'][1],
                 shared_part='COMMON_FALLBACK.pt'))
        write('CURVATURE_ADMISSION.json',dict(status='PASS',anchors=audits,mapped=len(med_blocks),
            common_fallback=61-len(med_blocks),total_trace_matched=True,
            global_spectrum_matched=False,operator_space_matching=True,counts=COUNTS))
    finally:
        hook.detach()
    del expert,legacy,xpieces,ypieces,full_blocks,med_blocks,gen_blocks,paths
    shared = {k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in fallback.items()}
    for name,cache in caches.items():
        cache['parts'] = [shared,{k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v
                                  for k,v in cache['parts'][1].items()}]
    original_penalty = r.mm.operator_penalty
    r.mm.operator_penalty = lambda state,cache: curvature.penalty(state,cache,r.mm.factors) if 'parts' in cache else original_penalty(state,cache)
    template = inputs['tasks'][0]
    for task in inputs['tasks']:
        rows = inputs['source_rows'][str(task['order'])]
        runtime.model.half()
        try:
            prepared = r.s.prepare(runtime,bindings,task,rows,True)
        finally:
            r.old.fp32(runtime)
        batches = [r.old.batch_on(runtime,v[1]) for v in prepared]
        reference = r.c.read(PARENT/'private/results/C'/f"{task['order']}.json")
        C = r.p.load_state(PARENT/'private/candidates/C'/f"{task['order']}.pt")['states']
        endpoint = {k:v.to(runtime.device) for k,v in C['FINAL'].items()}
        denominator = float(r.mm.operator_penalty(endpoint,caches['FULL_CTX']))
        assert len(reference['gain']) == 1
        coefficient = training_math.coefficient(reference['gain'][0],denominator)
        targets = [v['operator_step_norm'] for v in reference['curve']]
        write(f"calibration/{task['order']}.json",dict(C_gain=reference['gain'][0],
            C_FULL_CTX_penalty=denominator,coefficient=coefficient,targets=targets,
            shared_across_arms=True,source_only=True))
        for arm in LOCK['arms']:
            seed_everything(task['seed'])
            initial = r.p.tr.TT4(task['seed'],8,8).state_dict()
            assert all(torch.equal(initial[k],C['BASE'][k].cpu()) for k in initial)
            state = r.mm.canonicalize(initial)[0]
            expert = r.p.expert(state,task['seed'],runtime.device)
            hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
            hook.attach()
            scalar = dict(step=0,second_moments=[0.,0.]);curve=[]
            try:
                with PrefixCache(runtime,batches):
                    for step in range(1,161):
                        budget()
                        before = r.old.snapshot(expert)
                        trajectory.source_gradients(r,runtime,hook,expert,batches,step)
                        COUNTS['backwards'] += 2
                        gradients = {k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                        selected,scalar,audit = training_math.step(r,before,gradients,caches[arm],
                            coefficient,scalar,targets[step-1],arm='M13P')
                        expert.load_state_dict(selected)
                        COUNTS['updates'] += 1; COUNTS['operator_backwards'] += 1
                        curve.append(dict(step=step,**audit))
                        if step%20 == 0:
                            save(f"resume/{arm}_{task['order']}.pt",dict(state=selected,scalar=scalar,
                                 step=step,curve=curve,rng=r.c.rng()))
                            write('PROGRESS.json',dict(phase='TRAIN',arm=arm,owner=task['order'],
                                step=step,counts=COUNTS,time=time.time()))
                            print('TRAIN',arm,task['order'],step,flush=True)
                final = r.old.snapshot(expert)
                save(f"candidates/{arm}_{task['order']}.pt",final)
                loaded = torch.load(ROOT/'candidates'/f"{arm}_{task['order']}.pt",weights_only=True)
                assert all(torch.equal(v,loaded[k]) for k,v in final.items())
                write(f"training/{arm}_{task['order']}.json",dict(curve=curve,steps=160,
                    coefficient=coefficient,save_load_exact=True,counts=COUNTS))
            finally:
                hook.detach()
    assert COUNTS['updates'] == 1920
    write('ALL_CANDIDATES_FROZEN.json',dict(time=time.time(),counts=COUNTS))
    del caches,shared,batches,prepared,fallback
    torch.cuda.empty_cache()
    for task in inputs['tasks']:
        rows = [v for v in inputs['consumers'] if v['owner']==task['order']]
        for arm in LOCK['arms']:
            state = torch.load(ROOT/'candidates'/f"{arm}_{task['order']}.pt",map_location=runtime.device,weights_only=True)
            expert = r.p.expert(state,task['seed'],runtime.device)
            hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
            hook.attach()
            try:
                runtime.model.half()
                try:
                    prepared = r.s.prepare(runtime,bindings,task,[v['query'] for v in rows])
                finally:
                    r.old.fp32(runtime)
                for consumer,item in zip(rows,prepared):
                    budget()
                    COUNTS['generations'] += 1
                    answer,trace = r.s.fit.short.generate(runtime,hook,item[2],item[3])
                    tokens = list(answer.raw_token_ids)
                    name = r.c.digest(consumer['query']['query_id'])
                    write(f"outputs/{arm}/{task['order']}/{name}.json",dict(
                        arm=arm,owner=task['order'],query=consumer['query'],panels=consumer['panels'],
                        binding=item[4],raw_answer=answer.decoded_text,raw_token_ids=tokens,
                        EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,
                        generation_trace=trace,legacy_output=consumer['legacy_output']))
                    write('PROGRESS.json',dict(phase='GENERATE',arm=arm,owner=task['order'],
                        generated=COUNTS['generations'],counts=COUNTS,time=time.time()))
            finally:
                hook.detach()
    assert COUNTS['generations'] == 207
    write('GENERATION_COMPLETE.json',dict(time=time.time(),counts=COUNTS,all_candidates_before_evaluation=True,
        scoring='PENDING; full answers and immutable consumer bindings are durable'))
    fh.remove();vh.remove()
    write('WORKER_COMPLETE.json',dict(time=time.time(),counts=COUNTS,
        cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException as error:
        code = 1
        write('FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),counts=COUNTS,time=time.time()))
        raise
    finally:
        write('WORKER_EXIT.json',dict(code=code,counts=COUNTS,time=time.time()))
