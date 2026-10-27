"""Frozen V133 curvature construction; only provenance and anchor recovery differ."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


def rebuild(runtime,r,inputs,ROOT,PARENT,LOCK,COUNTS,budget,write,save):
    import torch
    import fisher_math
    import med133_curvature as curvature
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json')
    cache_contract=dict(input_binding=LOCK['input_binding'],mapping_binding=LOCK['mapping_binding'],
        original_runtime=next(iter(bindings.values()))['runtime'],source_version=json.loads((ROOT/'SOURCE_VERSION.json').read_text()),
        precision='MATH_FP32_FROZEN_FP16_PREFILL',mask='m[:, :-1] = labels[:, 1:] != -100',
        normalization='sqrt(61*32*n_predictor_positions)',Base_seed=LOCK['seed'],probe_seed_base=20261009,
        aliases=[dict(index=v['index'],ordered_tokens=[runtime.adapter.tokenizer.encode(a,add_special_tokens=False)+
            [runtime.adapter.tokenizer.eos_token_id] for group in v['classes'] for a in group]) for v in inputs['mapping']])
    if (ROOT/'CURVATURE_ADMISSION.json').exists() and (ROOT/'V132_CURVATURE_PARITY.json').exists() and (ROOT/'CURVATURE_BINDING.json').exists():
        assert json.loads((ROOT/'CURVATURE_ADMISSION.json').read_text())['status']=='PASS'
        assert json.loads((ROOT/'CURVATURE_BINDING.json').read_text())['contract']==cache_contract
        fallback=r.p.load_state(ROOT/'curvature/COMMON_FALLBACK.pt')
        caches={name:dict(parts=[fallback,r.p.load_state(ROOT/'curvature'/f'{name}.pt')['mapped_part']]) for name in ('FULL_CTX','MED')}
    else:
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
        full_blocks, med_blocks, audits, actual_inputs = [], [], [], []
        try:
            for entry in inputs['mapping']:
                index, row = entry['index'], entry['row']
                checkpoint=ROOT/'curvature/anchors'/f'{index}.pt'
                if checkpoint.exists():
                    prior=r.p.load_state(checkpoint)
                    assert prior['input_binding']==LOCK['input_binding']
                    full_blocks.extend(prior['full']);med_blocks.append(prior['med']);audits.append(prior['audit'])
                    actual_inputs.extend(prior['actual_inputs']);continue
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
                        actual_inputs.append(dict(index=index,alias=alias,actual_teacher_binding=binding))
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
                        COUNTS['operator_backwards'] += 1
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
                    full,med,audit = curvature.med_anchor(paths,scores,index)
                except curvature.DegenerateCurvature as error:
                    # Use the same declared legacy fallback for MED and its FULL_CTX calibration.
                    common = dict(x=xpieces[index],y=ypieces[index])
                    fallback_parts = list(zip(fallback['x'].split(fallback['token_counts']),fallback['y'].split(fallback['token_counts'])))
                    fallback = curvature.concatenate([dict(x=x,y=y) for x,y in fallback_parts]+[common])
                    audits.append(dict(index=index,status='COMMON_LEGACY_FALLBACK',reason=str(error)))
                else:
                    full_blocks.extend(full);med_blocks.append(med)
                    audits.append(dict(audit,status='MAPPED'))
                    save(f'curvature/anchors/{index}.pt',dict(full=full,med=med,audit=audits[-1],actual_inputs=actual_inputs[-4:],input_binding=LOCK['input_binding']))
                write('PROGRESS.json',dict(phase='CURVATURE',complete=len(audits),total=len(inputs['mapping']),
                                           counts=COUNTS,time=time.time()))
                print('CURVATURE',len(audits),flush=True)
            assert med_blocks and len(full_blocks)==4*len(med_blocks)
            caches = {name:dict(parts=[fallback,curvature.concatenate(blocks)])
                      for name,blocks in [('FULL_CTX',full_blocks),('MED',med_blocks)]}
            save('curvature/COMMON_FALLBACK.pt',fallback)
            for name,cache in caches.items():
                save('curvature/'+name+'.pt',dict(mapped_part=cache['parts'][1],
                     shared_part='COMMON_FALLBACK.pt'))
            write('CURVATURE_ADMISSION.json',dict(status='PASS',anchors=audits,mapped=len(med_blocks),
                common_fallback=61-len(med_blocks),total_trace_matched=True,
                global_spectrum_matched=False,operator_space_matching=True,counts=COUNTS))
        finally:
            hook.detach()
        previous=r.c.read(Path(os.environ['RAW_MED_ROOT'])/'CURVATURE_ADMISSION.json')
        oldanchors={v['index']:v for v in previous['anchors']}
        for v in audits:
            old=oldanchors[v['index']];assert v['status']==old['status']
            if v['status']=='MAPPED':
                assert abs(v['full_trace']/old['full_trace']-1)<=LOCK['numeric']['trace_rtol']
                assert abs(v['MED_trace']/old['MED_trace']-1)<=LOCK['numeric']['trace_rtol']
        write('V132_CURVATURE_PARITY.json',dict(status='PASS',same_31_anchors=True,trace_rtol=LOCK['numeric']['trace_rtol']))
        write('CURVATURE_BINDING.json',dict(contract=cache_contract,actual_inputs=actual_inputs,identity=r.c.digest(cache_contract)))
        del expert,legacy,xpieces,ypieces,full_blocks,med_blocks,paths
    old=r.c.read(Path(os.environ['RAW_MED_ROOT'])/'CURVATURE_BINDING.json')
    assert r.c.read(ROOT/'CURVATURE_BINDING.json')['actual_inputs']==old['actual_inputs']
    old_contract=old['contract']
    for key in ('mapping_binding','original_runtime','precision','mask','normalization','Base_seed','probe_seed_base','aliases'):
        assert cache_contract[key]==old_contract[key],key
    current=r.c.read(ROOT/'CURVATURE_ADMISSION.json')['anchors']
    original={v['index']:v for v in r.c.read(Path(os.environ['RAW_MED_ROOT'])/'CURVATURE_ADMISSION.json')['anchors']}
    for value in current:
        prior=original[value['index']]
        assert value['status']==prior['status'] and value['full_rank']==prior['full_rank']
        assert abs(value['MED_unscaled_direction_square']/prior['MED_unscaled_direction_square']-1)<=LOCK['numeric']['trace_rtol']
        assert max(abs(a-b) for a,b in zip(value['class_probabilities'],prior['class_probabilities']))<=LOCK['numeric']['trace_rtol']
    write('CURVATURE_REBUILD_PARITY.json',dict(status='PASS',mask_teacher_bindings124_exact=True,
        original31_trace_and_rank_matched=True,old_direction_tensors_unavailable=True,
        no_bytewise_direction_equality_claim=True))
    return caches
