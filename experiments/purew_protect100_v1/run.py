"""One frozen KL-protected continuous trajectory; original146 is never modified."""
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
import common as c
import base_run as b
from logic import clocks, resume_check
from protection_math import preservation_kl, proposal, reference

R = c.RUN
b.ARM = 'K'
c.available = b.available
PREFIXES = (1, 50, 100)


class AdmissionRejected(Exception):
    pass


def teacher_batch(runtime, row, tokens, expected):
    import torch
    raw = runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])), row['question'], None)
    assert raw['image_sha256'] == expected['image_sha256']
    assert raw['input_ids'][0].tolist() == expected['prompt_ids']
    assert raw['attention_mask'][0].tolist() == expected['attention_mask']
    assert 0 < len(tokens) <= 128 and tokens[-1] == runtime.adapter.tokenizer.eos_token_id
    answer = torch.tensor([tokens], device=runtime.device, dtype=raw['input_ids'].dtype)
    ids = torch.cat([raw['input_ids'], answer], 1)
    attention = torch.cat([raw['attention_mask'], torch.ones_like(answer)], 1)
    labels = torch.cat([torch.full_like(raw['input_ids'], -100), answer], 1)
    embeds, attention, positions, labels = runtime._expand_multimodal(
        raw_input_ids=ids, attention_mask=attention, labels=labels, images=raw['images'])
    assert embeds.dtype == torch.float16 and labels[0, labels[0] != -100].tolist() == tokens
    kwargs = dict(inputs_embeds=embeds.detach().cpu(), attention_mask=attention.detach().cpu(),
                  position_ids=positions.detach().cpu() if positions is not None else None,
                  labels=None, use_cache=False, return_dict=True)
    return kwargs, labels.detach().cpu()


def teachers(runtime, bindings):
    import torch
    rows = c.read(R/'private/BASIS_BUFFER.json')
    assert len(rows) == 61
    out = []
    for index, row in enumerate(rows):
        old = c.read(row['response_path'])
        expected = old['binding']['judge_input']
        assert runtime.generation_config == expected['generation']
        token_ids = old['R0']['raw_token_ids']
        batch = teacher_batch(runtime, row, token_ids, expected)
        cache_path = R/'private/teacher'/(str(index)+'.pt')
        lock = c.digest(dict(row=row, tokens=token_ids, runtime=next(iter(bindings.values()))['runtime'],
                             generation=runtime.generation_config, native_dtype='torch.float16'))
        if cache_path.exists():
            cache = torch.load(cache_path, map_location='cpu', weights_only=True)
            assert cache['lock'] == lock
        else:
            key = 'BASIS_' + str(index)
            bound = {key: expected}
            query = dict(row, opaque_Base_id=key)
            output = b.generate(runtime, bound, query, R/'private/base_basis'/(str(index)+'.json'),
                                dict(stage='UNEDITED_NATIVE_BASE_BASIS', index=index))
            if output['R0'] != old['R0']:
                c.write(R/'public/ADMISSION_REJECTED.json', dict(status='BLOCKED_BASE_OUTPUT_MISMATCH',
                        matched_before=index, required=61, old_scores_unchanged=True, new_labels=0))
                raise AdmissionRejected('Frozen BASIS Base text/tokens differ; no replacement or rejudge')
            with torch.no_grad():
                logits, targets = b.logits(runtime, batch)
            assert targets.tolist() == token_ids
            cache = dict(lock=lock, logits=logits.detach().cpu(), tokens=token_ids)
            b.save(cache_path, cache)
        assert cache['tokens'] == token_ids and not cache['logits'].requires_grad
        assert cache['logits'].shape == (len(token_ids), runtime.model.config.vocab_size)
        out.append((batch, cache['logits']))
        b.write_progress(status='REAL_VLM_ADMISSION_BASIS', matched=index+1, required=61)
        print('BASE_CHECK', index+1, 61, flush=True)
    return out


def admission(runtime, bindings, tasks, weight, shape, frozen, schedule):
    marker = R/'private/ADMISSION_STARTED.json'
    assert not marker.exists(), 'Admission attempt requires retained-state review; never replay'
    c.write(marker, dict(epoch=time.time(), required_Base_outputs=64))
    old_write = c.write
    def capture(path, data):
        if Path(path) == R/'public/REAL_VLM_ADMISSION.json':
            path = R/'private/BASE3_MECHANICAL.json'
        old_write(path, data)
    def mechanical(g, master):
        after, _, n = proposal(master, g, reference(schedule, 1, 1))
        b.COUNTS['committed_updates'] += 1; b.call('C'); b.counts()
        return after, dict(proposal_norm=float(n), gradient_norm=float(g.norm()), temporary=True)
    b.proposals = mechanical
    c.write = capture
    try:
        b.admission(runtime, bindings, tasks, weight, shape, frozen)
    finally:
        c.write = old_write
    protected = teachers(runtime, bindings)
    b.guard(runtime, weight, shape, frozen)
    result = dict(c.read(R/'private/BASE3_MECHANICAL.json'), basis61_outputs_exact=True,
                  teacher_native_FP16=True, source_only_buffer=True, required_Base_outputs_exact=64,
                  evaluation_training=False, temporary_updates=1, formal_updates=0)
    c.write(R/'public/REAL_VLM_ADMISSION.json', result)
    c.write(R/'private/ADMISSION_COMPLETE.json', dict(status='PASS', epoch=time.time()))
    b.write_progress(status='REAL_VLM_ADMISSION_PASS', formal_updates=0)
    return protected


def optimize(runtime, weight, master, components, teacher, target, sampled=False):
    import torch
    import torch.nn.functional as F
    gradient = torch.zeros_like(master); losses = []
    source = history = None
    for index, (batch, factor) in enumerate(components):
        weight.grad = None
        z, y = b.logits(runtime, batch); loss = F.cross_entropy(z, y)
        assert torch.isfinite(loss)
        b.COUNTS['backwards'] += 1; b.call('B'); b.counts()
        (128 * factor * loss).backward()
        assert weight.grad is not None
        g = weight.grad.float() / 128
        assert torch.isfinite(g).all()
        gradient.add_(g); losses.append(loss.detach())
        if sampled and index == 1: source = gradient.clone()
        if sampled and index == 2: history = g.clone()
        del z, loss, g
    weight.grad = None
    batch, base_logits = teacher
    z, _ = b.logits(runtime, batch)
    kl = preservation_kl(z, base_logits.to(runtime.device))
    assert torch.isfinite(kl)
    b.COUNTS['backwards'] += 1; b.call('B'); b.counts()
    (128 * kl).backward()
    assert weight.grad is not None
    protected = weight.grad.float() / 128
    assert torch.isfinite(protected).all()
    diagnostic = None
    if sampled:
        def cosine(a, bb):
            if a is None or bb is None or a.norm() == 0 or bb.norm() == 0: return None
            return float((a.flatten().dot(bb.flatten()) / (a.norm() * bb.norm())).detach())
        diagnostic = dict(source_gradient_norm=float(source.norm()),
                          history_gradient_norm=float(history.norm()) if history is not None else 0.,
                          KL_gradient_norm=float(protected.norm()), source_history_cosine=cosine(source, history),
                          source_KL_cosine=cosine(source, protected), history_KL_cosine=cosine(history, protected))
    gradient.add_(protected)
    before = weight.detach().float()
    after, native, norm = proposal(master, gradient, target)
    values = torch.stack([x.to(torch.float64) for x in losses + [kl.detach(), gradient.norm(), norm,
                          (native.float()-before).norm(), (native != weight.detach()).sum(),
                          (native.float()-after).norm()]]).cpu().tolist()
    event = dict(source_losses=values[:-6], BASIS_KL=values[-6], gradient_norm=values[-5],
                 proposal_norm=values[-4], target_proposal_norm=target, actual_native_norm=values[-3],
                 changed_elements=int(values[-2]), native_master_discrepancy=values[-1], diagnostics=diagnostic)
    assert abs(event['proposal_norm'] - target) <= max(1e-8, 1e-5 * target)
    with torch.no_grad(): weight.copy_(native)
    weight.grad = None
    return after, event


def save_state(active, master, weight, owner, step, completed, event, edit_initial):
    import torch
    import numpy as np
    nr = np.random.get_state()
    state = dict(lock=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')), owner=owner, step=step,
                 committed=completed, master=master.detach().cpu(), native=weight.detach().cpu(),
                 clocks=clocks(owner, step), rng=c.rng(), event=event, edit_initial=edit_initial.detach().cpu(),
                 numpy_rng=(nr[0], torch.from_numpy(nr[1].astype('int64')), nr[2], nr[3], nr[4]))
    b.save(active, state)
    return state


def panel(runtime, bindings, tasks, owner, prefix, insertion=False):
    ledger = c.read(R/'private/EVAL_LEDGER.json')
    if insertion:
        t = tasks[-1]
        ids = [t['edit_id']] + list(dict.fromkeys(q for e in t['events'] if e['task']=='T2G'
                      for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]))
    else:
        ids = list(dict.fromkeys(q for t in tasks for e in t['events'] for q in e['all_probe_query_ids']))
    mode = 'insertion_R0' if insertion else 'bank_R0'
    phase = dict(arm='K', mode=mode, owner=owner if insertion else 0, prefix=prefix,
                 protocol=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json')),
                 code=c.read(R/'private/GPU_SOURCE_VERSION.json')['commit'])
    folder = R/'private/outputs'/mode/str(phase['owner'])/str(prefix)
    for q in ids:
        target = folder/(q+'.json')
        prior = R/'private/outputs/insertion_R0'/str(prefix)/str(prefix)/(q+'.json')
        if not insertion and not target.exists() and prior.exists():
            d = c.read(prior)
            assert d['binding']['phase']['prefix'] == prefix and d['binding']['input'] == ledger['queries'][q]
            d['binding']['phase'] = phase
            c.write(target, d)
        else:
            b.generate(runtime, bindings, ledger['queries'][q], target, phase)
    c.write(folder/'COMPLETE.json', dict(phase=phase, outputs=len(ids)))


def trajectory(runtime, bindings, tasks, weight, original, shape, frozen, protected, schedule):
    import torch
    import numpy as np
    native = [b.batch_cpu(runtime, t) for t in tasks]
    active = R/'private/active/bank_R0.pt'
    units = {int(p.stem): c.read(p)['step'] for p in (R/'private/units/bank_R0').glob('*.json')}
    state = torch.load(active, map_location='cpu', weights_only=True) if active.exists() else None
    resume_check(state, units)
    if state:
        assert state['lock'] == c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
        assert torch.equal(state['native'], state['master'].half())
        c.restore_rng(state['rng']); nr = state['numpy_rng']
        np.random.set_state((nr[0], nr[1].numpy().astype('uint32'), nr[2], nr[3], nr[4]))
        with torch.no_grad(): weight.copy_(state['native'].to(runtime.device))
        c.write(R/'private/steps/bank_R0'/str(state['owner'])/(str(state['step'])+'.json'), state['event'])
    else:
        assert not list((R/'private/steps/bank_R0').glob('*/*.json')), 'Unsaved executed work needs explicit review'
    master = state['master'].to(runtime.device) if state else original.float().clone()
    completed = state['committed'] if state else 0
    for owner, task in enumerate(tasks, 1):
        done = R/'private/units/bank_R0'/(str(owner)+'.json')
        if done.exists(): continue
        first = state['step'] if state and state['owner'] == owner else 0
        if not first:
            from m3bench_repro.editors.llava_runtime import seed_everything
            seed_everything(task['seed'])
        fits = [b.batch_cpu(runtime, task, q) for q in task['fit_questions']]
        edit_initial = weight.detach().float().clone()
        if state and state['owner'] == owner:
            edit_initial = state['edit_initial'].to(runtime.device)
        weight.requires_grad_(True)
        for step in range(first+1, 161):
            c.budget(); started = time.perf_counter()
            clock = clocks(owner, step)
            batches = [(native[owner-1], .5), (fits[(step-1)%4], .5)]
            if clock['history'] is not None: batches.append((native[clock['history']], .5))
            master, event = optimize(runtime, weight, master, batches, protected[clock['protection']],
                                     reference(schedule, owner, step), step in (1, 40, 80, 160))
            completed += 1
            b.COUNTS['committed_updates'] += 1; b.call('C'); b.counts()
            event.update(owner=owner, step=step, mode='bank_R0', clocks=clock,
                         C_native_norm=schedule['native_step_norms'][completed-1],
                         edit_native_net_displacement=float((weight.detach().float()-edit_initial).norm()),
                         optimization_seconds=time.perf_counter()-started)
            if step%20 == 0 or b.STOP_REQUESTED:
                state = save_state(active, master, weight, owner, step, completed, event, edit_initial)
            c.write(R/'private/steps/bank_R0'/str(owner)/(str(step)+'.json'), event)
            if b.STOP_REQUESTED: raise SystemExit('Cooperative stop: native/master/RNG and clocks saved')
            if step%20 == 0:
                b.guard(runtime, weight, shape, frozen)
                b.write_progress(status='TRAINING', arm='K', owner=owner, step=step, committed=completed)
                print('TRAIN', owner, step, completed, flush=True)
        weight.requires_grad_(False); b.guard(runtime, weight, shape, frozen)
        panel(runtime, bindings, tasks[:owner], owner, owner, True)
        if owner in PREFIXES: panel(runtime, bindings, tasks[:owner], 0, owner)
        c.write(done, dict(status='COMPLETE', step=160, committed=completed))
    assert completed == 16000
    b.save(R/'private/final.pt', dict(native=weight.detach().cpu(), schema=shape,
           parameters=sum(p.numel() for p in runtime.model.parameters()), lock=c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))))
    c.write(R/'private/TRAINING_GENERATION_COMPLETE.json', dict(status='COMPLETE', formal_updates=completed))


def gpu():
    import torch
    signal.signal(signal.SIGTERM, b.request_stop); signal.signal(signal.SIGINT, b.request_stop)
    tasks = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']
    schedule = c.read(R/'private/C_REFERENCE.json')
    with c.lease(5):
        runtime, bindings = c.load(5)
        def forward(*_): b.COUNTS['forwards'] += 1; b.call('F')
        h = runtime.model.register_forward_pre_hook(forward)
        try:
            weight = runtime.get_module(c.LAYER).weight; original = weight.detach().clone()
            shape = b.schema(runtime.model)
            frozen = {n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            if not (R/'public/REAL_VLM_ADMISSION.json').exists():
                protected = admission(runtime, bindings, tasks, weight, shape, frozen, schedule)
            else:
                assert c.read(R/'public/REAL_VLM_ADMISSION.json')['status'] == 'PASS'
                protected = teachers(runtime, bindings)
            trajectory(runtime, bindings, tasks[:100], weight, original, shape, frozen, protected, schedule)
        finally: h.remove(); b.counts()


def reload():
    import torch
    with c.lease(5):
        runtime, bindings = c.load(5)
        def forward(*_): b.COUNTS['forwards'] += 1; b.call('F')
        h = runtime.model.register_forward_pre_hook(forward)
        try:
            saved = torch.load(R/'private/final.pt', map_location='cpu', weights_only=True)
            assert b.schema(runtime.model) == saved['schema']
            assert sum(p.numel() for p in runtime.model.parameters()) == saved['parameters']
            weight = runtime.get_module(c.LAYER).weight
            with torch.no_grad(): weight.copy_(saved['native'].to(runtime.device))
            assert not any(p.requires_grad for p in runtime.model.parameters())
            assert not runtime.get_module(c.LAYER)._forward_hooks and not runtime.get_module(c.LAYER)._forward_pre_hooks
            task = c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][99]
            ledger = c.read(R/'private/EVAL_LEDGER.json')
            ids = [task['edit_id'], next(e['all_probe_query_ids'][0] for e in task['events'] if e['task']=='T2G')]
            for q in ids:
                d = b.generate(runtime, bindings, ledger['queries'][q], R/'private/reload'/(q+'.json'), dict(stage='FRESH_NATIVE_RELOAD'))
                assert d['R0'] == c.read(R/'private/outputs/bank_R0/0/100'/(q+'.json'))['R0']
            c.write(R/'public/NATIVE_RELOAD.json', dict(status='PASS', two_outputs_exact=True,
                    added_parameters=0, schema_exact=True, final_native_bytes=(R/'private/final.pt').stat().st_size))
            owned = [R/'private/final.pt', R/'private/active/bank_R0.pt'] + list((R/'private/teacher').glob('*.pt'))
            c.write(R/'private/FINAL_CONSUMED.json', dict(reason='All panels durable and native fresh reload exact',
                    owned_files=len(owned), bytes=sum(p.stat().st_size for p in owned if p.exists()), historical_deleted=0))
            for p in owned:
                assert not p.is_symlink() and R in p.parents
                if p.exists(): p.unlink()
        finally: h.remove(); b.counts()


if __name__ == '__main__':
    try:
        {'gpu': gpu, 'reload': reload}[os.environ['ACTION']]()
        c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'), dict(exit_code=0, epoch=time.time()))
    except BaseException as error:
        b.counts()
        c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),
                dict(error=repr(error), traceback=traceback.format_exc(), counts=b.COUNTS, epoch=time.time()))
        raise
