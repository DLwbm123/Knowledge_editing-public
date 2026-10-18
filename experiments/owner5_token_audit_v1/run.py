"""Two-state, one-input teacher-forcing audit; no training or generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

P = Path(os.environ['PARENT_RUN'])
SOURCE = Path(os.environ['AUDIT_SOURCE'])
OUT = Path(os.environ['OUTPUT_RUN'])
sys.path[:0] = [str(P/'private/tools'), str(P/'private/source'), str(P/'private/official_llava')]
import torch
import torch.nn.functional as F
import module_run as r
from scripts.medtrace.run_realmodel_core import load_real_runtime
from prefix_cache import PrefixCache


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def main():
    start = time.time()
    with (OUT/'ATTEMPT').open('x') as f:
        f.write(str(os.getpid()))
    torch.use_deterministic_algorithms(True)
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'))
    runtime.model.eval()
    assert not any(p.requires_grad for p in runtime.model.parameters())
    assert not runtime.get_module(r.c.LAYER)._forward_hooks
    task = read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    row = read(SOURCE/'CONSTRUCT.json')[1]
    assert row['query_id'] == 'negative_construct' and row['reference'] == 'no'
    bindings = read(P/'private/EVAL_BINDINGS.json')
    runtime.model.half()
    item = r.s.prepare(runtime, bindings, task, [row], True)[0]
    _, cpu_batch, raw, _, binding = item
    r.old.fp32(runtime)
    batch = r.old.batch_on(runtime, cpu_batch)
    assert tuple(batch.target_token_ids) == (708, 28705, 2)
    old_binding = read(SOURCE/'bindings/negative_construct.json')
    for key in ('prompt_ids', 'image_sha256', 'question', 'reference'):
        assert binding[key] == old_binding[key], key
    prefix = raw['input_ids'][0].tolist()
    candidates = {}
    for word in ('no', 'yes', 'No', 'Yes'):
        ids = runtime._tokenize_prompt(raw['prompt']+' '+word)[0].tolist()
        assert ids[:len(prefix)] == prefix and len(ids) == len(prefix)+1
        candidates[word] = ids[-1]
    initial = r.p.tr.TT4(task['seed'], 8, 8).state_dict()
    saved = r.p.load_state(P/'private/candidates/C/5.pt')['states']['BASE']
    assert all(torch.equal(initial[k], saved[k]) for k in initial)
    base = r.mm.canonicalize(initial)[0]
    expert = r.p.expert(base, task['seed'], runtime.device)
    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
    positions = torch.where(batch.labels[0] != -100)[0]
    assert positions.tolist() == list(range(int(positions[0]), int(positions[0])+3))
    labels = batch.labels[0, positions]
    write('INPUT.json', dict(binding=binding, seed=task['seed'], candidates=candidates,
          raw_labels_reconstructed=[[-100]*len(prefix)+list(batch.target_token_ids)], labels=batch.labels.cpu().tolist(),
          attention_mask=batch.forward_kwargs()['attention_mask'].cpu().tolist(), target_positions=positions.tolist(),
          prediction_positions=(positions-1).tolist(), initial_state_exact=True))
    write('PROCESS.json', dict(commands=subprocess.check_output(
          ['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True),
          gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)))
    counts = dict(decoder_forwards=0, backwards=0, updates=0, generations=0)
    def count(*_):
        counts['decoder_forwards'] += 1
        assert counts['decoder_forwards'] <= 2
    handle = runtime.model.register_forward_pre_hook(count)
    results = {}
    try:
        with PrefixCache(runtime, [batch]):
            for state in ('NATIVE_BASE', 'ZERO_EXPERT_BASE'):
                if state == 'ZERO_EXPERT_BASE':
                    hook.attach()
                    hook.set_teacher_routing(batch.labels)
                captured = {}
                def capture(_module, _inputs, output):
                    captured['logits'] = output.logits[0, positions-1].detach().float()
                capture_handle = runtime.model.register_forward_hook(capture)
                try:
                    with torch.no_grad():
                        mean = float(runtime.compute_loss(batch))
                finally:
                    capture_handle.remove()
                logits = captured['logits']
                losses = F.cross_entropy(logits, labels, reduction='none')
                cpu_losses = F.cross_entropy(logits.cpu(), labels.cpu(), reduction='none')
                parity_delta = float(losses.mean())-mean
                logp = logits[0].log_softmax(-1)
                values = {w: float(logp[i]) for w, i in candidates.items()}
                lower = values['no']-values['yes']
                capital = values['No']-values['Yes']
                combined = float(torch.logsumexp(logp[[candidates['no'],candidates['No']]],0)
                                 - torch.logsumexp(logp[[candidates['yes'],candidates['Yes']]],0))
                result = dict(mean_nll=mean, token_losses=losses.tolist(),
                              decomposition_delta=parity_delta, decomposition_pass=abs(parity_delta)<2e-6,
                              cpu_token_losses=cpu_losses.tolist(), cpu_decomposition_delta=float(cpu_losses.mean())-mean,
                              format_loss_share=float(losses[1:].sum()/losses.sum()),
                              candidate_log_probabilities=values,
                              margin_lower=lower, margin_capital=capital, margin_combined=combined)
                results[state] = result
                torch.save(dict(logits=logits.cpu(), labels=labels.cpu(), positions=positions.cpu()), OUT/(state+'.pt'))
                write(state+'.json', result)
                if state == 'ZERO_EXPERT_BASE':
                    hook.detach()
    finally:
        handle.remove()
    historical = read(SOURCE/'BASE_CONSTRUCT.json')['NLL']
    assert counts['decoder_forwards'] == 2
    write('RESULTS.json', dict(states=results, historical_mean_nll=historical,
          historical_difference=results['ZERO_EXPERT_BASE']['mean_nll']-historical,
          counts=counts, seconds=time.time()-start, status='COMPLETE', automatic_next_run=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        write('FAILURE.json', dict(traceback=traceback.format_exc()))
        raise
