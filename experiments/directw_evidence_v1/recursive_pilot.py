"""Bounded language-W adaptation of M-ORE; private frozen inputs via RUN_DIR."""
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback

import torch
from torch.func import functional_call
from experiments.alphaedit_stage17 import runtime as native
from experiments.alphaedit_stage17.run import configure, consume, dump, hooks, space
from experiments.alphaedit_vlm.core import answer_nll, prefix_inputs
from .recursive_locality import initial_inverse, propose


def pooled_key(model, module, inputs):
    found = []
    def collect(_, args):
        values = args[0][0].detach().float()
        mask = inputs.get('attention_mask')
        if mask is not None:
            values = values[mask[0].bool()]
        found.append(values.mean(0))
    handle = module.register_forward_pre_hook(collect)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    if len(found) != 1 or not torch.isfinite(found[0]).all():
        raise RuntimeError('invalid native pooled key')
    return found[0]


def run(c, root, save):
    configure()
    rt = native.load(c, root)
    model = rt.model
    layer = c['layer']
    name = f'model.layers.{layer}.mlp.down_proj.weight'
    module = model.get_submodule(name.rsplit('.', 1)[0])
    weight = module.weight
    original = weight.detach().clone()
    versions = {n: p._version for n, p in model.named_parameters() if n != name}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    def verify():
        if hooks(model) or versions != {n:p._version for n,p in model.named_parameters() if n != name}:
            raise RuntimeError('residual hooks or noneditable parameter mutation')
        if any(not torch.equal(b, buffers[n]) for n,b in model.named_buffers()):
            raise RuntimeError('buffer mutation')
    torch.manual_seed(c['seed'])
    a = torch.empty(c['rank'], weight.shape[1], device=weight.device, dtype=torch.float32)
    torch.nn.init.orthogonal_(a)
    orthogonality = float((a @ a.T - torch.eye(c['rank'], device=a.device)).abs().max())
    if orthogonality > 1e-4:
        raise RuntimeError('coordinate orthogonality failure')
    qids = list(dict.fromkeys(qid for task in c['tasks'] for qid in task['query_ids']))
    def panel(folder, ids):
        for j, qid in enumerate(ids, 1):
            save(dict(state='GENERATING',panel=str(folder.relative_to(root)),completed=j-1,intended=len(ids)))
            q = c['queries'][qid]
            inp, labels = native.prepared(rt, q)
            with torch.no_grad():
                nll = float(answer_nll(model(**inp).logits, labels))
            dump(folder/f'{j:03d}.private.json',dict(query_id=qid,nll=nll,**native.generated(rt,q)))
    base = root/'base';base.mkdir()
    panel(base, qids)
    first_write = None
    for arm in c['arms']:
        folder = root/arm;folder.mkdir()
        with torch.no_grad():
            weight.copy_(original)
        inverse = initial_inverse(c['rank'], c['ridge'], device=weight.device)
        b = torch.zeros(weight.shape[0], c['rank'], device=weight.device)
        for index, task in enumerate(c['tasks'], 1):
            space(root)
            save(dict(state='EDITING',arm=arm,completed=index-1,intended=len(c['tasks'])))
            q = c['queries'][task['edit_id']]
            inputs, labels = native.prepared(rt, q)
            prefix, _ = prefix_inputs((inputs, labels))
            # Persistent B and fixed A, merged from W0 each time to avoid accumulated rounding.
            candidate = b.detach().requires_grad_(True)
            merged = (original.float() + c['scale'] * (candidate @ a)).to(weight.dtype)
            loss = answer_nll(functional_call(model, {name:merged}, (), inputs).logits, labels)
            gradient, = torch.autograd.grad(loss * c['loss_scale'], candidate)
            gradient = gradient.detach() / c['loss_scale']
            if not torch.isfinite(gradient).all() or not float(gradient.norm()) > 0:
                raise RuntimeError('nonfinite or zero native gradient')
            # First propose the W update using P[t-1]; history is computed POST-write.
            direction, _ = propose(gradient, inverse, torch.zeros(c['rank'], device=a.device), c['lr'])
            before = weight.detach().float().clone()
            with torch.no_grad():
                b.add_(direction)
                weight.copy_((original.float() + c['scale'] * (b @ a)).to(weight.dtype))
            if not torch.isfinite(weight).all():
                raise RuntimeError('nonfinite merged W')
            z = a @ pooled_key(model, module, prefix)
            _, next_inverse = propose(gradient, inverse, z, c['lr'])
            if arm == 'recursive':
                inverse = next_inverse
            actual = weight.detach().float() - before
            with torch.no_grad():
                post_nll = float(answer_nll(model(**inputs).logits, labels))
            verify()
            first_parity = None
            if index == 1:
                if first_write is None:
                    first_write = weight.detach().cpu().clone()
                else:
                    first_parity = torch.equal(first_write, weight.detach().cpu())
                    if not first_parity:
                        raise RuntimeError('first edit differs despite identical P0')
            dump(folder/f'{index:03d}.RECEIPT.json',dict(index=index,arm=arm,nll_before=float(loss.detach()),
                 nll_after=post_nll,gradient_norm=float(gradient.norm()),actual_write_norm=float(actual.norm()),
                 changed_elements=int((actual!=0).sum()),pooled_key_norm=float(z.norm()),inverse_trace=float(inverse.trace()),
                 first_edit_control_parity=first_parity,frozen_parameters=True,frozen_buffers=True,hooks=0))
            state=folder/'resume.private.pt';tmp=state.with_suffix('.tmp')
            torch.save(dict(completed=index,b=b.cpu(),inverse=inverse.cpu(),a=a.cpu(),weight=weight.detach().cpu()),tmp);tmp.replace(state)
            dump(folder/f'{index:03d}.TARGET.private.json',dict(query_id=task['edit_id'],**native.generated(rt,q)))
            del candidate,merged,loss,gradient,direction,before,actual,inputs,labels,prefix
            gc.collect();torch.cuda.empty_cache()
        panel_dir=folder/'final';panel_dir.mkdir();panel(panel_dir,qids)
        matrix=folder/'active.private.pt';torch.save(weight.detach().cpu(),matrix)
        q=c['queries'][c['tasks'][-1]['edit_id']]
        expected=native.logits(rt,q);generation=native.generated(rt,q)
        save(dict(state='CLEAN_RELOAD',arm=arm,completed=len(c['tasks'])))
        result=consume(c,folder,{layer:matrix},q)
        error=float((expected-result['logits']).abs().max())
        if error or generation!=result['generation'] or result['hooks'] or result['algorithm_loaded']:
            raise RuntimeError('native clean reload mismatch')
        verify()
        dump(folder/'COMPLETE.json',dict(state='GENERATED_NOT_SEMANTICALLY_SCORED',edits=len(c['tasks']),
             final_queries=len(qids),reload_logit_error=error,reload_tokens_identical=True,orthogonality_error=orthogonality))
        # These exact generated files have no remaining registered weight consumers.
        size=matrix.stat().st_size+state.stat().st_size
        matrix.unlink();state.unlink()
        dump(folder/'DELETION.json',dict(files=2,bytes=size,retained_copy=False,
             consumers=['eight_ordered_edits','target_and_final_panel','native_clean_reload'],reconstruction='rerun frozen pilot'))
    dump(root/'COMPLETE.json',dict(state='GENERATED_NOT_SEMANTICALLY_SCORED',arms=c['arms'],edits_per_arm=len(c['tasks']),queries=len(qids)))
    save(dict(state='GENERATED_NOT_SEMANTICALLY_SCORED',arms_completed=len(c['arms'])))


def main():
    root=Path(os.environ['RUN_DIR']);c=json.loads((root/'CONFIG.private.json').read_text())
    def save(status):dump(root/'STATUS.json',dict(**status,updated_epoch=time.time()))
    def alarm(*_):raise TimeoutError('fixed pilot deadline reached; no automatic continuation')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(c['deadline_epoch']-time.time())))
    try:
        save(dict(state='LOADING'));run(c,root,save)
    except BaseException as exc:
        dump(root/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        save(dict(state='FAILED',type=type(exc).__name__));raise
