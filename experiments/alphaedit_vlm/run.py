"""Finite background workers. Config/data/output paths arrive via environment.

GPU 5 estimates statistics; GPU 6/7 consume them for independent single edits.
No scientific retries, parameter search, external judge, or follow-on campaign.
"""
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch

from experiments.directw_evidence_v1.native_io import load, prepare, generation_snapshot
from .core import answer_nll, capture, optimize_z, prefix_inputs, write_edit


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def configure():
    torch.manual_seed(20261003)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)


def hooks_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks)
               for m in model.modules())


def snapshot(model, tokenizer, processor, row):
    prepared = prepare(model, tokenizer, processor, row)
    with torch.no_grad():
        logits = model(**prepared[0]).logits.float()
    labels = prepared[1]
    mask = (labels[0,1:]!=-100) & (labels[0,1:]!=tokenizer.eos_token_id)
    logp = logits[0,:-1][mask].log_softmax(-1).cpu()
    ids = labels[0,1:][mask].cpu()
    score = logp.gather(1, ids[:,None]).mean().item()
    if not torch.isfinite(logp).all():
        raise FloatingPointError('nonfinite probe')
    return score, logp


def stats(config, root, save):
    status = {'state':'LOADING', 'completed':0, 'intended':len(config['statistics_rows'])}
    save(status)
    model, tokenizer, processor = load(config)
    layers = config['statistics_layers']
    before_versions = {n:p._version for n,p in model.named_parameters()}
    # Reserved mechanical rows are excluded from targets/probes and statistics.
    smoke = [prepare(model,tokenizer,processor,r) for r in config['smoke_rows']]
    hp = dict(config['hparams'], v_num_grad_steps=2)
    for layer in sorted({config['arms']['single']['layers'][-1], config['arms']['multi']['layers'][-1]}):
        z, receipt = optimize_z(model, smoke[0], smoke[1], layer, hp)
        if receipt['trace'][0].get('gradient_norm',0)<=0 or not torch.isfinite(z).all():
            raise RuntimeError('native latent gradient qualification failed')
    if hooks_count(model) or before_versions!={n:p._version for n,p in model.named_parameters()}:
        raise RuntimeError('mechanical qualification changed frozen model or retained hooks')
    dump(root/'NATIVE_MECHANICS.json',dict(state='PASS', layers_checked=[21,23],
        frozen_parameter_versions_unchanged=True, hooks=0, latent_gradient_finite_nonzero=True,
        scope='two reserved rows, two objective evaluations, no physical weight update'))
    del smoke,z
    gc.collect();torch.cuda.empty_cache()
    moments = {}
    for layer in layers:
        width = model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight.shape[1]
        moments[layer] = torch.zeros(width,width,device='cuda',dtype=torch.float32)
    captures = {}; hooks=[]; count=0
    for layer in layers:
        def collect(_, args, layer=layer):
            captures[layer] = args[0][0].detach().float()
        hooks.append(model.get_submodule(f'model.layers.{layer}.mlp.down_proj').register_forward_pre_hook(collect))
    try:
        for i,row in enumerate(config['statistics_rows']):
            prepared = prepare(model,tokenizer,processor,row)
            inputs,_ = prefix_inputs(prepared)
            with torch.no_grad():
                model(**inputs)
                lengths = {value.shape[0] for value in captures.values()}
                if len(captures)!=len(layers) or len(lengths)!=1:
                    raise RuntimeError('incomplete layer statistics')
                count += lengths.pop()
                for layer,keys in captures.items():
                    moments[layer].addmm_(keys.T, keys)
            captures.clear()
            status.update(state='COLLECTING',completed=i+1,token_count=count)
            save(status)
            if (i+1)%32==0:print(json.dumps(status),flush=True)
    finally:
        for hook in hooks:hook.remove()
    del model,prepared,inputs,processor,tokenizer
    gc.collect();torch.cuda.empty_cache()
    records=[]
    for layer in layers:
        status.update(state='EIGENDECOMPOSITION',layer=layer);save(status)
        covariance=moments.pop(layer).div_(count)
        covariance=(covariance+covariance.T)*.5
        values,vectors=torch.linalg.eigh(covariance)
        selected=values<config['hparams']['nullspace_threshold']
        basis=vectors[:,selected].contiguous()
        if not torch.isfinite(values).all() or not torch.isfinite(basis).all():
            raise FloatingPointError('nonfinite eigensystem')
        sample_indices=torch.linspace(0,max(0,basis.shape[1]-1),min(32,basis.shape[1]),device='cuda').long()
        sample=basis[:,sample_indices]
        gram_error=float((sample.T@sample-torch.eye(sample.shape[1],device='cuda')).abs().max()) if sample.shape[1] else 0
        if gram_error>1e-3:raise RuntimeError('nullspace basis lost orthonormality')
        temporary=root/f'basis-{layer}.tmp'
        torch.save(basis.cpu(),temporary);temporary.replace(root/f'basis-{layer}.private.pt')
        record=dict(layer=layer,input_dimension=covariance.shape[0],nullspace_rank=basis.shape[1],
                    threshold=config['hparams']['nullspace_threshold'],sample_orthogonality_max_error=gram_error,
                    eigenvalue_min=float(values.min()),eigenvalue_max=float(values.max()),
                    negative_eigenvalues=int((values<0).sum()),token_count=count,
                    eigenvalues=values.cpu().tolist())
        dump(root/f'SPECTRUM_{layer}.json',record);records.append({k:v for k,v in record.items() if k!='eigenvalues'})
        del covariance,values,vectors,basis,sample
        gc.collect();torch.cuda.empty_cache()
    status.update(state='COMPLETE',layers=records);save(status)
    dump(root/'READY.json',dict(state='READY',layers=layers,token_count=count))


def consume(config, matrices, target_row, output):
    """Separate ordinary native loader; this code is embedded in a fresh process."""
    script = '''import json,os,torch
from pathlib import Path
from experiments.directw_evidence_v1.native_io import load,prepare,generation_snapshot
torch.manual_seed(20261003)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(8)
c=json.loads(Path(os.environ['CONSUMER_CONFIG']).read_text())
model,tok,processor=load(c['loader'])
for layer,path in c['matrices'].items():
 w=model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight
 value=torch.load(path,map_location='cpu',weights_only=True)
 assert value.shape==w.shape and value.dtype==w.dtype
 with torch.no_grad():w.copy_(value.to(w.device))
p=prepare(model,tok,processor,c['row'])
with torch.no_grad():logits=model(**p[0]).logits[0,:-1][p[1][0,1:]!=-100].float().cpu()
g=generation_snapshot(model,tok,p[2],c['row']['answer'],64)
hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks) for m in model.modules())
torch.save(dict(logits=logits,generation=g,hooks=hooks,editor_imported=any(k.startswith('experiments.alphaedit_vlm') for k in __import__('sys').modules)),c['output'])
'''
    packet=output.with_suffix('.config.private.json')
    dump(packet,dict(loader={k:config[k] for k in ('model_path','vision_path','native_source','attention_backend','native_deployment_dtype')},
                     matrices={str(k):str(v) for k,v in matrices.items()},row=target_row,output=str(output)))
    env=dict(os.environ,CONSUMER_CONFIG=str(packet))
    with output.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-'],input=script,text=True,stdout=log,stderr=subprocess.STDOUT,
                       env=env,check=True,timeout=min(600,max(1,int(config['deadline_epoch']-time.time()))))
    return torch.load(output,map_location='cpu',weights_only=False)


def edit(config, root, arm, save):
    stats_root=root.parent/'statistics'
    status=dict(state='WAITING_FOR_STATISTICS',completed=0,intended=len(config['cases']),cases=[])
    save(status)
    while not (stats_root/'READY.json').exists():
        if (stats_root/'FAILURE.json').exists():raise RuntimeError('statistics dependency failed')
        time.sleep(5)
    status['state']='LOADING';save(status)
    model,tokenizer,processor=load(config)
    layers=config['arms'][arm]['layers']
    weights={layer:model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight for layer in layers}
    originals={layer:w.detach().cpu().clone() for layer,w in weights.items()}
    bases={layer:torch.load(stats_root/f'basis-{layer}.private.pt',map_location='cpu',weights_only=True) for layer in layers}
    selected_names={f'model.layers.{layer}.mlp.down_proj.weight' for layer in layers}
    frozen_versions={n:p._version for n,p in model.named_parameters() if n not in selected_names}
    buffer_versions={n:b._version for n,b in model.named_buffers()}
    def restore():
        with torch.no_grad():
            for layer,w in weights.items():w.copy_(originals[layer].to(w.device))
    try:
        for i,pair in enumerate(config['cases']):
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            status.update(state='EDITING',active_case=i);save(status)
            target_row,reference_row=pair
            target=prepare(model,tokenizer,processor,target_row)
            reference=prepare(model,tokenizer,processor,reference_row)
            before=generation_snapshot(model,tokenizer,target[2],target_row['answer'],64)
            probes=config['diagnostic_probe_rows'][str(i)]
            probe_before=[snapshot(model,tokenizer,processor,r) for r in probes]
            with torch.no_grad():base_nll=float(answer_nll(model(**target[0]).logits,target[1]))
            target_z,latent=optimize_z(model,target,reference,layers[-1],config['hparams'])
            writes=write_edit(model,target,layers,target_z,bases,config['hparams'])
            if hooks_count(model) or frozen_versions!={n:p._version for n,p in model.named_parameters() if n not in selected_names} or buffer_versions!={n:b._version for n,b in model.named_buffers()}:
                raise RuntimeError('unexpected frozen parameter/buffer mutation or retained hooks')
            with torch.no_grad():
                logits=model(**target[0]).logits
                after_nll=float(answer_nll(logits,target[1]))
                expected_logits=logits[0,:-1][target[1][0,1:]!=-100].float().cpu()
            after=generation_snapshot(model,tokenizer,target[2],target_row['answer'],64)
            target_tokens=target[1][0][target[1][0]!=-100].cpu().tolist()
            probe_records=[]
            for row,(before_score,before_logp) in zip(probes,probe_before):
                after_score,after_logp=snapshot(model,tokenizer,processor,row)
                kl=float((before_logp.exp()*(before_logp-after_logp)).sum(-1).mean())
                probe_records.append(dict(role=row['diagnostic_role'],score_change=after_score-before_score,KL=kl))
            raw=dict(before=before,after=after,probe_records=probe_records,target_tokens=target_tokens)
            dump(root/f'{i}.OUTPUT.private.json',raw)
            matrices={layer:root/f'{i}.{layer}.matrix.private.pt' for layer in layers}
            for layer,path in matrices.items():torch.save(weights[layer].detach().cpu(),path)
            del logits,target_z,probe_before
            gc.collect();torch.cuda.empty_cache()
            status['state']='CLEAN_RELOAD';save(status)
            output=root/f'{i}.consumer.private.pt'
            result=consume(config,matrices,target_row,output)
            parity=float((expected_logits-result['logits']).abs().max())
            if parity!=0 or after!=result['generation'] or result['hooks']!=0 or result['editor_imported']:
                raise RuntimeError('independent native consumer mismatch')
            deleted_bytes=sum(p.stat().st_size for p in matrices.values())
            # Last consumers complete for these independent edits. No sequence uses them.
            for path in matrices.values():path.unlink()
            dump(root/f'{i}.DELETION.json',dict(files=len(matrices),bytes=deleted_bytes,
                consumers_complete=['post_edit_generation','all_probes','independent_native_reload'],
                retained_copy=False,reconstruction='rerun frozen independent edit from Base'))
            output.unlink() # Compact parity receipt and raw generated outputs are retained.
            restore()
            restored=all(torch.equal(w.detach().cpu(),originals[layer]) for layer,w in weights.items())
            if not restored:raise RuntimeError('Base restoration failed')
            record=dict(index=i,state='COMPLETE',base_nll=base_nll,edited_nll=after_nll,
                target_tokens_exact_before=before['tokens']==target_tokens,
                target_tokens_exact_after=after['tokens']==target_tokens,
                before_metrics=before['metrics'],after_metrics=after['metrics'],
                probes=probe_records,latent=latent,writes=writes,
                clean_reload=dict(max_logit_difference=parity,generation_identical=True,hooks=0,editor_imported=False),
                Base_restored=restored,deleted_matrix_bytes=deleted_bytes,elapsed_seconds=time.monotonic()-started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
            status['cases'].append(record);status.update(state='BETWEEN_CASES',completed=i+1);save(status)
            print(json.dumps(dict(arm=arm,case=i,exact_before=record['target_tokens_exact_before'],
                                  exact_after=record['target_tokens_exact_after'],elapsed=record['elapsed_seconds'])),flush=True)
            del target,reference,result,expected_logits
            gc.collect();torch.cuda.empty_cache()
        status.update(state='COMPLETE',final_Base_restored=True);save(status)
    finally:
        restore()


def main():
    root=Path(os.environ['RUN_DIR']);config=json.loads((root.parent/'CONFIG.private.json').read_text())
    mode=os.environ['WORKER_MODE'];started=time.monotonic()
    def save(status):
        status.update(elapsed_seconds=time.monotonic()-started,mode=mode)
        dump(root/'STATUS.json',status)
    def expired(*_):raise TimeoutError('frozen campaign deadline reached')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(max(1,int(config['deadline_epoch']-time.time())))
    configure()
    try:
        if mode=='statistics':stats(config,root,save)
        else:edit(config,root,mode,save)
    except BaseException as error:
        dump(root/'FAILURE.json',dict(error_type=type(error).__name__,message=str(error),
             elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
        raise


if __name__=='__main__':main()
