"""Frozen Stage17 single/sequential AlphaEdit workers; private inputs via environment."""
import gc
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import torch
from . import runtime as native
from .algebra import update
from experiments.alphaedit_vlm.core import capture, optimize_z, prefix_inputs


def dump(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');tmp.replace(path)


def configure():
    torch.manual_seed(20260912)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(8)


def hooks(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks) for m in model.modules())


def space(root):
    import shutil
    if shutil.disk_usage(root).free < 8*1024**3:raise RuntimeError('8 GiB storage reserve reached')


def statistics(c,root,save):
    save(dict(state='LOADING',completed=0))
    rt=native.load(c,root);model=rt.model;layers=c['statistics_layers']
    q=c['queries'][c['tasks'][0]['edit_id']]
    native.generated(rt,q) # Exact original prompt/image binding gate.
    frozen={n:p._version for n,p in model.named_parameters()}
    target=native.prepared(rt,q);reference=native.prepared(rt,c['tasks'][0]['reference'])
    for layer in (21,23):
        z,receipt=optimize_z(model,target,reference,layer,dict(c['hparams'],v_num_grad_steps=2))
        if not torch.isfinite(z).all() or receipt['trace'][0].get('gradient_norm',0)<=0:raise RuntimeError('native latent gradient gate failed')
    if hooks(model) or frozen!={n:p._version for n,p in model.named_parameters()}:raise RuntimeError('mechanics changed frozen Base')
    dump(root/'NATIVE_MECHANICS.json',dict(state='PASS',FP16=True,gradient_layers=[21,23],frozen_versions_unchanged=True,prompt_image_binding=True,hooks=0))
    del target,reference,z
    moments={l:torch.zeros(model.get_submodule(f'model.layers.{l}.mlp.down_proj').weight.shape[1],
                          model.get_submodule(f'model.layers.{l}.mlp.down_proj').weight.shape[1],device='cuda') for l in layers}
    captured={};handles=[];count=0
    class StopForward(Exception):pass
    for layer in layers:
        def collect(_,args,layer=layer):captured[layer]=args[0][0].detach().float()
        handles.append(model.get_submodule(f'model.layers.{layer}.mlp.down_proj').register_forward_pre_hook(collect))
    def stop(*args):raise StopForward()
    handles.append(model.get_submodule(f'model.layers.{max(layers)}').register_forward_hook(stop))
    try:
        with gzip.open(root.parent/'wikipedia.jsonl.gz','rt') as stream:
            for i,line in enumerate(stream):
                if i>=c['statistics_documents']:break
                text=json.loads(line)['text']
                ids=rt.adapter.tokenizer(text,return_tensors='pt',truncation=True,max_length=c['statistics_max_tokens'])['input_ids'].cuda()
                with torch.no_grad():
                    try:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                    except StopForward:pass
                    if set(captured)!=set(layers):raise RuntimeError('missing covariance captures')
                    count+=ids.shape[1]
                    for layer,keys in captured.items():moments[layer].addmm_(keys.T,keys)
                captured.clear()
                if (i+1)%25==0:save(dict(state='COLLECTING',completed=i+1,intended=c['statistics_documents'],tokens=count))
        if i+1<c['statistics_documents']:raise RuntimeError('statistics corpus too short')
    finally:
        for handle in handles:handle.remove()
    del rt,model,ids;gc.collect();torch.cuda.empty_cache()
    records=[]
    for layer in layers:
        save(dict(state='EIGENDECOMPOSITION',completed=c['statistics_documents'],layer=layer,tokens=count))
        covariance=moments.pop(layer)/count;covariance=(covariance+covariance.T)*.5
        values,vectors=torch.linalg.eigh(covariance)
        basis=vectors[:,values<c['hparams']['nullspace_threshold']].contiguous()
        if not torch.isfinite(values).all() or not torch.isfinite(basis).all():raise FloatingPointError('invalid eigensystem')
        sample=basis[:,::max(1,basis.shape[1]//32)]
        error=float((sample.T@sample-torch.eye(sample.shape[1],device='cuda')).abs().max()) if sample.shape[1] else 0
        if error>1e-3:raise RuntimeError('projection orthogonality failed')
        space(root);torch.save(basis.cpu(),root/f'basis-{layer}.private.pt')
        record=dict(layer=layer,rank=basis.shape[1],dimension=basis.shape[0],minimum=float(values.min()),maximum=float(values.max()),sample_orthogonality_error=error)
        dump(root/f'SPECTRUM_{layer}.json',dict(**record,eigenvalues=values.cpu().tolist()));records.append(record)
        del covariance,values,vectors,basis,sample;gc.collect();torch.cuda.empty_cache()
    dump(root/'READY.json',dict(state='READY',layers=records,tokens=count,documents=c['statistics_documents']))
    # The accepted cohort and masks remain frozen even if hardware causes Base output drift.
    save(dict(state='BASE_AUDIT_LOADING',completed=0));rt=native.load(c,root)
    drift=0
    with (root/'BASE_OUTPUTS.private.jsonl').open('w') as out:
        for i,(qid,q) in enumerate(c['queries'].items(),1):
            result=native.generated(rt,q)
            different=result['raw_token_ids']!=q['historical_base']['raw_token_ids'];drift+=different
            out.write(json.dumps(dict(query_id=qid,**result,token_drift=different))+'\n');out.flush()
            if i%10==0:save(dict(state='BASE_AUDIT',completed=i,intended=len(c['queries']),drift=drift))
    save(dict(state='COMPLETE',completed=len(c['queries']),drift=drift,statistics=records,tokens=count))


def consume(c,root,matrices,q):
    packet=root/'CONSUMER.private.json';output=root/'CONSUMER.private.pt'
    dump(packet,dict(config=c,matrices={str(l):str(p) for l,p in matrices.items()},query=q,output=str(output),root=str(root)))
    script='''import json,os,sys,torch
from pathlib import Path
from experiments.alphaedit_stage17 import runtime as native
torch.manual_seed(20260912);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(8)
p=json.loads(Path(os.environ['CONSUMER_PACKET']).read_text());rt=native.load(p['config'],p['root'])
for layer,path in p['matrices'].items():
 w=rt.model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight
 v=torch.load(path,map_location='cpu',weights_only=True)
 assert v.shape==w.shape and v.dtype==w.dtype
 with torch.no_grad():w.copy_(v.to(w.device))
count=sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks) for m in rt.model.modules())
torch.save(dict(logits=native.logits(rt,p['query']),generation=native.generated(rt,p['query']),hooks=count,
 algorithm_loaded=any(k in sys.modules for k in ['experiments.alphaedit_stage17.algebra','experiments.alphaedit_vlm.core'])),p['output'])
'''
    with (root/'consumer.private.log').open('a') as log:
        subprocess.run([sys.executable,'-'],input=script,text=True,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CONSUMER_PACKET=str(packet)),check=True,
            timeout=min(600,max(1,int(c['deadline_epoch']-time.time()))))
    result=torch.load(output,map_location='cpu',weights_only=False);output.unlink();return result


def edit(c,root,arm,save):
    sr=root.parent/'statistics';save(dict(state='WAITING_FOR_STATISTICS'))
    while not (sr/'READY.json').exists():
        if (sr/'FAILURE.json').exists():raise RuntimeError('statistics failed')
        time.sleep(5)
    save(dict(state='LOADING'));rt=native.load(c,root);model=rt.model;layers=c['arms'][arm]['layers']
    weights={l:model.get_submodule(f'model.layers.{l}.mlp.down_proj').weight for l in layers}
    originals={l:w.detach().cpu().clone() for l,w in weights.items()}
    bases={l:torch.load(sr/f'basis-{l}.private.pt',map_location='cpu',weights_only=True) for l in layers}
    names={f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
    versions={n:p._version for n,p in model.named_parameters() if n not in names}
    def restore():
        with torch.no_grad():
            for l,w in weights.items():w.copy_(originals[l].to(w.device))
    def verify():
        if hooks(model) or versions!={n:p._version for n,p in model.named_parameters() if n not in names}:raise RuntimeError('frozen model mutation or retained hooks')
    for phase in ('single','sequential'):
        folder=root/phase;folder.mkdir();restore()
        history={l:torch.empty(weights[l].shape[1],0,device='cuda') for l in layers}
        for index,task in enumerate(c['tasks'],1):
            started=time.monotonic();space(root)
            save(dict(state='EDITING',phase=phase,completed=index-1,intended=146,active=index))
            if phase=='single':restore();history={l:h[:,:0] for l,h in history.items()}
            q=c['queries'][task['edit_id']];target=native.prepared(rt,q);reference=native.prepared(rt,task['reference'])
            z,latent=optimize_z(model,target,reference,layers[-1],c['hparams'])
            inp,pos=prefix_inputs(target);writes=[]
            for offset,layer in enumerate(layers):
                k,_=capture(model,inp,layer,pos);_,current= capture(model,inp,layers[-1],pos)
                residual=(z-current)/(len(layers)-offset)
                basis=bases[layer].cuda();delta=update(k,residual,basis,history[layer],c['hparams']['L2'])
                w=weights[layer];before=w.detach().float().clone()
                with torch.no_grad():w.copy_((before+delta).to(w.dtype))
                actual=w.detach().float()-before
                if not torch.isfinite(w).all():raise FloatingPointError('nonfinite edited W')
                writes.append(dict(layer=layer,requested_norm=float(delta.norm()),actual_norm=float(actual.norm()),rounding_error=float((actual-delta).norm())))
                del basis,delta,before,actual
            # Official cache_c is updated with all POST-edit keys, after every layer write.
            if phase=='sequential':
                for layer in layers:
                    k,_=capture(model,inp,layer,pos);history[layer]=torch.cat((history[layer],k[:,None]),1)
            verify()
            matrices={l:folder/f'active-{l}.private.pt' for l in layers}
            for l,p in matrices.items():
                tmp=p.with_suffix('.tmp');torch.save(weights[l].detach().cpu(),tmp);tmp.replace(p)
            state=folder/'resume.private.pt';tmp=state.with_suffix('.tmp')
            torch.save(dict(completed=index,phase=phase,history={l:h.cpu() for l,h in history.items()},weights={l:w.detach().cpu() for l,w in weights.items()}),tmp);tmp.replace(state)
            # Every stored edit gets a native output; single gets its complete event panel.
            if phase=='single':qids=task['query_ids']
            elif index in c['prefixes']:qids=list(dict.fromkeys(qid for t in c['tasks'][:index] for qid in t['query_ids']))
            else:qids=[task['edit_id']]
            if task['edit_id'] not in qids:qids.insert(0,task['edit_id'])
            save(dict(state='GENERATING',phase=phase,completed=index-1,active=index,queries=len(qids)))
            generated={}
            for qid in qids:generated[qid]=native.generated(rt,c['queries'][qid])
            dump(folder/f'{index:03d}.OUTPUT.private.json',dict(index=index,edit_id=task['edit_id'],outputs=generated,
                 active_targets=c['active_targets'].get(str(index),{}) if phase=='sequential' else {task['edit_id']:q['reference']}))
            expected=native.logits(rt,q);gc.collect();torch.cuda.empty_cache()
            # Native fresh-process reload on every single edit and registered sequence prefix.
            parity=None
            if phase=='single' or index in c['prefixes']:
                save(dict(state='CLEAN_RELOAD',phase=phase,completed=index-1,active=index))
                result=consume(c,folder,matrices,q);difference=float((expected-result['logits']).abs().max())
                if difference!=0 or generated[task['edit_id']]!=result['generation'] or result['hooks'] or result['algorithm_loaded']:
                    raise RuntimeError('independent native reload mismatch')
                parity=dict(logit_max_error=difference,generation_identical=True,hooks=0,algorithm_loaded=False)
            verify()
            dump(folder/f'{index:03d}.RECEIPT.json',dict(state='COMPLETE',index=index,phase=phase,queries=len(qids),latent=latent,writes=writes,
                 clean_reload=parity,history_columns={str(l):h.shape[1] for l,h in history.items()},seconds=time.monotonic()-started))
            if phase=='single':
                files=list(matrices.values())+[state];size=sum(p.stat().st_size for p in files)
                for p in files:p.unlink()
                dump(folder/f'{index:03d}.DELETION.json',dict(files=len(files),bytes=size,consumers=['panel_generation','native_reload'],retained_copy=False,reconstruction='rerun frozen independent edit'))
            del target,reference,z,inp,expected;gc.collect();torch.cuda.empty_cache()
        if phase=='sequential':
            files=list(matrices.values())+[state];size=sum(p.stat().st_size for p in files)
            for p in files:p.unlink()
            dump(folder/'DELETION.json',dict(files=len(files),bytes=size,consumers=['all146insertions','all4prefixes','native_reload'],retained_copy=False,reconstruction='rerun frozen full ordered sequence'))
        dump(folder/'COMPLETE.json',dict(state='GENERATED_NOT_SCORED',completed=146,prefixes=c['prefixes'] if phase=='sequential' else None))
    restore();save(dict(state='COMPLETE',single=146,sequential=146,scoring='PENDING_ASTRA_HIGH'))


def main():
    configure();root=Path(os.environ['RUN_DIR']);c=json.loads((root.parent/'CONFIG.private.json').read_text());started=time.time()
    def alarm(*args):raise TimeoutError('frozen campaign wall deadline reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(c['deadline_epoch']-time.time())))
    def save(status):dump(root/'STATUS.json',dict(**status,elapsed_seconds=time.time()-started,updated_epoch=time.time()))
    try:
        if os.environ['WORKER_MODE']=='statistics':statistics(c,root,save)
        else:edit(c,root,os.environ['WORKER_MODE'],save)
    except BaseException as error:
        dump(root/'FAILURE.json',dict(state='FAILED',error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise


if __name__=='__main__':main()
